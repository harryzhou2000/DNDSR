#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <limits>
#include <climits>
#include <cstdlib>
#include <chrono>
#include <cstdio>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        using FiniteVolumeDetail::Sample;
    }

    RHSLocalDiagnostics::RHSLocalDiagnostics(std::vector<int> zones)
        : _zones(std::move(zones)), _forces(_zones.size())
    {
        DNDS_check_throw_info(std::is_sorted(_zones.begin(), _zones.end()) &&
                                  std::adjacent_find(_zones.begin(), _zones.end()) == _zones.end(),
                              "RHS diagnostic zones must be sorted and unique");
        Reset();
    }
    void RHSLocalDiagnostics::Reset()
    {
        _faceFallbacks = 0;
        std::fill(_forces.begin(), _forces.end(), Vector3{});
    }
    void RHSLocalDiagnostics::Merge(const RHSKernelDiagnostics &input)
    {
        // Standalone kernel users commonly need only the fallback count. Let an
        // empty aggregate adopt the immutable slot extent on first merge; named
        // BoundaryForce queries still require construction with the zone table.
        if (_zones.empty() && _forces.empty())
            _forces.resize(input._forces.size(), Vector3{});
        DNDS_check_throw_info(input._forces.size() == _forces.size(), "RHS diagnostic force-slot layout differs");
        _faceFallbacks += input._faceFallbacks;
        for (size_t slot = 0; slot < _forces.size(); ++slot)
            for (int d = 0; d < 3; ++d)
                _forces[slot][d] += input._forces[slot][d];
    }
    Vector3 RHSLocalDiagnostics::BoundaryForce(const std::vector<int> &zones) const
    {
        Vector3 force{};
        for (int zone : zones)
        {
            const auto found = std::lower_bound(_zones.begin(), _zones.end(), zone);
            DNDS_check_throw_info(found != _zones.end() && *found == zone, "force zone has no boundary condition");
            const auto slot = size_t(found - _zones.begin());
            for (int d = 0; d < 3; ++d)
                force[d] += _forces[slot][d];
        }
        return force;
    }
    FiniteVolume::FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation)
    {
        // Agree the actual selection before optional stages can diverge. The
        // metric plan retains this immutable policy for subsequent operators.
        R::CommunicationDetail::AgreePolicy(mesh->getMPI().comm, settings.implicit.collectiveChecks);
        R::CommunicationDetail::LocalStage(mesh->getMPI().comm, settings.implicit.collectiveChecks, [&]
                                           {
            settings.numerics.ValidatePositivity();
            DNDS_check_throw_info(settings.initialState.is_null() && settings.referenceState.is_null(),
                                  "resolve tagged physical inputs before constructing FiniteVolume directly");
            if (equation.HasRANS())
                DNDS_check_throw_info(std::isfinite(settings.numerics.wallDistanceFloor) && settings.numerics.wallDistanceFloor > 0 && mesh->HasNodeWallDist(),
                                      "RANS FV requires a published nodal wall field with a positive sampling floor");
            settings.numerics.limitedVR.Validate();
            DNDS_check_throw_info(settings.numerics.limiter != "Barth" ||
                                      settings.numerics.reconstruction == "gradient",
                                  "Barth needs direct gradient reconstruction");
            DNDS_check_throw_info(settings.numerics.limiter != "WBAP" ||
                                      settings.numerics.reconstruction != "constant",
                                  "WBAP needs gradient or variational reconstruction");
            DNDS_check_throw_info(!equation.HasMixture() || settings.numerics.limiter != "LVR",
                                  "mixture LVR policy is not connected yet");
            DNDS_check_throw_info(settings.numerics.limiter != "LVR" || (settings.numerics.reconstruction == "variational" && settings.numerics.cfv.maxOrder >= 2),
                                  "LVR requires degree2/3 variational reconstruction");
            DNDS_check_throw_info(settings.numerics.viscousGradient == "primitive" || settings.numerics.viscousGradient == "polynomial",
                                  "unknown Cauchy viscous gradient scheme"); }, "invalid Cauchy finite-volume setup");
        auto data = std::make_shared<Data>(std::move(mesh), ResolveSweepOrdering(settings), equation);
        if (equation.HasRANS())
            data->metrics.Communication()->LocalStage([&]
                                                      {
                data->wallDistance.emplace(WallDistanceSamples::Construct(data->mesh, settings.numerics.RHSIntegrationOrder(), settings.numerics.wallDistanceFloor));
                data->ransCellLength = data->metrics.BuildCellMaxLenScale(); });
        const bool limitedVR = settings.numerics.limiter == "LVR" && (settings.numerics.limitedVR.alphaMax > 0 || settings.numerics.limitedVR.contactAlphaMax > 0);
        for (const auto &[zone, condition] : data->settings.boundaries)
            data->forceZones.push_back(std::stoi(zone));
        std::sort(data->forceZones.begin(), data->forceZones.end());
        data->forceZones.erase(std::unique(data->forceZones.begin(), data->forceZones.end()), data->forceZones.end());
        for (index slot = 0; slot < index(data->forceZones.size()); ++slot)
            data->forceSlots.emplace(data->forceZones[size_t(slot)], slot);
        auto meshRead = data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto topology = meshRead.HostView();
        auto geometry = data->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        std::set<int> localZones;
        data->metrics.Communication()->LocalStage([&]
                                                  {
        data->offsets.resize(size_t(data->metrics.NumCell()) + 1, 0);
        index count = 0;
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            for (auto iFace : topology.CellFaces(iCell))
            {
                count += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                data->stencilFaces.push_back(iFace);
            }
            data->offsets[iCell + 1] = count;
        }
        std::sort(data->stencilFaces.begin(), data->stencilFaces.end());
        data->stencilFaces.erase(std::unique(data->stencilFaces.begin(), data->stencilFaces.end()), data->stencilFaces.end());
        data->samples.reserve(size_t(count));
        data->jacobianOffsets.resize(size_t(data->metrics.NumCell()) + 1);
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            data->jacobianOffsets[iCell] = index(data->jacobianSamples.size());
            const auto faces = topology.CellFaces(iCell);
            for (rowsize slot = 0; slot < rowsize(faces.size()); ++slot)
            {
                const auto iFace = faces[slot];
                const int side = topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
                const index other = topology.CellFaceOther(iCell, iFace, slot);
                DNDS_check_throw_info(other == UnInitIndex || (other >= 0 && other < data->metrics.NumCellProc()), "flux stencil has an unavailable neighbor cell");
                const index auxiliaryFace = index(std::lower_bound(data->stencilFaces.begin(), data->stencilFaces.end(), iFace) - data->stencilFaces.begin());
                BoundaryCondition condition;
                if (other == UnInitIndex)
                {
                    const int zone = topology.GetFaceZone(iFace);
                    condition = data->boundary.At(zone);
                    localZones.insert(zone);
                }
                real faceArea = 0;
                Geom::tPoint faceCenter = Geom::tPoint::Zero();
                for (index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                {
                    faceArea += metric.GetFaceWeight(q);
                    faceCenter += metric.GetFaceWeight(q) * metric.GetFacePoint(q);
                }
                DNDS_check_throw_info(faceArea > 0, "nonpositive viscous face area");
                faceCenter /= faceArea;
                const Geom::tPoint viscousDisplacement = other == UnInitIndex
                    ? Geom::tPoint(2 * (CFV::Detail::FacePointInCell(topology, iFace, side, faceCenter) - metric.GetCellBary(iCell)))
                    : Geom::tPoint(CFV::Detail::OtherPointInCell(topology, iFace, side, metric.GetCellBary(other)) - metric.GetCellBary(iCell));
                const real distanceScale = 2 * std::min(metric.GetCellVol(iCell), metric.GetCellVol(other == UnInitIndex ? iCell : other)) / faceArea;
                Sample jacobian{iCell, other, metric.FaceQuadratureBegin(iFace), side, topology.GetFaceZone(iFace),
                                metric.GetFaceMeanNorm(iFace), CFV::Detail::FacePointInCell(topology, iFace, side, metric.GetFaceCent(iFace)),
                                CFV::Detail::OtherPointInCell(topology, iFace, side, metric.GetCellCent(other == UnInitIndex ? iCell : other)),
                                CFV::Detail::OtherFrameInCell(topology, iFace, side), condition};
                jacobian.auxiliaryFace = auxiliaryFace;
                jacobian.face = iFace;
                jacobian.changesFrame = !(jacobian.rotation.array() == Geom::tGPoint::Identity().array()).all();
                if (side)
                    jacobian.normal = -jacobian.rotation * metric.GetFaceMeanNorm(iFace);
                data->jacobianSamples.push_back(std::move(jacobian));
                for (index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                {
                    Sample sample{iCell, other, q, side, topology.GetFaceZone(iFace), (side ? -1 : 1) * metric.GetFaceNorm(q), CFV::Detail::FacePointInCell(topology, iFace, side, metric.GetFacePoint(q)),
                                  CFV::Detail::OtherPointInCell(topology, iFace, side, metric.GetCellCent(other == UnInitIndex ? iCell : other)),
                                  CFV::Detail::OtherFrameInCell(topology, iFace, side), condition};
                    sample.auxiliaryFace = auxiliaryFace;
                    if (other == UnInitIndex &&
                        (condition.kind == BoundaryKind::SlipWall || condition.kind == BoundaryKind::NoSlipWall))
                        sample.forceSlot = data->forceSlots.at(sample.zone);
                    data->hasRotatedFrames = data->hasRotatedFrames || !(sample.rotation.array() == Geom::tGPoint::Identity().array()).all();
                    sample.face = iFace;
                    sample.changesFrame = !(sample.rotation.array() == Geom::tGPoint::Identity().array()).all();
                    const index entry = index(data->samples.size());
                    // Cached basis derivatives live in each parent's frame.
                    // Flux and neighbor state are mapped into this cell's frame.
                    if (side) sample.normal = -sample.rotation * metric.GetFaceNorm(q);
                    // Face-jump length: normal centroid separation, clamped
                    // by the two-cell volume/area scale. Retain this policy
                    // also for P0 convection (Euler instead uses full distance
                    // there). This is unrelated to geometric wall distance.
                    // Evaluate once, not during every residual iteration.
                    sample.viscousDistance = std::max(.25 * distanceScale, std::min(2 * distanceScale, std::abs(viscousDisplacement.dot(sample.normal))));
                    DNDS_check_throw_info(std::isfinite(sample.viscousDistance) && sample.viscousDistance > 0, "invalid polynomial viscous distance");
                    data->samples.push_back(std::move(sample));
                }
            }
        } });
        data->jacobianOffsets.back() = index(data->jacobianSamples.size());
        data->metrics.Communication()->LocalStage([&]
                                                  {
        data->jacobianColumns = std::make_shared<const std::vector<std::vector<index>>>(FiniteVolumeDetail::JacobianColumns(data->jacobianOffsets, data->jacobianSamples));
        // The assembled SGS/Jacobian writer visits each face incidence many
        // times for a frozen state. Resolve its dense column slot once at
        // setup instead of repeating a linear search in every matrix fill.
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            const auto &columns = (*data->jacobianColumns)[size_t(iCell)];
            for (index entry = data->jacobianOffsets[iCell]; entry < data->jacobianOffsets[iCell + 1]; ++entry)
            {
                auto &sample = data->jacobianSamples[size_t(entry)];
                if (sample.other == UnInitIndex || sample.other == iCell)
                    continue;
                const auto found = std::find(columns.begin(), columns.end(), sample.other);
                DNDS_check_throw_info(found != columns.end(), "Jacobian stencil has no canonical column slot");
                sample.jacobianColumn = index(found - columns.begin());
            }
        }
        data->boundaryZones.assign(localZones.begin(), localZones.end());
        for (auto iFace : data->stencilFaces)
        {
            const auto cells = topology.FaceCells(iFace);
            const index iCell = cells[0] >= 0 && cells[0] < data->metrics.NumCell() ? cells[0] : cells[1];
            DNDS_check_throw_info(iCell >= 0 && iCell < data->metrics.NumCell(), "auxiliary face has no owned parent");
            index entry = data->offsets[iCell];
            while (entry < data->offsets[iCell + 1] && data->samples[entry].point != metric.FaceQuadratureBegin(iFace)) ++entry;
            DNDS_check_throw_info(entry < data->offsets[iCell + 1], "auxiliary face missing its quadrature incidence");
            const index iAux = index(data->faceEntries.size());
            const auto other = data->samples[entry].other;
            (other == UnInitIndex || other < data->metrics.NumCell() ? data->auxiliaryCoreFaces : data->auxiliaryDependentFaces).push_back(iAux);
            real area = 0;
            for (index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                area += metric.GetFaceWeight(q);
            data->auxiliaryFaceArea.push_back(area);
            data->faceEntries.push_back(entry);
        }
        // Derived work lists retain the canonical face order of the original
        // face-centric reductions. Keep repeated incidences (e.g. a periodic
        // face whose two parents are the same cell); they contribute twice.
        data->auxiliaryCellOffsets.resize(size_t(data->metrics.NumCell()) + 1, 0);
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
            data->auxiliaryCellOffsets[iCell + 1] = data->auxiliaryCellOffsets[iCell] + index(topology.CellFaces(iCell).size());
        data->auxiliaryCellFaces.resize(size_t(data->auxiliaryCellOffsets.back()));
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            index cursor = data->auxiliaryCellOffsets[iCell];
            for (index entry = data->offsets[iCell]; entry < data->offsets[iCell + 1];)
            {
                const auto iAux = data->samples[entry].auxiliaryFace;
                data->auxiliaryCellFaces[cursor++] = iAux;
                const auto iFace = data->stencilFaces[iAux];
                entry += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
            }
            DNDS_check_throw_info(cursor == data->auxiliaryCellOffsets[iCell + 1], "auxiliary cell incidence count differs");
            std::sort(data->auxiliaryCellFaces.begin() + data->auxiliaryCellOffsets[iCell], data->auxiliaryCellFaces.begin() + cursor);
        }
        for (const auto iCell : data->partition.Core())
            for (index entry = data->auxiliaryCellOffsets[iCell]; entry < data->auxiliaryCellOffsets[iCell + 1]; ++entry)
            {
                const auto other = data->samples[data->faceEntries[data->auxiliaryCellFaces[entry]]].other;
                DNDS_check_throw_info(other == UnInitIndex || other < data->metrics.NumCell(), "auxiliary core depends on an input ghost");
            }
        const auto collectFaces = [&](const auto &cells, auto &faces)
        {
            for (const auto iCell : cells)
                for (index entry = data->auxiliaryCellOffsets[iCell]; entry < data->auxiliaryCellOffsets[iCell + 1]; ++entry)
                    faces.push_back(data->auxiliaryCellFaces[entry]);
            std::sort(faces.begin(), faces.end());
            faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
        };
        collectFaces(data->partition.Core(), data->rhsCoreFaces);
        collectFaces(data->partition.HaloDependent(), data->rhsDependentFaces);
        data->boundaryLayout=std::make_shared<Detail::BoundaryPointLayout>();
        data->boundaryLayout->mesh=data->mesh;
        data->boundaryLayout->stamp=data->mesh->versions.Stamp(R::Change::Topology|R::Change::Geometry|R::Change::Distribution|R::Change::Layout);
        data->boundaryLayout->points.reserve(data->boundaryEntries.size());
        for(const auto &[zone,boundary]:settings.boundaries)
            data->hasInterfaces=data->hasInterfaces||boundary.kind=="interface";
        for (auto &sample : data->jacobianSamples)
            if (sample.other == UnInitIndex)
            {
                sample.boundaryPoint = index(data->boundaryLayout->faces.size());
                data->boundaryLayout->faces.push_back({sample.cell, sample.face, -1, sample.zone,
                    {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]},
                    {sample.normal[0], sample.normal[1], sample.normal[2]}, sample.boundary.kind == BoundaryKind::Interface});
            }
        // Boundary trace rows belong to the FV face-quadrature publication,
        // not to variational reconstruction. Derive their stable order from
        // the canonical cell-incidence samples so P0/P1 need no Taylor plan.
        for (index entry = 0; entry < index(data->samples.size()); ++entry)
        {
            auto &sample=data->samples[entry];
            if (sample.other != UnInitIndex)
                continue;
            sample.boundaryPoint=index(data->boundaryEntries.size());
            data->boundaryEntries.push_back(entry);
            const bool interfaceRow = sample.boundary.kind == BoundaryKind::Interface;
            data->hasLocalInterfaceRows = data->hasLocalInterfaceRows || interfaceRow;
            data->boundaryLayout->points.push_back({sample.cell,sample.face,
                int(sample.point-metric.FaceQuadratureBegin(sample.face)),sample.zone,
                {sample.ownPoint[0],sample.ownPoint[1],sample.ownPoint[2]},
                {sample.normal[0],sample.normal[1],sample.normal[2]},interfaceRow});
        }
        data->gradientBoundaryPoints.resize(size_t(data->gradient.NumBoundaryQuadrature()));
        for(index iPoint=0;iPoint<data->gradient.NumBoundaryQuadrature();++iPoint)
        {
            const auto point=data->gradient.GetBoundaryQuadrature(iPoint);
            const auto globalPoint=metric.FaceQuadratureBegin(point.face)+point.quadrature;
            index entry=data->offsets[point.cell];
            while(entry<data->offsets[point.cell+1]&&data->samples[entry].point!=globalPoint) ++entry;
            DNDS_check_throw_info(entry<data->offsets[point.cell+1]&&data->samples[entry].boundaryPoint!=UnInitIndex,"gradient boundary point has no canonical boundary row");
            data->gradientBoundaryPoints[iPoint]=data->samples[entry].boundaryPoint;
        }
        if (settings.numerics.positivity || limitedVR)
        {
            std::vector<int> volumeOrders;
            if (settings.numerics.positivityVolumePoints) volumeOrders.push_back(Geom::Elem::INT_ORDER_MAX);
            if ((settings.equation.source == "chemical" || equation.HasRANS()) && std::find(volumeOrders.begin(), volumeOrders.end(), settings.numerics.RHSIntegrationOrder()) == volumeOrders.end())
                volumeOrders.push_back(settings.numerics.RHSIntegrationOrder());
            data->positivityOffsets.resize(size_t(data->metrics.NumCell()) + 1, 0);
            for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
            {
                data->positivityOffsets[iCell + 1] = data->positivityOffsets[iCell] + data->offsets[iCell + 1] - data->offsets[iCell];
                for (auto order : volumeOrders)
                    data->positivityOffsets[iCell + 1] += Geom::Elem::Quadrature(topology.GetCellElement(iCell), order).GetNumPoints();
            }
            const auto count = data->positivityOffsets.back();
            R::Array<index> cells(R::RowLayout::Dense(count, 1));
            R::Array<real> points(R::RowLayout::Dense(count, 3));
            {
                auto cellWrite = cells.Write(); auto pointWrite = points.Write();
                const auto c = cellWrite.View<R::FixedRows<1>>(); const auto p = pointWrite.View<R::FixedRows<3>>();
                Geom::tSmallCoords coords;
                for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
                {
                    index iPoint = data->positivityOffsets[iCell];
                    const auto append = [&](const Geom::tPoint &point)
                    {
                        c(iPoint, 0) = iCell;
                        for (int d = 0; d < 3; ++d)
                            p(iPoint, d) = point[d];
                        data->positivityDisplacements.push_back(point - metric.GetCellCent(iCell));
                        ++iPoint;
                    };
                    for (index entry = data->offsets[iCell]; entry < data->offsets[iCell + 1]; ++entry) append(data->samples[entry].ownPoint);
                    for (auto order : volumeOrders)
                    {
                        topology.GetCoordsOnCell(iCell, coords);
                        const auto element = topology.GetCellElement(iCell);
                        const Geom::Elem::Quadrature quadrature(element, order);
                        for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                            append(Geom::Elem::PPhysicsCoordD01Nj(coords, Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ)));
                    }
                }
            }
            if (data->reconstruction)
                data->positivitySamples.emplace(*data->reconstruction, cells, points, false);
            DNDS_check_throw_info(index(data->positivityDisplacements.size()) == count,
                                  "direct-P1 positivity geometry size differs");
        }
        if (limitedVR) data->limitedVRGeometry = std::make_shared<LimitedVRGeometry>(*data); });
        // Zone presence is a domain-global property; do not reject a condition
        // just because this partition has no wall faces. Only small BC IDs move.
        std::vector<int> zones(localZones.begin(), localZones.end()), counts(data->mesh->mpi.size), starts(counts.size() + 1, 0);
        int localCount = int(zones.size());
        MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, data->metrics.Communicator());
        for (index i = 0; i < index(counts.size()); ++i)
            starts[i + 1] = starts[i] + counts[i];
        std::vector<int> global(size_t(starts.back()));
        MPI_Allgatherv(zones.data(), localCount, MPI_INT, global.data(), counts.data(), starts.data(), MPI_INT, data->metrics.Communicator());
        data->boundary.ValidateUsed(std::set<int>(global.begin(), global.end()));
        _data = std::move(data);
    }
    const CFV::FiniteVolumePlan &FiniteVolume::Metrics() const { return _data->metrics; }
    R::TransferBinding FiniteVolume::BindHalo(CFV::FVField &field) const { return _data->BindHalo(field); }
    const EquationSystem &FiniteVolume::Equation() const { return _data->equation; }
    const DomainSettings &FiniteVolume::Settings() const { return _data->settings; }
    OptionalRef<const CFV::VariationalReconstructionPlan> FiniteVolume::PolynomialReconstruction() const
    {
        return _data->reconstruction ? OptionalRef<const CFV::VariationalReconstructionPlan>(*_data->reconstruction)
                                     : OptionalRef<const CFV::VariationalReconstructionPlan>();
    }
    const Geom::UnstructuredMesh &FiniteVolume::Mesh() const { return *_data->mesh; }
    const RigidMotion &FiniteVolume::Motion() const { return _data->motion; }

    Json FiniteVolume::ReconstructionMetadata() const
    {
        index components = 0;
        Json blocks = Json::array();
        for (const auto &block : _data->equation.Schema().blocks)
        {
            components += index(block.components.size());
            blocks.push_back({{"name", block.name}, {"components", block.components.size()}});
        }
        return {{"dimension", Metrics().Dimension()},
                {"family", _data->settings.numerics.reconstruction},
                {"coefficient_dofs", _data->reconstruction ? _data->reconstruction->NRecDOF() : 0},
                {"components", components},
                {"blocks", std::move(blocks)},
                {"coefficient_role", "unlimitedGuess"},
                {"cfv", _data->settings.numerics.cfv}};
    }
    namespace
    {
        template <class MeshView>
        FlowState CellAverage(const MeshView &mesh, index iCell, const FlowProfileSettings &profile,
                              const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame)
        {
            if (profile.kind == "constant")
                return frame.LocalState(gas.Conservative(primitive));
            Geom::tSmallCoords coords;
            mesh.GetCoordsOnCell(iCell, coords);
            const auto element = mesh.GetCellElement(iCell);
            constexpr int order = Geom::Elem::INT_ORDER_MAX;
            Geom::Elem::Quadrature quadrature(element, order);
            FlowState average{};
            real volume = 0;
            for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
            {
                const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                const Geom::tPoint point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                const real weight = Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                const auto value = frame.LocalState(ProfileValue(profile, primitive, gas, frame.WorldPoint({point[0], point[1], point[2]}), time));
                for (int j = 0; j < 5; ++j)
                    average[j] += weight * value[j];
                volume += weight;
            }
            for (auto &value : average)
                value /= volume;
            return average;
        }
    }
    void FiniteVolume::Initialize(EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive) const
    {
        _data->equation.ValidateStateSchema(state);
        DNDS_check_throw_info(!_data->equation.HasMixture() && !_data->equation.HasRANS() && state.ComponentNumber() == 5,
                              "analytic flow-profile initialization currently requires a pure flow5 equation");
        Metrics().Communication()->LocalStage([&]
                                              {
            auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow");
            const auto frame = _data->motion.At(0);
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                const auto average = CellAverage(mesh, iCell, profile, primitive, _data->equation.Gas(), 0, frame);
                for (int j = 0; j < 5; ++j)
                    flow(iCell, j) = average[j];
            } });
    }
    std::array<FlowState, 3> FiniteVolume::Error(const EquationDofs &state, const FlowProfileSettings &profile, const FlowState &primitive, real time) const
    {
        _data->equation.ValidateStateSchema(state);
        std::array<FlowState, 3> errors{};
        real volume = 0;
        Metrics().Communication()->LocalStage([&]
                                              {
            auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            auto geometry = Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            const auto read = state.ReadOwned();
            const auto flow = read.Block("flow");
            const auto frame = _data->motion.At(time);
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                const auto exact = CellAverage(mesh, iCell, profile, primitive, _data->equation.Gas(), time, frame);
                const real weight = metric.GetCellVol(iCell);
                volume += weight;
                for (int j = 0; j < 5; ++j)
                {
                    const real error = std::abs(flow(iCell, j) - exact[j]);
                    errors[0][j] += weight * error;
                    errors[1][j] += weight * error * error;
                    errors[2][j] = std::max(errors[2][j], error);
                }
            } });
        volume = MPI::AllreduceValue<MPI::Reduction::Sum>(volume, Metrics().Communicator());
        for (int i = 0; i < 3; ++i)
            if (i == 2)
                MPI::AllreduceInPlace<MPI::Reduction::Max>(errors[i], Metrics().Communicator());
            else
                MPI::AllreduceInPlace<MPI::Reduction::Sum>(errors[i], Metrics().Communicator());
        for (int j = 0; j < 5; ++j)
        {
            errors[0][j] /= volume;
            errors[1][j] = std::sqrt(errors[1][j] / volume);
        }
        return errors;
    }
    EquationAdmissibilityPolicy FiniteVolume::PrepareAdmissibility(
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        DNDS_check_throw_info(
            _data->equation.HasMixture() == bool(workers),
            "equation admissibility needs exactly its mixture worker pool");
        std::optional<EquationAdmissibilityPolicy> result;
        Metrics().Communication()->LocalStage([&]
                                              { result.emplace(
                                                    _data->equation, _data->settings.numerics.densityFloor,
                                                    _data->settings.numerics.temperatureFloor,
                                                    _data->settings.numerics.sensibleEnergyFloor,
                                                    _data->settings.numerics.pressureFloor,
                                                    _data->equation.HasRANS()
                                                        ? ResolveRANSStatePolicy(
                                                              _data->equation,
                                                              _data->settings.numerics.ransState)
                                                        : RANSStateView{},
                                                    workers ? workers->Acquire()
                                                            : Physics::Chemistry::ChemicalWorkerPool::Lease{}); });
        return std::move(*result);
    }
    const ThermalBounds &FiniteVolume::ThermalConstraints() const
    {
        return _data->FixedFlowThermalBounds();
    }
    bool FiniteVolume::Admissible(const EquationDofs &state) const
    {
        DNDS_check_throw_info(
            _data->defaultAdmissibility,
            "mixture finite-volume admissibility needs an explicitly prepared policy");
        return Admissible(state, *_data->defaultAdmissibility);
    }
    bool FiniteVolume::Admissible(
        const EquationDofs &state,
        const EquationAdmissibilityPolicy &policy) const
    {
        _data->equation.ValidateStateSchema(state);
        DNDS_check_throw_info(
            policy.PropertyKey() == _data->equation.PropertyKey(),
            "admissibility policy belongs to another equation material");
        const auto bounds = policy.View();
        DNDS_check_throw_info(
            bounds.thermal.components == state.ComponentNumber(),
            "admissibility policy and equation state widths differ");
        int valid = 1;
        Metrics().Communication()->LocalStage([&]
                                              {
            const auto read = state.ReadOwned();
            const auto layout = read.ComponentLayout();
            const auto values = layout.View();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                valid &= bounds.Admissible(
                    EquationPositivityDetail::PointRow<decltype(values)>{
                        values, iCell}); });
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, Metrics().Communicator());
        return valid;
    }
}
