#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Discretization/FiniteVolumeKernels.hpp"
#include "Cauchy/Discretization/LimitedVRInternal.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
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
        EquationDofs BindFields(const EquationDofs &prototype, Algebra::DofBundle fields, index modes)
        {
            auto schema = prototype.Schema();
            std::vector<EquationDofBinding> bindings;
            for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
            {
                const auto location = prototype.LocateBlock(iBlock);
                auto &block = schema.blocks[iBlock];
                bindings.push_back({block.name, prototype.Fields().Name(location.physicalField), location.firstComponent * modes});
                if (modes != 1)
                {
                    auto components = std::move(block.components);
                    block.components.clear();
                    block.vectorStarts.clear(); // Basis coefficients are not ordinary adjacent vector triples.
                    for (const auto &component : components)
                        for (index mode = 0; mode < modes; ++mode)
                            block.components.push_back({component.name + ":mode" + std::to_string(mode), component.units, "polynomial-coefficient"});
                }
            }
            if (modes != 1)
                schema.family = "polynomial:" + schema.family;
            return {std::move(schema), std::move(fields), std::move(bindings)};
        }
        void PreflightHalo(EquationDofs &fields)
        {
            auto read = fields.ReadOwned();
            std::vector<R::Array<real>::WriteLease> ghosts;
            ghosts.reserve(size_t(fields.PhysicalFieldNumber()));
            try
            {
                for (index iField = 0; iField < fields.PhysicalFieldNumber(); ++iField)
                    ghosts.emplace_back(fields.PhysicalField(iField).Ghost().Write());
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy equation finite-volume preparation failed");
                for (auto &lease : ghosts)
                    lease.CancelUnmodified();
                throw;
            }
            for (auto &lease : ghosts)
                lease.CancelUnmodified();
        }
        using namespace EquationFVDetail;
        using FiniteVolumeDetail::Load;
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
        data->boundaryEntries.resize(size_t(data->reconstruction.NumBoundaryQuadrature()));
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
        for (index iPoint = 0; iPoint < data->reconstruction.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto point = data->reconstruction.GetBoundaryQuadrature(iPoint);
            const auto globalPoint = metric.FaceQuadratureBegin(point.face) + point.quadrature;
            auto &entry = data->boundaryEntries[iPoint];
            entry = data->offsets[point.cell];
            while (entry < data->offsets[point.cell + 1] && data->samples[entry].point != globalPoint) ++entry;
            DNDS_check_throw_info(entry < data->offsets[point.cell + 1], "boundary quadrature missing from cell incidences");
            auto &sample=data->samples[entry];
            sample.boundaryPoint=iPoint;
            data->boundaryLayout->points.push_back({point.cell,point.face,point.quadrature,sample.zone,
                {sample.ownPoint[0],sample.ownPoint[1],sample.ownPoint[2]},
                {sample.normal[0],sample.normal[1],sample.normal[2]},sample.boundary.kind==BoundaryKind::Interface});
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
                    { c(iPoint, 0) = iCell; for (int d = 0; d < 3; ++d) p(iPoint, d) = point[d]; ++iPoint; };
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
            data->positivitySamples.emplace(data->reconstruction, cells, points, false);
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
    const CFV::VariationalReconstructionPlan &FiniteVolume::Reconstruction() const { return _data->reconstruction; }
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
        return {{"dimension", Metrics().Dimension()}, {"coefficient_dofs", _data->reconstruction.NRecDOF()}, {"components", components}, {"blocks", std::move(blocks)}, {"coefficient_role", "unlimitedGuess"}, {"cfv", _data->settings.numerics.cfv}};
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
        MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, Metrics().Communicator());
        for (int i = 0; i < 3; ++i)
            MPI_Allreduce(MPI_IN_PLACE, errors[i].data(), 5, MPI_DOUBLE, i == 2 ? MPI_MAX : MPI_SUM, Metrics().Communicator());
        for (int j = 0; j < 5; ++j)
        {
            errors[0][j] /= volume;
            errors[1][j] = std::sqrt(errors[1][j] / volume);
        }
        return errors;
    }
    const ThermalBounds &FiniteVolume::ThermalConstraints() const { return _data->PerfectGasBounds(); }
    bool FiniteVolume::Admissible(const EquationDofs &state) const
    {
        _data->equation.ValidateStateSchema(state);
        DNDS_check_throw_info(!_data->equation.HasMixture() && !_data->equation.HasRANS(),
                              "finite-volume admissibility without a worker/model policy is available only for pure flow5");
        int valid = 1;
        Metrics().Communication()->LocalStage([&]
                                              {
            const auto read = state.ReadOwned();
            const auto flow = read.Block("flow");
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                FlowState value{};
                for (int j = 0; j < 5; ++j)
                    value[j] = flow(iCell, j);
                valid &= ThermallyAdmissible(value, _data->PerfectGasBounds());
            } });
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, Metrics().Communicator());
        return valid;
    }
    struct FiniteVolume::ReconstructionState
    {
        R::SessionAccess session;
        std::shared_ptr<const Data> owner;
        EquationDofOperations::Binding binding;
        /// Durable warm/unlimited coefficients advanced by the reconstruction
        /// iteration. They are solver state, never workspace scratch.
        EquationDofs coefficients;
        /// Durable finalized polynomial consumed by flux/source/Jacobian
        /// operators after limiter and PP. DITR stages need independent copies
        /// even while serially sharing every temporary workspace allocation.
        EquationDofs fluxCoefficients;
        /// Provenance of the finalized owned polynomial. Limiter/PP profiles
        /// require an Evaluate/PrepareReconstruction after the warm guess
        /// changes; unfiltered profiles can republish by a direct owned copy.
        std::optional<EquationDofOperations::Binding> finalizedFrom;
        std::vector<Algebra::DofHalo<>> meanHalo;
        uint64_t meanEpoch = 0;
        std::vector<R::TransferBinding> coefficientHalo, fluxHalo;
        std::optional<R::Array<real>> limitedVRHistory;

        bool FinalizedCurrent() const
        {
            if (!finalizedFrom || finalizedFrom->snapshot.PhysicalFieldNumber() != coefficients.PhysicalFieldNumber())
                return false;
            for (index iField = 0; iField < coefficients.PhysicalFieldNumber(); ++iField)
            {
                const auto &saved = finalizedFrom->snapshot.PhysicalField(iField).Owned();
                const auto &current = coefficients.PhysicalField(iField).Owned();
                if (saved.SharedStorage() != current.SharedStorage() ||
                    finalizedFrom->versions[size_t(iField)] != current.Version())
                    return false;
            }
            return true;
        }
        void PublishFinalized() { finalizedFrom.emplace(coefficients); }

        static EquationDofs BuildCoefficients(const Data &data, const EquationDofs &state)
        {
            Algebra::DofBundle fields(state.Space());
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                const auto &physical = state.PhysicalField(iField);
                fields.Add(state.Fields().Name(iField), data.reconstruction.BuildCoefficients(
                                                            index(physical.Width()),
                                                            FiniteVolumeDetail::FieldLayout(physical.Owned().Layout(), data.metrics.NumCell(),
                                                                                            index(physical.Width()) * data.reconstruction.NRecDOF())));
            }
            return BindFields(state, std::move(fields), data.reconstruction.NRecDOF());
        }
        ReconstructionState(std::shared_ptr<const Data> data, EquationDofs &state)
            : owner(std::move(data)), binding(state), coefficients(BuildCoefficients(*owner, state)),
              fluxCoefficients(BuildCoefficients(*owner, state))
        {
            for (index iField = 0; iField < coefficients.PhysicalFieldNumber(); ++iField)
            {
                Algebra::Fill(coefficients.PhysicalField(iField), 0.);
                Algebra::Fill(fluxCoefficients.PhysicalField(iField), 0.);
                meanHalo.push_back(owner->BindTrackedHalo(state.PhysicalField(iField)));
                coefficientHalo.push_back(owner->BindHalo(coefficients.PhysicalField(iField)));
                fluxHalo.push_back(owner->BindHalo(fluxCoefficients.PhysicalField(iField)));
            }
            if (owner->limitedVRGeometry)
            {
                limitedVRHistory.emplace(R::RowLayout::Dense(owner->metrics.NumCell(), 2));
                auto write = limitedVRHistory->Write();
                const auto values = write.View<R::FixedRows<2>>();
                for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                    values(iCell, 0) = values(iCell, 1) = 0;
            }
            finalizedFrom.emplace(coefficients);
        }
    };
    struct FiniteVolume::Workspace
    {
        R::SessionAccess session;
        struct Field
        {
            CFV::ReconstructionWorkspace reconstruction;
            CFV::ReconstructionBoundaryValues boundary;
            std::optional<CFV::FVField> gradient;
            std::optional<R::Array<real>> gradientBoundary;
            std::vector<index> vectorStarts;
        };
        std::shared_ptr<const Data> owner;
        enum ProfilePhase
        {
            BoundaryPhase,
            GradientPhase,
            VRPhase,
            PPPhase,
            RHSPhase,
            SamplingPhase,
            SourcePhase,
            TotalPhase,
            ProfilePhases
        };
        bool profiling = std::getenv("DNDS_CAUCHY_PROFILE_EQUATION_FV") != nullptr;
        bool profileAllRanks = std::getenv("DNDS_CAUCHY_PROFILE_ALL_RANKS") != nullptr;
        std::array<real, ProfilePhases> profileSeconds{};
        std::array<index, ProfilePhases> profileCalls{};
        struct PhaseTime
        {
            using Clock = std::chrono::steady_clock;
            Workspace *work;
            ProfilePhase phase;
            Clock::time_point start{};
            PhaseTime(Workspace &input, ProfilePhase section) : work(input.profiling ? &input : nullptr), phase(section)
            {
                if (work)
                    start = Clock::now();
            }
            ~PhaseTime()
            {
                if (!work)
                    return;
                work->profileSeconds[phase] += std::chrono::duration<real>(Clock::now() - start).count();
                const int rank = work->owner->mesh->getMPI().rank;
                if (++work->profileCalls[phase] % 32 == 0 && phase == TotalPhase && (!rank || work->profileAllRanks))
                {
                    const auto record = Json{{"rank", rank}, {"owned_cells", work->owner->metrics.NumCell()}, {"meanOnly", work->meanOnly}, {"reconstruction", work->owner->settings.numerics.reconstruction}, {"firstOrderConvection", work->owner->settings.numerics.firstOrderConvection}, {"phases", {"boundary", "gradient", "vr", "pp", "rhs", "sampling", "source", "evaluate"}}, {"seconds", work->profileSeconds}, {"calls", work->profileCalls}}.dump();
                    std::fprintf(stdout, "CAUCHY_EQUATION_FV_PROFILE %s\n", record.c_str());
                }
            }
        };
        const bool meanOnly;
        index componentNumber = 0, physicalFieldNumber = 0;
        std::vector<Field> fields;
        EquationDofs boundary;
        /// Operation-scoped retained binding. It is installed by the
        /// orchestration call and cleared on every exit; the workspace never
        /// becomes the durable coefficient owner.
        std::optional<EquationDofs> activeCoefficients, activeFluxCoefficients;
        // Accepted direct-FV physical gradients, separate from source sampling.
        // WBAP/PP consume their explicit polynomial projection; the direct RHS
        // reads only these physical gradients and never evaluates a Taylor basis.
        std::vector<CFV::FVField> directGradients, directViscousGradients;
        /// Legacy-compatible primitive-gradient viscosity is prepared inside
        /// this workspace but consumed by the same fixed-flow RHS leaf.
        std::optional<CFV::FVField> primitive, primitiveGradient;
        std::optional<R::Array<real>> primitiveBoundary;
        std::optional<R::TransferBinding> primitiveGradientHalo;
        struct DirectGradientPublication
        {
            // Weak identities reject rebinding/ABA without retaining old fields.
            std::weak_ptr<const R::ArrayStorage> input, output;
            uint64_t inputShape = 0, inputVersion = 0, outputShape = 0, outputVersion = 0;
        };
        std::shared_ptr<EquationPositivityWorkspace> positivity;
        std::unique_ptr<LimitedVRWorkspace> limitedVR;
        EquationThermalPolicy thermal;
        /// Setup-resolved fixed-flow leaf. It owns immutable kernel reads; the
        /// surrounding Workspace still owns the only reconstruction/RHS
        /// schedule and all operand leases.
        std::optional<RHSOperator> fixedFlowRHS;
        RHSLocalDiagnostics diagnostics;
        std::array<index, 2> faceFallbacks{};
        std::optional<CFV::FaceCharacteristicTransform> flowCharacteristic;
        index flowCharacteristicField = -1;
        index characteristicFallbacks = 0;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        /// Published receiver-frame interface traces. The workspace retains
        /// their allocation/publication identity; each evaluation pins the
        /// numeric rows through its final dependent consumer.
        std::optional<BoundaryInputs> incoming;
        /// Operator-owned packed auxiliary channels needed by flux profiles
        /// whose RHS cannot be evaluated from one face pair alone. Ordinary
        /// fluxes avoid this allocation and their existing fused path.
        std::optional<AuxiliaryValues> rhsAuxiliary;
        std::vector<index> logicalFirst, vectorStarts;
        index flowFirst = -1, speciesFirst = -1, ransFirst = -1;
        std::vector<index> sourceOffsets;
        R::Array<real> sourceWeights;
        std::optional<CFV::PolynomialSamplingPlan> sourceSampling;
        std::vector<R::Array<real>> sourceValues;
        bool sourceValuesAreMeans = false;
        std::vector<std::array<R::Array<real>, 3>> sourceGradients;
        // The default approximate source Jacobian uses cell means plus
        // gradients at one cell center. The opt-in rhsQuadrature mode reuses
        // the separately published RHS source samples.
        std::optional<CFV::PolynomialSamplingPlan> sourceJacobianSampling;
        std::vector<std::array<R::Array<real>, 3>> sourceJacobianGradients;
        R::Array<real> boundaryMomentumFlux;
        /// Integrated residual contribution in the canonical owned-parent
        /// frame, one row per local stencil face. Refilled on every RHS call.
        R::Array<real> faceFlux;
        /// Optional first-order conservative flux path used by explicit flow5
        /// positivity. It is an output of the same equation RHS operator, not
        /// a second finite-volume frontend or a second reconstruction pass.
        std::optional<EquationDofs> lowOrderResidual;
        std::optional<R::Array<real>> lowOrderFaceFlux;
        bool boundaryMomentumCurrent = false;
        std::optional<EquationMeanGradients> meanGradients;
        bool chemical;
        int iterations = 0;
        real relativeUpdate = 0;
        struct WeakPublication
        {
            std::vector<std::weak_ptr<const R::ArrayStorage>> storage;
            std::vector<uint64_t> shapes, versions;
            explicit WeakPublication(const EquationDofs &values)
            {
                storage.reserve(size_t(values.PhysicalFieldNumber()));
                shapes.reserve(storage.capacity());
                versions.reserve(storage.capacity());
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    storage.push_back(array.SharedStorage());
                    shapes.push_back(array.ShapeVersion());
                    versions.push_back(array.Version());
                }
            }
            bool Current(const EquationDofs &values) const
            {
                if (storage.size() != size_t(values.PhysicalFieldNumber()))
                    return false;
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &array = values.PhysicalField(iField).Owned();
                    if (storage[size_t(iField)].lock() != array.SharedStorage() ||
                        shapes[size_t(iField)] != array.ShapeVersion() ||
                        versions[size_t(iField)] != array.Version())
                        return false;
                }
                return true;
            }
        };
        std::optional<EquationDofOperations::Binding> sourceState;
        std::optional<WeakPublication> sourceCoefficients;
        real sourceTime = 0;
        std::optional<EquationDofOperations::Binding> rhsState;
        std::optional<WeakPublication> rhsCoefficients;
        real rhsTime = 0;
        // Keep auxiliary cache metadata after the existing numerical fields.
        std::vector<std::array<DirectGradientPublication, 2>> directGradientPublications;

        class ActiveReconstruction
        {
            Workspace *_work = nullptr;

        public:
            ActiveReconstruction(Workspace &work, const EquationDofs &coefficients,
                                 const EquationDofs &fluxCoefficients) : _work(&work)
            {
                DNDS_check_throw_info(!work.activeCoefficients && !work.activeFluxCoefficients,
                                      "workspace already has an active reconstruction binding");
                work.activeCoefficients.emplace(coefficients);
                work.activeFluxCoefficients.emplace(fluxCoefficients);
            }
            ActiveReconstruction(const ActiveReconstruction &) = delete;
            ActiveReconstruction &operator=(const ActiveReconstruction &) = delete;
            ActiveReconstruction(ActiveReconstruction &&other) noexcept : _work(other._work) { other._work = nullptr; }
            ~ActiveReconstruction()
            {
                if (_work)
                {
                    _work->activeCoefficients.reset();
                    _work->activeFluxCoefficients.reset();
                }
            }
        };
        ActiveReconstruction BindReconstruction(const EquationDofs &coefficients,
                                                const EquationDofs &fluxCoefficients)
        {
            return ActiveReconstruction(*this, coefficients, fluxCoefficients);
        }
        const EquationDofs &Coefficients() const
        {
            DNDS_check_throw_info(activeCoefficients, "workspace has no active reconstruction binding");
            return *activeCoefficients;
        }
        const EquationDofs &FluxCoefficients() const
        {
            DNDS_check_throw_info(activeFluxCoefficients, "workspace has no active finalized reconstruction binding");
            return *activeFluxCoefficients;
        }
        EquationDofs &FluxCoefficients()
        {
            DNDS_check_throw_info(activeFluxCoefficients, "workspace has no active finalized reconstruction binding");
            return *activeFluxCoefficients;
        }

        static std::vector<Field> BuildFields(const Data &data, const EquationDofs &state, bool meanOnly)
        {
            std::vector<Field> result;
            result.reserve(size_t(state.PhysicalFieldNumber()));
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                const auto &physical = state.PhysicalField(iField);
                std::vector<index> vectors;
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto location = state.LocateBlock(iBlock);
                    if (location.physicalField == iField)
                        for (const auto first : state.Schema().blocks[iBlock].vectorStarts)
                            vectors.push_back(location.firstComponent + first);
                }
                std::sort(vectors.begin(), vectors.end());
                Field field{data.reconstruction.BuildWorkspace(index(physical.Width()),
                                                               FiniteVolumeDetail::FieldLayout(physical.Owned().Layout(), data.metrics.NumCell(), index(physical.Width()) * data.reconstruction.NRecDOF()), vectors),
                            data.reconstruction.BuildBoundaryValues(index(physical.Width())),
                            {},
                            {},
                            vectors};
                Algebra::Fill(field.reconstruction.temporary, 0.);
                if (!meanOnly && (data.settings.numerics.reconstruction == "gradient" ||
                                  (data.equation.HasRANS() && data.settings.numerics.reconstruction == "constant")))
                {
                    field.gradient.emplace(data.metrics.BuildField(index(physical.Width()) * data.metrics.Dimension()));
                    field.gradientBoundary.emplace(data.gradient.BuildBoundaryValues(index(physical.Width())));
                }
                else if (!meanOnly && data.limitedVRGeometry)
                    field.gradientBoundary.emplace(data.gradient.BuildBoundaryValues(index(physical.Width())));
                result.push_back(std::move(field));
            }
            return result;
        }
        void BuildSourceSampling(const EquationDofs &state)
        {
            if (!owner->equation.HasRANS() && (!chemical || owner->equation.Mixture().SpeciesNumber() == 1))
                return;
            auto read = owner->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = read.HostView();
            const int order = owner->settings.numerics.RHSIntegrationOrder();
            sourceOffsets.resize(size_t(owner->metrics.NumCell()) + 1, 0);
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
            {
                const Geom::Elem::Quadrature quadrature(mesh.GetCellElement(iCell), order);
                sourceOffsets[iCell + 1] = sourceOffsets[iCell] + quadrature.GetNumPoints();
            }
            const auto count = sourceOffsets.back();
            R::Array<index> cells(R::RowLayout::Dense(count, 1));
            R::Array<real> points(R::RowLayout::Dense(count, 3));
            sourceWeights = R::Array<real>(R::RowLayout::Dense(count, 1));
            const auto geometry = owner->metrics.AcquireRead();
            const auto metric = geometry.HostView();
            {
                auto cellWrite = cells.Write();
                auto pointWrite = points.Write(), weightWrite = sourceWeights.Write();
                const auto c = cellWrite.View<R::FixedRows<1>>();
                const auto p = pointWrite.View<R::FixedRows<3>>();
                const auto w = weightWrite.View<R::FixedRows<1>>();
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                {
                    Geom::tSmallCoords coords;
                    mesh.GetCoordsOnCell(iCell, coords);
                    const auto element = mesh.GetCellElement(iCell);
                    const Geom::Elem::Quadrature quadrature(element, order);
                    for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                    {
                        const index iPoint = sourceOffsets[iCell] + iQ;
                        const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                        const auto point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                        c(iPoint, 0) = iCell;
                        for (int d = 0; d < 3; ++d)
                            p(iPoint, d) = point[d];
                        w(iPoint, 0) = order == 1 ? metric.GetCellVol(iCell) : Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                    }
                }
            }
            sourceSampling.emplace(owner->reconstruction, cells, points, owner->equation.HasRANS());
            sourceValuesAreMeans = owner->settings.numerics.reconstruction != "variational" && count == owner->metrics.NumCell();
            for (index iCell = 0; sourceValuesAreMeans && iCell < owner->metrics.NumCell(); ++iCell)
                sourceValuesAreMeans = sourceOffsets[iCell] == iCell && sourceOffsets[iCell + 1] == iCell + 1;
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                // Keep source sampling scratch independent of any particular
                // mean allocation so serialized stage states can reuse it.
                sourceValues.emplace_back(FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), count,
                                                                          index(state.PhysicalField(iField).Width())));
                if (owner->equation.HasRANS())
                {
                    sourceGradients.emplace_back();
                    // One-point direct FV can read its ready physical gradient
                    // field directly; polynomial/multi-point paths own samples.
                    if (!sourceValuesAreMeans)
                        for (int d = 0; d < owner->metrics.Dimension(); ++d)
                            sourceGradients.back()[d] = R::Array<real>(sourceValues.back().Layout());
                }
            }
            if (owner->equation.HasRANS())
            {
                const index owned = owner->metrics.NumCell();
                R::Array<index> centerCells(R::RowLayout::Dense(owned, 1));
                R::Array<real> centers(R::RowLayout::Dense(owned, 3));
                {
                    auto cellWrite = centerCells.Write();
                    auto pointWrite = centers.Write();
                    for (index iCell = 0; iCell < owned; ++iCell)
                    {
                        cellWrite.View()(iCell, 0) = iCell;
                        const auto center = metric.GetCellCent(iCell);
                        for (int d = 0; d < 3; ++d)
                            pointWrite.View()(iCell, d) = center[d];
                    }
                }
                sourceJacobianSampling.emplace(owner->reconstruction, centerCells, centers);
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    sourceJacobianGradients.emplace_back();
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        sourceJacobianGradients.back()[d] = R::Array<real>(FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owned, state.PhysicalField(iField).Width()));
                }
            }
        }
        static EquationDofs BindBoundary(const EquationDofs &state, const std::vector<Field> &fields, index rows)
        {
            const Algebra::DofSpace space(size_t(rows), 0, "equation boundary quadrature");
            Algebra::DofBundle bundle(space);
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                bundle.Add(state.Fields().Name(iField), CFV::FVField(space, fields[iField].boundary.values,
                                                                     R::Array<real>(R::RowLayout::Dense(0, state.PhysicalField(iField).Width()))));
            return BindFields(state, std::move(bundle), 1);
        }
        Workspace(std::shared_ptr<const Data> data, EquationDofs &state,
                  std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, bool meanTrial)
            : owner(std::move(data)), meanOnly(meanTrial), componentNumber(state.ComponentNumber()), physicalFieldNumber(state.PhysicalFieldNumber()),
              fields(BuildFields(*owner, state, meanOnly)),
              boundary(BindBoundary(state, fields, owner->reconstruction.NumBoundaryQuadrature())),
              thermal(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                      owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                      pool ? pool->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}),
              diagnostics(owner->forceZones),
              workers(std::move(pool)),
              boundaryMomentumFlux(R::RowLayout::Dense(owner->reconstruction.NumBoundaryQuadrature(), 3)),
              faceFlux(R::RowLayout::Dense(index(owner->stencilFaces.size()), state.ComponentNumber())),
              chemical(owner->settings.equation.source == "chemical")
        {
            index first = 0;
            for (const auto &block : state.Schema().blocks)
            {
                logicalFirst.push_back(first);
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                if (block.name == "rans")
                    ransFirst = first;
                for (const auto vector : block.vectorStarts)
                    vectorStarts.push_back(first + vector);
                first += index(block.components.size());
            }
            if (!meanOnly && owner->settings.numerics.positivity)
            {
                // Positivity retains only its own reference/sample buffers.
                // This construction prototype is discarded immediately; the
                // actual finalized coefficients belong to ReconstructionState.
                auto coefficientPrototype = ReconstructionState::BuildCoefficients(*owner, state);
                positivity = std::make_shared<EquationPositivityWorkspace>(owner, state, coefficientPrototype, workers);
                // Conservative explicit flux blending is currently the
                // established pure-flow policy. Extended RANS/species stages
                // reject and retry their raw update instead of constructing a
                // false independently blended scalar system.
                if (!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                    state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1)
                {
                    lowOrderResidual.emplace(EquationDofOperations::NewVector(state));
                    lowOrderFaceFlux.emplace(R::RowLayout::Dense(index(owner->stencilFaces.size()), state.ComponentNumber()));
                }
            }
            if (!meanOnly && owner->limitedVRGeometry)
            {
                DNDS_check_throw_info(!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                                          state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                      "LVR requires one coupled perfect-gas flow5 field");
                limitedVR = std::make_unique<LimitedVRWorkspace>(*owner, fields.front().reconstruction);
            }
            if (!meanOnly && (owner->settings.numerics.limiter == "WBAP" || limitedVR) && owner->settings.numerics.characteristicLimiter &&
                !owner->equation.HasMixture())
            {
                const auto flow = state.LocateBlock("flow");
                DNDS_check_throw_info(flow.componentNumber == 5, "characteristic limiter requires one five-component flow block");
                flowCharacteristicField = flow.physicalField;
                flowCharacteristic.emplace(owner->reconstruction.BuildFaceTransform(flow.firstComponent, 5));
            }
            if (owner->settings.numerics.reconstruction != "variational")
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    directGradients.push_back(owner->metrics.BuildField(
                        FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owner->metrics.NumCell(),
                                                        state.PhysicalField(iField).Width() * owner->metrics.Dimension())));
                    if (owner->equation.Viscous() && owner->settings.numerics.viscousGradient == "polynomial" &&
                        !owner->settings.numerics.viscousLimited)
                        directViscousGradients.push_back(owner->metrics.BuildField(
                            FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owner->metrics.NumCell(),
                                                            state.PhysicalField(iField).Width() * owner->metrics.Dimension())));
                }
            if (!meanOnly && owner->equation.Viscous() && owner->settings.numerics.viscousGradient == "primitive")
            {
                DNDS_check_throw_info(!owner->equation.HasMixture() && !owner->equation.HasRANS() &&
                                          state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                      "primitive viscosity requires one perfect-gas flow5 field");
                primitive.emplace(owner->metrics.BuildField(5));
                primitiveGradient.emplace(owner->metrics.BuildField(5 * owner->metrics.Dimension()));
                primitiveBoundary.emplace(owner->gradient.BuildBoundaryValues(5));
                primitiveGradientHalo.emplace(owner->BindHalo(*primitiveGradient, "cauchy-primitive-gradient"));
            }
            BuildSourceSampling(state);
            if (ransFirst >= 0)
                directGradientPublications.resize(directGradients.size());
        }
        void BindHalos(EquationDofs &state)
        {
            (void)state;
        }
        void RefreshDirectGradients(bool ghost)
        {
            if (owner->settings.numerics.reconstruction == "variational")
                return;
            for (index iField = 0; iField < index(fields.size()); ++iField)
            {
                const bool fromGradientField = owner->settings.numerics.reconstruction == "constant" && fields[iField].gradient;
                // Constant face fluxes never read remote source gradients.
                if (fromGradientField && ghost)
                    continue;
                const auto refresh = [&]
                {
                    if (fromGradientField)
                    {
                        // The cache check may retain a compatible input read.
                        // Use the explicit engine entry: the immediate algebra
                        // wrapper deliberately requires completely idle fields.
                        R::EagerEngine engine;
                        const auto stream = engine.CreateCPUStream();
                        Algebra::Copy(engine, stream, directGradients[iField], *fields[iField].gradient);
                        engine.Synchronize();
                    }
                    else
                        owner->gradient.ExtractLinearGradients(FluxCoefficients().PhysicalField(iField), owner->reconstruction,
                                                               directGradients[iField], ghost);
                };
                // Only RANS source sampling repeats this pass. Flow-only and
                // chemical-only paths retain their original preparation path.
                if (ransFirst < 0)
                {
                    refresh();
                    continue;
                }
                const auto &source = fromGradientField ? fields[iField].gradient->Owned()
                                     : ghost           ? FluxCoefficients().PhysicalField(iField).Ghost()
                                                       : FluxCoefficients().PhysicalField(iField).Owned();
                const auto &destination = ghost ? directGradients[iField].Ghost() : directGradients[iField].Owned();
                // Pin the input through both the check and a possible refresh:
                // an unfinished writer has not published its next generation.
                const auto inputRead = source.Read();
                const auto inputStorage = source.SharedStorage();
                auto &published = directGradientPublications[iField][ghost ? 1 : 0];
                if (published.input.lock() == inputStorage && published.output.lock() == destination.SharedStorage() &&
                    published.inputShape == source.ShapeVersion() && published.inputVersion == source.Version())
                {
                    const auto outputRead = destination.Read();
                    if (published.outputShape == destination.ShapeVersion() && published.outputVersion == destination.Version())
                    {
                        owner->metrics.Validate();
                        continue;
                    }
                }
                // The independently callable source path still executes this
                // on a miss. P0 sources keep their physical gradients, and P1
                // extracts the final limiter/PP coefficients, never raw slopes.
                refresh();
                const auto outputRead = destination.Read();
                published = {inputStorage, destination.SharedStorage(), source.ShapeVersion(), source.Version(),
                             destination.ShapeVersion(), destination.Version()};
            }
            if (!directViscousGradients.empty())
            {
                DNDS_check_throw_info(directViscousGradients.size() == fields.size(),
                                      "unlimited direct viscous-gradient fields are incomplete");
                for (index iField = 0; iField < index(fields.size()); ++iField)
                    owner->gradient.ExtractLinearGradients(Coefficients().PhysicalField(iField), owner->reconstruction,
                                                           directViscousGradients[iField], ghost);
            }
        }

        void PreparePrimitiveGradient(const EquationDofs &state, CFV::ReconstructionRegion region)
        {
            if (!primitiveGradient)
                return;
            DNDS_assert_info(primitive && primitiveBoundary && state.PhysicalFieldNumber() == 1,
                             "primitive-gradient workspace is incomplete");
            const auto fill = [&](const R::Array<real> &source, R::Array<real> &destination)
            {
                const auto read = source.Read();
                auto write = destination.Write();
                const auto from = read.View<R::UniformTableRows>();
                const auto to = write.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < to.RowCount(); ++iCell)
                {
                    FlowState conservative;
                    for (int j = 0; j < 5; ++j)
                        conservative[j] = from(iCell, j);
                    const auto value = owner->equation.Gas().Primitive(conservative);
                    const real converted[5]{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                    for (int j = 0; j < 5; ++j)
                        to(iCell, j) = converted[j];
                }
            };
            if (region != CFV::ReconstructionRegion::HaloDependent)
                fill(state.PhysicalField(0).Owned(), primitive->Owned());
            if (region != CFV::ReconstructionRegion::Core)
                fill(state.PhysicalField(0).Ghost(), primitive->Ghost());
            owner->gradient.Evaluate(*primitive, *primitiveBoundary, *primitiveGradient, {1}, region);
        }

        void FillBoundary(const EquationDofs &state, bool polynomial, real time, EquationPointEvaluator &evaluator,
                          RHSScalarView incoming = {})
        {
            const PhaseTime timing(*this, BoundaryPhase);
            const auto mean = state.ReadOwned(), rec = Coefficients().ReadOwned();
            const auto meanBlocks = BlockViews(state, mean), recBlocks = BlockViews(Coefficients(), rec);
            std::optional<CFV::ReconstructionBasisRead> basisRead;
            const bool direct = owner->settings.numerics.reconstruction != "variational";
            if (!direct)
                basisRead.emplace(owner->reconstruction.AcquireBasisRead());
            CFV::Math::SmallMatrix scratch;
            PointScratch point(state.ComponentNumber()), cellMean(state.ComponentNumber());
            const auto *ransModel = owner->equation.HasRANS() ? &owner->equation.RANS() : nullptr;
            const auto ransMean = ransModel ? mean.Block("rans") : EquationBlockReadView{};
            std::optional<WallDistanceRead> wallRead;
            if (ransModel && ransModel->ComponentNumber() == 2)
                wallRead.emplace(owner->wallDistance->AcquireRead());
            const auto wallValues = wallRead ? wallRead->HostView() : WallDistanceView{};
            const auto frame = owner->motion.At(time);
            const int modes = owner->reconstruction.NRecDOF();
            {
                auto output = boundary.WriteOwned();
                for (index iPoint = 0; iPoint < index(owner->boundaryEntries.size()); ++iPoint)
                {
                    const auto &sample = owner->samples[owner->boundaryEntries[iPoint]];
                    Gather(meanBlocks, sample.cell, cellMean);
                    point.values = cellMean.values;
                    if (polynomial)
                    {
                        const auto basis = basisRead->FaceBasis(sample.face, sample.side, sample.point, 0, scratch);
                        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                        {
                            const auto row = recBlocks[iBlock][sample.cell];
                            for (index j = 0; j < meanBlocks[iBlock].ComponentNumber(); ++j)
                                for (int mode = 0; mode < modes; ++mode)
                                    point.values[logicalFirst[iBlock] + j] += row[j * modes + mode] * basis(0, mode);
                        }
                    }
                    if (ransModel)
                        RANSStateView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairReconstruction(ransMean[sample.cell], point.values.data() + ransFirst);
                    BoundaryValue(point, flowFirst, speciesFirst, owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0,
                                  sample, evaluator, frame, time, true,
                                  {ransModel, ransFirst, &cellMean, wallRead ? wallValues.cellMean(sample.cell, 0) : 0}, incoming);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    // Euler's reconstruction callback uses fixUL=true: its
                    // returned SA target is zero, rather than a negative ghost.
                    if (ransModel && ransModel->Kind() == Physics::RANS::ModelKind::SA && sample.boundary.kind == BoundaryKind::NoSlipWall)
                        point.values[ransFirst] = 0;
#endif
                    if (owner->settings.numerics.reconstruction != "variational" && sample.boundary.kind != BoundaryKind::Interface)
                        for (index j = 0; j < state.ComponentNumber(); ++j)
                            point.values[j] = .5 * (cellMean.values[j] + point.values[j]);
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto target = output.Block(iBlock);
                        for (index j = 0; j < target.ComponentNumber(); ++j)
                            target(iPoint, j) = point.values[logicalFirst[iBlock] + j];
                    }
                }
            }
            // GradientPlan may select another face quadrature order. Its setup
            // already maps these rows to the canonical boundary quadrature.
            for (auto &field : fields)
                if (field.gradientBoundary)
                {
                    const auto source = field.boundary.values.Read();
                    auto target = field.gradientBoundary->Write();
                    const auto a = source.View<R::UniformTableRows>();
                    const auto b = target.View<R::UniformTableRows>();
                    for (index iPoint = 0; iPoint < b.RowCount(); ++iPoint)
                        for (index j = 0; j < b.Width(); ++j)
                            b(iPoint, j) = a(owner->gradientBoundaryPoints[iPoint], j);
                }
            if (primitiveBoundary)
            {
                auto target = primitiveBoundary->Write();
                const auto values = target.View<R::UniformTableRows>();
                for (index iPoint = 0; iPoint < owner->gradient.NumBoundaryQuadrature(); ++iPoint)
                {
                    const auto &sample = owner->samples[owner->boundaryEntries[owner->gradientBoundaryPoints[iPoint]]];
                    Gather(meanBlocks, sample.cell, cellMean);
                    const auto state = FiniteVolumeDetail::BoundaryGradientValue(
                        sample, owner->equation.Gas(), cellMean.Flow(flowFirst), time, frame, incoming);
                    const auto value = owner->equation.Gas().Primitive(state);
                    const real converted[5]{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                    for (int j = 0; j < 5; ++j)
                        values(iPoint, j) = converted[j];
                }
            }
        }
        void ApplyFixedFlowRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                               real time, std::optional<std::reference_wrapper<const AuxiliaryValues>> auxiliary,
                               std::optional<std::reference_wrapper<R::Array<real>>> faceOutput,
                               RHSScalarView incoming)
        {
            DNDS_assert_info(fixedFlowRHS && state.PhysicalFieldNumber() == 1 && result.PhysicalFieldNumber() == 1 &&
                                 state.ComponentNumber() == 5 && result.ComponentNumber() == 5,
                             "fixed-flow RHS leaf requires one flow5 physical field");
            const bool ghosts = region != CFV::ReconstructionRegion::Core;
            std::vector<R::Array<real>::ReadLease> reads;
            reads.reserve(16);
            const auto readField = [&](const CFV::FVField &field)
            {
                const size_t owned = reads.size();
                reads.emplace_back(field.Owned().Read());
                if (ghosts)
                    reads.emplace_back(field.Ghost().Read());
                return RHSFieldView{reads[owned].View<R::UniformTableRows>(),
                                    ghosts ? reads[owned + 1].View<R::UniformTableRows>() : RHSScalarView{}};
            };
            RHSReadViews input;
            input.mean = readField(state.PhysicalField(0));
            input.includeViscosity = !meanOnly;
            input.meanOnly = meanOnly;
            if (owner->settings.numerics.reconstruction == "variational")
                input.coefficients = readField(FluxCoefficients().PhysicalField(0));
            else if (!meanOnly && !owner->settings.numerics.firstOrderConvection &&
                     owner->settings.numerics.reconstruction == "gradient")
            {
                DNDS_assert_info(directGradients.size() == 1, "fixed-flow direct gradient is unavailable");
                input.conservativeGradient = readField(directGradients.front());
            }
            if (!owner->settings.numerics.viscousLimited &&
                (owner->settings.numerics.limiter == "WBAP" || limitedVR || owner->settings.numerics.positivity))
            {
                if (owner->settings.numerics.reconstruction == "variational")
                    input.viscousCoefficients = readField(Coefficients().PhysicalField(0));
                else if (!directViscousGradients.empty())
                    input.viscousConservativeGradient = readField(directViscousGradients.front());
            }
            if (primitiveGradient)
                input.primitiveGradient = readField(*primitiveGradient);
            std::optional<R::Array<real>::ReadLease> thermodynamics;
            std::optional<DerivedFields::ReadScope> derivedOwned, derivedGhost;
            if (auxiliary)
            {
                const auto &values = auxiliary->get();
                if (values.meanThermodynamics)
                {
                    thermodynamics.emplace(values.meanThermodynamics->Read());
                    input.auxiliary.meanThermodynamics.emplace(thermodynamics->View<R::UniformTableRows>());
                }
                else if (values.derivedOwned)
                {
                    const auto typed = EquationDofs::Coupled(owner->equation.Schema(), state.PhysicalField(0));
                    derivedOwned.emplace(values.derivedOwned->ReadCurrent(typed, owner->equation));
                    if (ghosts && values.derivedGhost)
                        derivedGhost.emplace(values.derivedGhost->ReadCurrent(typed, owner->equation));
                    input.auxiliary.meanThermodynamics.emplace(MeanThermodynamicView::Derived(
                        *derivedOwned, derivedGhost ? std::optional(std::cref(*derivedGhost)) : std::nullopt));
                }
                if (values.cellWaveJump)
                    input.auxiliary.cellWaveJump = readField(*values.cellWaveJump);
                if (values.cellPressureMinimum)
                    input.auxiliary.cellPressureMinimum = readField(*values.cellPressureMinimum);
            }
            if (incoming.data)
                input.boundary = incoming;
            auto outputWrite = result.PhysicalField(0).Owned().Write();
            RHSWriteViews output{outputWrite.View<R::UniformTableRows>()};
            std::optional<R::Array<real>::WriteLease> lowWrite, faceWrite;
            if (lowOrderResidual)
            {
                lowWrite.emplace(lowOrderResidual->PhysicalField(0).Owned().Write());
                output.lowOrder = lowWrite->View<R::UniformTableRows>();
            }
            if (faceOutput)
            {
                faceWrite.emplace(faceOutput->get().Write());
                output.faceFlux = faceWrite->View<R::UniformTableRows>();
            }
            diagnostics.Merge(fixedFlowRHS->ApplyFluxBorrowedFatal(input, output, region, time));
        }
        void ApplyRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                      EquationPointEvaluator &evaluator, real time,
                      std::optional<std::reference_wrapper<const AuxiliaryValues>> auxiliary = std::nullopt,
                      std::optional<std::reference_wrapper<R::Array<real>>> faceOutput = std::nullopt,
                      RHSScalarView incoming = {})
        {
            const PhaseTime timing(*this, RHSPhase);
            if (fixedFlowRHS)
            {
                ApplyFixedFlowRHS(state, result, region, time, auxiliary, faceOutput, incoming);
                return;
            }
            const auto mean = region != CFV::ReconstructionRegion::Core ? state.ReadWithGhost() : state.ReadOwned();
            auto output = result.WriteOwned();
            const EquationViewLayout<const real> owned(state, mean), remote(state, mean, true);
            const EquationViewLayout<real> destination(result, output);
            if (lowOrderResidual)
            {
                auto lowOutput = lowOrderResidual->WriteOwned();
                const EquationViewLayout<real> lowDestination(*lowOrderResidual, lowOutput);
                const auto kernel = [&](auto u, auto ghost, auto rhs, auto low)
                { ApplyRHSRows(state, region, evaluator, time, auxiliary, mean, u, ghost, rhs, low, true, faceOutput, incoming); };
                if (owned.Table() && remote.Table() && destination.Table() && lowDestination.Table() &&
                    owned.Flow5() && remote.Flow5() && destination.Flow5() && lowDestination.Flow5())
                    DispatchCPUEquationViews<EquationViewKind::CombinedAoS>(kernel, owned, remote, destination, lowDestination);
                else
                    DispatchCPUEquationViews<>(kernel, owned, remote, destination, lowDestination);
                return;
            }
            // The disabled low-order view deliberately aliases the high-order
            // descriptor. ApplyRHSRows receives produceLow=false and never
            // reads or writes it, avoiding a dummy allocation in non-PP runs.
            const auto kernel = [&](auto u, auto ghost, auto rhs, auto unusedLow)
            { ApplyRHSRows(state, region, evaluator, time, auxiliary, mean, u, ghost, rhs, unusedLow, false, faceOutput, incoming); };
            // RHS opts into one ordered AoS family. Arbitrary block bindings
            // and physical layouts keep the same generic numerical loop.
            // Ghost shape may be absent during Core; no ghost value is read.
            if (owned.Table() && remote.Table() && destination.Table() && owned.Flow5() && remote.Flow5() && destination.Flow5())
                DispatchCPUEquationViews<EquationViewKind::CombinedAoS>(kernel, owned, remote, destination, destination);
            else
                DispatchCPUEquationViews<>(kernel, owned, remote, destination, destination);
        }
        template <class Input, class Output>
        void ApplyRHSRows(const EquationDofs &state, CFV::ReconstructionRegion region,
                          EquationPointEvaluator &evaluator, real time,
                          std::optional<std::reference_wrapper<const AuxiliaryValues>> auxiliary,
                          const EquationDofs::ReadScope &mean, Input owned, Input remote, Output destination,
                          Output lowDestination, bool produceLow,
                          std::optional<std::reference_wrapper<R::Array<real>>> faceOutput, RHSScalarView incoming)
        {
            index localFallbacks = 0;
            auto geometry = owner->metrics.AcquireRead();
            const auto metric = geometry.HostView();
            std::optional<CFV::ReconstructionBasisRead> basisRead;
            const bool direct = owner->settings.numerics.reconstruction != "variational";
            if (!direct)
                basisRead.emplace(owner->reconstruction.AcquireBasisRead());
            const bool ghosts = region != CFV::ReconstructionRegion::Core;
            // Preparation owns writes; the point kernel holds only numeric
            // read views. These means are not reconstructed thermodynamics.
            std::optional<R::Array<real>::ReadLease> meanProperties;
            std::optional<RHSScalarView> preparedMean;
            if (auxiliary && auxiliary->get().meanThermodynamics)
            {
                meanProperties.emplace(auxiliary->get().meanThermodynamics->Read());
                preparedMean.emplace(meanProperties->View<R::UniformTableRows>());
            }
            std::array<std::optional<R::Array<real>::ReadLease>, 4> stencilReads;
            RHSFieldView waveJump, pressureMinimum;
            const auto readStencil = [&](const std::optional<CFV::FVField> &field, int slot, RHSFieldView &view)
            {
                if (!field)
                    return;
                stencilReads[slot].emplace(field->Owned().Read());
                if (ghosts)
                    stencilReads[slot + 1].emplace(field->Ghost().Read());
                view = {stencilReads[slot]->View<R::UniformTableRows>(),
                        ghosts ? stencilReads[slot + 1]->View<R::UniformTableRows>() : RHSScalarView{}};
            };
            if (auxiliary)
            {
                readStencil(auxiliary->get().cellWaveJump, 0, waveJump);
                readStencil(auxiliary->get().cellPressureMinimum, 2, pressureMinimum);
            }
            std::optional<EquationDofs::ReadScope> rec, viscousRec;
            std::vector<EquationBlockReadView> recBlocks, viscousRecBlocks;
            if (!direct)
            {
                rec.emplace(ghosts ? FluxCoefficients().ReadWithGhost() : FluxCoefficients().ReadOwned());
                recBlocks = BlockViews(FluxCoefficients(), *rec);
                if (owner->equation.Viscous() && !owner->settings.numerics.viscousLimited &&
                    (owner->settings.numerics.limiter == "WBAP" || limitedVR || owner->settings.numerics.positivity))
                {
                    viscousRec.emplace(ghosts ? Coefficients().ReadWithGhost() : Coefficients().ReadOwned());
                    viscousRecBlocks = BlockViews(Coefficients(), *viscousRec);
                }
            }
            const auto meanBlocks = BlockViews(state, mean);
            std::vector<EquationBlockLocation> directLocations;
            for (index iBlock = 0; direct && iBlock < state.BlockNumber(); ++iBlock)
                directLocations.push_back(state.LocateBlock(iBlock));
            std::vector<R::Array<real>::ReadLease> directOwned, directGhost, directViscousOwned, directViscousGhost;
            std::vector<RHSFieldView> directViews, directViscousViews;
            if (direct && owner->settings.numerics.reconstruction == "gradient" && !owner->settings.numerics.firstOrderConvection)
            {
                DNDS_check_throw_info(directGradients.size() == size_t(state.PhysicalFieldNumber()), "direct FV requires prepared physical gradients");
                for (const auto &gradient : directGradients)
                {
                    directOwned.emplace_back(gradient.Owned().Read());
                    if (ghosts)
                        directGhost.emplace_back(gradient.Ghost().Read());
                    directViews.push_back({directOwned.back().View<R::UniformTableRows>(), ghosts ? directGhost.back().View<R::UniformTableRows>() : RHSScalarView{}});
                }
                for (const auto &gradient : directViscousGradients)
                {
                    directViscousOwned.emplace_back(gradient.Owned().Read());
                    if (ghosts)
                        directViscousGhost.emplace_back(gradient.Ghost().Read());
                    directViscousViews.push_back({directViscousOwned.back().View<R::UniformTableRows>(),
                                                  ghosts ? directViscousGhost.back().View<R::UniformTableRows>() : RHSScalarView{}});
                }
            }
            std::optional<R::Array<real>::ReadLease> primitiveOwned, primitiveGhost;
            RHSFieldView primitiveGradientValues;
            if (primitiveGradient)
            {
                primitiveOwned.emplace(primitiveGradient->Owned().Read());
                if (ghosts)
                    primitiveGhost.emplace(primitiveGradient->Ghost().Read());
                primitiveGradientValues = {primitiveOwned->View<R::UniformTableRows>(),
                                           primitiveGhost ? primitiveGhost->View<R::UniformTableRows>() : RHSScalarView{}};
            }
            auto boundaryWrite = boundaryMomentumFlux.Write();
            const auto wallFlux = boundaryWrite.View<R::FixedRows<3>>();
            auto faceWrite = (faceOutput ? faceOutput->get() : faceFlux).Write();
            const auto faceValues = faceWrite.View<R::UniformTableRows>();
            std::optional<R::Array<real>::WriteLease> lowFaceWrite;
            decltype(faceWrite.View<R::UniformTableRows>()) lowFaceValues;
            if (produceLow)
            {
                DNDS_check_throw_info(lowOrderFaceFlux && lowOrderResidual &&
                                          state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                      "low-order equation flux requires the coupled flow5 PP profile");
                lowFaceWrite.emplace(lowOrderFaceFlux->Write());
                lowFaceValues = lowFaceWrite->View<R::UniformTableRows>();
            }
            const auto gatherMean = [&](index iCell, PointScratch &point)
            {
                for (index j = 0; j < owned.ComponentNumber(); ++j)
                    point.values[j] = iCell < owned.RowCount() ? owned(iCell, j) : remote(iCell - owned.RowCount(), j);
                point.gradient.setZero();
            };
            const int modes = owner->reconstruction.NRecDOF(), dim = owner->metrics.Dimension();
            const bool viscous = !meanOnly && owner->equation.Viscous();
            const auto &fluxPolicy = owner->fluxPolicy;
            const bool noRsOnWall = fluxPolicy.noRsOnWall;
            const bool noGRPOnWall = fluxPolicy.noGRPOnWall;
            const bool rusanovFlux = fluxPolicy.rusanov;
            const auto &fluxRequirements = fluxPolicy.required;
            const auto &baseFluxSettings = fluxPolicy.settings;
            const bool useWaveJump = owner->settings.numerics.riemann.useStencil && fluxPolicy.transverseSignal;
            const bool usePressureMinimum = owner->settings.numerics.riemann.useStencil && fluxPolicy.pressureStencil;
            const bool preparedWaveJump = auxiliary && bool(auxiliary->get().cellWaveJump);
            const bool preparedPressureMinimum = auxiliary && bool(auxiliary->get().cellPressureMinimum);
            DNDS_check_throw_info((!useWaveJump || preparedWaveJump) && (!usePressureMinimum || preparedPressureMinimum),
                                  "required equation RHS stencil auxiliary is absent");
            const int species = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() : 0;
            const int independent = std::max(0, species - 1);
            const index width = state.ComponentNumber();
            const auto frame = owner->motion.At(time);
            PointScratch left(width), right(width), face(width), cellMean(width);
            const auto *ransModel = owner->equation.HasRANS() ? &owner->equation.RANS() : nullptr;
            const auto thermalBounds = thermal.View();
            const auto ransMean = ransModel ? mean.Block("rans") : EquationBlockReadView{};
            std::vector<real> integrated(size_t(width), 0), flux(size_t(width), 0), speciesFlux(size_t(independent), 0);
            std::vector<real> integratedLow(produceLow ? size_t(width) : 0, 0), lowFlux(produceLow ? size_t(width) : 0, 0);
            EquationTransportScratch transport(species);
            std::optional<WallDistanceRead> wallRead;
            if (owner->wallDistance)
                wallRead.emplace(owner->wallDistance->AcquireRead());
            const auto wallValues = wallRead ? wallRead->HostView() : WallDistanceView{};
            const auto boundaryInput = [&](const FiniteVolumeDetail::Sample &sample)
            {
                if (ransModel && ransModel->ComponentNumber() == 2)
                    gatherMean(sample.cell, cellMean);
                return RANSBoundaryInput{ransModel, ransFirst, &cellMean, wallRead ? wallValues.cellMean(sample.cell, 0) : 0};
            };
            CFV::Math::SmallMatrix basisScratch;
            const auto reconstruct = [&](index iCell, const FiniteVolumeDetail::Sample &sample, int side, PointScratch &point)
            {
                gatherMean(iCell, point);
                // P0 uses mean states and only the normal face jump for
                // viscosity. This does not suppress volume-source gradients:
                // RANS/chemistry sampling retains the configured derivatives.
                if (!owner->settings.numerics.firstOrderConvection && (!direct || owner->settings.numerics.reconstruction == "gradient"))
                {
                    if (direct)
                    {
                        const auto displacement = FiniteVolumeDetail::DirectFVDisplacement(metric, sample, iCell != sample.cell || side != sample.side);
                        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                        {
                            const auto location = directLocations[iBlock];
                            const auto row = directViews[location.physicalField][iCell];
                            const auto viscousRow = directViscousViews.empty() ? row : directViscousViews[location.physicalField][iCell];
                            for (index j = 0; j < location.componentNumber; ++j)
                                for (int d = 0; d < dim; ++d)
                                {
                                    const real value = row[(location.firstComponent + j) * dim + d];
                                    point.values[logicalFirst[iBlock] + j] += value * displacement[d];
                                    if (viscous)
                                        point.gradient(logicalFirst[iBlock] + j, d) = viscousRow[(location.firstComponent + j) * dim + d];
                                }
                        }
                    }
                    else
                    {
                        const auto basis = basisRead->FaceBasis(sample.face, side, sample.point, viscous ? 1 : 0, basisScratch);
                        for (index iBlock = 0; iBlock < index(meanBlocks.size()); ++iBlock)
                        {
                            const auto row = recBlocks[iBlock][iCell];
                            const auto viscousRow = viscousRecBlocks.empty() ? row : viscousRecBlocks[iBlock][iCell];
                            for (index j = 0; j < meanBlocks[iBlock].ComponentNumber(); ++j)
                                for (int mode = 0; mode < modes; ++mode)
                                {
                                    const real value = row[j * modes + mode];
                                    const index logical = logicalFirst[iBlock] + j;
                                    point.values[logical] += value * basis(0, mode);
                                    if (viscous)
                                        for (int d = 0; d < dim; ++d)
                                            point.gradient(logical, d) += viscousRow[j * modes + mode] * basis(1 + d, mode);
                                }
                        }
                    }
                }
                if (ransModel)
                    RANSStateView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairReconstruction(ransMean[iCell], point.values.data() + ransFirst);
                if (primitiveGradient)
                {
                    const auto row = primitiveGradientValues[iCell];
                    for (int j = 0; j < 5; ++j)
                        for (int d = 0; d < dim; ++d)
                            point.gradient(flowFirst + j, d) = row[j * dim + d];
                }
                // Equation reconstruction PP deliberately uses Euler's final
                // property-repaired thermal postcheck. Requiring the raw
                // species simplex again here would reject coefficients that
                // the PP publication has already accepted. Standalone face
                // fallback retains the stricter raw-state predicate.
                const bool requireRawSpecies = !owner->settings.numerics.positivity;
                if ((owner->settings.numerics.faceFallback || owner->settings.numerics.positivity) &&
                    !thermalBounds.Admissible(point.values, requireRawSpecies))
                {
                    ++localFallbacks;
                    gatherMean(iCell, point);
                }
            };
            const auto computeFace = [&](index iAux)
            {
                const index first = owner->faceEntries[iAux];
                const auto &anchor = owner->samples[first];
                const index iCell = anchor.cell;
                const index end = first + metric.FaceQuadratureEnd(anchor.face) - metric.FaceQuadratureBegin(anchor.face);
                std::fill(integrated.begin(), integrated.end(), 0);
                if (produceLow)
                    std::fill(integratedLow.begin(), integratedLow.end(), 0);
                for (index entry = first; entry < end; ++entry)
                {
                    const auto &sample = owner->samples[entry];
                    const bool internal = sample.other != UnInitIndex;
                    const auto boundaryKind = sample.boundary.kind;
                    const real weight = metric.GetFaceWeight(sample.point);
                    reconstruct(iCell, sample, sample.side, left);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    real wallSAViscousInterior = 0;
#endif
                    if (internal)
                    {
                        reconstruct(sample.other, sample, 1 - sample.side, right);
                        if (sample.changesFrame)
                            RotatePoint(right, sample.rotation, vectorStarts);
                    }
                    else
                    {
                        right.values = left.values;
                        right.gradient = left.gradient;
                        BoundaryValue(right, flowFirst, speciesFirst, independent, sample, evaluator, frame, time, true,
                                      boundaryInput(sample), incoming);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                        if (ransModel && ransModel->Kind() == Physics::RANS::ModelKind::SA && sample.boundary.kind == BoundaryKind::NoSlipWall)
                        {
                            // Euler retains ULxyUnlim before its BC zeros the
                            // convective SA pair. Preserve that interior value
                            // for the viscous jump without mutating cell means.
                            wallSAViscousInterior = left.values[ransFirst];
                            left.values[ransFirst] = right.values[ransFirst] = 0;
                        }
#endif
                    }
                    auto leftFlow = left.Flow(flowFirst), rightFlow = right.Flow(flowFirst);
                    const bool wallFluxState = noRsOnWall && !internal && boundaryKind == BoundaryKind::NoSlipWall;
                    const auto prepareWallState = [&]
                    {
                        face.values = left.values;
                        const auto original = left.Flow(flowFirst);
                        const Physics::Chemistry::SpeciesBufferView wallSpecies{independent ? face.values.data() + speciesFirst : nullptr, independent};
                        const auto wall = sample.boundary.FaceValue(evaluator, original, wallSpecies,
                                                                    {sample.normal[0], sample.normal[1], sample.normal[2]},
                                                                    {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time, &frame);
                        for (int j = 0; j < 5; ++j)
                            face.values[flowFirst + j] = wall[j];
                        if (ransModel)
                            for (int j = 0; j < ransModel->ComponentNumber(); ++j)
                                face.values[ransFirst + j] *= wall[0] / original[0];
                        return wall;
                    };
                    if (wallFluxState)
                        leftFlow = rightFlow = prepareWallState();
                    const auto leftData = EquationPointEvaluator::FluxState(leftFlow, evaluator.Evaluate(leftFlow, wallFluxState ? face.Species(speciesFirst, independent) : left.Species(speciesFirst, independent), 0, {}));
                    const auto rightData = EquationPointEvaluator::FluxState(rightFlow, evaluator.Evaluate(rightFlow, wallFluxState ? face.Species(speciesFirst, independent) : right.Species(speciesFirst, independent), 0, {}));
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto grid = frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                    auto meanLeftData = leftData, meanRightData = rightData;
                    if (fluxRequirements.meanStates || produceLow)
                    {
                        // Mean eigenvalue preparation follows the same path as
                        // flow5. Storage/block traversal stays in this adapter.
                        Gather(meanBlocks, iCell, cellMean);
                        const auto meanLeft = cellMean.Flow(flowFirst);
                        auto meanRight = meanLeft;
                        if (internal)
                        {
                            Gather(meanBlocks, sample.other, cellMean);
                            meanRight = sample.changesFrame ? FiniteVolumeDetail::RotateState(cellMean.Flow(flowFirst), sample.rotation) : cellMean.Flow(flowFirst);
                        }
                        else
                            meanRight = sample.boundary.kind == BoundaryKind::Interface
                                            ? FiniteVolumeDetail::Load(incoming, sample.boundaryPoint)
                                            : FiniteVolumeDetail::BoundaryExterior(sample, owner->equation.Gas(), meanLeft, time, frame, {});
                        meanLeftData = preparedMean
                                           ? Riemann::AcousticState{{meanLeft, (*preparedMean)(iCell, 0)}, (*preparedMean)(iCell, 1)}
                                           : owner->equation.PrepareFluxState(meanLeft);
                        meanRightData = preparedMean && internal
                                            ? Riemann::AcousticState{{meanRight, (*preparedMean)(sample.other, 0)}, (*preparedMean)(sample.other, 1)}
                                            : owner->equation.PrepareFluxState(meanRight);
                    }
                    auto fluxSettings = baseFluxSettings;
                    if (useWaveJump)
                    {
                        real jump = waveJump(iCell, 0);
                        if (internal)
                            jump = std::max(jump, waveJump(sample.other, 0));
                        fluxSettings.dLambda = std::max(fluxSettings.dLambda, jump);
                    }
                    if (usePressureMinimum)
                    {
                        real minimum = pressureMinimum(iCell, 0);
                        if (internal)
                            minimum = std::min(minimum, pressureMinimum(sample.other, 0));
                        fluxSettings.pressureStencilMin = std::min(fluxSettings.pressureStencilMin, minimum);
                    }
                    const auto convection = owner->equation.EvaluateFlowFlux(owner->flux, leftData, rightData, meanLeftData, meanRightData,
                                                                             normal, grid, fluxSettings);
                    const auto &flowFlux = convection.flux;
                    FlowState lowFlowFlux{};
                    if (produceLow)
                        lowFlowFlux = Riemann::Rusanov(wallFluxState ? leftData : meanLeftData,
                                                       wallFluxState ? rightData : meanRightData, normal, grid);
                    real speedLeft = 0, speedRight = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        speedLeft += (leftFlow[1 + d] / leftFlow[0] - grid[d]) * normal[d];
                        speedRight += (rightFlow[1 + d] / rightFlow[0] - grid[d]) * normal[d];
                    }
                    // The scalar kernel receives a fixed speed, not the flow5
                    // acoustic flux. Roe's prepared contact magnitude is the
                    // Euler velocity/entropy-fix policy. Rusanov's diagnostic
                    // contact is acoustic, so use an explicit velocity speed.
                    const real alpha = rusanovFlux
                                           ? .5 * (std::abs(speedLeft) + std::abs(speedRight))
                                           : convection.contact;
                    for (index j = 0; j < width; ++j)
                        if (j < flowFirst || j >= flowFirst + 5)
                            flux[j] = wallFluxState ? face.values[j] * speedLeft : AdvectedScalarFlux(left.values[j], right.values[j], speedLeft, speedRight, alpha);
                    for (int j = 0; j < 5; ++j)
                    {
                        flux[flowFirst + j] = flowFlux[j];
                        if (produceLow)
                            lowFlux[flowFirst + j] = lowFlowFlux[j];
                    }
                    if (ransModel && ransModel->ComponentNumber() == 2)
                    {
                        FlowState average;
                        for (int j = 0; j < 5; ++j)
                            average[j] = .5 * (leftFlow[j] + rightFlow[j]);
                        const auto stress = TurbulentNormalFlux(average, wallFluxState ? face.values[ransFirst] : .5 * (left.values[ransFirst] + right.values[ransFirst]), normal);
                        for (int j = 0; j < 5; ++j)
                            flux[flowFirst + j] += stress[j];
                    }
                    const bool wall = !internal && boundaryKind == BoundaryKind::NoSlipWall;
                    if (viscous && (internal || wall))
                    {
                        if (primitiveGradient)
                        {
                            DNDS_assert_info(!ransModel && independent == 0 && flowFirst == 0 && width == 5,
                                             "primitive-gradient leaf requires pure flow5");
                            const auto &gas = owner->equation.Gas();
                            auto primitive = gas.Primitive(leftFlow);
                            FlowGradient gradient{};
                            for (int j = 0; j < 5; ++j)
                                for (int d = 0; d < 3; ++d)
                                    gradient[j][d] = left.gradient(flowFirst + j, d);
                            Gather(meanBlocks, iCell, cellMean);
                            const auto ownMean = cellMean.Flow(flowFirst);
                            const auto ownPrimitive = gas.Primitive(ownMean);
                            FlowState rightMean;
                            if (internal)
                            {
                                Gather(meanBlocks, sample.other, cellMean);
                                rightMean = sample.changesFrame
                                                ? FiniteVolumeDetail::RotateState(cellMean.Flow(flowFirst), sample.rotation)
                                                : cellMean.Flow(flowFirst);
                            }
                            else
                                rightMean = FiniteVolumeDetail::BoundaryGradientValue(sample, gas, ownMean, time, frame, incoming);
                            const auto rightPrimitive = gas.Primitive(rightMean);
                            const Geom::tPoint distance = !internal
                                                              ? Geom::tPoint(sample.ownPoint - metric.GetCellBary(iCell))
                                                              : Geom::tPoint(sample.otherPoint - metric.GetCellBary(iCell));
                            const real lengthSquared = distance.squaredNorm();
                            DNDS_assert_info(lengthSquared > 0, "zero primitive viscous stencil displacement");
                            const std::array<real, 5> own{ownPrimitive.density, ownPrimitive.velocity[0], ownPrimitive.velocity[1],
                                                          ownPrimitive.velocity[2], ownPrimitive.temperature};
                            const std::array<real, 5> neighbor{rightPrimitive.density, rightPrimitive.velocity[0], rightPrimitive.velocity[1],
                                                               rightPrimitive.velocity[2], rightPrimitive.temperature};
                            for (int j = 0; j < 5; ++j)
                            {
                                real projected = 0;
                                for (int d = 0; d < 3; ++d)
                                {
                                    if (internal)
                                        gradient[j][d] = .5 * (gradient[j][d] + right.gradient(flowFirst + j, d));
                                    projected += gradient[j][d] * distance[d];
                                }
                                for (int d = 0; d < 3; ++d)
                                    gradient[j][d] += (neighbor[j] - own[j] - projected) * distance[d] / lengthSquared;
                            }
                            if (internal)
                            {
                                const auto other = gas.Primitive(rightFlow);
                                for (int d = 0; d < 3; ++d)
                                    primitive.velocity[d] = .5 * (primitive.velocity[d] + other.velocity[d]);
                            }
                            else
                            {
                                FlowState faceMean{};
                                for (int j = 0; j < 5; ++j)
                                    faceMean[j] = .5 * (leftFlow[j] + rightFlow[j]);
                                primitive = gas.Primitive(faceMean);
                                if (!sample.boundary.isothermal)
                                {
                                    real heat = 0;
                                    for (int d = 0; d < 3; ++d)
                                        heat += gradient[4][d] * normal[d];
                                    for (int d = 0; d < 3; ++d)
                                        gradient[4][d] -= heat * normal[d];
                                }
                            }
                            const auto diffusion = gas.ViscousFlux(primitive, gradient, normal);
                            for (int j = 0; j < 5; ++j)
                            {
                                flux[j] -= diffusion[j];
                                if (produceLow)
                                    lowFlux[j] -= diffusion[j];
                            }
                        }
                        else
                        {
                            const real inverseDistance = wall && noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                            for (index j = 0; j < width; ++j)
                            {
                                face.values[j] = .5 * (left.values[j] + right.values[j]);
                                for (int d = 0; d < 3; ++d)
                                    face.gradient(j, d) = .5 * (left.gradient(j, d) + right.gradient(j, d)) +
                                                          normal[d] * (right.values[j] - left.values[j]) * inverseDistance;
                            }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                            if (wallSAViscousInterior != 0)
                                for (int d = 0; d < 3; ++d)
                                    face.gradient(ransFirst, d) -= normal[d] * wallSAViscousInterior * inverseDistance;
#endif
                            if (wallFluxState)
                                prepareWallState();
                            const auto faceFlow = face.Flow(flowFirst);
                            const auto point = evaluator.Evaluate(faceFlow, face.Species(speciesFirst, independent), 0, {true, true, true, false, false});
                            struct SpeciesGradient
                            {
                                const PointScratch &point;
                                index first;
                                real operator()(int k, int d) const { return point.gradient(first + k, d); }
                            };
                            EquationTransportOptions transportOptions{wall && !sample.boundary.isothermal, wall};
                            if (ransFirst >= 0)
                            {
                                const auto &model = owner->equation.RANS();
                                const index iQ = sample.point - metric.FaceQuadratureBegin(sample.face);
                                // Transport does not consume the DES source length.
                                const auto ransPoint = RANSPoint(face, flowFirst, ransFirst, model, point, wallValues.faces(sample.face, iQ), 0);
                                const auto turbulent = model.Transport(ransPoint);
                                transportOptions.turbulentViscosity = turbulent.viscosity;
                                transportOptions.turbulentConductivity = turbulent.conductivity;
                                transportOptions.turbulentSpeciesDiffusivity = turbulent.speciesDiffusivity;
                                transportOptions.useQCR = model.Settings().useQCR;
                                const auto diffusionRANS = model.DiffusionFlux(ransPoint, Eigen::Vector3d{normal[0], normal[1], normal[2]}, turbulent.viscosity);
                                for (int j = 0; j < model.ComponentNumber(); ++j)
                                    flux[ransFirst + j] -= diffusionRANS[j];
                            }
                            const auto diffusion = PrepareViscousFlux(faceFlow, face.FlowDerivative(flowFirst), point, face.Species(speciesFirst, independent),
                                                                      SpeciesGradient{face, speciesFirst}, normal, speciesFlux, transport, transportOptions);
                            for (int j = 0; j < 5; ++j)
                            {
                                flux[flowFirst + j] -= diffusion[j];
                                if (produceLow)
                                    lowFlux[flowFirst + j] -= diffusion[j];
                            }
                            for (int k = 0; k < independent; ++k)
                                flux[speciesFirst + k] -= speciesFlux[k];
                        }
                    }
                    if (sample.boundaryPoint != UnInitIndex)
                        for (int d = 0; d < 3; ++d)
                            wallFlux(sample.boundaryPoint, d) = wall || boundaryKind == BoundaryKind::SlipWall
                                                                    ? weight * flux[flowFirst + 1 + d]
                                                                    : 0;
                    for (index j = 0; j < width; ++j)
                    {
                        integrated[j] -= weight * flux[j];
                        if (produceLow)
                            integratedLow[j] -= weight * lowFlux[j];
                    }
                }
                for (index j = 0; j < width; ++j)
                {
                    faceValues(iAux, j) = integrated[j];
                    if (produceLow)
                        lowFaceValues(iAux, j) = integratedLow[j];
                }
            };
            if (region == CFV::ReconstructionRegion::All)
                for (index iAux = 0; iAux < index(owner->faceEntries.size()); ++iAux)
                    computeFace(iAux);
            else
                for (const auto iAux : region == CFV::ReconstructionRegion::Core ? owner->rhsCoreFaces : owner->rhsDependentFaces)
                    computeFace(iAux);
            const auto compute = [&](index iCell)
            {
                std::fill(integrated.begin(), integrated.end(), 0);
                if (produceLow)
                    std::fill(integratedLow.begin(), integratedLow.end(), 0);
                for (index entry = owner->offsets[iCell]; entry < owner->offsets[iCell + 1];)
                {
                    const auto &sample = owner->samples[entry];
                    const index iAux = sample.auxiliaryFace;
                    const auto &anchor = owner->samples[owner->faceEntries[iAux]];
                    const bool opposite = sample.side != anchor.side;
                    for (index j = 0; j < width; ++j)
                    {
                        flux[j] = (opposite ? -1 : 1) * faceValues(iAux, j);
                        if (produceLow)
                            lowFlux[j] = (opposite ? -1 : 1) * lowFaceValues(iAux, j);
                    }
                    // Scalar fluxes change sign; vector fluxes additionally
                    // rotate from the other parent's periodic component frame.
                    // Side, not cell identity, distinguishes self-periodic faces.
                    if (opposite && sample.changesFrame)
                        for (const index first : vectorStarts)
                        {
                            const Geom::tPoint value{flux[first], flux[first + 1], flux[first + 2]};
                            const Geom::tPoint rotated = sample.rotation * value;
                            for (int d = 0; d < 3; ++d)
                                flux[first + d] = rotated[d];
                            if (produceLow)
                            {
                                const Geom::tPoint lowValue{lowFlux[first], lowFlux[first + 1], lowFlux[first + 2]};
                                const Geom::tPoint lowRotated = sample.rotation * lowValue;
                                for (int d = 0; d < 3; ++d)
                                    lowFlux[first + d] = lowRotated[d];
                            }
                        }
                    for (index j = 0; j < width; ++j)
                    {
                        integrated[j] += flux[j];
                        if (produceLow)
                            integratedLow[j] += lowFlux[j];
                    }
                    entry += metric.FaceQuadratureEnd(sample.face) - metric.FaceQuadratureBegin(sample.face);
                }
                for (index j = 0; j < destination.ComponentNumber(); ++j)
                {
                    destination(iCell, j) = integrated[j];
                    if (produceLow)
                        lowDestination(iCell, j) = integratedLow[j];
                }
            };
            if (region == CFV::ReconstructionRegion::All)
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                    compute(iCell);
            else
                for (const auto iCell : region == CFV::ReconstructionRegion::Core ? owner->partition.Core() : owner->partition.HaloDependent())
                    compute(iCell);
            const size_t slot = region == CFV::ReconstructionRegion::HaloDependent ? 1 : 0;
            faceFallbacks[slot] = localFallbacks;
            if (region == CFV::ReconstructionRegion::All)
                faceFallbacks[1] = 0;
            if ((!owner->settings.numerics.faceFallback || owner->settings.numerics.positivity) && localFallbacks)
                DNDS_check_throw_info(false, "inadmissible reconstructed face with faceFallback disabled");
        }
        void SampleSourceState(const EquationDofs &state)
        {
            const PhaseTime timing(*this, SamplingPhase);
            sourceState.reset();
            sourceCoefficients.reset();
            if (!sourceSampling)
                return;
            // P0 face convection/viscosity does not disable these derivatives.
            // Direct-FV sources consume physical gradients without a basis.
            if (ransFirst >= 0 && owner->settings.numerics.reconstruction != "variational")
                RefreshDirectGradients(false);
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                if (ransFirst >= 0 && owner->settings.numerics.reconstruction == "variational")
                {
                    CFV::PolynomialSampleOutputs outputs;
                    outputs.values = &sourceValues[iField];
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        outputs.gradients[d] = &sourceGradients[iField][d];
                    sourceSampling->EvaluateValueGradient(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), outputs);
                    continue;
                }
                if (owner->settings.numerics.reconstruction == "variational")
                    sourceSampling->Evaluate(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), sourceValues[iField]);
                else
                {
                    // Euler's direct FV source uses the cell mean at its
                    // center, together with the direct gradient. In a curved
                    // or distorted cell the reference center need not be the
                    // volume barycenter: do not add a reconstructed offset.
                    const auto read = state.PhysicalField(iField).Owned().Read();
                    auto write = sourceValues[iField].Write();
                    const auto means = read.View<R::UniformTableRows>();
                    const auto values = write.View<R::UniformTableRows>();
                    for (index iCell = 0; iCell < means.RowCount(); ++iCell)
                        for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
                            for (index j = 0; j < means.Width(); ++j)
                                values(iPoint, j) = means(iCell, j);
                }
                if (ransFirst >= 0)
                {
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        if (owner->settings.numerics.reconstruction == "variational")
                            sourceSampling->Evaluate(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), sourceGradients[iField][d], 1 + d);
                        else if (!sourceValuesAreMeans)
                        {
                            const auto read = directGradients[iField].Owned().Read();
                            auto write = sourceGradients[iField][d].Write();
                            const auto input = read.View<R::UniformTableRows>();
                            const auto output = write.View<R::UniformTableRows>();
                            const int dim = owner->metrics.Dimension();
                            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                                for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
                                    for (index j = 0; j < output.Width(); ++j)
                                        output(iPoint, j) = input(iCell, j * dim + d);
                        }
                }
            }
        }
        /// Opt-in source linearization at the already sampled RHS source
        /// quadrature. This is deliberately separate from the retained
        /// one-point approximate Jacobian and uses only the current published
        /// limited/PP polynomial values/gradients.
        bool AddRHSQuadratureSourceJacobian(const EquationDofs &state,
                                            const R::Array<real>::WriteLease &diagonal,
                                            DiagonalStorage storage, real scale,
                                            EquationPointEvaluator &evaluator,
                                            const MeanLinearizationOutput *linearization) const
        {
            if (!sourceSampling)
                return false;
            if (owner->settings.numerics.reconstruction != "variational")
                return false;
            const auto means = state.ReadOwned();
            const auto blocks = BlockViews(state, means);
            const auto geometry = owner->metrics.AcquireRead();
            const auto metric = geometry.HostView();
            std::optional<WallDistanceRead> wallRead;
            std::optional<R::Array<real>::ReadLease> lengthRead;
            if (ransFirst >= 0)
            {
                wallRead.emplace(owner->wallDistance->AcquireRead());
                lengthRead.emplace(owner->ransCellLength.Read());
            }
            const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
            const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>()
                                            : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
            const auto matrix = diagonal.View<R::TableRows>();
            const index nv = state.ComponentNumber();
            const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
            std::vector<EquationBlockLocation> locations;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                locations.push_back(state.LocateBlock(iBlock));
            PointScratch point(nv);
            // The linearization consumer still needs mean pressure and its
            // conservative derivative. Fill it separately from quadrature
            // source points so the high-order source path does not silently
            // replace the mean-state EOS contract.
            if (linearization)
            {
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                {
                    Gather(blocks, iCell, point, false);
                    const auto properties = evaluator.Evaluate(
                        point.Flow(flowFirst), point.Species(speciesFirst, independent), 0,
                        {ransFirst >= 0, false, true, false, false});
                    linearization->Publish(iCell, point, properties);
                }
            }
            std::vector<R::Array<real>::ReadLease> sourceReads;
            sourceReads.reserve(sourceValues.size());
            for (const auto &field : sourceValues)
                sourceReads.push_back(field.Read());
            std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> gradientReads(sourceGradients.size());
            for (index iField = 0; iField < index(sourceGradients.size()); ++iField)
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    gradientReads[iField][d].emplace(sourceGradients[iField][d].Read());
            const auto weightRead = sourceWeights.Read();
            const auto weights = weightRead.View<R::FixedRows<1>>();
            std::vector<RHSScalarView> samples;
            samples.reserve(sourceReads.size());
            for (const auto &read : sourceReads)
                samples.push_back(read.View<R::UniformTableRows>());
            std::vector<std::array<RHSScalarView, 3>> gradients(sourceGradients.size());
            for (index iField = 0; iField < index(sourceGradients.size()); ++iField)
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    gradients[iField][d] = gradientReads[iField][d]->View<R::UniformTableRows>();
            const auto ransMean = ransFirst >= 0 ? means.Block("rans") : EquationBlockReadView{};
            const auto addTerms = [&](index iCell, const PointScratch &at,
                                      const EquationPointProperties &properties, real weight,
                                      real wallDistance)
            {
                if (ransFirst >= 0)
                {
                    const auto &model = owner->equation.RANS();
                    const auto input = RANSPoint(at, flowFirst, ransFirst, model, properties, wallDistance, lengths(iCell, 0));
                    const auto destruction = model.ImplicitSourceDiagonal(input);
                    for (int j = 0; j < model.ComponentNumber(); ++j)
                    {
                        DNDS_check_throw_info(std::isfinite(destruction[j]), "nonfinite quadrature RANS source derivative");
                        const index row = ransFirst + j;
                        matrix(iCell, storage == DiagonalStorage::Block ? row + nv * row : row) +=
                            scale * weight * destruction[j];
                    }
                }
                if (chemical)
                    for (int j = 0; j < 5 + independent; ++j)
                    {
                        const index column = j < 5 ? flowFirst + j : speciesFirst + j - 5;
                        for (int k = 0; k < independent; ++k)
                        {
                            const index row = speciesFirst + k;
                            const real derivative = properties.sourceJacobian(k, j);
                            DNDS_check_throw_info(std::isfinite(derivative), "nonfinite quadrature source derivative");
                            if (storage == DiagonalStorage::Block || row == column)
                                matrix(iCell, storage == DiagonalStorage::Block ? row + nv * column : row) -=
                                    scale * weight * derivative;
                        }
                    }
            };
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
            {
                for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
                {
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto location = locations[iBlock];
                        const auto values = samples[location.physicalField][iPoint];
                        for (index j = 0; j < location.componentNumber; ++j)
                            point.values[logicalFirst[iBlock] + j] = values[location.firstComponent + j];
                        if (ransFirst >= 0)
                            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                            {
                                const auto values = gradients[location.physicalField][d][iPoint];
                                for (index j = 0; j < location.componentNumber; ++j)
                                    point.gradient(logicalFirst[iBlock] + j, d) =
                                        values[location.firstComponent + j];
                            }
                    }
                    if (ransFirst >= 0)
                    {
                        const auto &model = owner->equation.RANS();
                        RANSStateView{model.Kind(), model.ComponentNumber()}.RepairReconstruction(
                            ransMean[iCell], point.values.data() + ransFirst);
                    }
                    const auto properties = evaluator.Evaluate(
                        point.Flow(flowFirst), point.Species(speciesFirst, independent), 0,
                        {ransFirst >= 0, false, false, chemical, chemical});
                    const real wallDistance = ransFirst >= 0
                                                  ? walls.cells(iCell, iPoint - sourceOffsets[iCell])
                                                  : real(1);
                    addTerms(iCell, point, properties, weights(iPoint, 0), wallDistance);
                }
            }
            return true;
        }

        /// Add the approximate source matrix to caller-owned output.
        /// The default is one mean point per cell; rhsQuadrature selects the
        /// explicit same-quadrature source linearization.
        /// The controller retains current means/coefficients and ready source
        /// gradients. Optional linearization outputs consume this same mean
        /// property evaluation; false means no samples existed and no outputs
        /// were filled. This entry neither samples RHS quadrature nor publishes
        /// source freshness; the production wrapper validates its provenance.
        bool AddMeanSourceJacobian(const EquationDofs &state, const R::Array<real>::WriteLease &diagonal,
                                   DiagonalStorage storage, real scale, EquationPointEvaluator &evaluator,
                                   const MeanLinearizationOutput *linearization = nullptr) const
        {
            if (!sourceSampling)
                return false;
            if (owner->settings.implicit.sourceJacobianMode == "rhsQuadrature")
                return AddRHSQuadratureSourceJacobian(state, diagonal, storage, scale, evaluator, linearization);
            // The default approximate source matrix is a one-point,
            // mean-state operator. The opt-in quadrature path is a separate
            // branch selected by the explicit implicit source mode.
            // RANS still consumes the current polynomial's center gradients.
            std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> gradientReads;
            // Direct gradients already contain every direction in one physical
            // field. Retain that allocation's read lease and select the direction
            // through indexing, rather than copying whole fields into scratch.
            // Polynomial center sampling keeps its separate per-direction arrays.
            struct SourceGradientView
            {
                RHSScalarView values;
                int componentStride = 1, direction = 0;
                real operator()(index iCell, index component) const
                {
                    return values(iCell, component * componentStride + direction);
                }
            };
            std::vector<std::array<SourceGradientView, 3>> gradients;
            if (ransFirst >= 0)
            {
                gradientReads.resize(size_t(state.PhysicalFieldNumber()));
                gradients.resize(size_t(state.PhysicalFieldNumber()));
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    if (owner->settings.numerics.reconstruction != "variational")
                    {
                        gradientReads[iField][0].emplace(directGradients[iField].Owned().Read());
                        const auto view = gradientReads[iField][0]->View<R::UniformTableRows>();
                        for (int d = 0; d < owner->metrics.Dimension(); ++d)
                            gradients[iField][d] = {view, owner->metrics.Dimension(), d};
                        continue;
                    }
                    CFV::PolynomialSampleOutputs outputs;
                    // Workspace identity is const; shallow mutable handles
                    // retain its existing scratch allocations for the write.
                    auto scratch = sourceJacobianGradients[iField];
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        outputs.gradients[d] = &scratch[d];
                    // One center, gradients only. The sampling facility owns
                    // bounded layout specialization and generic fallback; no
                    // unused sampled-value field is allocated for the matrix.
                    sourceJacobianSampling->EvaluateValueGradient(state.PhysicalField(iField), FluxCoefficients().PhysicalField(iField), outputs);
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    {
                        gradientReads[iField][d].emplace(sourceJacobianGradients[iField][d].Read());
                        gradients[iField][d] = {gradientReads[iField][d]->View<R::UniformTableRows>(), 1, 0};
                    }
                }
            }
            const auto means = state.ReadOwned();
            const auto blocks = BlockViews(state, means);
            const auto geometry = owner->metrics.AcquireRead();
            const auto metric = geometry.HostView();
            std::optional<WallDistanceRead> wallRead;
            std::optional<R::Array<real>::ReadLease> lengthRead;
            if (ransFirst >= 0)
            {
                wallRead.emplace(owner->wallDistance->AcquireRead());
                lengthRead.emplace(owner->ransCellLength.Read());
            }
            const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
            const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>() : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
            const auto matrix = diagonal.View<R::TableRows>();
            const index nv = state.ComponentNumber();
            const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
            PointScratch point(nv);
            std::vector<EquationBlockLocation> locations;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                locations.push_back(state.LocateBlock(iBlock));
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
            {
                Gather(blocks, iCell, point, false);
                if (ransFirst >= 0)
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto location = locations[iBlock];
                        for (index j = 0; j < location.componentNumber; ++j)
                            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                                point.gradient(logicalFirst[iBlock] + j, d) = gradients[location.physicalField][d](iCell, location.firstComponent + j);
                    }
                const auto properties = evaluator.Evaluate(point.Flow(flowFirst), point.Species(speciesFirst, independent), 0,
                                                           {ransFirst >= 0, false, bool(linearization), chemical, chemical});
                if (linearization)
                    linearization->Publish(iCell, point, properties);
                const real volume = metric.GetCellVol(iCell);
                if (ransFirst >= 0)
                {
                    const auto &model = owner->equation.RANS();
                    const auto input = RANSPoint(point, flowFirst, ransFirst, model, properties, walls.cellMean(iCell, 0), lengths(iCell, 0));
                    const auto destruction = model.ImplicitSourceDiagonal(input);
                    for (int j = 0; j < model.ComponentNumber(); ++j)
                    {
                        DNDS_check_throw_info(std::isfinite(destruction[j]), "nonfinite mean-state RANS destruction diagonal");
                        const index row = ransFirst + j;
                        matrix(iCell, storage == DiagonalStorage::Block ? row + nv * row : row) += scale * volume * destruction[j];
                    }
                }
                if (chemical)
                    for (int j = 0; j < 5 + independent; ++j)
                    {
                        const index column = j < 5 ? flowFirst + j : speciesFirst + j - 5;
                        for (int k = 0; k < independent; ++k)
                        {
                            const index row = speciesFirst + k;
                            const real derivative = properties.sourceJacobian(k, j);
                            DNDS_check_throw_info(std::isfinite(derivative), "nonfinite mean-state source derivative");
                            if (storage == DiagonalStorage::Block || row == column)
                                matrix(iCell, storage == DiagonalStorage::Block ? row + nv * column : row) -= scale * volume * derivative;
                        }
                    }
            }
            return true;
        }

        void PublishSourceState(const EquationDofs &state, real time)
        {
            if (!sourceSampling)
                return;
            sourceState.emplace(state.NumericAlias());
            sourceCoefficients.emplace(FluxCoefficients());
            sourceTime = time;
        }
        /// ValidateWorkspace has already fixed schema, binding and geometry.
        /// Source samples depend only on owned means, the published flux
        /// polynomial and physical time. Ghost refreshes do not change them.
        /// Reserve readers as well as checking generations: an active writer
        /// has not published its next generation yet and must still exclude us.
        bool SourceSamplesCurrent(const EquationDofs &state, real time) const
        {
            if (!sourceSampling)
                return true;
            if (!sourceState || !sourceCoefficients || sourceTime != time)
                return false;
            const auto means = state.ReadOwned(), coefficients = FluxCoefficients().ReadOwned();
            const auto unchanged = [](const EquationDofOperations::Binding &publication, const EquationDofs &values)
            {
                if (publication.snapshot.PhysicalFieldNumber() != values.PhysicalFieldNumber())
                    return false;
                for (index iField = 0; iField < values.PhysicalFieldNumber(); ++iField)
                {
                    const auto &saved = publication.snapshot.PhysicalField(iField).Owned();
                    const auto &current = values.PhysicalField(iField).Owned();
                    if (saved.SharedStorage() != current.SharedStorage() || saved.ShapeVersion() != current.ShapeVersion() ||
                        publication.versions[iField] != current.Version())
                        return false;
                }
                return true;
            };
            return unchanged(*sourceState, state) && sourceCoefficients->Current(FluxCoefficients());
        }
        /// High-order RHS source sampling supplies the opt-in quadrature
        /// source Jacobian or refreshes publication provenance only. The
        /// default Jacobian continues to sample its own center gradients.
        void PrepareVolumeSource(const EquationDofs &state,
                                 std::optional<std::reference_wrapper<EquationDofs>> result,
                                 EquationPointEvaluator &evaluator, real time = 0,
                                 std::optional<std::reference_wrapper<EquationDofs>> lowOrder = std::nullopt)
        {
            const PhaseTime timing(*this, SourcePhase);
            // Kinematic and constitutive sources share this additive entrance;
            // face-only calls never include either kind of source.
            const auto frame = owner->motion.At(time);
            if (result && frame.moving)
            {
                const auto read = state.ReadOwned();
                auto write = result->get().WriteOwned();
                std::optional<EquationDofs::WriteScope> lowWrite;
                if (lowOrder)
                {
                    state.RequireCompatible(lowOrder->get());
                    EquationDofOperations::RequireDistinct(result->get(), lowOrder->get());
                    lowWrite.emplace(lowOrder->get().WriteOwned());
                }
                const auto flow = read.Block("flow");
                const auto output = write.Block("flow");
                const auto lowOutput = lowWrite ? lowWrite->Block("flow") : EquationBlockArrayView<real>{};
                const auto geometry = owner->metrics.AcquireRead();
                const auto metric = geometry.HostView();
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                {
                    FlowState mean{};
                    for (int j = 0; j < 5; ++j)
                        mean[j] = flow(iCell, j);
                    const auto source = frame.RotationSource(mean);
                    for (int j = 0; j < 5; ++j)
                    {
                        output(iCell, j) += metric.GetCellVol(iCell) * source[j];
                        if (lowWrite)
                            lowOutput(iCell, j) += metric.GetCellVol(iCell) * source[j];
                    }
                }
            }
            if (!sourceSampling)
                return;
            SampleSourceState(state);
            if (!result)
                return;
            std::vector<R::Array<real>::ReadLease> sources;
            for (const auto &field : sourceValues)
                sources.push_back(field.Read());
            std::vector<std::array<std::optional<R::Array<real>::ReadLease>, 3>> derivatives(sourceGradients.size());
            for (index iField = 0; iField < index(sourceGradients.size()); ++iField)
                if (sourceValuesAreMeans)
                    derivatives[iField][0].emplace(directGradients[iField].Owned().Read());
                else
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        derivatives[iField][d].emplace(sourceGradients[iField][d].Read());
            // Resolve physical layouts once, outside the quadrature loop.
            using SampleView = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
            std::vector<SampleView> samples;
            struct SourceGradientView
            {
                SampleView values;
                int componentStride = 1, direction = 0;
                real operator()(index iPoint, index component) const
                {
                    return values(iPoint, component * componentStride + direction);
                }
            };
            std::vector<std::array<SourceGradientView, 3>> gradients(derivatives.size());
            for (const auto &read : sources)
                samples.push_back(read.View<R::UniformTableRows>());
            for (index iField = 0; iField < index(derivatives.size()); ++iField)
                for (int d = 0; d < owner->metrics.Dimension(); ++d)
                    gradients[iField][d] = sourceValuesAreMeans
                                               ? SourceGradientView{derivatives[iField][0]->View<R::UniformTableRows>(), owner->metrics.Dimension(), d}
                                               : SourceGradientView{derivatives[iField][d]->View<R::UniformTableRows>(), 1, 0};
            std::optional<WallDistanceRead> wallRead;
            if (owner->wallDistance)
                wallRead.emplace(owner->wallDistance->AcquireRead());
            const auto walls = wallRead ? wallRead->HostView() : WallDistanceView{};
            const auto weightRead = sourceWeights.Read();
            const auto weights = weightRead.View<R::FixedRows<1>>();
            std::optional<R::Array<real>::ReadLease> lengthRead;
            if (ransFirst >= 0)
                lengthRead.emplace(owner->ransCellLength.Read());
            const auto lengths = lengthRead ? lengthRead->View<R::FixedRows<1>>() : R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<1>>{};
            std::optional<EquationDofs::WriteScope> output;
            if (result)
                output.emplace(result->get().WriteOwned());
            const auto speciesOutput = result && speciesFirst >= 0 ? output->Block("species") : EquationBlockArrayView<real>{};
            const auto ransOutput = result && ransFirst >= 0 ? output->Block("rans") : EquationBlockArrayView<real>{};
            const int independent = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() - 1 : 0;
            PointScratch point(state.ComponentNumber());
            const auto mean = state.ReadOwned();
            const auto ransMean = ransFirst >= 0 ? mean.Block("rans") : EquationBlockReadView{};
            std::vector<EquationBlockLocation> locations;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                locations.push_back(state.LocateBlock(iBlock));
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                for (index iPoint = sourceOffsets[iCell]; iPoint < sourceOffsets[iCell + 1]; ++iPoint)
                {
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto location = locations[iBlock];
                        // Resolve each borrowed row before traversing its
                        // components. The span retains physical element stride;
                        // no packing or AoS storage assumption is introduced.
                        const auto values = samples[location.physicalField][iPoint];
                        for (index j = 0; j < location.componentNumber; ++j)
                            point.values[logicalFirst[iBlock] + j] = values[location.firstComponent + j];
                        if (ransFirst >= 0)
                            for (int d = 0; d < owner->metrics.Dimension(); ++d)
                            {
                                const auto &gradient = gradients[location.physicalField][d];
                                const auto values = gradient.values[iPoint];
                                for (index j = 0; j < location.componentNumber; ++j)
                                    point.gradient(logicalFirst[iBlock] + j, d) = values[(location.firstComponent + j) * gradient.componentStride + gradient.direction];
                            }
                    }
                    if (ransFirst >= 0)
                    {
                        const auto &model = owner->equation.RANS();
                        RANSStateView{model.Kind(), model.ComponentNumber()}.RepairReconstruction(ransMean[iCell], point.values.data() + ransFirst);
                    }
                    const auto properties = evaluator.Evaluate(point.Flow(flowFirst), point.Species(speciesFirst, independent), 0, {ransFirst >= 0, false, false, chemical && result, false});
                    if (chemical && result)
                        for (int k = 0; k < independent; ++k)
                            speciesOutput(iCell, k) += weights(iPoint, 0) * properties.chemicalSource[k];
                    if (ransFirst >= 0)
                    {
                        const auto &model = owner->equation.RANS();
                        const auto input = RANSPoint(point, flowFirst, ransFirst, model, properties, walls.cells(iCell, iPoint - sourceOffsets[iCell]), lengths(iCell, 0));
                        const auto source = result ? model.Source(input) : std::array<real, 2>{};
                        for (int j = 0; j < model.ComponentNumber(); ++j)
                        {
                            DNDS_check_throw_info(std::isfinite(source[j]), "nonfinite RANS source");
                            if (result)
                                ransOutput(iCell, j) += weights(iPoint, 0) * source[j];
                        }
                    }
                }
        }
    };

    struct FiniteVolume::MeanWorkspace
    {
        /// Zero-polynomial probe storage is temporary by definition; it is not
        /// the solver's warm reconstruction publication.
        std::shared_ptr<ReconstructionState> reconstruction;
        std::shared_ptr<Workspace> work;
    };

    void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction,
                                            Workspace &work, AuxiliaryValues &auxiliary,
                                            R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.fluxCoefficients);
        const MPIInfo mpi(Metrics().Communicator());
        reconstruction.binding.Check(state);
        const auto local = [&](auto &&action)
        {
            Metrics().Communication()->LocalStage(std::forward<decltype(action)>(action));
        };
        local([&]
              {
            ValidateWorkspace(state, work);
            if (reconstruction.FinalizedCurrent()) return;
            const bool filtered = _data->settings.numerics.limiter == "WBAP" ||
                                  bool(work.limitedVR) || _data->settings.numerics.positivity;
            DNDS_check_throw_info(!filtered,
                                  "correction source sampling requires a current limited/PP reconstruction publication");
            EquationDofOperations::Copy(reconstruction.fluxCoefficients, reconstruction.coefficients);
            reconstruction.PublishFinalized(); });
        if (work.meanGradients)
        {
            ValidateWorkspace(state, work);
            EquationAuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary = std::cref(*work.incoming);
            PrepareAuxValues(state, auxiliary, *work.meanGradients, engine, stream, time, work.workers, inputs);
            local([&]
                  {
                if (work.SourceSamplesCurrent(state, time)) return;
                auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {true});
                work.PrepareVolumeSource(state, std::nullopt, evaluator);
                work.PublishSourceState(state, time); });
            return;
        }
        std::optional<AuxiliaryPreparation> preparation;
        local([&]
              {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(stream.Context().device == R::Device::Host(), "equation correction requires a CPU stream");
            PreflightHalo(state);
            EquationAuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary = std::cref(*work.incoming);
            preparation.emplace(BeginAuxValues(state, auxiliary, time, work.workers, inputs)); });
        OperatorSchedule schedule(engine, stream, _data->settings.execution.overlap);
        std::vector<Algebra::DofHalo<>::Operation> transfers;
        transfers.reserve(reconstruction.meanHalo.size());
        ++reconstruction.meanEpoch;
        for (auto &halo : reconstruction.meanHalo)
            transfers.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        if (_data->settings.execution.overlap)
            schedule.Submit([&]
                            { preparation->Apply(CFV::ReconstructionRegion::Core); });
        OperatorSchedule::Wait(transfers);
        const auto rest = schedule.DependentRegion();
        local([&]
              {
            if (_data->settings.CheckHaloFreshness())
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch);
            schedule.Submit([&, rest]
            {
                preparation->Apply(rest);
                preparation->Finish(auxiliary);
                // Source Jacobians use the retained current polynomial. RANS
                // inputs must be current before publishing these samples.
                // No VR sweep or chemical-rate evaluation is hidden here.
                if (work.SourceSamplesCurrent(state, time)) return;
                if (_data->equation.HasRANS())
                {
                    auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                                        {true, false, false, false, false});
                    work.PrepareVolumeSource(state, std::nullopt, evaluator);
                }
                else work.SampleSourceState(state);
                work.PublishSourceState(state, time);
            });
            schedule.Join(); });
    }
    void FiniteVolume::RefreshCorrectionSource(EquationDofs &state,
                                               ReconstructionState &reconstruction,
                                               Workspace &work, real time) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients,
                                                            reconstruction.fluxCoefficients);
        Metrics().Communication()->LocalStage([&]
                                              {
            ValidateWorkspace(state, work);
            reconstruction.binding.Check(state);
            if (!reconstruction.FinalizedCurrent())
            {
                const bool filtered = _data->settings.numerics.limiter == "WBAP" ||
                                      bool(work.limitedVR) || _data->settings.numerics.positivity;
                DNDS_check_throw_info(!filtered,
                                      "source refresh requires a current limited/PP reconstruction publication");
                EquationDofOperations::Copy(reconstruction.fluxCoefficients,
                                            reconstruction.coefficients);
                reconstruction.PublishFinalized();
            }
            if (work.SourceSamplesCurrent(state, time)) return;
            if (_data->equation.HasRANS())
            {
                auto evaluator = _data->equation.MakePointEvaluator(
                    work.workers ? work.workers->Acquire()
                                 : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                    {true, false, false, false, false});
                work.PrepareVolumeSource(state, std::nullopt, evaluator);
            }
            else
                work.SampleSourceState(state);
            work.PublishSourceState(state, time); });
    }
    void FiniteVolume::InitializeConservative(EquationDofs &state, const std::vector<real> &values) const
    {
        Metrics().Communication()->LocalStage([&]
                                              {
            Metrics().Validate();
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            index width = 0;
            for (const auto &block : _data->equation.Schema().blocks) width += index(block.components.size());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && index(values.size()) == width &&
                                      std::all_of(values.begin(), values.end(), [](real v) { return std::isfinite(v); }),
                                  "conservative initializer row space or values differ from equation");
            FlowState flow;
            std::copy_n(values.begin(), 5, flow.begin());
            flow = _data->motion.At(0).LocalState(flow);
            auto output = state.WriteOwned();
            index offset = 0;
            for (const auto &block : _data->equation.Schema().blocks)
            {
                const auto rows = output.Block(block.name);
                for (index iCell = 0; iCell < rows.RowCount(); ++iCell)
                    for (index j = 0; j < rows.ComponentNumber(); ++j)
                        rows(iCell, j) = block.name == "flow" ? flow[j] : values[offset + j];
                offset += index(block.components.size());
            } });
    }
    void FiniteVolume::InitializeConstant(EquationDofs &state, const FlowState &primitive,
                                          const std::map<std::string, real> &massFractions, real temperature,
                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                          const std::map<std::string, real> &ransConservative) const
    {
        Metrics().Communication()->LocalStage([&]
                                              {
            Metrics().Validate();
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            const auto rans = ResolveRANSValues(_data->equation, ransConservative);
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && _data->equation.HasMixture() == bool(workers),
                                  "equation initialization row space/worker profile differs");
            auto fractions = _data->equation.IndependentMassFractions(massFractions);
            for (auto &value : fractions) value *= primitive[0];
            auto evaluator = _data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto value = _data->motion.At(0).LocalState(evaluator.FromPrimitive(primitive, {fractions.data(), int(fractions.size())}, temperature));
            // Resolve/validate EOS input before obtaining any state writer.
            auto output = state.WriteOwned();
            const auto flow = output.Block("flow");
            for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j) flow(iCell, j) = value[j];
            if (!fractions.empty())
            {
                const auto species = output.Block("species");
                for (index iCell = 0; iCell < species.RowCount(); ++iCell)
                    for (index k = 0; k < index(fractions.size()); ++k) species(iCell, k) = fractions[k];
            }
            if (_data->equation.HasRANS())
            {
                const auto scalars = output.Block("rans");
                for (index iCell = 0; iCell < scalars.RowCount(); ++iCell)
                    for (index j = 0; j < scalars.ComponentNumber(); ++j) scalars(iCell, j) = rans[j];
            } });
    }

    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspace(EquationDofs &state,
                                                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                                          WorkspaceOptions options) const
    {
        return BuildWorkspaceImpl(state, std::move(workers), false, options);
    }
    std::shared_ptr<FiniteVolume::ReconstructionState> FiniteVolume::BuildReconstructionState(EquationDofs &state) const
    {
        std::shared_ptr<ReconstructionState> result;
        Metrics().Communication()->LocalStage([&]
                                              {
            _data->equation.ValidateStateSchema(state);
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "reconstruction state belongs to another row space");
            result = std::make_shared<ReconstructionState>(_data, state); });
        return result;
    }
    std::shared_ptr<FiniteVolume::MeanWorkspace> FiniteVolume::BuildMeanWorkspace(EquationDofs &state,
                                                                                  std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        auto inner = BuildWorkspaceImpl(state, std::move(workers), true);
        auto reconstruction = BuildReconstructionState(state);
        std::shared_ptr<MeanWorkspace> result;
        Metrics().Communication()->LocalStage([&]
                                              { result = std::make_shared<MeanWorkspace>(MeanWorkspace{std::move(reconstruction), std::move(inner)}); });
        return result;
    }
    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspaceImpl(EquationDofs &state,
                                                                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                                              bool meanOnly, WorkspaceOptions options) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        const auto &n = _data->settings.numerics;
        // Resolve an operator-local fixed-width leaf without changing the one
        // EquationDofs workspace or outer schedule. Unsupported layouts and
        // physics retain the generic equation kernel.
        const bool flow5Profile = !_data->equation.HasMixture() && !_data->equation.HasRANS() &&
                                  state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1;
        std::string operationControls;
        Metrics().Communication()->LocalStage([&]
                                              {
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "equation FV state belongs to another row space");
            DNDS_check_throw_info(!HasInterfaces() || flow5Profile,
                                  "interface packets currently provide only one coupled flow5 state");
            DNDS_check_throw_info(options.rhsKernel != RHSKernelPolicy::FixedFlow5 || flow5Profile,
                                  "fixed flow5 RHS kernel requires one perfect-gas flow5 physical field");
            DNDS_check_throw_info((n.reconstruction == "constant" || n.reconstruction == "gradient" || n.reconstruction == "variational") &&
                                  n.reconstructionIterations >= 0, "invalid equation reconstruction controls");
            DNDS_check_throw_info(!_data->equation.HasMixture() || _data->fluxPolicy.rusanov,
                "extended EOS or stencil flux needs an explicit preparation provider");
            DNDS_check_throw_info(n.limiter == "none" || n.limiter == "LVR" ||
                                      (n.limiter == "WBAP" && (!n.characteristicLimiter || !_data->equation.HasMixture())),
                "requested extended FV flux/PP/limiter profile is not connected yet");
            DNDS_check_throw_info(!n.positivity || n.positivityReconstruction != "Segment" || flow5Profile,
                                  "Segment reconstruction PP is available only for one perfect-gas flow5 field");
            DNDS_check_throw_info(!_data->equation.Viscous() || n.viscousGradient == "polynomial" || flow5Profile,
                                  "extended viscosity requires the conservative polynomial gradient");
            DNDS_check_throw_info(_data->settings.equation.source == "none" ||
                (_data->settings.equation.source == "chemical" && _data->equation.HasMixture()), "unavailable equation volume source");
            for (const auto &sample : _data->samples)
                if (sample.other == UnInitIndex)
                    sample.boundary.ValidateEquationProducer();
            if (_data->equation.HasMixture())
            {
                DNDS_check_throw_info(workers, "mixture FV requires explicit chemistry workers");
                auto checked = _data->equation.MakePointEvaluator(workers->Acquire(), {});
            }
            else DNDS_check_throw_info(!workers, "perfect-gas FV does not require chemistry workers");
            operationControls = Json{{"meanOnly", meanOnly}, {"rhsKernel", int(options.rhsKernel)},
                {"numerics", _data->settings.numerics}, {"equation", _data->settings.equation},
                {"motion", _data->settings.motion}, {"boundaries", _data->settings.boundaries}, {"overlap", _data->settings.execution.overlap},
                {"collectiveChecks", _data->settings.implicit.collectiveChecks}, {"properties", _data->equation.PropertyKey().parameters},
                {"rans", _data->equation.HasRANS() ? Json(_data->equation.RANS().Settings()) : Json()}}.dump(); });
        AgreeEquationBinding(state, mpi.comm, operationControls);
        std::shared_ptr<Workspace> result;
        Metrics().Communication()->LocalStage([&]
                                              { result = std::make_shared<Workspace>(_data, state, std::move(workers), meanOnly); });
        result->BindHalos(state);
        if (flow5Profile && options.rhsKernel != RHSKernelPolicy::Generic)
            result->fixedFlowRHS.emplace(AcquireRHSOperator());
        auto request = CompleteRHSAuxiliaryRequest({});
        if (request.cellWaveJump || request.cellPressureMinimum)
        {
            request.meanThermodynamics = true;
            result->rhsAuxiliary.emplace(BuildAuxiliaryValues(request));
        }
        if (!meanOnly && _data->equation.HasRANS() && _data->equation.RANS().ComponentNumber() == 2)
            result->meanGradients.emplace(BuildMeanGradients(state));
        return result;
    }
    const EquationDofs &FiniteVolume::ReconstructionGuess(const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
        return state.coefficients;
    }
    void FiniteVolume::ValidateWorkspace(const EquationDofs &state, const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        _data->equation.ValidateStateSchema(state);
        DNDS_check_throw_info(state.Space().Same(Metrics().Space()) &&
                                  state.ComponentNumber() == work.componentNumber &&
                                  state.PhysicalFieldNumber() == work.physicalFieldNumber,
                              "equation workspace schema or row space differs");
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            DNDS_check_throw_info(state.PhysicalField(iField).Width() == work.fields[iField].reconstruction.components,
                                  "equation workspace physical field width differs");
    }
    bool FiniteVolume::HasRoeJacobianKernel(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        // Roe assembly is an operator-local flow5 leaf. It consumes the
        // canonical state/auxiliary views directly and does not require the
        // archived whole-flow reconstruction workspace.
        return !_data->equation.HasMixture() && !_data->equation.HasRANS() &&
               work.componentNumber == 5 && work.physicalFieldNumber == 1;
    }
    index FiniteVolume::PrepareFaceCharacteristics(const CFV::FVField &state, CFV::FaceCharacteristicTransform &transform,
                                                   index firstComponent) const
    {
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = meshRead.HostView();
        index fallbacks = 0;
        {
            auto owned = state.Owned().Read(), ghost = state.Ghost().Read();
            const RHSFieldView u{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
            DNDS_check_throw_info(firstComponent >= 0 && firstComponent + 5 <= state.Width() &&
                                      transform.first == firstComponent && transform.count == 5,
                                  "flow characteristic range differs from its physical field");
            const auto loadFlow = [&](index iCell)
            {
                FlowState value;
                for (int j = 0; j < 5; ++j)
                    value[j] = u(iCell, firstComponent + j);
                return value;
            };
            auto forward = transform.forward.Write(), inverse = transform.inverse.Write();
            const auto l = forward.View<R::TableRows>(), r = inverse.View<R::TableRows>();
            for (auto iFace : _data->stencilFaces)
            {
                const auto cells = mesh.FaceCells(iFace);
                if (cells[0] < 0 || cells[0] >= Metrics().NumCellProc())
                    continue;
                const auto left = loadFlow(cells[0]);
                auto right = left;
                if (cells[1] >= 0 && cells[1] < Metrics().NumCellProc())
                    right = FiniteVolumeDetail::RotateState(loadFlow(cells[1]), CFV::Detail::OtherFrameInCell(mesh, iFace, 0));
                const auto normal = metric.GetFaceNorm(metric.FaceQuadratureBegin(iFace));
                const auto matrices = _data->equation.PrepareCharacteristics(left, right, {normal[0], normal[1], normal[2]});
                fallbacks += matrices.componentwiseFallback;
                for (index j = 0; j < 25; ++j)
                    l(iFace, j) = matrices.forward[j], r(iFace, j) = matrices.inverse[j];
            }
        }
        _data->reconstruction.FinalizeFaceTransform(transform);
        return fallbacks;
    }
    const EquationDofs &FiniteVolume::FluxReconstruction(const ReconstructionState &state, const Workspace &work) const
    {
        DNDS_check_throw_info(state.owner == _data && work.owner == _data, "foreign equation reconstruction state or workspace");
        return state.fluxCoefficients;
    }
    void FiniteVolume::RestoreReconstructionGuess(ReconstructionState &state, const EquationDofs &guess,
                                                  std::optional<std::reference_wrapper<const Runtime::Array<real>>> limitedVRHistory) const
    {
        auto restore = ReserveReconstructionRestoreImpl(state, guess, limitedVRHistory, {});
        restore.Commit();
        if (_data->settings.numerics.limiter == "WBAP" || _data->limitedVRGeometry ||
            _data->settings.numerics.positivity)
            state.finalizedFrom.reset();
        else
            state.PublishFinalized();
    }
    void FiniteVolume::CopyReconstructionState(ReconstructionState &destination,
                                               const ReconstructionState &source) const
    {
        DNDS_check_throw_info(&destination != &source && destination.owner == _data && source.owner == _data,
                              "reconstruction state copy requires distinct states from this finite-volume plan");
        DNDS_check_throw_info(source.FinalizedCurrent(),
                              "source reconstruction has no current finalized polynomial");
        const auto history = source.limitedVRHistory
                                 ? std::optional(std::cref(*source.limitedVRHistory))
                                 : std::nullopt;
        auto restore = ReserveReconstructionRestoreImpl(destination, source.coefficients, history, {}, true,
                                                        std::cref(source.fluxCoefficients));
        restore.Commit();
        destination.PublishFinalized();
    }
    void FiniteVolume::RestoreReconstructionState(ReconstructionState &state,
                                                  const EquationDofs &warm, const EquationDofs &finalized,
                                                  std::optional<std::reference_wrapper<const Runtime::Array<real>>> limitedVRHistory) const
    {
        auto restore = ReserveReconstructionRestoreImpl(state, warm, limitedVRHistory, {}, true,
                                                        std::cref(finalized));
        restore.Commit();
        state.PublishFinalized();
    }
    std::optional<Runtime::Array<real>> FiniteVolume::LimitedVRHistory(const ReconstructionState &state) const
    {
        DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
        return state.limitedVRHistory;
    }
    std::optional<Runtime::Array<real>> FiniteVolume::LimitedVRAlpha(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.limitedVR ? std::optional<R::Array<real>>(work.limitedVR->alpha) : std::nullopt;
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestore(std::shared_ptr<ReconstructionState> state, const EquationDofs &guess,
                                                                                   std::optional<std::reference_wrapper<const Runtime::Array<real>>> limitedVRHistory,
                                                                                   bool collectiveChecks) const
    {
        const auto validate = [&]
        { DNDS_check_throw_info(bool(state), "null equation reconstruction state"); };
        R::CommunicationDetail::LocalStage(Metrics().Communicator(), collectiveChecks && _data->settings.implicit.collectiveChecks,
                                           validate, "reconstruction workspace validation failed");
        auto &reference = *state;
        return ReserveReconstructionRestoreImpl(reference, guess, limitedVRHistory, std::move(state), collectiveChecks);
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestoreImpl(ReconstructionState &state, const EquationDofs &guess,
                                                                                       std::optional<std::reference_wrapper<const Runtime::Array<real>>> limitedVRHistory,
                                                                                       std::shared_ptr<ReconstructionState> retained, bool collectiveChecks,
                                                                                       std::optional<std::reference_wrapper<const EquationDofs>> finalized) const
    {
        const auto *historyInput = limitedVRHistory ? &limitedVRHistory->get() : nullptr;
        const auto stage = [&](const auto &run)
        {
            R::CommunicationDetail::LocalStage(Metrics().Communicator(), collectiveChecks && _data->settings.implicit.collectiveChecks,
                                               run, "reconstruction restore preparation failed");
        };
        struct Reservation
        {
            std::shared_ptr<ReconstructionState> owner;
            R::SessionAccess::Lease session;
            std::optional<EquationDofs::ReadScope> source, fluxSource;
            std::optional<EquationDofs::WriteScope> destination, fluxDestination;
            std::optional<R::Array<real>::ReadLease> historySource;
            std::optional<R::Array<real>::WriteLease> historyDestination;
            std::vector<EquationBlockArrayView<const real>> input, fluxInput;
            std::vector<EquationBlockArrayView<real>> output, fluxOutput;
            bool started = false;
            ~Reservation()
            {
                if (!started && destination)
                    destination->CancelUnmodified();
                if (!started && fluxDestination)
                    fluxDestination->CancelUnmodified();
                if (!started && historyDestination)
                    historyDestination->CancelUnmodified();
            }
        };
        std::shared_ptr<Reservation> reservation;
        stage([&]
              {
                reservation = std::make_shared<Reservation>();
                reservation->owner = std::move(retained);
                reservation->session = state.session.Acquire();
                DNDS_check_throw_info(state.owner == _data, "foreign equation reconstruction state");
                state.coefficients.RequireCompatible(guess);
                state.fluxCoefficients.RequireCompatible(guess);
                EquationDofOperations::RequireDistinct(state.coefficients, guess);
                EquationDofOperations::RequireDistinct(state.fluxCoefficients, guess);
                const auto &finalInput = finalized ? finalized->get() : guess;
                state.fluxCoefficients.RequireCompatible(finalInput);
                EquationDofOperations::RequireDistinct(state.coefficients, finalInput);
                EquationDofOperations::RequireDistinct(state.fluxCoefficients, finalInput);
                DNDS_check_throw_info(state.limitedVRHistory || !historyInput,
                                      "LVR history requires an active matching reconstruction state");
                if (historyInput)
                {
                    DNDS_check_throw_info(historyInput->GetDevice() == R::Device::Host() &&
                                              !historyInput->Layout().IsVariable() &&
                                              historyInput->RowCount() == Metrics().NumCell() &&
                                              historyInput->Layout().Width() == 2,
                                          "LVR history shape or placement differs");
                    for (index iField = 0; iField < guess.PhysicalFieldNumber(); ++iField)
                    {
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(),
                                                                         state.coefficients.PhysicalField(iField).Owned().Storage()),
                                              "LVR history aliases reconstruction output");
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(),
                                                                         state.fluxCoefficients.PhysicalField(iField).Owned().Storage()),
                                              "LVR history aliases finalized reconstruction output");
                    }
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(historyInput->Storage(), state.limitedVRHistory->Storage()),
                                          "LVR history aliases its destination");
                    const auto history = historyInput->Read();
                    const auto values = history.View<R::UniformTableRows>();
                    const std::array<real, 2> caps{_data->settings.numerics.limitedVR.alphaMax,
                                                   _data->settings.numerics.limitedVR.contactAlphaMax};
                    for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                        for (int j = 0; j < 2; ++j)
                            DNDS_check_throw_info(std::isfinite(values(iCell, j)) && values(iCell, j) >= 0 && values(iCell, j) <= caps[j],
                                                  "LVR history exceeds its valid branch cap");
                }
                reservation->source.emplace(guess.ReadOwned());
                if (finalized)
                    reservation->fluxSource.emplace(finalized->get().ReadOwned());
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                {
                    const auto input = reservation->source->Block(iBlock);
                    for (index iCell = 0; iCell < input.owned.RowCount(); ++iCell)
                        for (index j = 0; j < input.ComponentNumber(); ++j)
                            DNDS_check_throw_info(std::isfinite(input(iCell, j)), "nonfinite equation reconstruction guess");
                    reservation->input.push_back(input.owned);
                    const auto fluxInput = finalized ? reservation->fluxSource->Block(iBlock) : input;
                    for (index iCell = 0; iCell < fluxInput.owned.RowCount(); ++iCell)
                        for (index j = 0; j < fluxInput.ComponentNumber(); ++j)
                            DNDS_check_throw_info(std::isfinite(fluxInput(iCell, j)),
                                                  "nonfinite finalized equation reconstruction");
                    reservation->fluxInput.push_back(fluxInput.owned);
                }
                reservation->destination.emplace(state.coefficients.WriteOwned());
                reservation->fluxDestination.emplace(state.fluxCoefficients.WriteOwned());
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                {
                    reservation->output.push_back(reservation->destination->Block(iBlock));
                    reservation->fluxOutput.push_back(reservation->fluxDestination->Block(iBlock));
                }
                if (historyInput)
                    reservation->historySource.emplace(historyInput->Read());
                if (state.limitedVRHistory)
                    reservation->historyDestination.emplace(state.limitedVRHistory->Write()); });
        std::optional<ReconstructionRestore> restore;
        stage([&]
              { restore.emplace(ReconstructionRestore([reservation]() noexcept
                                                      {
            reservation->started = true;
            for (index iBlock = 0; iBlock < index(reservation->input.size()); ++iBlock)
            {
                const auto input = reservation->input[iBlock];
                const auto output = reservation->output[iBlock];
                const auto fluxInput = reservation->fluxInput[iBlock];
                const auto fluxOutput = reservation->fluxOutput[iBlock];
                for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                    for (index j = 0; j < output.ComponentNumber(); ++j)
                    {
                        output(iCell, j) = input(iCell, j);
                        fluxOutput(iCell, j) = fluxInput(iCell, j);
                    }
            }
            if (reservation->historyDestination)
            {
                const auto output = reservation->historyDestination->View<R::UniformTableRows>();
                if (reservation->historySource)
                {
                    const auto input = reservation->historySource->View<R::UniformTableRows>();
                    for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                        for (int j = 0; j < 2; ++j)
                            output(iCell, j) = input(iCell, j);
                }
                else
                    for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                        for (int j = 0; j < 2; ++j)
                            output(iCell, j) = 0;
            } })); });
        return std::move(*restore);
    }
    void FiniteVolume::ResetReconstruction(ReconstructionState &state, Workspace &work) const
    {
        auto stateSession = state.session.Acquire();
        auto session = work.session.Acquire();
        DNDS_check_throw_info(state.owner == _data && work.owner == _data, "foreign equation reconstruction state or workspace");
        // Reserve every destination before changing any value. In particular,
        // a retained LVR history reader must not allow coefficients to be
        // cleared first and only then report the conflict.
        std::optional<EquationDofs::WriteScope> coefficients, fluxCoefficients;
        std::vector<R::Array<real>::WriteLease> temporary;
        std::optional<R::Array<real>::WriteLease> previous, alpha;
        try
        {
            coefficients.emplace(state.coefficients.WriteOwned());
            fluxCoefficients.emplace(state.fluxCoefficients.WriteOwned());
            temporary.reserve(work.fields.size());
            for (auto &field : work.fields)
                temporary.emplace_back(field.reconstruction.temporary.Owned().Write());
            if (work.limitedVR)
            {
                previous.emplace(state.limitedVRHistory->Write());
                alpha.emplace(work.limitedVR->alpha.Write());
            }
        }
        catch (...)
        {
            if (coefficients)
                coefficients->CancelUnmodified();
            if (fluxCoefficients)
                fluxCoefficients->CancelUnmodified();
            for (auto &write : temporary)
                write.CancelUnmodified();
            if (previous)
                previous->CancelUnmodified();
            if (alpha)
                alpha->CancelUnmodified();
            throw;
        }
        {
            const auto layout = coefficients->ComponentLayout();
            const auto values = layout.View();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) = 0;
        }
        {
            const auto layout = fluxCoefficients->ComponentLayout();
            const auto values = layout.View();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) = 0;
        }
        for (auto &write : temporary)
        {
            const auto values = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (index j = 0; j < values.Width(); ++j)
                    values(iCell, j) = 0;
        }
        for (auto *write : {previous ? &*previous : nullptr, alpha ? &*alpha : nullptr})
            if (write)
            {
                const auto values = write->View<R::UniformTableRows>();
                for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (index j = 0; j < values.Width(); ++j)
                        values(iCell, j) = 0;
            }
        work.sourceState.reset();
        work.sourceCoefficients.reset();
        work.rhsState.reset();
        work.rhsCoefficients.reset();
        state.finalizedFrom.reset();
    }
    int FiniteVolume::LastReconstructionIterations(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.iterations;
    }
    real FiniteVolume::LastReconstructionUpdate(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.relativeUpdate;
    }
    void FiniteVolume::LimitReconstructionPP(const EquationDofs &state, ReconstructionState &reconstruction, Workspace &work) const
    {
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        Metrics().Communication()->LocalStage([&]
                                              {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(work.positivity, "equation reconstruction PP was not requested");
            DNDS_check_throw_info(reconstruction.owner == _data, "foreign equation reconstruction state");
            work.positivity->Apply(state, reconstruction.coefficients, reconstruction.fluxCoefficients, true);
            reconstruction.PublishFinalized(); });
    }
    std::pair<index, real> FiniteVolume::PositivityReconstructionStats(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        auto result = work.positivity ? work.positivity->LocalStatistics() : std::pair<index, real>{0, 1};
        MPI_Allreduce(MPI_IN_PLACE, &result.first, 1, MPI_INT64_T, MPI_SUM, Metrics().Communicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.second, 1, MPI_DOUBLE, MPI_MIN, Metrics().Communicator());
        return result;
    }

    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &work,
                                R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, std::ref(residual), reconstruction, work, engine, stream, time);
    }
    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, ReconstructionState &reconstruction, Workspace &work,
                                AuxiliaryValues &auxiliary, R::EagerEngine &engine,
                                const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, std::ref(residual), reconstruction, work, engine, stream, time, std::ref(auxiliary));
    }
    void FiniteVolume::PrepareReconstruction(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, R::EagerEngine &engine,
                                             const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, std::nullopt, reconstruction, work, engine, stream, time);
    }
    void FiniteVolume::RefreshSamplingInputs(EquationDofs &state, ReconstructionState &reconstruction, Workspace &work, R::EagerEngine &engine,
                                             const R::ExecutionStream &stream) const
    {
        ValidateWorkspace(state, work);
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.fluxCoefficients);
        Metrics().Communication()->LocalStage([&]
                                              {
            DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == R::Device::Host(),
                                  "sampling refresh requires the workspace CPU engine");
            PreflightHalo(state);
            PreflightHalo(reconstruction.fluxCoefficients); });
        ++reconstruction.meanEpoch;
        std::vector<Algebra::DofHalo<>::Operation> means;
        means.reserve(reconstruction.meanHalo.size());
        for (auto &halo : reconstruction.meanHalo)
            means.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        for (auto &operation : means)
            operation.Wait();
        if (_data->settings.CheckHaloFreshness())
            Metrics().Communication()->LocalStage([&]
                                                  {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch); });
        for (auto &binding : reconstruction.fluxHalo)
        {
            auto operation = engine.ChecksAccess() ? binding.BeginChecked(stream) : binding.BeginScheduled(stream);
            operation.Wait();
        }
    }
    void FiniteVolume::EvaluateMean(EquationDofs &state, EquationDofs &residual, MeanWorkspace &work,
                                    R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        EvaluateImpl(state, std::ref(residual), *work.reconstruction, *work.work, engine, stream, time);
    }
    void FiniteVolume::EvaluateImpl(EquationDofs &state,
                                    std::optional<std::reference_wrapper<EquationDofs>> residual,
                                    ReconstructionState &reconstruction, Workspace &work, R::EagerEngine &engine,
                                    const R::ExecutionStream &stream, real time,
                                    std::optional<std::reference_wrapper<AuxiliaryValues>> auxiliary) const
    {
        if (!auxiliary && work.rhsAuxiliary)
            auxiliary = std::ref(*work.rhsAuxiliary);
        auto reconstructionSession = reconstruction.session.Acquire();
        auto session = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.fluxCoefficients);
        const Workspace::PhaseTime timing(work, Workspace::TotalPhase);
        work.sourceState.reset();
        work.sourceCoefficients.reset();
        work.rhsState.reset();
        work.rhsCoefficients.reset();
        work.relativeUpdate = !work.meanOnly && _data->settings.numerics.reconstruction == "variational" ? std::numeric_limits<real>::infinity() : 0;
        work.faceFallbacks = {};
        work.diagnostics.Reset();
        work.characteristicFallbacks = 0;
        const MPIInfo mpi(Metrics().Communicator());
        // Repeated entry/CPU-join agreement follows the same checked/eager
        // policy as later local kernels. Unchecked execution still validates
        // locally and propagates failures to the fail-fast frontend. Payload
        // transfer preflight remains mandatory and unchanged.
        const auto local = [&](auto &&action)
        {
            Metrics().Communication()->LocalStage(std::forward<decltype(action)>(action));
        };
        // Observe only this lane's producer prefix. MPI remains on the caller's
        // controller thread, including when the CPU launcher uses worker threads.
        local([&]
              { stream.Record().WaitHost(); });
        std::optional<EquationPointEvaluator> evaluator;
        std::optional<AuxiliaryPreparation> auxiliaryPreparation;
        std::optional<R::Array<real>::ReadLease> incomingRead;
        RHSScalarView incoming;
        // A Cantera auxiliary pass and RHS pass may share a single worker
        // context. Do not acquire the RHS lease until aux has published and
        // returned its worker. Perfect-gas passes need no such worker lease.
        const bool deferredEvaluator = auxiliary && _data->equation.HasMixture();
        const auto createEvaluator = [&]
        {
            const bool viscous = !work.meanOnly && _data->equation.Viscous();
            evaluator.emplace(_data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                                 {viscous || _data->equation.HasRANS(), viscous, viscous, work.chemical, false}));
        };
        local([&]
              {
            DNDS_check_throw_info(work.owner == _data && reconstruction.owner == _data && std::isfinite(time) && stream.Context().device == R::Device::Host(),
                                  "foreign equation workspace/state, invalid time or missing CPU backend");
            ValidateWorkspace(state, work);
            reconstruction.binding.Check(state);
            if (residual)
            {
                auto &output = residual->get();
                _data->equation.ValidateStateSchema(output);
                state.RequireCompatible(output);
                EquationDofOperations::RequireDistinct(state, output);
                EquationDofOperations::RequireDistinct(reconstruction.coefficients, output);
                EquationDofOperations::RequireDistinct(reconstruction.fluxCoefficients, output);
                auto write = output.WriteOwned(); write.CancelUnmodified();
            }
            PreflightHalo(state); PreflightHalo(reconstruction.coefficients); PreflightHalo(reconstruction.fluxCoefficients);
            DNDS_check_throw_info(!HasInterfaces() || work.incoming,
                                  "FV evaluation needs a ready interface packet");
            if (work.incoming)
            {
                ValidateBoundaryInputs(*work.incoming, time);
                incomingRead.emplace(work.incoming->flow.Read());
                incoming = incomingRead->View<R::UniformTableRows>();
            }
            _data->partition.Validate();
            work.boundaryMomentumCurrent = false;
            if (!deferredEvaluator) createEvaluator(); });
        if (auxiliary && !work.meanGradients)
            local([&]
                  {
                      EquationAuxiliaryInputs inputs;
                      if (work.incoming)
                          inputs.boundary = std::cref(*work.incoming);
                      auxiliaryPreparation.emplace(BeginAuxValues(state, auxiliary->get(), time, work.workers, inputs)); });
        const bool overlap = _data->settings.execution.overlap;
        OperatorSchedule schedule(engine, stream, overlap);
        const auto rest = schedule.DependentRegion();
        const auto sweep = [&](CFV::ReconstructionRegion region)
        {
            if (work.limitedVR)
            {
                auto &field = work.fields.front();
                _data->reconstruction.SubmitReconstructionIterLimitedTo(
                    engine, stream, state.CoupledField(), reconstruction.coefficients.CoupledField(),
                    field.reconstruction, field.boundary, work.limitedVR->targetCoefficients,
                    work.limitedVR->alpha, region);
                return;
            }
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                auto &field = work.fields[iField];
                _data->reconstruction.SubmitReconstructionIterTo(
                    engine, stream, state.PhysicalField(iField), reconstruction.coefficients.PhysicalField(iField),
                    field.reconstruction, field.boundary, region);
            }
        };
        const auto gradient = [&](CFV::ReconstructionRegion region)
        {
            const Workspace::PhaseTime timing(work, Workspace::GradientPhase);
            if (work.limitedVR)
                _data->limitedVRGeometry->gradient.Evaluate(state.CoupledField(), *work.fields.front().gradientBoundary,
                                                            work.limitedVR->gradients, {1}, region);
            else
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                {
                    auto &field = work.fields[iField];
                    if (field.gradient)
                        _data->gradient.Evaluate(state.PhysicalField(iField), *field.gradientBoundary, *field.gradient, field.vectorStarts, region);
                }
            work.PreparePrimitiveGradient(state, region);
        };
        // Locals captured by worker commands remain alive even on exceptions.
        std::vector<Algebra::DofHalo<>::Operation> meanOperations;
        meanOperations.reserve(reconstruction.meanHalo.size());
        ++reconstruction.meanEpoch;
        for (auto &halo : reconstruction.meanHalo)
            meanOperations.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        std::vector<R::TransferOperation> firstCoefficients;
        const auto &numerics = _data->settings.numerics;
        const bool variational = !work.meanOnly && numerics.reconstruction == "variational";
        const bool firstSweep = overlap && variational && !work.limitedVR && numerics.reconstructionIterations > 0 && !deferredEvaluator;
        if (firstSweep)
            firstCoefficients = schedule.Begin(reconstruction.coefficientHalo);
        // Post both transfers before queueing core work. A CPU-visible MPI
        // bridge may otherwise join that work while checking producer readiness.
        if (auxiliaryPreparation && overlap)
            schedule.Submit([&]
                            { auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });
        if (!deferredEvaluator)
            schedule.Submit([&, variational]
                            { work.FillBoundary(state, variational && _data->settings.numerics.reconstructionBoundaryFromPolynomial,
                                                time, *evaluator, incoming); });
        const bool directGradientWork = !work.meanOnly && (numerics.reconstruction == "gradient" ||
                                                           (_data->equation.HasRANS() && numerics.reconstruction == "constant"));
        const bool gradientWork = directGradientWork || bool(work.limitedVR) || bool(work.primitiveGradient);
        if (gradientWork && overlap && !deferredEvaluator)
            schedule.Submit([&]
                            { gradient(CFV::ReconstructionRegion::Core); });
        if (firstSweep)
            sweep(CFV::ReconstructionRegion::Core);
        OperatorSchedule::Wait(meanOperations);
        if (auxiliaryPreparation)
        {
            schedule.Submit([&]
                            {
                auxiliaryPreparation->Apply(rest);
                auxiliaryPreparation->Finish(auxiliary->get()); });
            local([&]
                  { schedule.Join(); });
        }
        else if (auxiliary)
        {
            // Two-equation transport explicitly needs mean gradients; those
            // are independent of the reconstruction being computed below.
            EquationAuxiliaryInputs inputs;
            if (work.incoming)
                inputs.boundary = std::cref(*work.incoming);
            PrepareAuxValues(state, auxiliary->get(), *work.meanGradients, engine, stream, time, work.workers, inputs);
        }
        if (deferredEvaluator)
        {
            local(createEvaluator);
            schedule.Submit([&, variational]
                            { work.FillBoundary(state, variational && numerics.reconstructionBoundaryFromPolynomial,
                                                time, *evaluator, incoming); });
        }
        if (_data->settings.CheckHaloFreshness())
            local([&]
                  {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                reconstruction.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), reconstruction.meanEpoch); });
        if (gradientWork)
        {
            schedule.Submit([&, rest]
                            { gradient(deferredEvaluator ? CFV::ReconstructionRegion::All : rest); });
            if (directGradientWork)
                schedule.Submit([&]
                                {
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                {
                    auto &field = work.fields[iField];
                    if (numerics.reconstruction == "gradient")
                        _data->gradient.ConvertToCoefficients(*field.gradient, _data->reconstruction,
                                                             reconstruction.coefficients.PhysicalField(iField));
                    else
                        Algebra::Fill(reconstruction.coefficients.PhysicalField(iField), 0.);
                } });
            local([&]
                  { schedule.Join(); });
        }
        std::optional<R::TransferOperation> primitiveGradientOperation;
        if (work.primitiveGradientHalo)
            primitiveGradientOperation.emplace(schedule.Begin(*work.primitiveGradientHalo));
        if (work.limitedVR)
        {
            if (work.flowCharacteristic)
                local([&]
                      {
                const auto flow = state.LocateBlock("flow");
                work.characteristicFallbacks = PrepareFaceCharacteristics(
                    state.PhysicalField(flow.physicalField), *work.flowCharacteristic, flow.firstComponent); });
            DNDS_check_throw_info(reconstruction.limitedVRHistory, "LVR reconstruction state has no history");
            work.limitedVR->Build(*_data, state.CoupledField(), *reconstruction.limitedVRHistory,
                                  work.flowCharacteristic ? CFV::FaceCharacteristicTransformRef(std::cref(*work.flowCharacteristic))
                                                          : std::nullopt,
                                  engine, stream);
        }
        if (variational)
        {
            for (int iteration = 0; iteration < numerics.reconstructionIterations; ++iteration)
            {
                std::vector<R::TransferOperation> operations;
                if (iteration == 0 && firstSweep)
                    operations = std::move(firstCoefficients);
                else
                {
                    local([&]
                          { schedule.Join(); });
                    if (numerics.reconstructionBoundaryFromPolynomial)
                        local([&]
                              { work.FillBoundary(state, true, time, *evaluator, incoming); });
                    operations = schedule.Begin(reconstruction.coefficientHalo);
                    if (overlap)
                        sweep(CFV::ReconstructionRegion::Core);
                }
                OperatorSchedule::Wait(operations);
                sweep(rest);
                local([&]
                      { schedule.Join(); });
                bool converged = false;
                if (numerics.reconstructionTolerance > 0 && (iteration % 4 == 3 || iteration + 1 == numerics.reconstructionIterations))
                {
                    real norms[2]{};
                    local([&]
                          {
                        for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        {
                            const auto aRead = reconstruction.coefficients.PhysicalField(iField).Owned().Read();
                            const auto bRead = work.fields[iField].reconstruction.temporary.Owned().Read();
                            const auto a = aRead.View<R::UniformTableRows>(), b = bRead.View<R::UniformTableRows>();
                            for (index iCell = 0; iCell < a.RowCount(); ++iCell)
                                for (index j = 0; j < a.Width(); ++j)
                                {
                                    const real delta = b(iCell, j) - a(iCell, j);
                                    norms[0] += delta * delta; norms[1] += b(iCell, j) * b(iCell, j);
                                }
                        } });
                    MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
                    work.relativeUpdate = std::sqrt(norms[0] / std::max(norms[1], real(1e-300)));
                    converged = work.relativeUpdate <= numerics.reconstructionTolerance;
                }
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    _data->reconstruction.SubmitCommitReconstruction(engine, stream,
                                                                     reconstruction.coefficients.PhysicalField(iField),
                                                                     work.fields[iField].reconstruction);
                local([&]
                      { schedule.Join(); });
                work.iterations = iteration + 1;
                if (converged)
                    break;
            }
        }
        else if (!directGradientWork)
            local([&]
                  {
                schedule.Join();
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    Algebra::Fill(reconstruction.coefficients.PhysicalField(iField), 0.); });
        const bool useFluxScratch = numerics.limiter == "WBAP" || bool(work.limitedVR) || bool(work.positivity);
        local([&]
              { EquationDofOperations::Copy(reconstruction.fluxCoefficients, reconstruction.coefficients); });
        if (numerics.limiter == "WBAP")
        {
            if (work.flowCharacteristic && !work.limitedVR)
                local([&]
                      {
                          const auto flow = state.LocateBlock("flow");
                          DNDS_check_throw_info(flow.physicalField == work.flowCharacteristicField,
                                                "characteristic flow allocation changed after workspace setup");
                          work.characteristicFallbacks = PrepareFaceCharacteristics(
                              state.PhysicalField(flow.physicalField), *work.flowCharacteristic, flow.firstComponent); });
            for (int order = numerics.cfv.maxOrder; order >= 1; --order)
            {
                auto operations = schedule.Begin(reconstruction.fluxHalo);
                const auto limit = [&, order](CFV::ReconstructionRegion region)
                {
                    for (index iField = 0; iField < index(work.fields.size()); ++iField)
                        _data->reconstruction.SubmitLimiterWBAP_3OrderTo(
                            engine, stream, reconstruction.fluxCoefficients.PhysicalField(iField), work.fields[iField].reconstruction, order,
                            iField == work.flowCharacteristicField && work.flowCharacteristic
                                ? CFV::FaceCharacteristicTransformRef(std::cref(*work.flowCharacteristic))
                                : std::nullopt,
                            region);
                };
                if (overlap)
                    limit(CFV::ReconstructionRegion::Core);
                OperatorSchedule::Wait(operations);
                limit(rest);
                local([&]
                      { schedule.Join(); });
                for (index iField = 0; iField < index(work.fields.size()); ++iField)
                    _data->reconstruction.SubmitCommitReconstruction(engine, stream,
                                                                     reconstruction.fluxCoefficients.PhysicalField(iField),
                                                                     work.fields[iField].reconstruction);
                local([&]
                      { schedule.Join(); });
            }
        }
        if (work.positivity)
            local([&]
                  { const Workspace::PhaseTime timing(work, Workspace::PPPhase);
                    work.positivity->Apply(state, reconstruction.coefficients, reconstruction.fluxCoefficients, false); });
        local([&]
              { reconstruction.PublishFinalized(); });
        auto operations = schedule.Begin(reconstruction.fluxHalo);
        std::vector<R::TransferOperation> unlimitedViscousOperations;
        if (useFluxScratch && _data->equation.Viscous() && numerics.viscousGradient == "polynomial" && !numerics.viscousLimited)
            unlimitedViscousOperations = schedule.Begin(reconstruction.coefficientHalo);
        // Publish the direct physical gradients after limiter/PP, once per row.
        // The physical-gradient RHS itself never interprets coefficient data.
        schedule.Submit([&]
                        { work.RefreshDirectGradients(false); });
        if (overlap && residual)
            schedule.Submit([&]
                            { work.ApplyRHS(state, residual->get(), CFV::ReconstructionRegion::Core, *evaluator, time,
                                            auxiliary ? std::optional(std::cref(auxiliary->get())) : std::nullopt,
                                            std::nullopt, incoming); });
        OperatorSchedule::Wait(operations);
        OperatorSchedule::Wait(unlimitedViscousOperations);
        if (primitiveGradientOperation)
            primitiveGradientOperation->Wait();
        schedule.Submit([&]
                        { work.RefreshDirectGradients(true); });
        if (residual)
        {
            schedule.Submit([&, rest]
                            { work.ApplyRHS(state, residual->get(), rest, *evaluator, time,
                                            auxiliary ? std::optional(std::cref(auxiliary->get())) : std::nullopt,
                                            std::nullopt, incoming); });
            schedule.Submit([&]
                            { work.PrepareVolumeSource(
                                  state, std::ref(residual->get()), *evaluator, time,
                                  work.lowOrderResidual
                                      ? std::optional<std::reference_wrapper<EquationDofs>>(std::ref(*work.lowOrderResidual))
                                      : std::nullopt); });
        }
        local([&]
              { schedule.Join(); });
        if (residual && (work.chemical || work.ransFirst >= 0) && work.sourceSampling)
            local([&]
                  { work.PublishSourceState(state, time); });
        work.boundaryMomentumCurrent = bool(residual);
        work.rhsState.emplace(state.NumericAlias());
        work.rhsCoefficients.emplace(reconstruction.fluxCoefficients);
        work.rhsTime = time;
    }

    R::Array<real> FiniteVolume::BuildFaceFluxBuffer(index components) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(components > 0, "face flux requires positive component count");
        return R::Array<real>(R::RowLayout::Dense(index(_data->stencilFaces.size()), components));
    }
    void FiniteVolume::EvaluateFlux(const EquationDofs &state, EquationDofs &residual,
                                    const ReconstructionState &reconstruction, Workspace &work,
                                    CFV::ReconstructionRegion region, real time,
                                    std::optional<std::reference_wrapper<R::Array<real>>> faceOutput,
                                    std::optional<std::reference_wrapper<const AuxiliaryValues>> auxiliary) const
    {
        ValidateWorkspace(state, work);
        DNDS_check_throw_info(reconstruction.owner == _data, "foreign equation reconstruction state");
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.fluxCoefficients);
        state.RequireCompatible(residual);
        EquationDofOperations::RequireDistinct(state, residual);
        EquationDofOperations::RequireDistinct(reconstruction.fluxCoefficients, residual);
        DNDS_check_throw_info(work.rhsState && work.rhsCoefficients && work.rhsTime == time, "flux needs a completed spatial preparation at this time");
        work.rhsState->Check(state, true);
        DNDS_check_throw_info(work.rhsCoefficients->Current(reconstruction.fluxCoefficients),
                              "flux reconstruction changed after spatial preparation");
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                  region == CFV::ReconstructionRegion::HaloDependent,
                              "unknown flux region");
        if (faceOutput)
        {
            const auto &output = faceOutput->get();
            DNDS_check_throw_info(output.GetDevice() == R::Device::Host() && !output.Layout().IsVariable() &&
                                      output.RowCount() == index(_data->stencilFaces.size()) && output.Layout().Width() == state.ComponentNumber(),
                                  "equation face output shape/device differs");
            for (const auto *fields : std::array<const EquationDofs *, 3>{&state, &residual, &reconstruction.fluxCoefficients})
                for (index iField = 0; iField < fields->PhysicalFieldNumber(); ++iField)
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(output.Storage(), fields->PhysicalField(iField).Owned().Storage()) &&
                                              !Algebra::Detail::Overlaps(output.Storage(), fields->PhysicalField(iField).Ghost().Storage()),
                                          "face output aliases equation fields");
        }
        if (auxiliary)
            ValidateAuxValues(state, auxiliary->get(), time);
        std::optional<R::Array<real>::ReadLease> incomingRead;
        RHSScalarView incoming;
        DNDS_check_throw_info(!HasInterfaces() || work.incoming,
                              "equation flux evaluation needs an incoming interface packet");
        if (work.incoming)
        {
            ValidateBoundaryInputs(*work.incoming, time);
            incomingRead.emplace(work.incoming->flow.Read());
            incoming = incomingRead->View<R::UniformTableRows>();
        }
        auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                            {true, true, true, false, false});
        work.ApplyRHS(state, residual, region, evaluator, time, auxiliary, faceOutput, incoming);
        work.boundaryMomentumCurrent = region == CFV::ReconstructionRegion::All;
    }
    void FiniteVolume::AddSource(const EquationDofs &state, EquationDofs &residual,
                                 const ReconstructionState &reconstruction, Workspace &work, real time) const
    {
        ValidateWorkspace(state, work);
        DNDS_check_throw_info(reconstruction.owner == _data, "foreign equation reconstruction state");
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients, reconstruction.fluxCoefficients);
        state.RequireCompatible(residual);
        EquationDofOperations::RequireDistinct(state, residual);
        EquationDofOperations::RequireDistinct(reconstruction.fluxCoefficients, residual);
        DNDS_check_throw_info(work.rhsState && work.rhsCoefficients && work.rhsTime == time, "source needs a completed spatial preparation at this time");
        work.rhsState->Check(state, true);
        DNDS_check_throw_info(work.rhsCoefficients->Current(reconstruction.fluxCoefficients),
                              "source reconstruction changed after spatial preparation");
        auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                            {true, false, false, work.chemical, false});
        work.PrepareVolumeSource(state, std::ref(residual), evaluator, time);
        work.PublishSourceState(state, time);
    }

    Vector3 FiniteVolume::LastBoundaryForce(const Workspace &work, const std::vector<int> &zones) const
    {
        if (work.fixedFlowRHS)
        {
            DNDS_check_throw_info(work.owner == _data && work.boundaryMomentumCurrent,
                                  "fixed-flow boundary force requires a completed RHS evaluation");
            auto force = work.diagnostics.BoundaryForce(zones);
            MPI_Allreduce(MPI_IN_PLACE, force.data(), 3, MPI_DOUBLE, MPI_SUM, Metrics().Communicator());
            return force;
        }
        Vector3 force{};
        const MPIInfo mpi(Metrics().Communicator());
        Metrics().Communication()->LocalStage([&]
                                              {
            DNDS_check_throw_info(work.owner == _data && work.boundaryMomentumCurrent,
                "equation boundary force requires a completed RHS evaluation");
            for (const int zone : zones)
                DNDS_check_throw_info(_data->settings.boundaries.count(std::to_string(zone)), "force zone has no boundary condition");
            const auto read = work.boundaryMomentumFlux.Read();
            const auto values = read.View<Runtime::FixedRows<3>>();
            for (index iPoint = 0; iPoint < index(_data->boundaryEntries.size()); ++iPoint)
            {
                const auto &sample = _data->samples[_data->boundaryEntries[iPoint]];
                if (std::find(zones.begin(), zones.end(), sample.zone) != zones.end())
                    for (int d = 0; d < 3; ++d) force[d] += values(iPoint, d);
            } });
        MPI_Allreduce(MPI_IN_PLACE, force.data(), 3, MPI_DOUBLE, MPI_SUM, mpi.comm);
        return force;
    }

    void FiniteVolume::SetBoundaryInputs(Workspace &work, const BoundaryInputs &input) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        work.incoming = AliasBoundaryInputs(input);
    }

    void FiniteVolume::SetBoundaryInputs(MeanWorkspace &work, const BoundaryInputs &input) const
    {
        DNDS_check_throw_info(work.work && work.work->owner == _data, "foreign equation mean workspace");
        work.work->incoming = AliasBoundaryInputs(input);
    }

    index FiniteVolume::LastFaceFallbacks(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        if (work.fixedFlowRHS)
        {
            auto count = work.diagnostics.FaceFallbacks();
            MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT64_T, MPI_SUM, Metrics().Communicator());
            return count;
        }
        return work.faceFallbacks[0] + work.faceFallbacks[1];
    }

    index FiniteVolume::LastCharacteristicFallbacks(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.characteristicFallbacks;
    }

    const EquationDofs &FiniteVolume::LowOrderResidual(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        DNDS_check_throw_info(work.lowOrderResidual, "low-order residual requires the positivity workspace");
        return *work.lowOrderResidual;
    }

    void FiniteVolume::ValidateEquationSourceJacobian(
        const EquationDofs &state, const JacobianContext &context, real time) const
    {
        DNDS_check_throw_info(context.reconstruction,
                              "source Jacobian context has no reconstruction publication");
        const auto &reconstruction = context.reconstruction->get();
        auto &work = context.workspace.get();
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients,
                                                            reconstruction.fluxCoefficients);
        DNDS_check_throw_info(work.owner == _data && (work.chemical || work.ransFirst >= 0), "foreign or source-free reconstruction");
        if (!work.sourceSampling)
            return;
        DNDS_check_throw_info(work.sourceState && work.sourceCoefficients && work.sourceTime == time, "source quadrature requires a current completed sample publication");
        work.sourceState->Check(state, true);
        DNDS_check_throw_info(work.sourceCoefficients->Current(reconstruction.fluxCoefficients),
                              "source reconstruction changed after sample publication");
    }
    bool FiniteVolume::AddEquationSourceJacobian(const EquationDofs &state,
                                                 const JacobianContext &context,
                                                 const R::Array<real>::WriteLease &diagonal, DiagonalStorage storage, real scale, EquationPointEvaluator &evaluator,
                                                 const EquationFVDetail::MeanLinearizationOutput *linearization) const
    {
        DNDS_check_throw_info(context.reconstruction,
                              "source Jacobian context has no reconstruction publication");
        const auto &reconstruction = context.reconstruction->get();
        auto &work = context.workspace.get();
        auto reconstructionSession = reconstruction.session.Acquire();
        auto workspaceSession = work.session.Acquire();
        auto activeReconstruction = work.BindReconstruction(reconstruction.coefficients,
                                                            reconstruction.fluxCoefficients);
        return work.AddMeanSourceJacobian(state, diagonal, storage, scale, evaluator, linearization);
    }
}
