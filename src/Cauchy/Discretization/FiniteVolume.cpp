#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
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
        // Allocate/fill setup scratch is discarded after canonical publication.
        struct StencilBuilder
        {
            std::vector<index> offsets, jacobianOffsets, boundaryEntries, stencilFaces, faceEntries;
            std::vector<index> auxiliaryCoreFaces, auxiliaryDependentFaces, rhsCoreFaces, rhsDependentFaces;
            std::vector<index> auxiliaryCellOffsets, auxiliaryCellFaces, positivityOffsets;
            std::vector<real> auxiliaryFaceArea;
            std::vector<Geom::tPoint> positivityDisplacements;
            std::vector<Sample> samples, jacobianSamples;
            std::vector<BoundaryPoint> boundaryPoints, boundaryFaces;
        };

    }

    FiniteVolume::FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation)
    {
        StencilBuilder build;
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
        build.offsets.resize(size_t(data->metrics.NumCell()) + 1, 0);
        index count = 0;
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            for (auto iFace : topology.CellFaces(iCell))
            {
                count += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
                build.stencilFaces.push_back(iFace);
            }
            build.offsets[iCell + 1] = count;
        }
        std::sort(build.stencilFaces.begin(), build.stencilFaces.end());
        build.stencilFaces.erase(std::unique(build.stencilFaces.begin(), build.stencilFaces.end()), build.stencilFaces.end());
        build.samples.reserve(size_t(count));
        build.jacobianOffsets.resize(size_t(data->metrics.NumCell()) + 1);
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            build.jacobianOffsets[iCell] = index(build.jacobianSamples.size());
            const auto faces = topology.CellFaces(iCell);
            for (rowsize slot = 0; slot < rowsize(faces.size()); ++slot)
            {
                const auto iFace = faces[slot];
                const int side = topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
                const index other = topology.CellFaceOther(iCell, iFace, slot);
                DNDS_check_throw_info(other == UnInitIndex || (other >= 0 && other < data->metrics.NumCellProc()), "flux stencil has an unavailable neighbor cell");
                const index auxiliaryFace = index(std::lower_bound(build.stencilFaces.begin(), build.stencilFaces.end(), iFace) - build.stencilFaces.begin());
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
                build.jacobianSamples.push_back(std::move(jacobian));
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
                    const index entry = index(build.samples.size());
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
                    build.samples.push_back(std::move(sample));
                }
            }
        } });
        build.jacobianOffsets.back() = index(build.jacobianSamples.size());
        data->metrics.Communication()->LocalStage([&]
                                                  {
        const auto jacobianColumns = FiniteVolumeDetail::JacobianColumns(build.jacobianOffsets, build.jacobianSamples);
        data->jacobianColumns = std::make_shared<LinearPatternStorage>(jacobianColumns,
            std::vector<index>{}, std::vector<index>{}, std::vector<index>{}, std::vector<index>{});
        // The assembled SGS/Jacobian writer visits each face incidence many
        // times for a frozen state. Resolve its dense column slot once at
        // setup instead of repeating a linear search in every matrix fill.
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            const auto &columns = jacobianColumns[size_t(iCell)];
            for (index entry = build.jacobianOffsets[iCell]; entry < build.jacobianOffsets[iCell + 1]; ++entry)
            {
                auto &sample = build.jacobianSamples[size_t(entry)];
                if (sample.other == UnInitIndex || sample.other == iCell)
                    continue;
                const auto found = std::find(columns.begin(), columns.end(), sample.other);
                DNDS_check_throw_info(found != columns.end(), "Jacobian stencil has no canonical column slot");
                sample.jacobianColumn = index(found - columns.begin());
            }
        }
        data->boundaryZones.assign(localZones.begin(), localZones.end());
        for (auto iFace : build.stencilFaces)
        {
            const auto cells = topology.FaceCells(iFace);
            const index iCell = cells[0] >= 0 && cells[0] < data->metrics.NumCell() ? cells[0] : cells[1];
            DNDS_check_throw_info(iCell >= 0 && iCell < data->metrics.NumCell(), "auxiliary face has no owned parent");
            index entry = build.offsets[iCell];
            while (entry < build.offsets[iCell + 1] && build.samples[entry].point != metric.FaceQuadratureBegin(iFace)) ++entry;
            DNDS_check_throw_info(entry < build.offsets[iCell + 1], "auxiliary face missing its quadrature incidence");
            const index iAux = index(build.faceEntries.size());
            const auto other = build.samples[entry].other;
            (other == UnInitIndex || other < data->metrics.NumCell() ? build.auxiliaryCoreFaces : build.auxiliaryDependentFaces).push_back(iAux);
            real area = 0;
            for (index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                area += metric.GetFaceWeight(q);
            build.auxiliaryFaceArea.push_back(area);
            build.faceEntries.push_back(entry);
        }
        // Derived work lists retain the canonical face order of the original
        // face-centric reductions. Keep repeated incidences (e.g. a periodic
        // face whose two parents are the same cell); they contribute twice.
        build.auxiliaryCellOffsets.resize(size_t(data->metrics.NumCell()) + 1, 0);
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
            build.auxiliaryCellOffsets[iCell + 1] = build.auxiliaryCellOffsets[iCell] + index(topology.CellFaces(iCell).size());
        build.auxiliaryCellFaces.resize(size_t(build.auxiliaryCellOffsets.back()));
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            index cursor = build.auxiliaryCellOffsets[iCell];
            for (index entry = build.offsets[iCell]; entry < build.offsets[iCell + 1];)
            {
                const auto iAux = build.samples[entry].auxiliaryFace;
                build.auxiliaryCellFaces[cursor++] = iAux;
                const auto iFace = build.stencilFaces[iAux];
                entry += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
            }
            DNDS_check_throw_info(cursor == build.auxiliaryCellOffsets[iCell + 1], "auxiliary cell incidence count differs");
            std::sort(build.auxiliaryCellFaces.begin() + build.auxiliaryCellOffsets[iCell], build.auxiliaryCellFaces.begin() + cursor);
        }
        for (const auto iCell : data->partition.Core())
            for (index entry = build.auxiliaryCellOffsets[iCell]; entry < build.auxiliaryCellOffsets[iCell + 1]; ++entry)
            {
                const auto other = build.samples[build.faceEntries[build.auxiliaryCellFaces[entry]]].other;
                DNDS_check_throw_info(other == UnInitIndex || other < data->metrics.NumCell(), "auxiliary core depends on an input ghost");
            }
        const auto collectFaces = [&](const auto &cells, auto &faces)
        {
            for (const auto iCell : cells)
                for (index entry = build.auxiliaryCellOffsets[iCell]; entry < build.auxiliaryCellOffsets[iCell + 1]; ++entry)
                    faces.push_back(build.auxiliaryCellFaces[entry]);
            std::sort(faces.begin(), faces.end());
            faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
        };
        collectFaces(data->partition.Core(), build.rhsCoreFaces);
        collectFaces(data->partition.HaloDependent(), build.rhsDependentFaces);
        data->boundaryLayout=std::make_shared<Detail::BoundaryPointLayout>();
        data->boundaryLayout->mesh=data->mesh;
        data->boundaryLayout->stamp=data->mesh->versions.Stamp(R::Change::Topology|R::Change::Geometry|R::Change::Distribution|R::Change::Layout);
        build.boundaryPoints.reserve(build.boundaryEntries.size());
        for(const auto &[zone,boundary]:settings.boundaries)
            data->hasInterfaces=data->hasInterfaces||boundary.kind=="interface";
        for (auto &sample : build.jacobianSamples)
            if (sample.other == UnInitIndex)
            {
                sample.boundaryPoint = index(build.boundaryFaces.size());
                build.boundaryFaces.push_back({sample.cell, sample.face, -1, sample.zone,
                    {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]},
                    {sample.normal[0], sample.normal[1], sample.normal[2]}, sample.boundary.kind == BoundaryKind::Interface});
            }
        // Boundary trace rows belong to the FV face-quadrature publication,
        // not to variational reconstruction. Derive their stable order from
        // the canonical cell-incidence samples so P0/P1 need no Taylor plan.
        for (index entry = 0; entry < index(build.samples.size()); ++entry)
        {
            auto &sample=build.samples[entry];
            if (sample.other != UnInitIndex)
                continue;
            sample.boundaryPoint=index(build.boundaryEntries.size());
            build.boundaryEntries.push_back(entry);
            const bool interfaceRow = sample.boundary.kind == BoundaryKind::Interface;
            data->hasLocalInterfaceRows = data->hasLocalInterfaceRows || interfaceRow;
            build.boundaryPoints.push_back({sample.cell,sample.face,
                int(sample.point-metric.FaceQuadratureBegin(sample.face)),sample.zone,
                {sample.ownPoint[0],sample.ownPoint[1],sample.ownPoint[2]},
                {sample.normal[0],sample.normal[1],sample.normal[2]},interfaceRow});
        }
        std::vector<index> gradientBoundaryPoints(size_t(data->gradient.NumBoundaryQuadrature()));
        for(index iPoint=0;iPoint<data->gradient.NumBoundaryQuadrature();++iPoint)
        {
            const auto point=data->gradient.GetBoundaryQuadrature(iPoint);
            const auto globalPoint=metric.FaceQuadratureBegin(point.face)+point.quadrature;
            index entry=build.offsets[point.cell];
            while(entry<build.offsets[point.cell+1]&&build.samples[entry].point!=globalPoint) ++entry;
            DNDS_check_throw_info(entry<build.offsets[point.cell+1]&&build.samples[entry].boundaryPoint!=UnInitIndex,"gradient boundary point has no canonical boundary row");
            gradientBoundaryPoints[iPoint]=build.samples[entry].boundaryPoint;
        }
        // Publish boundary kernel fields once. These are entity-shaped input
        // arrays even though geometry and row maps remain immutable afterward.
        data->boundaryTraceGeometry = R::Array<BoundaryTraceGeometry>(
            R::RowLayout::Dense(build.boundaryEntries.size(), 1));
        data->gradientBoundaryRows = R::Array<index>(
            R::RowLayout::Dense(gradientBoundaryPoints.size(), 1));
        for (const int zone : data->forceZones)
            data->boundaryConditions.push_back(data->boundary.At(zone));
        {
            auto pointWrite = data->boundaryTraceGeometry.Write();
            auto rowWrite = data->gradientBoundaryRows.Write();
            const auto points = pointWrite.View<R::FixedRows<1>>();
            const auto rows = rowWrite.View<R::FixedRows<1>>();
            for (index iPoint = 0; iPoint < points.RowCount(); ++iPoint)
            {
                const auto &sample = build.samples[build.boundaryEntries[iPoint]];
                const auto condition = std::lower_bound(
                    data->forceZones.begin(), data->forceZones.end(), sample.zone);
                DNDS_check_throw_info(condition != data->forceZones.end() &&
                                          *condition == sample.zone,
                                      "boundary trace lacks a resolved condition");
                points(iPoint, 0) = {
                    sample.cell, sample.face, sample.point,
                    index(condition - data->forceZones.begin()),
                    sample.boundaryPoint, sample.side,
                    {sample.normal[0], sample.normal[1], sample.normal[2]},
                    {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}};
            }
            for (index iPoint = 0; iPoint < rows.RowCount(); ++iPoint)
                rows(iPoint, 0) = gradientBoundaryPoints[size_t(iPoint)];
        }
        if (settings.numerics.positivity || limitedVR)
        {
            std::vector<int> volumeOrders;
            if (settings.numerics.positivityVolumePoints) volumeOrders.push_back(Geom::Elem::INT_ORDER_MAX);
            if ((settings.equation.source == "chemical" || equation.HasRANS()) && std::find(volumeOrders.begin(), volumeOrders.end(), settings.numerics.RHSIntegrationOrder()) == volumeOrders.end())
                volumeOrders.push_back(settings.numerics.RHSIntegrationOrder());
            build.positivityOffsets.resize(size_t(data->metrics.NumCell()) + 1, 0);
            for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
            {
                build.positivityOffsets[iCell + 1] = build.positivityOffsets[iCell] + build.offsets[iCell + 1] - build.offsets[iCell];
                for (auto order : volumeOrders)
                    build.positivityOffsets[iCell + 1] += Geom::Elem::Quadrature(topology.GetCellElement(iCell), order).GetNumPoints();
            }
            const auto count = build.positivityOffsets.back();
            R::Array<index> cells(R::RowLayout::Dense(count, 1));
            R::Array<real> points(R::RowLayout::Dense(count, 3));
            {
                auto cellWrite = cells.Write(); auto pointWrite = points.Write();
                const auto c = cellWrite.View<R::FixedRows<1>>(); const auto p = pointWrite.View<R::FixedRows<3>>();
                Geom::tSmallCoords coords;
                for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
                {
                    index iPoint = build.positivityOffsets[iCell];
                    const auto append = [&](const Geom::tPoint &point)
                    {
                        c(iPoint, 0) = iCell;
                        for (int d = 0; d < 3; ++d)
                            p(iPoint, d) = point[d];
                        build.positivityDisplacements.push_back(point - metric.GetCellCent(iCell));
                        ++iPoint;
                    };
                    for (index entry = build.offsets[iCell]; entry < build.offsets[iCell + 1]; ++entry) append(build.samples[entry].ownPoint);
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
            DNDS_check_throw_info(index(build.positivityDisplacements.size()) == count,
                                  "direct-P1 positivity geometry size differs");
        } });
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
        // Publish canonical numerical operands before any executable operator
        // is constructed. Interior records use the final, inert BC descriptor.
        {
            const index interiorCondition = index(data->boundaryConditions.size());
            data->boundaryConditions.emplace_back();
            const auto publishSamples = [&](const auto &rows, auto &array)
            {
                array = R::Array<FiniteVolumeDetail::SampleRecord>(R::RowLayout::Dense(rows.size(), 1));
                auto write = array.Write();
                const auto view = write.template View<R::FixedRows<1>>();
                for (index iRow = 0; iRow < index(rows.size()); ++iRow)
                {
                    const auto &sample = rows[iRow];
                    const index condition = sample.other == UnInitIndex
                                                ? data->forceSlots.at(sample.zone)
                                                : interiorCondition;
                    view(iRow, 0) = FiniteVolumeDetail::PackSample(sample, condition);
                }
            };
            publishSamples(build.samples, data->stencil.samples);
            publishSamples(build.jacobianSamples, data->stencil.jacobianSamples);
            const auto publishIndices = [](const auto &values, auto &array)
            {
                array = R::Array<index>(R::RowLayout::Dense(values.size(), 1));
                auto write = array.Write();
                const auto view = write.template View<R::FixedRows<1>>();
                for (index iRow = 0; iRow < index(values.size()); ++iRow)
                    view(iRow, 0) = values[iRow];
            };
            publishIndices(build.offsets, data->stencil.offsets);
            publishIndices(build.jacobianOffsets, data->stencil.jacobianOffsets);
            publishIndices(build.stencilFaces, data->stencil.stencilFaces);
            publishIndices(build.faceEntries, data->stencil.faceEntries);
            publishIndices(build.auxiliaryCoreFaces, data->stencil.auxiliaryCoreFaces);
            publishIndices(build.auxiliaryDependentFaces, data->stencil.auxiliaryDependentFaces);
            publishIndices(build.rhsCoreFaces, data->stencil.rhsCoreFaces);
            publishIndices(build.rhsDependentFaces, data->stencil.rhsDependentFaces);
            publishIndices(build.auxiliaryCellOffsets, data->stencil.auxiliaryCellOffsets);
            publishIndices(build.auxiliaryCellFaces, data->stencil.auxiliaryCellFaces);
            publishIndices(build.positivityOffsets, data->stencil.positivityOffsets);
            publishIndices(build.boundaryEntries, data->stencil.boundaryEntries);
            publishIndices(data->partition.Core(), data->stencil.coreCells);
            publishIndices(data->partition.HaloDependent(), data->stencil.dependentCells);
            data->stencil.auxiliaryFaceArea = R::Array<real>(R::RowLayout::Dense(build.auxiliaryFaceArea.size(), 1));
            data->stencil.positivityDisplacements = R::Array<real>(R::RowLayout::Dense(build.positivityDisplacements.size(), 3));
            auto areas = data->stencil.auxiliaryFaceArea.Write();
            auto displacements = data->stencil.positivityDisplacements.Write();
            const auto a = areas.View<R::FixedRows<1>>();
            const auto x = displacements.View<R::FixedRows<3>>();
            for (index iRow = 0; iRow < a.RowCount(); ++iRow)
                a(iRow, 0) = build.auxiliaryFaceArea[iRow];
            for (index iRow = 0; iRow < x.RowCount(); ++iRow)
                for (int d = 0; d < 3; ++d)
                    x(iRow, d) = build.positivityDisplacements[iRow][d];
        }
        const auto publishBoundary = [](const auto &values)
        {
            R::Array<BoundaryPoint> rows(R::RowLayout::Dense(values.size(), 1));
            {
                auto write = rows.Write();
                const auto view = write.View<R::FixedRows<1>>();
                for (index iRow = 0; iRow < index(values.size()); ++iRow)
                    view(iRow, 0) = values[iRow];
            }
            return rows;
        };
        data->boundaryLayout->points = publishBoundary(build.boundaryPoints);
        data->boundaryLayout->faces = publishBoundary(build.boundaryFaces);
        if (limitedVR)
            data->limitedVRGeometry = std::make_shared<LimitedVRGeometry>(*data);
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
            CPU::InitializeFlowProfile(mesh, flow, profile, primitive, _data->equation.Gas(), frame); });
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
            const auto result = CPU::MeasureFlowError(mesh, metric, flow.owned, profile, primitive, _data->equation.Gas(), time, frame);
            errors = result.errors; volume = result.volume; });
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
            valid = CPU::AdmissibleMeans(values, bounds); });
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, Metrics().Communicator());
        return valid;
    }
}
