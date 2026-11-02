#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include <optional>
#include <set>
#include <atomic>
#include <fstream>
#include <iomanip>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        using UniformView = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        using FieldView = R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows>;
        using FiniteVolumeDetail::Load;
        using FiniteVolumeDetail::RotateState;
        using FiniteVolumeDetail::Sample;
        R::RowLayout FieldLayout(const R::RowLayout &prototype, index rows, index width)
        {
            if (prototype.Kind() == R::RowLayoutKind::ColMajorTable)
                return R::RowLayout::ColMajor(rows, width, prototype.PaddingOptions());
            if (prototype.Kind() == R::RowLayoutKind::TiledColMajorTable)
                return R::RowLayout::TiledColMajor(rows, width, prototype.Stride());
            return R::RowLayout::Dense(rows, width);
        }
    }
    struct FiniteVolume::Workspace
    {
        CFV::ReconstructionWorkspace reconstruction;
        CFV::FVField gradient;
        CFV::FVField primitive;
        CFV::FVField lowResidual;
        R::Array<real> positivityValues;
        real betaMin = 1;
        index betaLimited = 0;
        index characteristicFallbacks = 0;
        CFV::ReconstructionBoundaryValues boundary;
        R::Array<real> gradientBoundary;
        R::TransferBinding meanHalo, coefficientHalo, gradientHalo;
        AuxiliaryValues auxiliary;
        std::optional<R::TransferBinding> waveHalo, pressureHalo;
        const Data *owner;
        uint64_t ownedIdentity, ghostIdentity;
        std::atomic<index> fallbacks{0};
        int iterations = 0;
        real relativeUpdate = 0;
        CFV::FaceCharacteristicTransform characteristic;
        // Only a handful of physical boundary zones. Kernels accumulate into
        // disjoint atomic counters; no mutable mesh or reconstruction policy.
        std::map<int, std::array<std::atomic<real>, 3>> forces;
        Workspace(const Data &data, CFV::FVField &state)
            : reconstruction(data.reconstruction.BuildWorkspace(5, FieldLayout(state.Owned().Layout(), data.metrics.NumCell(), data.reconstruction.NRecDOF() * 5), {1})),
              gradient(data.metrics.BuildField(5 * data.metrics.Dimension())), primitive(data.metrics.BuildField(5)), lowResidual(data.metrics.BuildField(5)),
              positivityValues(R::RowLayout::Dense(data.positivitySamples ? data.positivitySamples->Size() : 0, 5)), boundary(data.reconstruction.BuildBoundaryValues(5)),
              gradientBoundary(data.gradient.BuildBoundaryValues(5)), meanHalo(data.metrics.BindHalo(state)),
              coefficientHalo(data.metrics.BindHalo(reconstruction.coefficients)), gradientHalo(data.metrics.BindHalo(gradient)), owner(&data),
              ownedIdentity(state.Owned().Identity()), ghostIdentity(state.Ghost().Identity()), characteristic(data.reconstruction.BuildFaceTransform(0, 5))
        {
            Algebra::Fill(reconstruction.coefficients, 0.);
            Algebra::Fill(reconstruction.temporary, 0.);
            const bool wave = data.settings.numerics.riemann.useStencil && data.flux.UsesTransverseSignal();
            const bool pressure = data.settings.numerics.riemann.useStencil && data.flux.UsesPressureStencil();
            int stencilFlags[4]{int(wave), -int(wave), int(pressure), -int(pressure)};
            MPI_Allreduce(MPI_IN_PLACE, stencilFlags, 4, MPI_INT, MPI_MIN, data.metrics.Communicator());
            DNDS_check_throw_info(stencilFlags[0] == -stencilFlags[1] && stencilFlags[2] == -stencilFlags[3], "Riemann stencil capability differs between domain ranks");
            for (const auto &[zone, condition] : data.settings.boundaries)
                forces.try_emplace(std::stoi(zone));
        }
    };
    FiniteVolume::FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation)
    {
        auto data = std::make_shared<Data>(std::move(mesh), ResolveSweepOrdering(settings), equation);
        auto meshRead = data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto topology = meshRead.HostView();
        auto geometry = data->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        std::set<int> localZones;
        MPI::LocalStage(data->mesh->getMPI(), [&]
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
        const int n = data->reconstruction.NRecDOF();
        data->basis = R::Array<real>(R::RowLayout::Dense(count, 2 * 4 * n));
        auto basisWrite = data->basis.Write();
        const auto basisOut = basisWrite.View<R::TableRows>();
        for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
        {
            const auto faces = topology.CellFaces(iCell);
            for (rowsize slot = 0; slot < rowsize(faces.size()); ++slot)
            {
                const auto iFace = faces[slot];
                const int side = topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
                const index other = topology.CellFaceOther(iCell, iFace, slot);
                const index auxiliaryFace = index(std::lower_bound(data->stencilFaces.begin(), data->stencilFaces.end(), iFace) - data->stencilFaces.begin());
                BoundaryCondition condition;
                if (other == UnInitIndex)
                {
                    const int zone = topology.GetFaceZone(iFace);
                    condition = data->boundary.At(zone);
                    localZones.insert(zone);
                }
                for (index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                {
                    const int localQ = int(q - metric.FaceQuadratureBegin(iFace));
                    Sample sample{iCell, other, q, side, topology.GetFaceZone(iFace), (side ? -1 : 1) * metric.GetFaceNorm(q), CFV::Detail::FacePointInCell(topology, iFace, side, metric.GetFacePoint(q)),
                                  CFV::Detail::OtherPointInCell(topology, iFace, side, metric.GetCellCent(other == UnInitIndex ? iCell : other)),
                                  CFV::Detail::OtherFrameInCell(topology, iFace, side), condition};
                    sample.auxiliaryFace = auxiliaryFace;
                    const index entry = index(data->samples.size());
                    // Cached basis derivatives live in each parent's frame.
                    // Flux and neighbor state are mapped into this cell's frame.
                    if (side) sample.normal = -sample.rotation * metric.GetFaceNorm(q);
                    for (int parent = 0; parent < 2; ++parent)
                    {
                        if (parent == 1 && other == UnInitIndex) continue;
                        const int faceSide = parent == 0 ? side : 1 - side;
                        const auto matrix = data->reconstruction.GetCachedFaceBasis(iFace, faceSide, localQ);
                        for (int derivative = 0; derivative <= data->metrics.Dimension(); ++derivative)
                            for (int mode = 0; mode < n; ++mode) basisOut(entry, (parent * 4 + derivative) * n + mode) = matrix(derivative, mode);
                    }
                    data->samples.push_back(std::move(sample));
                }
            }
        } });
        MPI::LocalStage(data->mesh->getMPI(), [&]
                        {
        for (auto iFace : data->stencilFaces)
        {
            const auto cells = topology.FaceCells(iFace);
            const index iCell = cells[0] >= 0 && cells[0] < data->metrics.NumCell() ? cells[0] : cells[1];
            DNDS_check_throw_info(iCell >= 0 && iCell < data->metrics.NumCell(), "auxiliary face has no owned parent");
            index entry = data->offsets[iCell];
            while (entry < data->offsets[iCell + 1] && data->samples[entry].point != metric.FaceQuadratureBegin(iFace)) ++entry;
            DNDS_check_throw_info(entry < data->offsets[iCell + 1], "auxiliary face missing its quadrature incidence");
            data->faceEntries.push_back(entry);
        }
        data->boundaryEntries.resize(size_t(data->reconstruction.NumBoundaryQuadrature()));
        for (index iPoint = 0; iPoint < data->reconstruction.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto point = data->reconstruction.GetBoundaryQuadrature(iPoint);
            const auto globalPoint = metric.FaceQuadratureBegin(point.face) + point.quadrature;
            auto &entry = data->boundaryEntries[iPoint];
            entry = data->offsets[point.cell];
            while (entry < data->offsets[point.cell + 1] && data->samples[entry].point != globalPoint) ++entry;
            DNDS_check_throw_info(entry < data->offsets[point.cell + 1], "boundary quadrature missing from cell incidences");
        }
        if (settings.numerics.positivity)
        {
            data->positivityOffsets.resize(size_t(data->metrics.NumCell()) + 1, 0);
            for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
                data->positivityOffsets[iCell + 1] = data->positivityOffsets[iCell] + data->offsets[iCell + 1] - data->offsets[iCell] +
                    Geom::Elem::Quadrature(topology.GetCellElement(iCell), Geom::Elem::INT_ORDER_MAX).GetNumPoints();
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
                    topology.GetCoordsOnCell(iCell, coords);
                    const auto element = topology.GetCellElement(iCell);
                    const Geom::Elem::Quadrature quadrature(element, Geom::Elem::INT_ORDER_MAX);
                    for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                        append(Geom::Elem::PPhysicsCoordD01Nj(coords, Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(Geom::Elem::INT_ORDER_MAX).at(iQ)));
                }
            }
            data->positivitySamples.emplace(data->reconstruction, cells, points, false);
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
        _data = std::move(data);
    }
    const CFV::FiniteVolumePlan &FiniteVolume::Metrics() const { return _data->metrics; }
    const Geom::UnstructuredMesh &FiniteVolume::Mesh() const { return *_data->mesh; }
    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspace(CFV::FVField &state, AuxiliaryRequest request) const
    {
        Metrics().ValidateField(state, 5);
        auto work = std::make_shared<Workspace>(*_data, state);
        request.meanThermodynamics = true;
        request.cellWaveJump = request.cellWaveJump || (_data->settings.numerics.riemann.useStencil && _data->flux.UsesTransverseSignal());
        request.cellPressureMinimum = request.cellPressureMinimum || (_data->settings.numerics.riemann.useStencil && _data->flux.UsesPressureStencil());
        work->auxiliary = BuildAuxiliaryValues(request);
        if (work->auxiliary.cellWaveJump)
            work->waveHalo.emplace(Metrics().BindHalo(*work->auxiliary.cellWaveJump));
        if (work->auxiliary.cellPressureMinimum)
            work->pressureHalo.emplace(Metrics().BindHalo(*work->auxiliary.cellPressureMinimum));
        return work;
    }
    void FiniteVolume::RequireWorkspaceChannels(const Workspace &work, AuxiliaryRequest required) const
    {
        DNDS_check_throw_info(work.owner == _data.get(), "foreign finite-volume workspace");
        const auto &values = work.auxiliary;
        DNDS_check_throw_info((!required.meanThermodynamics || values.meanThermodynamics) &&
                                  (!required.faceSpectralRadius || values.faceSpectralRadius) &&
                                  (!required.faceWaveJump || values.faceWaveJump) && (!required.facePressureMinimum || values.facePressureMinimum) &&
                                  (!required.cellSpectralSum || values.cellSpectralSum) && (!required.cellWaveJump || values.cellWaveJump) &&
                                  (!required.cellPressureMinimum || values.cellPressureMinimum),
                              "finite-volume workspace lacks required auxiliary channels");
    }
    Json FiniteVolume::ReconstructionMetadata() const
    {
        return {{"dimension", Metrics().Dimension()}, {"coefficient_dofs", _data->reconstruction.NRecDOF()}, {"components", 5}, {"cfv", _data->settings.numerics.cfv}};
    }
    void FiniteVolume::ResetReconstruction(Workspace &workspace) const
    {
        DNDS_check_throw_info(workspace.owner == _data.get(), "foreign FV workspace");
        Algebra::Fill(workspace.reconstruction.coefficients, 0.);
    }
    const CFV::FVField &FiniteVolume::ReconstructionGuess(const Workspace &workspace) const
    {
        DNDS_check_throw_info(workspace.owner == _data.get(), "foreign reconstruction workspace");
        return workspace.reconstruction.coefficients;
    }
    void FiniteVolume::RestoreReconstructionGuess(Workspace &workspace, const R::Array<real> &coefficients) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            DNDS_check_throw_info(workspace.owner == _data.get() && coefficients.GetDevice() == R::Device::Host() &&
                                      !coefficients.Layout().IsVariable() && coefficients.RowCount() == Metrics().NumCell() &&
                                      coefficients.Layout().Width() == workspace.reconstruction.coefficients.Width(),
                                  "restart reconstruction guess schema mismatch");
            auto read = coefficients.Read();
            const auto values = read.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (index j = 0; j < values.RowSize(iCell); ++j)
                    DNDS_check_throw_info(std::isfinite(values(iCell, j)), "nonfinite restart reconstruction guess"); });
        std::optional<R::Array<real>::ReadLease> source;
        std::optional<R::Array<real>::WriteLease> destination;
        try
        {
            MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                            {
                source.emplace(coefficients.Read());
                destination.emplace(workspace.reconstruction.coefficients.Owned().Write()); });
        }
        catch (...)
        {
            if (destination)
                destination->CancelUnmodified();
            if (source)
                source->CancelUnmodified();
            throw;
        }
        const auto input = source->View<R::UniformTableRows>();
        const auto output = destination->View<R::UniformTableRows>();
        for (index iCell = 0; iCell < output.RowCount(); ++iCell)
            for (index j = 0; j < output.RowSize(iCell); ++j)
                output(iCell, j) = input(iCell, j);
    }
    void FiniteVolume::Evaluate(CFV::FVField &state, CFV::FVField &residual, Workspace &work,
                                R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
        Metrics().ValidateField(state, 5);
        Metrics().ValidateField(residual, 5);
        DNDS_check_throw_info(work.owner == _data.get() && stream.Context().device == R::Device::Host() &&
                                  state.Owned().Identity() == work.ownedIdentity && state.Ghost().Identity() == work.ghostIdentity,
                              "foreign FV workspace, rebound field, or unavailable execution backend");
        DNDS_check_throw_info(!Algebra::Detail::Overlaps(state.Owned().Storage(), residual.Owned().Storage()) &&
                                  !Algebra::Detail::Overlaps(state.Ghost().Storage(), residual.Owned().Storage()),
                              "Cauchy residual output overlaps its state input");
        // Residuals are synchronous at this public boundary. Preflight their
        // debug-visible output before any rank starts halo communication; no
        // row-loop check or extra mutation generation is introduced.
        auto unused = residual.Owned().Write();
        unused.CancelUnmodified(); });
        work.auxiliary._origin.reset();
        DNDS_check_throw_info(Admissible(state), "inadmissible Cauchy mean state");
        work.fallbacks = 0;
        work.iterations = 0;
        work.relativeUpdate = 0;
        work.betaMin = 1;
        work.betaLimited = 0;
        work.characteristicFallbacks = 0;
        for (auto &[zone, force] : work.forces)
            for (auto &component : force)
                component.store(0);
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = meshRead.HostView();
        const auto &gas = _data->equation.Gas();
        const bool overlap = _data->settings.execution.overlap;
        auto exchange = [&](R::TransferBinding &binding)
        {
            auto operation = engine.ChecksAccess() ? binding.BeginChecked(stream) : binding.BeginScheduled(stream.Context());
            operation.Wait();
        };
        std::optional<R::TransferOperation> meanOperation;
        if (overlap)
            meanOperation.emplace(engine.ChecksAccess() ? work.meanHalo.BeginChecked(stream) : work.meanHalo.BeginScheduled(stream.Context()));
        // Explicit quadrature values keep BC policy out of CFV kernels. All
        // stages refresh these values from their own current mean state. They
        // read owned means only, so the mean ghost write can remain active.
        {
            auto mean = state.Owned().Read();
            const auto values = mean.View<R::UniformTableRows>();
            auto fill = [&](auto &array, const auto &plan)
            {
                auto write = array.Write();
                const auto target = write.template View<R::TableRows>();
                for (index iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
                {
                    const auto q = plan.GetBoundaryQuadrature(iPoint);
                    const auto &condition = _data->boundary.At(mesh.GetFaceZone(q.face));
                    const auto face = condition.FaceValue(gas, Load(values, q.cell), {q.normal[0], q.normal[1], q.normal[2]}, {q.point[0], q.point[1], q.point[2]}, time);
                    for (int j = 0; j < 5; ++j)
                        target(iPoint, j) = face[j];
                }
            };
            fill(work.boundary.values, _data->reconstruction);
            fill(work.gradientBoundary, _data->gradient);
        }
        const auto updateReconstructionBoundary = [&]
        {
            auto mean = state.Owned().Read(), coefficients = work.reconstruction.coefficients.Owned().Read(), basis = _data->basis.Read();
            auto boundary = work.boundary.values.Write();
            const auto u = mean.View<R::UniformTableRows>(), rec = coefficients.View<R::UniformTableRows>();
            const auto b = basis.View<R::TableRows>();
            const auto values = boundary.View<R::TableRows>();
            const int modes = _data->reconstruction.NRecDOF();
            for (index iPoint = 0; iPoint < index(_data->boundaryEntries.size()); ++iPoint)
            {
                const auto entry = _data->boundaryEntries[iPoint];
                const auto &sample = _data->samples[entry];
                auto inside = Load(u, sample.cell);
                for (int j = 0; j < 5; ++j)
                    for (int mode = 0; mode < modes; ++mode)
                        inside[j] += rec(sample.cell, j * modes + mode) * b(entry, mode);
                if (!gas.Admissible(inside))
                    inside = Load(u, sample.cell);
                const auto face = sample.boundary.FaceValue(gas, inside, {sample.normal[0], sample.normal[1], sample.normal[2]},
                                                            {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time);
                for (int j = 0; j < 5; ++j)
                    values(iPoint, j) = face[j];
            }
        };
        std::optional<R::TransferOperation> firstCoefficientOperation;
        if (overlap && _data->settings.numerics.reconstruction == "variational")
        {
            if (_data->settings.numerics.reconstructionBoundaryFromPolynomial)
                updateReconstructionBoundary();
            firstCoefficientOperation.emplace(engine.ChecksAccess() ? work.coefficientHalo.BeginChecked(stream) : work.coefficientHalo.BeginScheduled(stream.Context()));
            // The first VR core consumes only owned means and coefficients.
            // It is independent of both active ghost writes and of auxiliary
            // preparation, which remains a separate mean-state operator.
            _data->reconstruction.DoReconstructionIterTo(state, work.reconstruction.coefficients, work.reconstruction, work.boundary, CFV::ReconstructionRegion::Core);
        }
        if (meanOperation)
            meanOperation->Wait();
        else
            exchange(work.meanHalo);
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        { PrepareAuxValues(state, work.auxiliary, time); });
        if (work.waveHalo)
            exchange(*work.waveHalo);
        if (work.pressureHalo)
            exchange(*work.pressureHalo);
        // Keep the previous converged/limited polynomial as the initial guess.
        // Resetting it every stage turns a fixed sweep cap into a spatial error.
        if (_data->settings.numerics.reconstruction != "variational")
            Algebra::Fill(work.reconstruction.coefficients, 0.);
        if (_data->settings.numerics.reconstruction == "gradient")
        {
            _data->gradient.Evaluate(state, work.gradientBoundary, work.gradient, {1});
            _data->gradient.ConvertToCoefficients(work.gradient, _data->reconstruction, work.reconstruction.coefficients);
        }
        else if (_data->settings.numerics.reconstruction == "variational")
        {
            // Frozen sends and a separate Jacobi/SOR output. Split SOR uses
            // core-first logical order; core never acquires the ghost destination.
            for (int iteration = 0; iteration < _data->settings.numerics.reconstructionIterations; ++iteration)
            {
                // Project the current boundary polynomial onto the physical
                // BC, rather than clamping unconstrained quantities to a cell
                // mean. CFV still receives plain quadrature values, no functor.
                const bool corePrepared = iteration == 0 && bool(firstCoefficientOperation);
                if (!corePrepared && _data->settings.numerics.reconstructionBoundaryFromPolynomial)
                    updateReconstructionBoundary();
                R::TransferOperation operation;
                if (corePrepared)
                {
                    operation = std::move(*firstCoefficientOperation);
                    firstCoefficientOperation.reset();
                }
                else
                    operation = engine.ChecksAccess() ? work.coefficientHalo.BeginChecked(stream) : work.coefficientHalo.BeginScheduled(stream.Context());
                if (overlap && !corePrepared)
                    _data->reconstruction.DoReconstructionIterTo(state, work.reconstruction.coefficients, work.reconstruction, work.boundary, CFV::ReconstructionRegion::Core);
                operation.Wait();
                _data->reconstruction.DoReconstructionIterTo(state, work.reconstruction.coefficients, work.reconstruction, work.boundary,
                                                             overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
                work.iterations = iteration + 1;
                bool converged = false;
                if ((_data->settings.numerics.reconstructionTolerance > 0 && iteration % 4 == 3) || iteration + 1 == _data->settings.numerics.reconstructionIterations)
                {
                    real norms[2]{};
                    auto before = work.reconstruction.coefficients.Owned().Read(), after = work.reconstruction.temporary.Owned().Read();
                    const auto a = before.View<R::UniformTableRows>(), b = after.View<R::UniformTableRows>();
                    for (index iCell = 0; iCell < a.RowCount(); ++iCell)
                        for (index j = 0; j < a.RowSize(iCell); ++j)
                        {
                            const real delta = b(iCell, j) - a(iCell, j);
                            norms[0] += delta * delta;
                            norms[1] += b(iCell, j) * b(iCell, j);
                        }
                    MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, Metrics().Communicator());
                    work.relativeUpdate = std::sqrt(norms[0] / std::max(norms[1], real(1e-300)));
                    converged = _data->settings.numerics.reconstructionTolerance > 0 && work.relativeUpdate <= _data->settings.numerics.reconstructionTolerance;
                }
                _data->reconstruction.CommitReconstruction(work.reconstruction);
                if (converged)
                    break;
            }
        }
        if (_data->settings.numerics.limiter == "WBAP")
        {
            MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                            {
            if (_data->settings.numerics.characteristicLimiter)
            {
                auto owned = state.Owned().Read(), ghost = state.Ghost().Read();
                const FieldView u{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
                auto forward = work.characteristic.forward.Write(), inverse = work.characteristic.inverse.Write();
                auto l = forward.View<R::TableRows>(), r = inverse.View<R::TableRows>();
                for (auto iFace : _data->stencilFaces)
                {
                    const auto cells = mesh.FaceCells(iFace);
                    // Extra closure faces can have no owned parent. Their
                    // matrix is never consumed by an owned-cell limiter.
                    if (cells[0] < 0 || cells[0] >= Metrics().NumCellProc())
                        continue;
                    const auto left = Load(u, cells[0]);
                    auto right = left;
                    if (cells[1] >= 0 && cells[1] < Metrics().NumCellProc())
                        right = RotateState(Load(u, cells[1]), CFV::Detail::OtherFrameInCell(mesh, iFace, 0));
                    const auto normal = metric.GetFaceNorm(metric.FaceQuadratureBegin(iFace));
                    const auto matrices = _data->equation.PrepareCharacteristics(left, right, {normal[0], normal[1], normal[2]});
                    work.characteristicFallbacks += matrices.componentwiseFallback;
                    for (index j = 0; j < 25; ++j)
                    {
                        l(iFace, j) = matrices.forward[j];
                        r(iFace, j) = matrices.inverse[j];
                    }
                }
            }
            if (_data->settings.numerics.characteristicLimiter)
                _data->reconstruction.FinalizeFaceTransform(work.characteristic); });
            for (int order = _data->settings.numerics.cfv.maxOrder; order >= 1; --order)
            {
                auto operation = engine.ChecksAccess() ? work.coefficientHalo.BeginChecked(stream) : work.coefficientHalo.BeginScheduled(stream.Context());
                if (overlap)
                    _data->reconstruction.DoLimiterWBAP_3OrderTo(work.reconstruction, order, _data->settings.numerics.characteristicLimiter ? &work.characteristic : nullptr,
                                                                 CFV::ReconstructionRegion::Core);
                operation.Wait();
                _data->reconstruction.DoLimiterWBAP_3OrderTo(work.reconstruction, order, _data->settings.numerics.characteristicLimiter ? &work.characteristic : nullptr,
                                                             overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
                _data->reconstruction.CommitReconstruction(work.reconstruction);
            }
        }
        if (_data->settings.numerics.positivity)
        {
            _data->positivitySamples->Evaluate(state, work.reconstruction.coefficients, work.positivityValues);
            MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                            {
                auto mean = state.Owned().Read(), points = work.positivityValues.Read();
                auto coefficients = work.reconstruction.coefficients.Owned().Write();
                const auto u = mean.View<R::UniformTableRows>(), p = points.View<R::UniformTableRows>();
                const auto rec = coefficients.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                {
                    const auto base = Load(u, iCell);
                    real beta = 1;
                    for (index iPoint = _data->positivityOffsets[iCell]; iPoint < _data->positivityOffsets[iCell + 1]; ++iPoint)
                    {
                        const auto target = Load(p, iPoint);
                        for (auto value : target) DNDS_check_throw_info(std::isfinite(value), "nonfinite reconstruction in positivity input");
                        beta = std::min(beta, AdmissibleFraction(base, target, _data->thermalBounds));
                    }
                    if (beta < _data->settings.numerics.positivityBetaCutoff) beta = 0;
                    if (beta < 1)
                    {
                        ++work.betaLimited; work.betaMin = std::min(work.betaMin, beta);
                        for (index j = 0; j < rec.RowSize(iCell); ++j) rec(iCell, j) *= beta;
                    }
                    for (index iPoint = _data->positivityOffsets[iCell]; iPoint < _data->positivityOffsets[iCell + 1]; ++iPoint)
                    {
                        auto value = base; const auto target = Load(p, iPoint);
                        for (int j = 0; j < 5; ++j) value[j] = (1 - beta) * base[j] + beta * target[j];
                        DNDS_check_throw_info(ThermallyAdmissible(value, _data->thermalBounds), "reconstruction positivity verification failed");
                    }
                } });
            // Re-evaluate the scaled coefficients themselves. Algebraically
            // blending sampled states alone would miss cancellation/indexing
            // mistakes in the stored polynomial and its cache layout.
            _data->positivitySamples->Evaluate(state, work.reconstruction.coefficients, work.positivityValues);
            MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                            {
                auto points = work.positivityValues.Read(); const auto p = points.View<R::UniformTableRows>();
                for (index iPoint = 0; iPoint < p.RowCount(); ++iPoint)
                    DNDS_check_throw_info(ThermallyAdmissible(Load(p, iPoint), _data->thermalBounds), "scaled reconstruction violates temperature/internal-energy bound"); });
        }
        if (_data->equation.Viscous())
        {
            // Viscous gradients are gradients of rho,u,v,w,T, not a facewise
            // linearization of conservative gradients. The latter introduces
            // artificial heat conduction even in exact linear Couette flow.
            auto fillPrimitive = [&](const auto &input, auto &output)
            {
                const auto read = input.Read();
                auto write = output.Write();
                for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                {
                    const auto value = gas.Primitive(Load(read.template View<R::UniformTableRows>(), iCell));
                    const FlowState primitive{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                    for (int j = 0; j < 5; ++j)
                        write.View()(iCell, j) = primitive[j];
                }
            };
            fillPrimitive(state.Owned(), work.primitive.Owned());
            fillPrimitive(state.Ghost(), work.primitive.Ghost());
            {
                auto mean = state.Owned().Read();
                auto write = work.gradientBoundary.Write();
                for (index q = 0; q < _data->gradient.NumBoundaryQuadrature(); ++q)
                {
                    const auto point = _data->gradient.GetBoundaryQuadrature(q);
                    const auto value = gas.Primitive(_data->boundary.At(mesh.GetFaceZone(point.face)).FaceValue(gas, Load(mean.View<R::UniformTableRows>(), point.cell), {point.normal[0], point.normal[1], point.normal[2]}, {point.point[0], point.point[1], point.point[2]}, time));
                    const FlowState primitive{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                    for (int j = 0; j < 5; ++j)
                        write.View()(q, j) = primitive[j];
                }
            }
            _data->gradient.Evaluate(work.primitive, work.gradientBoundary, work.gradient, {1});
            exchange(work.gradientHalo);
        }
        std::optional<R::Array<real>::WriteLease> lowWrite;
        R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows> lowOutput;
        if (_data->settings.numerics.positivity)
        {
            lowWrite.emplace(work.lowResidual.Owned().Write());
            lowOutput = lowWrite->View<R::UniformTableRows>();
        }
        std::optional<R::Array<real>::ReadLease> gradOwned, gradGhost;
        FieldView gradients;
        if (_data->equation.Viscous())
        {
            gradOwned.emplace(work.gradient.Owned().Read());
            gradGhost.emplace(work.gradient.Ghost().Read());
            gradients = {gradOwned->View<R::UniformTableRows>(), gradGhost->View<R::UniformTableRows>()};
        }
        ValidateAuxValues(state, work.auxiliary, time);
        const auto thermoRead = work.auxiliary.meanThermodynamics->Read();
        const auto thermo = thermoRead.View<R::UniformTableRows>();
        std::array<std::optional<R::Array<real>::ReadLease>, 4> auxiliaryReads;
        FieldView wave, pressure;
        const auto readAuxiliary = [&](const auto &field, int slot, FieldView &view)
        {
            if (!field)
                return;
            auxiliaryReads[slot].emplace(field->Owned().Read());
            auxiliaryReads[slot + 1].emplace(field->Ghost().Read());
            view = {auxiliaryReads[slot]->View<R::UniformTableRows>(), auxiliaryReads[slot + 1]->View<R::UniformTableRows>()};
        };
        readAuxiliary(work.auxiliary.cellWaveJump, 0, wave);
        readAuxiliary(work.auxiliary.cellPressureMinimum, 2, pressure);
        const auto rhs = AcquireRHSOperator();
        RHSAuxiliaryViews auxiliary;
        auxiliary.meanThermodynamics = thermo;
        if (work.auxiliary.cellWaveJump)
            auxiliary.cellWaveJump = wave;
        if (work.auxiliary.cellPressureMinimum)
            auxiliary.cellPressureMinimum = pressure;
        const auto kernel = [rhs, time, lowOutput, produceLow = _data->settings.numerics.positivity,
                             forces = &work.forces, diagnostics = &work.fallbacks](CFV::ReconstructionRegion region,
                                                                                   const auto &mean, const auto &coefficient, const auto &gradientInput,
                                                                                   const RHSAuxiliaryViews &auxiliaryInput, auto result)
        {
            RHSWriteViews outputs{result, std::nullopt, diagnostics, forces};
            if (produceLow)
                outputs.lowOrder = lowOutput;
            rhs.Apply({mean, coefficient, gradientInput, auxiliaryInput}, outputs, region, time);
        };
        auto operation = engine.ChecksAccess() ? work.coefficientHalo.BeginChecked(stream) : work.coefficientHalo.BeginScheduled();
        auto coreAuxiliary = auxiliary;
        if (coreAuxiliary.cellWaveJump)
            coreAuxiliary.cellWaveJump->ghost = {};
        if (coreAuxiliary.cellPressureMinimum)
            coreAuxiliary.cellPressureMinimum->ghost = {};
        if (overlap)
            R::SubmitArrays(engine, stream, [kernel, gradientInput = FieldView{gradients.owned, {}}, coreAuxiliary](auto, auto mean, auto rec, auto result)
                            { kernel(CFV::ReconstructionRegion::Core, FieldView{mean, {}}, FieldView{rec, {}}, gradientInput, coreAuxiliary, result); }, R::ReadAs<R::UniformTableRows>(state.Owned()), R::ReadAs<R::UniformTableRows>(work.reconstruction.coefficients.Owned()), R::WriteAs<R::UniformTableRows>(residual.Owned()));
        operation.Wait();
        R::SubmitArrays(engine, stream, [kernel, gradients, auxiliary, overlap](auto, auto mean, auto meanGhost, auto rec, auto recGhost, auto result)
                        { kernel(overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All, FieldView{mean, meanGhost}, FieldView{rec, recGhost}, gradients, auxiliary, result); }, R::ReadAs<R::UniformTableRows>(state.Owned()), R::ReadAs<R::UniformTableRows>(state.Ghost()), R::ReadAs<R::UniformTableRows>(work.reconstruction.coefficients.Owned()), R::ReadAs<R::UniformTableRows>(work.reconstruction.coefficients.Ghost()), R::WriteAs<R::UniformTableRows>(residual.Owned()));
        // Local read scopes pin geometry/cache pointers through both submitted
        // phases. A later async public wrapper can retain these same scopes;
        // this CPU synchronous convenience observes only its own stream prefix.
        stream.Record().WaitHost();
        if (!_data->settings.numerics.faceFallback || _data->settings.numerics.positivity)
            DNDS_check_throw_info(LastFaceFallbacks(work) == 0, "inadmissible reconstructed face with faceFallback disabled");
    }
    namespace
    {
        template <class MeshView>
        FlowState CellAverage(const MeshView &mesh, index iCell, const FlowProfileSettings &profile,
                              const FlowState &primitive, const PerfectGas &gas, real time)
        {
            if (profile.kind == "constant")
                return gas.Conservative(primitive);
            Geom::tSmallCoords coords;
            mesh.GetCoordsOnCell(iCell, coords);
            const auto element = mesh.GetCellElement(iCell);
            constexpr int order = Geom::Elem::INT_ORDER_MAX;
            Geom::Elem::Quadrature quadrature(element, order);
            FlowState average{};
            real volume = 0;
            for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
            {
                const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(order).at(iQ);
                const Geom::tPoint point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                const real weight = Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                const auto value = ProfileValue(profile, primitive, gas, {point[0], point[1], point[2]}, time);
                for (int j = 0; j < 5; ++j)
                    average[j] += weight * value[j];
                volume += weight;
            }
            for (auto &value : average)
                value /= volume;
            return average;
        }
    }
    void FiniteVolume::Initialize(CFV::FVField &state, const FlowProfileSettings &profile, const FlowState &primitive) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            Metrics().ValidateField(state, 5);
            auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            auto write = state.Owned().Write();
            auto values = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                const auto average = CellAverage(mesh, iCell, profile, primitive, _data->equation.Gas(), 0);
                for (int j = 0; j < 5; ++j) values(iCell, j) = average[j];
            } });
    }
    std::array<FlowState, 3> FiniteVolume::Error(const CFV::FVField &state, const FlowProfileSettings &profile, const FlowState &primitive, real time) const
    {
        std::array<FlowState, 3> errors{};
        real volume = 0;
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = meshRead.HostView();
            auto geometry = Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto read = state.Owned().Read(); const auto values = read.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                const auto exact = CellAverage(mesh, iCell, profile, primitive, _data->equation.Gas(), time);
                const real weight = metric.GetCellVol(iCell); volume += weight;
                for (int j = 0; j < 5; ++j)
                {
                    const real error = std::abs(values(iCell, j) - exact[j]);
                    errors[0][j] += weight * error; errors[1][j] += weight * error * error;
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
    int FiniteVolume::LastReconstructionIterations(const Workspace &work) const { return work.iterations; }
    const CFV::FVField &FiniteVolume::LowOrderResidual(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data.get() && _data->settings.numerics.positivity, "low-order residual requires this plan's PP workspace");
        return work.lowResidual;
    }
    const AuxiliaryValues &FiniteVolume::PreparedAuxValues(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data.get() && work.auxiliary._origin, "auxiliary values require a completed preparation for this workspace");
        return work.auxiliary;
    }
    const ThermalBounds &FiniteVolume::ThermalConstraints() const { return _data->thermalBounds; }
    index FiniteVolume::LastCharacteristicFallbacks(const Workspace &work) const
    {
        auto count = work.characteristicFallbacks;
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT64_T, MPI_SUM, Metrics().Communicator());
        return count;
    }
    std::pair<index, real> FiniteVolume::PositivityReconstructionStats(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data.get(), "foreign positivity workspace");
        auto count = work.betaLimited;
        auto beta = work.betaMin;
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT64_T, MPI_SUM, Metrics().Communicator());
        MPI_Allreduce(MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_MIN, Metrics().Communicator());
        return {count, beta};
    }
    real FiniteVolume::LastReconstructionUpdate(const Workspace &work) const { return work.relativeUpdate; }
    Vector3 FiniteVolume::LastBoundaryForce(const Workspace &work, const std::vector<int> &zones) const
    {
        Vector3 force{};
        for (int zone : zones)
        {
            const auto found = work.forces.find(zone);
            DNDS_check_throw_info(found != work.forces.end(), "force zone has no boundary condition");
            for (int d = 0; d < 3; ++d)
                force[d] += found->second[d].load();
        }
        MPI_Allreduce(MPI_IN_PLACE, force.data(), 3, MPI_DOUBLE, MPI_SUM, Metrics().Communicator());
        return force;
    }
    void FiniteVolume::WriteVisualization(const CFV::FVField &state, const Workspace &work, const std::string &prefix, real time, int hdfDeflateLevel) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        { DNDS_check_throw_info(hdfDeflateLevel >= 0 && hdfDeflateLevel <= 9, "VTKHDF compression level must be in [0,9]"); });
        auto outputSettings = _data->mesh->hdf5OutSetting;
        outputSettings.deflateLevel = hdfDeflateLevel;
        auto read = state.Owned().Read();
        const auto u = read.View<R::UniformTableRows>();
        const auto &gas = _data->equation.Gas();
        static const std::array<std::string, 10> names{"rho", "rhou", "rhov", "rhow", "rhoE", "p", "u", "v", "w", "Mach"};
        _data->mesh->PrintParallelVTKHDFDataArray(prefix, "", 10, 0, 0, 0, [](int i)
                                                  { return names.at(i); }, [&](int i, index iCell)
                                                  {
                if (i < 5) return u(iCell, i);
                const auto value = gas.Primitive(Load(u, iCell));
                if (i == 5) return value.pressure;
                if (i < 9) return value.velocity[i - 6];
                real speed = 0; for (auto component : value.velocity) speed += component * component;
                return std::sqrt(speed) / value.soundSpeed; }, {}, {}, {}, {}, {}, {}, time, Metrics().Communicator(), &outputSettings);
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            std::ofstream surface(prefix + "_surface_rank" + std::to_string(_data->mesh->mpi.rank) + ".csv");
            DNDS_check_throw_info(bool(surface), "cannot open Cauchy surface output");
            surface << "cell_id,zone,x,y,z,nx,ny,nz,weight,rho,p,u,v,w\n" << std::setprecision(17);
            auto rec = work.reconstruction.coefficients.Owned().Read(), basis = _data->basis.Read();
            const auto coefficient = rec.View<R::UniformTableRows>();
            const auto b = basis.View<R::TableRows>();
            auto geometry = Metrics().AcquireRead(); const auto metric = geometry.HostView();
            const int modes = _data->reconstruction.NRecDOF();
            for (index entry = 0; entry < index(_data->samples.size()); ++entry)
            {
                const auto &sample = _data->samples[entry];
                if (sample.other != UnInitIndex) continue;
                auto value = Load(u, sample.cell);
                for (int j = 0; j < 5; ++j) for (int mode = 0; mode < modes; ++mode)
                    value[j] += coefficient(sample.cell, j * modes + mode) * b(entry, mode);
                if (!gas.Admissible(value)) value = Load(u, sample.cell);
                const auto physical = gas.Primitive(sample.boundary.FaceValue(gas, value, {sample.normal[0], sample.normal[1], sample.normal[2]},
                    {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time));
                surface << Mesh().OriginalEntityId(Geom::EntityKind::Cell, sample.cell) << ',' << sample.zone;
                for (int d = 0; d < 3; ++d) surface << ',' << sample.ownPoint[d];
                for (int d = 0; d < 3; ++d) surface << ',' << sample.normal[d];
                surface << ',' << metric.GetFaceWeight(sample.point) << ',' << physical.density << ',' << physical.pressure;
                for (auto component : physical.velocity) surface << ',' << component;
                surface << '\n';
            } });
    }
    bool FiniteVolume::Admissible(const CFV::FVField &state) const
    {
        int valid = 1;
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            Metrics().ValidateField(state, 5);
            auto read = state.Owned().Read();
            const auto input = read.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
                valid &= ThermallyAdmissible(Load(input, iCell), _data->thermalBounds); });
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, Metrics().Communicator());
        return valid;
    }
    index FiniteVolume::LastFaceFallbacks(const Workspace &workspace) const
    {
        DNDS_check_throw_info(workspace.owner == _data.get(), "foreign FV diagnostics workspace");
        index count = workspace.fallbacks.load();
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT64_T, MPI_SUM, Metrics().Communicator());
        return count;
    }

}
