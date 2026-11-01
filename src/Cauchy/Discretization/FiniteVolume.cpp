#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include <optional>
#include <set>
#include <atomic>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        using UniformView = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
        using FieldView = R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows>;
        struct Sample
        {
            index cell, other, point;
            int side;
            Geom::tPoint normal, ownPoint, otherPoint;
            Geom::tGPoint rotation;
            BoundaryCondition boundary;
        };
        template <class View>
        FlowState Load(const View &input, index iCell)
        {
            FlowState state;
            for (int j = 0; j < 5; ++j)
                state[j] = input(iCell, j);
            return state;
        }
        FlowState RotateState(FlowState state, const Geom::tGPoint &rotation)
        {
            const Geom::tPoint momentum{state[1], state[2], state[3]};
            const Geom::tPoint rotated = rotation * momentum;
            for (int d = 0; d < 3; ++d)
                state[1 + d] = rotated[d];
            return state;
        }
        FlowGradient RotateGradient(const FlowGradient &gradient, const Geom::tGPoint &rotation)
        {
            FlowGradient result{};
            for (int component = 0; component < 5; ++component)
                for (int d = 0; d < 3; ++d)
                    for (int k = 0; k < 3; ++k)
                        if (component == 0 || component == 4)
                            result[component][d] += rotation(d, k) * gradient[component][k];
                        else
                            for (int v = 0; v < 3; ++v)
                                result[component][d] += rotation(component - 1, v) * gradient[1 + v][k] * rotation(d, k);
            return result;
        }
        R::RowLayout FieldLayout(const R::RowLayout &prototype, index rows, index width)
        {
            if (prototype.Kind() == R::RowLayoutKind::ColMajorTable)
                return R::RowLayout::ColMajor(rows, width, prototype.PaddingOptions());
            if (prototype.Kind() == R::RowLayoutKind::TiledColMajorTable)
                return R::RowLayout::TiledColMajor(rows, width, prototype.Stride());
            return R::RowLayout::Dense(rows, width);
        }
    }
    struct FiniteVolume::Data
    {
        ssp<Geom::UnstructuredMesh> mesh;
        DomainSettings settings;
        EquationSystem equation;
        Riemann::RiemannFlux flux;
        BoundarySystem boundary;
        CFV::FiniteVolumePlan metrics;
        CFV::VariationalReconstructionPlan reconstruction;
        CFV::GradientPlan gradient;
        Geom::EntityWorkPartition partition;
        std::vector<index> offsets;
        std::vector<Sample> samples;
        // For each incidence quadrature, each side: [value,dx,dy,dz]*n modes.
        // Allocate once then fill from CFV's canonical basis/moments cache.
        R::Array<real> basis;
        Data(ssp<Geom::UnstructuredMesh> source, DomainSettings controls, EquationSystem physics)
            : mesh(std::move(source)), settings(std::move(controls)), equation(std::move(physics)), flux(Riemann::RiemannFlux::Resolve(settings.numerics.flux)),
              boundary(settings.boundaries, equation.Gas()), metrics(CFV::FiniteVolumePlan::ConstructMetrics(mesh, settings.numerics.cfv.intOrder)),
              reconstruction(CFV::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings.numerics.cfv)),
              gradient(CFV::GradientPlan::Prepare(*mesh, metrics, settings.numerics.gradient == "greenGauss" ? CFV::GradientMethod::GreenGauss : settings.numerics.gradient == "ggmp" ? CFV::GradientMethod::GGMP
                                                                                                                                                                                      : CFV::GradientMethod::LeastSquares)),
              partition(Geom::EntityWorkPartition::CellNodeCell(*mesh)) {}
    };
    struct FiniteVolume::Workspace
    {
        CFV::ReconstructionWorkspace reconstruction;
        CFV::FVField gradient;
        CFV::FVField primitive;
        CFV::ReconstructionBoundaryValues boundary;
        R::Array<real> gradientBoundary;
        R::TransferBinding meanHalo, coefficientHalo, gradientHalo;
        const Data *owner;
        uint64_t ownedIdentity, ghostIdentity;
        std::atomic<index> fallbacks{0};
        Workspace(const Data &data, CFV::FVField &state)
            : reconstruction(data.reconstruction.BuildWorkspace(5, FieldLayout(state.Owned().Layout(), data.metrics.NumCell(), data.reconstruction.NRecDOF() * 5), {1})),
              gradient(data.metrics.BuildField(5 * data.metrics.Dimension())), primitive(data.metrics.BuildField(5)), boundary(data.reconstruction.BuildBoundaryValues(5)),
              gradientBoundary(data.gradient.BuildBoundaryValues(5)), meanHalo(data.metrics.BindHalo(state)),
              coefficientHalo(data.metrics.BindHalo(reconstruction.coefficients)), gradientHalo(data.metrics.BindHalo(gradient)), owner(&data),
              ownedIdentity(state.Owned().Identity()), ghostIdentity(state.Ghost().Identity()) {}
    };
    FiniteVolume::FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation)
    {
        auto data = std::make_shared<Data>(std::move(mesh), settings, equation);
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
            for (auto iFace : topology.CellFaces(iCell)) count += metric.FaceQuadratureEnd(iFace) - metric.FaceQuadratureBegin(iFace);
            data->offsets[iCell + 1] = count;
        }
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
                    Sample sample{iCell, other, q, side, (side ? -1 : 1) * metric.GetFaceNorm(q), CFV::Detail::FacePointInCell(topology, iFace, side, metric.GetFacePoint(q)),
                                  CFV::Detail::OtherPointInCell(topology, iFace, side, metric.GetCellCent(other == UnInitIndex ? iCell : other)),
                                  CFV::Detail::OtherFrameInCell(topology, iFace, side), condition};
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
    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspace(CFV::FVField &state) const
    {
        Metrics().ValidateField(state, 5);
        return std::make_shared<Workspace>(*_data, state);
    }
    void FiniteVolume::Evaluate(CFV::FVField &state, CFV::FVField &residual, Workspace &work,
                                R::EagerEngine &engine, const R::ExecutionStream &stream) const
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
        DNDS_check_throw_info(Admissible(state), "inadmissible Cauchy mean state");
        work.fallbacks = 0;
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        auto meshRead = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = meshRead.HostView();
        const auto &gas = _data->equation.Gas();
        // Explicit quadrature values keep BC policy out of CFV kernels. All
        // stages refresh these values from their own current mean state.
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
                    const auto face = condition.FaceValue(gas, Load(values, q.cell), {q.normal[0], q.normal[1], q.normal[2]});
                    for (int j = 0; j < 5; ++j)
                        target(iPoint, j) = face[j];
                }
            };
            fill(work.boundary.values, _data->reconstruction);
            fill(work.gradientBoundary, _data->gradient);
        }
        auto exchange = [&](R::TransferBinding &binding)
        {
            auto operation = engine.ChecksAccess() ? binding.BeginChecked(stream) : binding.BeginScheduled(stream.Context());
            operation.Wait();
        };
        exchange(work.meanHalo);
        Algebra::Fill(work.reconstruction.coefficients, 0.);
        if (_data->settings.numerics.reconstruction == "gradient")
        {
            _data->gradient.Evaluate(state, work.gradientBoundary, work.gradient, {1});
            _data->gradient.ConvertToCoefficients(work.gradient, _data->reconstruction, work.reconstruction.coefficients);
        }
        else if (_data->settings.numerics.reconstruction == "variational")
        {
            // Frozen sends and an out-of-place Jacobi target. Core never
            // acquires the active ghost destination. SOR is rejected at preflight.
            for (int iteration = 0; iteration < _data->settings.numerics.reconstructionIterations; ++iteration)
            {
                auto operation = engine.ChecksAccess() ? work.coefficientHalo.BeginChecked(stream) : work.coefficientHalo.BeginScheduled();
                _data->reconstruction.DoReconstructionIterTo(state, work.reconstruction.coefficients, work.reconstruction, work.boundary, CFV::ReconstructionRegion::Core);
                operation.Wait();
                _data->reconstruction.DoReconstructionIterTo(state, work.reconstruction.coefficients, work.reconstruction, work.boundary, CFV::ReconstructionRegion::HaloDependent);
                _data->reconstruction.CommitReconstruction(work.reconstruction);
            }
        }
        if (_data->settings.numerics.limiter == "WBAP")
        {
            exchange(work.coefficientHalo);
            for (int order = _data->settings.numerics.cfv.maxOrder; order >= 1; --order)
            {
                _data->reconstruction.DoLimiterWBAP_3OrderTo(work.reconstruction, order);
                _data->reconstruction.CommitReconstruction(work.reconstruction);
                if (order > 1)
                    exchange(work.coefficientHalo);
            }
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
                    const auto value = gas.Primitive(_data->boundary.At(mesh.GetFaceZone(point.face)).FaceValue(gas, Load(mean.View<R::UniformTableRows>(), point.cell), {point.normal[0], point.normal[1], point.normal[2]}));
                    const FlowState primitive{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                    for (int j = 0; j < 5; ++j)
                        write.View()(q, j) = primitive[j];
                }
            }
            _data->gradient.Evaluate(work.primitive, work.gradientBoundary, work.gradient, {1});
            exchange(work.gradientHalo);
        }
        auto basisRead = _data->basis.Read();
        const auto basis = basisRead.View<R::TableRows>();
        const int modes = _data->reconstruction.NRecDOF(), dim = Metrics().Dimension();
        std::optional<R::Array<real>::ReadLease> gradOwned, gradGhost;
        FieldView gradients;
        if (_data->equation.Viscous())
        {
            gradOwned.emplace(work.gradient.Owned().Read());
            gradGhost.emplace(work.gradient.Ghost().Read());
            gradients = {gradOwned->View<R::UniformTableRows>(), gradGhost->View<R::UniformTableRows>()};
        }
        const auto kernel = [data = _data, metric, basis, modes, dim, gas, diagnostics = &work.fallbacks](const std::vector<index> &cells, const auto &mean, const auto &coefficient, const auto &gradientInput, auto result)
        {
            for (auto iCell : cells)
            {
                FlowState integrated{};
                for (index entry = data->offsets[iCell]; entry < data->offsets[iCell + 1]; ++entry)
                {
                    const auto &sample = data->samples[entry];
                    auto reconstruct = [&](index cell, int parent)
                    {
                        auto state = Load(mean, cell);
                        for (int component = 0; component < 5; ++component)
                            for (int mode = 0; mode < modes; ++mode)
                            {
                                const real value = coefficient(cell, component * modes + mode);
                                state[component] += value * basis(entry, parent * 4 * modes + mode);
                            }
                        // Explicit local first-order fallback keeps the mean
                        // unchanged when a high-order face value is inadmissible.
                        if (!gas.Admissible(state, data->settings.numerics.densityFloor, data->settings.numerics.pressureFloor))
                        {
                            diagnostics->fetch_add(1, std::memory_order_relaxed);
                            state = Load(mean, cell);
                        }
                        FlowGradient primitiveGradient{};
                        if (data->equation.Viscous())
                            for (int component = 0; component < 5; ++component)
                                for (int d = 0; d < dim; ++d)
                                    primitiveGradient[component][d] = gradientInput(cell, component * dim + d);
                        return std::make_pair(state, primitiveGradient);
                    };
                    const auto left = reconstruct(iCell, 0);
                    FlowState right;
                    FlowGradient rightGradient{};
                    if (sample.other != UnInitIndex)
                    {
                        auto other = reconstruct(sample.other, 1);
                        right = RotateState(other.first, sample.rotation);
                        rightGradient = RotateGradient(other.second, sample.rotation);
                    }
                    else
                        right = sample.boundary.Exterior(gas, left.first, {sample.normal[0], sample.normal[1], sample.normal[2]});
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto leftFlux = data->equation.PrepareFluxState(left.first), rightFlux = data->equation.PrepareFluxState(right);
                    const auto leftMean = Load(mean, iCell), rightMean = sample.other == UnInitIndex ? sample.boundary.Exterior(gas, leftMean, normal) : RotateState(Load(mean, sample.other), sample.rotation);
                    const auto prepared = data->equation.PrepareRiemann(data->flux, leftFlux, rightFlux, leftMean, rightMean, normal);
                    auto flux = data->flux.Evaluate(prepared.Input(leftFlux, rightFlux, normal), data->settings.numerics.riemann.KernelSettings()).flux;
                    if (data->equation.Viscous())
                    {
                        auto primitive = gas.Primitive(left.first);
                        auto gradient = left.second;
                        const auto ownPrimitive = gas.Primitive(Load(mean, iCell));
                        const auto rightPrimitive = gas.Primitive(sample.other == UnInitIndex ? sample.boundary.FaceValue(gas, Load(mean, iCell), normal) : RotateState(Load(mean, sample.other), sample.rotation));
                        // Evaluate into an owning vector: an Eigen expression
                        // would retain a dangling temporary cell barycenter.
                        const Geom::tPoint distance = sample.other == UnInitIndex ? Geom::tPoint(sample.ownPoint - metric.GetCellBary(iCell)) : Geom::tPoint(sample.otherPoint - metric.GetCellBary(iCell));
                        const real lengthSquared = distance.squaredNorm();
                        DNDS_check_throw_info(lengthSquared > 0, "zero viscous stencil displacement");
                        const std::array<real, 5> own{ownPrimitive.density, ownPrimitive.velocity[0], ownPrimitive.velocity[1], ownPrimitive.velocity[2], ownPrimitive.temperature};
                        const std::array<real, 5> neighbor{rightPrimitive.density, rightPrimitive.velocity[0], rightPrimitive.velocity[1], rightPrimitive.velocity[2], rightPrimitive.temperature};
                        for (int component = 0; component < 5; ++component)
                        {
                            real projected = 0;
                            for (int d = 0; d < 3; ++d)
                            {
                                if (sample.other != UnInitIndex)
                                    gradient[component][d] = .5 * (gradient[component][d] + rightGradient[component][d]);
                                projected += gradient[component][d] * distance[d];
                            }
                            for (int d = 0; d < 3; ++d)
                                gradient[component][d] += (neighbor[component] - own[component] - projected) * distance[d] / lengthSquared;
                        }
                        if (sample.other != UnInitIndex)
                        {
                            const auto p = gas.Primitive(right);
                            for (int d = 0; d < 3; ++d)
                                primitive.velocity[d] = .5 * (primitive.velocity[d] + p.velocity[d]);
                        }
                        else if (sample.boundary.kind == BoundaryKind::SlipWall || sample.boundary.kind == BoundaryKind::NoSlipWall)
                        {
                            primitive = gas.Primitive(sample.boundary.FaceValue(gas, left.first, normal));
                            if (!sample.boundary.isothermal)
                            {
                                real heat = 0;
                                for (int d = 0; d < 3; ++d)
                                    heat += gradient[4][d] * normal[d];
                                for (int d = 0; d < 3; ++d)
                                    gradient[4][d] -= heat * normal[d];
                            }
                        }
                        const auto viscous = gas.ViscousFlux(primitive, gradient, normal);
                        for (int component = 0; component < 5; ++component)
                            flux[component] -= viscous[component];
                    }
                    for (int component = 0; component < 5; ++component)
                        integrated[component] -= metric.GetFaceWeight(sample.point) * flux[component];
                }
                for (int component = 0; component < 5; ++component)
                    result(iCell, component) = integrated[component];
            }
        };
        auto operation = engine.ChecksAccess() ? work.coefficientHalo.BeginChecked(stream) : work.coefficientHalo.BeginScheduled();
        R::SubmitArrays(engine, stream, [kernel, data = _data, gradientInput = FieldView{gradients.owned, {}}](auto, auto mean, auto rec, auto result)
                        { kernel(data->partition.Core(), FieldView{mean, {}}, FieldView{rec, {}}, gradientInput, result); }, R::ReadAs<R::UniformTableRows>(state.Owned()), R::ReadAs<R::UniformTableRows>(work.reconstruction.coefficients.Owned()), R::WriteAs<R::UniformTableRows>(residual.Owned()));
        operation.Wait();
        R::SubmitArrays(engine, stream, [kernel, data = _data, gradients](auto, auto mean, auto meanGhost, auto rec, auto recGhost, auto result)
                        { kernel(data->partition.HaloDependent(), FieldView{mean, meanGhost}, FieldView{rec, recGhost}, gradients, result); }, R::ReadAs<R::UniformTableRows>(state.Owned()), R::ReadAs<R::UniformTableRows>(state.Ghost()), R::ReadAs<R::UniformTableRows>(work.reconstruction.coefficients.Owned()), R::ReadAs<R::UniformTableRows>(work.reconstruction.coefficients.Ghost()), R::WriteAs<R::UniformTableRows>(residual.Owned()));
        // Local read scopes pin geometry/cache pointers through both submitted
        // phases. A later async public wrapper can retain these same scopes;
        // this CPU synchronous convenience observes only its own stream prefix.
        stream.Record().WaitHost();
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
                valid &= _data->equation.Gas().Admissible(Load(input, iCell), _data->settings.numerics.densityFloor, _data->settings.numerics.pressureFloor); });
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
    real FiniteVolume::StableStep(const CFV::FVField &state, real cfl) const
    {
        real dt = std::numeric_limits<real>::max();
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
        DNDS_check_throw_info(std::isfinite(cfl) && cfl > 0, "positive CFL required");
        Metrics().ValidateField(state, 5);
        auto geometry = Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        auto read = state.Owned().Read();
        const auto input = read.View<R::UniformTableRows>();
        for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
        {
            real radius = 0;
            const auto value = Load(input, iCell);
            DNDS_check_throw_info(_data->equation.Gas().Admissible(value), "invalid mean in CFL estimate");
            for (index entry = _data->offsets[iCell]; entry < _data->offsets[iCell + 1]; ++entry)
            {
                const auto &sample = _data->samples[entry];
                const real h = metric.cells(iCell, 7);
                radius += metric.GetFaceWeight(sample.point) * (Riemann::SignalSpeed(_data->equation.PrepareFluxState(value), {sample.normal[0], sample.normal[1], sample.normal[2]}) +
                                                                (_data->equation.Viscous() ? 4 * _data->equation.Gas().viscosity / (value[0] * h) : 0));
            }
            dt = std::min(dt, cfl * metric.GetCellVol(iCell) / radius);
        } });
        MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, Metrics().Communicator());
        return dt;
    }
}
