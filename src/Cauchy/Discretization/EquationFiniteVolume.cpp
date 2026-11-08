#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
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
                for (auto &lease : ghosts)
                    lease.CancelUnmodified();
                throw;
            }
            for (auto &lease : ghosts)
                lease.CancelUnmodified();
        }
        using namespace EquationFVDetail;
        struct StreamJoin
        {
            const R::ExecutionStream &stream;
            int exceptions = std::uncaught_exceptions();
            ~StreamJoin()
            {
                if (std::uncaught_exceptions() > exceptions)
                    try
                    {
                        stream.Record().WaitHost();
                    }
                    catch (...)
                    {
                    }
            }
        };
    }

    struct FiniteVolume::EquationWorkspace
    {
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
            EquationWorkspace *work;
            ProfilePhase phase;
            Clock::time_point start{};
            PhaseTime(EquationWorkspace &input, ProfilePhase section) : work(input.profiling ? &input : nullptr), phase(section)
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
        EquationDofs prototype;
        EquationDofOperations::Binding binding;
        std::vector<Field> fields, limited;
        EquationDofs coefficients, fluxCoefficients, boundary;
        std::vector<Algebra::DofHalo<>> meanHalo;
        uint64_t meanEpoch = 0;
        std::vector<R::TransferBinding> coefficientHalo, fluxHalo;
        // Accepted direct-FV physical gradients, separate from source sampling.
        // WBAP/PP may still use an explicit coefficient compatibility bridge;
        // the direct RHS never reads that bridge or evaluates a Taylor basis.
        std::vector<CFV::FVField> directGradients;
        struct DirectGradientPublication
        {
            // Weak identities reject rebinding/ABA without retaining old fields.
            std::weak_ptr<const R::ArrayStorage> input, output;
            uint64_t inputShape = 0, inputVersion = 0, outputShape = 0, outputVersion = 0;
        };
        std::shared_ptr<EquationPositivityWorkspace> positivity;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
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
        bool boundaryMomentumCurrent = false;
        std::optional<EquationMeanGradients> meanGradients;
        bool chemical;
        int iterations = 0;
        real relativeUpdate = 0;
        std::optional<EquationDofOperations::Binding> sourceState, sourceCoefficients;
        real sourceTime = 0;
        std::optional<EquationDofOperations::Binding> rhsState, rhsCoefficients;
        real rhsTime = 0;
        // Keep auxiliary cache metadata after the existing numerical fields.
        std::vector<std::array<DirectGradientPublication, 2>> directGradientPublications;

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
                Algebra::Fill(field.reconstruction.coefficients, 0.);
                Algebra::Fill(field.reconstruction.temporary, 0.);
                if (!meanOnly && (data.settings.numerics.reconstruction == "gradient" ||
                                  (data.equation.HasRANS() && data.settings.numerics.reconstruction == "constant")))
                {
                    field.gradient.emplace(data.metrics.BuildField(index(physical.Width()) * data.metrics.Dimension()));
                    field.gradientBoundary.emplace(data.gradient.BuildBoundaryValues(index(physical.Width())));
                }
                result.push_back(std::move(field));
            }
            return result;
        }
        void BuildSourceSampling()
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
                        const auto &shape = Geom::Elem::detail::NBufferAtQuadrature.buf.at(element.type).at(order).at(iQ);
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
            for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
            {
                // Direct FV midpoint values are exactly the cell means. Pin
                // their Array storage instead of copying the whole field per
                // RHS. Workspace binding checks reject later state rebinding.
                if (sourceValuesAreMeans)
                    sourceValues.push_back(prototype.PhysicalField(iField).Owned());
                else
                    sourceValues.emplace_back(FiniteVolumeDetail::FieldLayout(prototype.PhysicalField(iField).Owned().Layout(), count,
                                                                              index(prototype.PhysicalField(iField).Width())));
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
                for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
                {
                    sourceJacobianGradients.emplace_back();
                    for (int d = 0; d < owner->metrics.Dimension(); ++d)
                        sourceJacobianGradients.back()[d] = R::Array<real>(FiniteVolumeDetail::FieldLayout(prototype.PhysicalField(iField).Owned().Layout(), owned, prototype.PhysicalField(iField).Width()));
                }
            }
        }
        static EquationDofs BindCoefficients(const EquationDofs &state, const std::vector<Field> &fields, index modes)
        {
            Algebra::DofBundle bundle(state.Space());
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                bundle.Add(state.Fields().Name(iField), fields[iField].reconstruction.coefficients);
            return BindFields(state, std::move(bundle), modes);
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
        EquationWorkspace(std::shared_ptr<const Data> data, EquationDofs &state,
                          std::optional<Physics::Chemistry::ChemicalWorkerPool> pool, bool meanTrial)
            : owner(std::move(data)), meanOnly(meanTrial), prototype(state), binding(state), fields(BuildFields(*owner, state, meanOnly)),
              coefficients(BindCoefficients(state, fields, owner->reconstruction.NRecDOF())), fluxCoefficients(coefficients),
              boundary(BindBoundary(state, fields, owner->reconstruction.NumBoundaryQuadrature())), workers(std::move(pool)),
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
            if (!meanOnly && (owner->settings.numerics.limiter == "WBAP" || owner->settings.numerics.positivity))
            {
                limited = fields;
                for (auto &field : limited)
                    field.reconstruction.coefficients = owner->metrics.BuildField(field.reconstruction.coefficients.Owned().Layout());
                fluxCoefficients = BindCoefficients(state, limited, owner->reconstruction.NRecDOF());
            }
            if (!meanOnly && owner->settings.numerics.positivity)
                positivity = std::make_shared<EquationPositivityWorkspace>(owner, state, fluxCoefficients, workers);
            if (owner->settings.numerics.reconstruction != "variational")
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    directGradients.push_back(owner->metrics.BuildField(
                        FiniteVolumeDetail::FieldLayout(state.PhysicalField(iField).Owned().Layout(), owner->metrics.NumCell(),
                                                        state.PhysicalField(iField).Width() * owner->metrics.Dimension())));
            BuildSourceSampling();
            if (ransFirst >= 0)
                directGradientPublications.resize(directGradients.size());
        }
        void BindHalos(EquationDofs &state)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                meanHalo.push_back(owner->BindTrackedHalo(state.PhysicalField(iField)));
                coefficientHalo.push_back(owner->BindHalo(fields[iField].reconstruction.coefficients));
                if (!limited.empty())
                    fluxHalo.push_back(owner->BindHalo(limited[iField].reconstruction.coefficients));
            }
        }
        void RefreshDirectGradients(bool ghost)
        {
            if (owner->settings.numerics.reconstruction == "variational")
                return;
            for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
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
                        owner->gradient.ExtractLinearGradients(fluxCoefficients.PhysicalField(iField), owner->reconstruction,
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
                                     : ghost           ? fluxCoefficients.PhysicalField(iField).Ghost()
                                                       : fluxCoefficients.PhysicalField(iField).Owned();
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
        }

        void FillBoundary(const EquationDofs &state, bool polynomial, real time, EquationPointEvaluator &evaluator)
        {
            const PhaseTime timing(*this, BoundaryPhase);
            const auto mean = state.ReadOwned(), rec = coefficients.ReadOwned();
            const auto meanBlocks = BlockViews(state, mean), recBlocks = BlockViews(coefficients, rec);
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
                                  {ransModel, ransFirst, &cellMean, wallRead ? wallValues.cellMean(sample.cell, 0) : 0});
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
        }
        void ApplyRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                      EquationPointEvaluator &evaluator, real time, const AuxiliaryValues *auxiliary = nullptr,
                      R::Array<real> *faceOutput = nullptr)
        {
            const PhaseTime timing(*this, RHSPhase);
            const auto mean = region != CFV::ReconstructionRegion::Core ? state.ReadWithGhost() : state.ReadOwned();
            auto output = result.WriteOwned();
            const EquationViewLayout<const real> owned(state, mean), remote(state, mean, true);
            const EquationViewLayout<real> destination(result, output);
            const auto kernel = [&](auto u, auto ghost, auto rhs)
            { ApplyRHSRows(state, region, evaluator, time, auxiliary, mean, u, ghost, rhs, faceOutput); };
            // RHS opts into one ordered AoS family. Arbitrary block bindings
            // and physical layouts keep the same generic numerical loop.
            // Ghost shape may be absent during Core; no ghost value is read.
            if (owned.Table() && remote.Table() && destination.Table() && owned.Flow5() && remote.Flow5() && destination.Flow5())
                DispatchCPUEquationViews<EquationViewKind::CombinedAoS>(kernel, owned, remote, destination);
            else
                DispatchCPUEquationViews<>(kernel, owned, remote, destination);
        }
        template <class Input, class Output>
        void ApplyRHSRows(const EquationDofs &state, CFV::ReconstructionRegion region,
                          EquationPointEvaluator &evaluator, real time, const AuxiliaryValues *auxiliary,
                          const EquationDofs::ReadScope &mean, Input owned, Input remote, Output destination,
                          R::Array<real> *faceOutput)
        {
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
            if (auxiliary && auxiliary->meanThermodynamics)
            {
                meanProperties.emplace(auxiliary->meanThermodynamics->Read());
                preparedMean.emplace(meanProperties->View<R::UniformTableRows>());
            }
            std::optional<EquationDofs::ReadScope> rec;
            std::vector<EquationBlockReadView> recBlocks;
            if (!direct)
            {
                rec.emplace(ghosts ? fluxCoefficients.ReadWithGhost() : fluxCoefficients.ReadOwned());
                recBlocks = BlockViews(fluxCoefficients, *rec);
            }
            const auto meanBlocks = BlockViews(state, mean);
            std::vector<EquationBlockLocation> directLocations;
            for (index iBlock = 0; direct && iBlock < state.BlockNumber(); ++iBlock)
                directLocations.push_back(state.LocateBlock(iBlock));
            std::vector<R::Array<real>::ReadLease> directOwned, directGhost;
            std::vector<RHSFieldView> directViews;
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
            }
            auto boundaryWrite = boundaryMomentumFlux.Write();
            const auto wallFlux = boundaryWrite.View<R::FixedRows<3>>();
            auto faceWrite = (faceOutput ? *faceOutput : faceFlux).Write();
            const auto faceValues = faceWrite.View<R::UniformTableRows>();
            const auto gatherMean = [&](index iCell, PointScratch &point)
            {
                for (index j = 0; j < owned.ComponentNumber(); ++j)
                    point.values[j] = iCell < owned.RowCount() ? owned(iCell, j) : remote(iCell - owned.RowCount(), j);
                point.gradient.setZero();
            };
            const int modes = owner->reconstruction.NRecDOF(), dim = owner->metrics.Dimension();
            const bool viscous = !meanOnly && owner->equation.Viscous();
            const int species = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() : 0;
            const int independent = std::max(0, species - 1);
            const index width = state.ComponentNumber();
            const auto frame = owner->motion.At(time);
            PointScratch left(width), right(width), face(width), cellMean(width);
            const auto *ransModel = owner->equation.HasRANS() ? &owner->equation.RANS() : nullptr;
            const auto ransMean = ransModel ? mean.Block("rans") : EquationBlockReadView{};
            std::vector<real> integrated(size_t(width), 0), flux(size_t(width), 0), speciesFlux(size_t(independent), 0);
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
                            for (index j = 0; j < location.componentNumber; ++j)
                                for (int d = 0; d < dim; ++d)
                                {
                                    const real value = row[(location.firstComponent + j) * dim + d];
                                    point.values[logicalFirst[iBlock] + j] += value * displacement[d];
                                    if (viscous)
                                        point.gradient(logicalFirst[iBlock] + j, d) = value;
                                }
                        }
                    }
                    else
                    {
                        const auto basis = basisRead->FaceBasis(sample.face, side, sample.point, viscous ? 1 : 0, basisScratch);
                        for (index iBlock = 0; iBlock < index(meanBlocks.size()); ++iBlock)
                        {
                            const auto row = recBlocks[iBlock][iCell];
                            for (index j = 0; j < meanBlocks[iBlock].ComponentNumber(); ++j)
                                for (int mode = 0; mode < modes; ++mode)
                                {
                                    const real value = row[j * modes + mode];
                                    const index logical = logicalFirst[iBlock] + j;
                                    point.values[logical] += value * basis(0, mode);
                                    if (viscous)
                                        for (int d = 0; d < dim; ++d)
                                            point.gradient(logical, d) += value * basis(1 + d, mode);
                                }
                        }
                    }
                }
                if (ransModel)
                    RANSStateView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairReconstruction(ransMean[iCell], point.values.data() + ransFirst);
            };
            const auto computeFace = [&](index iAux)
            {
                const index first = owner->faceEntries[iAux];
                const auto &anchor = owner->samples[first];
                const index iCell = anchor.cell;
                const index end = first + metric.FaceQuadratureEnd(anchor.face) - metric.FaceQuadratureBegin(anchor.face);
                std::fill(integrated.begin(), integrated.end(), 0);
                for (index entry = first; entry < end; ++entry)
                {
                    const auto &sample = owner->samples[entry];
                    reconstruct(iCell, sample, sample.side, left);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    real wallSAViscousInterior = 0;
#endif
                    if (sample.other != UnInitIndex)
                    {
                        reconstruct(sample.other, sample, 1 - sample.side, right);
                        if (sample.changesFrame)
                            RotatePoint(right, sample.rotation, vectorStarts);
                    }
                    else
                    {
                        right.values = left.values;
                        right.gradient = left.gradient;
                        BoundaryValue(right, flowFirst, speciesFirst, independent, sample, evaluator, frame, time, true, boundaryInput(sample));
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
                    const bool noRsWall = owner->settings.numerics.noRsOnWall && sample.other == UnInitIndex && sample.boundary.kind == BoundaryKind::NoSlipWall;
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
                    if (noRsWall)
                        leftFlow = rightFlow = prepareWallState();
                    const auto leftData = EquationPointEvaluator::FluxState(leftFlow, evaluator.Evaluate(leftFlow, noRsWall ? face.Species(speciesFirst, independent) : left.Species(speciesFirst, independent), 0, {}));
                    const auto rightData = EquationPointEvaluator::FluxState(rightFlow, evaluator.Evaluate(rightFlow, noRsWall ? face.Species(speciesFirst, independent) : right.Species(speciesFirst, independent), 0, {}));
                    const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                    const auto grid = frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                    auto meanLeftData = leftData, meanRightData = rightData;
                    if (owner->flux.RequiredInputs().meanStates)
                    {
                        // Mean eigenvalue preparation follows the same path as
                        // flow5. Storage/block traversal stays in this adapter.
                        Gather(meanBlocks, iCell, cellMean);
                        const auto meanLeft = cellMean.Flow(flowFirst);
                        auto meanRight = meanLeft;
                        if (sample.other != UnInitIndex)
                        {
                            Gather(meanBlocks, sample.other, cellMean);
                            meanRight = sample.changesFrame ? FiniteVolumeDetail::RotateState(cellMean.Flow(flowFirst), sample.rotation) : cellMean.Flow(flowFirst);
                        }
                        else
                            meanRight = FiniteVolumeDetail::BoundaryExterior(sample, owner->equation.Gas(), meanLeft, time, frame, {});
                        meanLeftData = preparedMean
                                           ? Riemann::AcousticState{{meanLeft, (*preparedMean)(iCell, 0)}, (*preparedMean)(iCell, 1)}
                                           : owner->equation.PrepareFluxState(meanLeft);
                        meanRightData = preparedMean && sample.other != UnInitIndex
                                            ? Riemann::AcousticState{{meanRight, (*preparedMean)(sample.other, 0)}, (*preparedMean)(sample.other, 1)}
                                            : owner->equation.PrepareFluxState(meanRight);
                    }
                    const auto convection = owner->equation.EvaluateFlowFlux(owner->flux, leftData, rightData, meanLeftData, meanRightData,
                                                                             normal, grid, owner->settings.numerics.riemann.KernelSettings());
                    const auto &flowFlux = convection.flux;
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
                    const real alpha = owner->flux.Kind() == Riemann::Solver::Rusanov
                                           ? .5 * (std::abs(speedLeft) + std::abs(speedRight))
                                           : convection.contact;
                    for (index j = 0; j < width; ++j)
                        if (j < flowFirst || j >= flowFirst + 5)
                            flux[j] = noRsWall ? face.values[j] * speedLeft : AdvectedScalarFlux(left.values[j], right.values[j], speedLeft, speedRight, alpha);
                    for (int j = 0; j < 5; ++j)
                        flux[flowFirst + j] = flowFlux[j];
                    if (ransModel && ransModel->ComponentNumber() == 2)
                    {
                        FlowState average;
                        for (int j = 0; j < 5; ++j)
                            average[j] = .5 * (leftFlow[j] + rightFlow[j]);
                        const auto stress = TurbulentNormalFlux(average, noRsWall ? face.values[ransFirst] : .5 * (left.values[ransFirst] + right.values[ransFirst]), normal);
                        for (int j = 0; j < 5; ++j)
                            flux[flowFirst + j] += stress[j];
                    }
                    const bool wall = sample.other == UnInitIndex && sample.boundary.kind == BoundaryKind::NoSlipWall;
                    if (viscous && (sample.other != UnInitIndex || wall))
                    {
                        const real inverseDistance = wall && owner->settings.numerics.noGRPOnWall ? 0 : 1 / sample.viscousDistance;
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
                        if (noRsWall)
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
                            flux[flowFirst + j] -= diffusion[j];
                        for (int k = 0; k < independent; ++k)
                            flux[speciesFirst + k] -= speciesFlux[k];
                    }
                    if (sample.boundaryPoint != UnInitIndex)
                        for (int d = 0; d < 3; ++d)
                            wallFlux(sample.boundaryPoint, d) = wall || sample.boundary.kind == BoundaryKind::SlipWall
                                                                    ? metric.GetFaceWeight(sample.point) * flux[flowFirst + 1 + d]
                                                                    : 0;
                    for (index j = 0; j < width; ++j)
                        integrated[j] -= metric.GetFaceWeight(sample.point) * flux[j];
                }
                for (index j = 0; j < width; ++j)
                    faceValues(iAux, j) = integrated[j];
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
                for (index entry = owner->offsets[iCell]; entry < owner->offsets[iCell + 1];)
                {
                    const auto &sample = owner->samples[entry];
                    const index iAux = sample.auxiliaryFace;
                    const auto &anchor = owner->samples[owner->faceEntries[iAux]];
                    const bool opposite = sample.side != anchor.side;
                    for (index j = 0; j < width; ++j)
                        flux[j] = (opposite ? -1 : 1) * faceValues(iAux, j);
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
                        }
                    for (index j = 0; j < width; ++j)
                        integrated[j] += flux[j];
                    entry += metric.FaceQuadratureEnd(sample.face) - metric.FaceQuadratureBegin(sample.face);
                }
                for (index j = 0; j < destination.ComponentNumber(); ++j)
                    destination(iCell, j) = integrated[j];
            };
            if (region == CFV::ReconstructionRegion::All)
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                    compute(iCell);
            else
                for (const auto iCell : region == CFV::ReconstructionRegion::Core ? owner->partition.Core() : owner->partition.HaloDependent())
                    compute(iCell);
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
                    sourceSampling->EvaluateValueGradient(state.PhysicalField(iField), fluxCoefficients.PhysicalField(iField), outputs);
                    continue;
                }
                if (owner->settings.numerics.reconstruction == "variational")
                    sourceSampling->Evaluate(state.PhysicalField(iField), fluxCoefficients.PhysicalField(iField), sourceValues[iField]);
                else if (!sourceValuesAreMeans)
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
                            sourceSampling->Evaluate(state.PhysicalField(iField), fluxCoefficients.PhysicalField(iField), sourceGradients[iField][d], 1 + d);
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
                    sourceJacobianSampling->EvaluateValueGradient(state.PhysicalField(iField), fluxCoefficients.PhysicalField(iField), outputs);
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
            sourceCoefficients.emplace(fluxCoefficients.NumericAlias());
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
            const auto means = state.ReadOwned(), coefficients = fluxCoefficients.ReadOwned();
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
            return unchanged(*sourceState, state) && unchanged(*sourceCoefficients, fluxCoefficients);
        }
        /// High-order RHS source sampling supplies the opt-in quadrature
        /// source Jacobian or refreshes publication provenance only. The
        /// default Jacobian continues to sample its own center gradients.
        void PrepareVolumeSource(const EquationDofs &state, EquationDofs *result, EquationPointEvaluator &evaluator, real time = 0)
        {
            const PhaseTime timing(*this, SourcePhase);
            // Kinematic and constitutive sources share this additive entrance;
            // face-only calls never include either kind of source.
            const auto frame = owner->motion.At(time);
            if (result && frame.moving)
            {
                const auto read = state.ReadOwned();
                auto write = result->WriteOwned();
                const auto flow = read.Block("flow");
                const auto output = write.Block("flow");
                const auto geometry = owner->metrics.AcquireRead();
                const auto metric = geometry.HostView();
                for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                {
                    FlowState mean{};
                    for (int j = 0; j < 5; ++j)
                        mean[j] = flow(iCell, j);
                    const auto source = frame.RotationSource(mean);
                    for (int j = 0; j < 5; ++j)
                        output(iCell, j) += metric.GetCellVol(iCell) * source[j];
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
                output.emplace(result->WriteOwned());
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

    struct FiniteVolume::EquationMeanWorkspace
    {
        std::shared_ptr<EquationWorkspace> work;
    };

    void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, EquationWorkspace &work, AuxiliaryValues &auxiliary,
                                            R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        const auto local = [&](auto &&action)
        {
            if (_data->settings.implicit.collectiveChecks)
                MPI::LocalStage(mpi, action);
            else
                action();
        };
        if (work.meanGradients)
        {
            ValidateWorkspace(state, work);
            PrepareAuxValues(state, auxiliary, *work.meanGradients, engine, stream, time, work.workers);
            local([&]
                  {
                if (work.SourceSamplesCurrent(state, time)) return;
                auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {true});
                work.PrepareVolumeSource(state, nullptr, evaluator);
                work.PublishSourceState(state, time); });
            return;
        }
        std::optional<EquationAuxiliaryPreparation> preparation;
        local([&]
              {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(stream.Context().device == R::Device::Host(), "equation correction requires a CPU stream");
            PreflightHalo(state);
            preparation.emplace(BeginAuxValues(state, auxiliary, time, work.workers)); });
        StreamJoin join{stream};
        std::vector<Algebra::DofHalo<>::Operation> transfers;
        transfers.reserve(work.meanHalo.size());
        ++work.meanEpoch;
        for (auto &halo : work.meanHalo)
            transfers.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        if (_data->settings.execution.overlap)
            engine.Submit(stream, [&](auto)
                          { preparation->Apply(CFV::ReconstructionRegion::Core); });
        for (auto &transfer : transfers)
            transfer.Wait();
        local([&]
              {
            if (_data->settings.CheckHaloFreshness())
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    work.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), work.meanEpoch);
            engine.Submit(stream, [&](auto)
            {
                preparation->Apply(_data->settings.execution.overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
                preparation->Finish(auxiliary);
                // Source Jacobians use the retained current polynomial. RANS
                // inputs must be current before publishing these samples.
                // No VR sweep or chemical-rate evaluation is hidden here.
                if (work.SourceSamplesCurrent(state, time)) return;
                if (_data->equation.HasRANS())
                {
                    auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                                        {true, false, false, false, false});
                    work.PrepareVolumeSource(state, nullptr, evaluator);
                }
                else work.SampleSourceState(state);
                work.PublishSourceState(state, time);
            });
            stream.Record().WaitHost(); });
    }
    void FiniteVolume::InitializeConservative(EquationDofs &state, const std::vector<real> &values) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
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
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
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

    std::shared_ptr<FiniteVolume::EquationWorkspace> FiniteVolume::BuildEquationWorkspace(EquationDofs &state,
                                                                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        return BuildEquationWorkspaceImpl(state, std::move(workers), false);
    }
    std::shared_ptr<FiniteVolume::EquationMeanWorkspace> FiniteVolume::BuildEquationMeanWorkspace(EquationDofs &state,
                                                                                                  std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        auto inner = BuildEquationWorkspaceImpl(state, std::move(workers), true);
        std::shared_ptr<EquationMeanWorkspace> result;
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        { result = std::make_shared<EquationMeanWorkspace>(EquationMeanWorkspace{std::move(inner)}); });
        return result;
    }
    std::shared_ptr<FiniteVolume::EquationWorkspace> FiniteVolume::BuildEquationWorkspaceImpl(EquationDofs &state,
                                                                                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers, bool meanOnly) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        std::string operationControls;
        MPI::LocalStage(mpi, [&]
                        {
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "equation FV state belongs to another row space");
            const auto &n = _data->settings.numerics;
            DNDS_check_throw_info((n.reconstruction == "constant" || n.reconstruction == "gradient" || n.reconstruction == "variational") &&
                                  n.reconstructionIterations >= 0, "invalid equation reconstruction controls");
            DNDS_check_throw_info((!_data->equation.HasMixture() || _data->flux.Kind() == Riemann::Solver::Rusanov) &&
                (!n.riemann.useStencil || (!_data->flux.UsesTransverseSignal() && !_data->flux.UsesPressureStencil())),
                "extended EOS or stencil flux needs an explicit preparation provider");
            DNDS_check_throw_info(!n.faceFallback &&
                (n.limiter == "none" || (n.limiter == "WBAP" && !n.characteristicLimiter)),
                "requested extended FV flux/PP/limiter profile is not connected yet");
            DNDS_check_throw_info(!n.positivity || n.positivityReconstruction == "Euler", "equation PP uses the Euler reconstruction rule; archive Segment mode remains in the flow5 adapter");
            DNDS_check_throw_info(!_data->equation.Viscous() || n.viscousGradient == "polynomial", "extended viscosity requires the conservative polynomial gradient");
            DNDS_check_throw_info(!(_data->equation.Viscous() && n.limiter == "WBAP") || n.viscousLimited,
                                  "separate unlimited viscous polynomial is not connected in the equation workspace");
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
            operationControls = Json{{"meanOnly", meanOnly}, {"numerics", _data->settings.numerics}, {"equation", _data->settings.equation},
                {"motion", _data->settings.motion}, {"boundaries", _data->settings.boundaries}, {"overlap", _data->settings.execution.overlap},
                {"collectiveChecks", _data->settings.implicit.collectiveChecks}, {"properties", _data->equation.PropertyKey().parameters},
                {"rans", _data->equation.HasRANS() ? Json(_data->equation.RANS().Settings()) : Json()}}.dump(); });
        AgreeEquationBinding(state, mpi.comm, operationControls);
        std::shared_ptr<EquationWorkspace> result;
        MPI::LocalStage(mpi, [&]
                        { result = std::make_shared<EquationWorkspace>(_data, state, std::move(workers), meanOnly); });
        result->BindHalos(state);
        if (!meanOnly && _data->equation.HasRANS() && _data->equation.RANS().ComponentNumber() == 2)
            result->meanGradients.emplace(BuildMeanGradients(state));
        return result;
    }
    const EquationDofs &FiniteVolume::ReconstructionGuess(const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.coefficients;
    }
    void FiniteVolume::ValidateWorkspace(const EquationDofs &state, const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        work.binding.Check(state);
    }
    const EquationDofs &FiniteVolume::FluxReconstruction(const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.fluxCoefficients;
    }
    void FiniteVolume::RestoreReconstructionGuess(EquationWorkspace &work, const EquationDofs &guess) const
    {
        auto restore = ReserveReconstructionRestoreImpl(work, guess, {});
        restore.Commit();
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestore(std::shared_ptr<EquationWorkspace> work, const EquationDofs &guess, bool collectiveChecks) const
    {
        const auto validate = [&]
        { DNDS_check_throw_info(bool(work), "null equation reconstruction workspace"); };
        if (collectiveChecks)
            MPI::LocalStage(MPIInfo(Metrics().Communicator()), validate);
        else
            validate();
        auto &reference = *work;
        return ReserveReconstructionRestoreImpl(reference, guess, std::move(work), collectiveChecks);
    }
    FiniteVolume::ReconstructionRestore FiniteVolume::ReserveReconstructionRestoreImpl(EquationWorkspace &work, const EquationDofs &guess,
                                                                                       std::shared_ptr<EquationWorkspace> retained, bool collectiveChecks) const
    {
        const auto stage = [&](const auto &run)
        {
            if (collectiveChecks)
                MPI::LocalStage(MPIInfo(Metrics().Communicator()), run);
            else
                run();
        };
        struct Reservation
        {
            std::shared_ptr<EquationWorkspace> owner;
            std::optional<EquationDofs::ReadScope> source;
            std::optional<EquationDofs::WriteScope> destination;
            std::vector<EquationBlockArrayView<const real>> input;
            std::vector<EquationBlockArrayView<real>> output;
            bool started = false;
            ~Reservation()
            {
                if (!started && destination)
                    destination->CancelUnmodified();
            }
        };
        std::shared_ptr<Reservation> reservation;
        stage([&]
              {
                reservation = std::make_shared<Reservation>();
                reservation->owner = std::move(retained);
                DNDS_check_throw_info(work.owner == _data, "foreign equation reconstruction workspace");
                work.coefficients.RequireCompatible(guess);
                EquationDofOperations::RequireDistinct(work.coefficients, guess);
                reservation->source.emplace(guess.ReadOwned());
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                {
                    const auto input = reservation->source->Block(iBlock);
                    for (index iCell = 0; iCell < input.owned.RowCount(); ++iCell)
                        for (index j = 0; j < input.ComponentNumber(); ++j)
                            DNDS_check_throw_info(std::isfinite(input(iCell, j)), "nonfinite equation reconstruction guess");
                    reservation->input.push_back(input.owned);
                }
                reservation->destination.emplace(work.coefficients.WriteOwned());
                for (index iBlock = 0; iBlock < guess.BlockNumber(); ++iBlock)
                    reservation->output.push_back(reservation->destination->Block(iBlock)); });
        std::optional<ReconstructionRestore> restore;
        stage([&]
              { restore.emplace(ReconstructionRestore([reservation, &work]() noexcept
                                                      {
            reservation->started = true;
            for (index iBlock = 0; iBlock < index(reservation->input.size()); ++iBlock)
            {
                const auto input = reservation->input[iBlock];
                const auto output = reservation->output[iBlock];
                for (index iCell = 0; iCell < output.RowCount(); ++iCell)
                    for (index j = 0; j < output.ComponentNumber(); ++j)
                        output(iCell, j) = input(iCell, j);
            }
            work.sourceState.reset();
            work.sourceCoefficients.reset();
            work.rhsState.reset();
            work.rhsCoefficients.reset(); })); });
        return std::move(*restore);
    }
    void FiniteVolume::ResetReconstruction(EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation reconstruction workspace");
        EquationDofOperations::Fill(work.coefficients, 0);
        for (auto &field : work.fields)
            Algebra::Fill(field.reconstruction.temporary, 0.);
        work.sourceState.reset();
        work.sourceCoefficients.reset();
        work.rhsState.reset();
        work.rhsCoefficients.reset();
    }
    int FiniteVolume::LastReconstructionIterations(const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.iterations;
    }
    real FiniteVolume::LastReconstructionUpdate(const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.relativeUpdate;
    }
    void FiniteVolume::LimitReconstructionPP(const EquationDofs &state, EquationWorkspace &work) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            ValidateWorkspace(state, work);
            DNDS_check_throw_info(work.positivity, "equation reconstruction PP was not requested");
            work.positivity->Apply(state, work.coefficients, work.fluxCoefficients, true); });
    }
    std::pair<index, real> FiniteVolume::PositivityReconstructionStats(const EquationWorkspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        auto result = work.positivity ? work.positivity->LocalStatistics() : std::pair<index, real>{0, 1};
        MPI_Allreduce(MPI_IN_PLACE, &result.first, 1, MPI_INT64_T, MPI_SUM, Metrics().Communicator());
        MPI_Allreduce(MPI_IN_PLACE, &result.second, 1, MPI_DOUBLE, MPI_MIN, Metrics().Communicator());
        return result;
    }

    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, EquationWorkspace &work,
                                R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        EvaluateEquationImpl(state, &residual, work, engine, stream, time);
    }
    void FiniteVolume::Evaluate(EquationDofs &state, EquationDofs &residual, EquationWorkspace &work,
                                AuxiliaryValues &auxiliary, R::EagerEngine &engine,
                                const R::ExecutionStream &stream, real time) const
    {
        EvaluateEquationImpl(state, &residual, work, engine, stream, time, &auxiliary);
    }
    void FiniteVolume::PrepareReconstruction(EquationDofs &state, EquationWorkspace &work, R::EagerEngine &engine,
                                             const R::ExecutionStream &stream, real time) const
    {
        EvaluateEquationImpl(state, nullptr, work, engine, stream, time);
    }
    void FiniteVolume::EvaluateMean(EquationDofs &state, EquationDofs &residual, EquationMeanWorkspace &work,
                                    R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const
    {
        EvaluateEquationImpl(state, &residual, *work.work, engine, stream, time);
    }
    void FiniteVolume::EvaluateEquationImpl(EquationDofs &state, EquationDofs *residual, EquationWorkspace &work,
                                            R::EagerEngine &engine, const R::ExecutionStream &stream, real time,
                                            AuxiliaryValues *auxiliary) const
    {
        const EquationWorkspace::PhaseTime timing(work, EquationWorkspace::TotalPhase);
        work.sourceState.reset();
        work.sourceCoefficients.reset();
        work.rhsState.reset();
        work.rhsCoefficients.reset();
        work.relativeUpdate = !work.meanOnly && _data->settings.numerics.reconstruction == "variational" ? std::numeric_limits<real>::infinity() : 0;
        const MPIInfo mpi(Metrics().Communicator());
        // Repeated entry/CPU-join agreement follows the same checked/eager
        // policy as later local kernels. Unchecked execution still validates
        // locally and propagates failures to the fail-fast frontend. Payload
        // transfer preflight remains mandatory and unchanged.
        const auto local = [&](auto &&action)
        {
            if (_data->settings.implicit.collectiveChecks)
                MPI::LocalStage(mpi, action);
            else
                action();
        };
        // Observe only this lane's producer prefix. MPI remains on the caller's
        // controller thread, including when the CPU launcher uses worker threads.
        local([&]
              { stream.Record().WaitHost(); });
        std::optional<EquationPointEvaluator> evaluator;
        std::optional<EquationAuxiliaryPreparation> auxiliaryPreparation;
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
            DNDS_check_throw_info(work.owner == _data && std::isfinite(time) && stream.Context().device == R::Device::Host(),
                                  "foreign equation workspace, invalid time or missing CPU backend");
            ValidateWorkspace(state, work);
            if (residual)
            {
                _data->equation.ValidateStateSchema(*residual);
                state.RequireCompatible(*residual);
                EquationDofOperations::RequireDistinct(state, *residual);
                EquationDofOperations::RequireDistinct(work.coefficients, *residual);
                if (!work.limited.empty()) EquationDofOperations::RequireDistinct(work.fluxCoefficients, *residual);
                auto write = residual->WriteOwned(); write.CancelUnmodified();
            }
            PreflightHalo(state); PreflightHalo(work.coefficients);
            if (!work.limited.empty()) PreflightHalo(work.fluxCoefficients);
            _data->partition.Validate();
            work.boundaryMomentumCurrent = false;
            if (!deferredEvaluator) createEvaluator(); });
        if (auxiliary && !work.meanGradients)
            local([&]
                  { auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers)); });
        const auto begin = [&](std::vector<R::TransferBinding> &bindings)
        {
            std::vector<R::TransferOperation> operations;
            operations.reserve(bindings.size());
            for (auto &binding : bindings)
                operations.push_back(engine.ChecksAccess() ? binding.BeginChecked(stream) : binding.BeginScheduled(stream.Context()));
            return operations;
        };
        const auto wait = [](auto &operations)
        { for (auto &operation : operations) operation.Wait(); operations.clear(); };
        const auto submit = [&](auto action)
        { engine.Submit(stream, [action = std::move(action)](auto)
                        { action(); }); };
        const bool overlap = _data->settings.execution.overlap;
        const auto rest = overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All;
        const auto sweep = [&](CFV::ReconstructionRegion region)
        {
            const EquationWorkspace::PhaseTime timing(work, EquationWorkspace::VRPhase);
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                auto &field = work.fields[iField];
                _data->reconstruction.DoReconstructionIterTo(state.PhysicalField(iField), field.reconstruction.coefficients,
                                                             field.reconstruction, field.boundary, region);
            }
        };
        const auto gradient = [&](CFV::ReconstructionRegion region)
        {
            const EquationWorkspace::PhaseTime timing(work, EquationWorkspace::GradientPhase);
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            {
                auto &field = work.fields[iField];
                _data->gradient.Evaluate(state.PhysicalField(iField), *field.gradientBoundary, *field.gradient, field.vectorStarts, region);
            }
        };
        // Locals captured by worker commands remain alive even on exceptions.
        StreamJoin join{stream};
        std::vector<Algebra::DofHalo<>::Operation> meanOperations;
        meanOperations.reserve(work.meanHalo.size());
        ++work.meanEpoch;
        for (auto &halo : work.meanHalo)
            meanOperations.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));
        std::vector<R::TransferOperation> firstCoefficients;
        const auto &numerics = _data->settings.numerics;
        const bool variational = !work.meanOnly && numerics.reconstruction == "variational";
        const bool firstSweep = overlap && variational && numerics.reconstructionIterations > 0 && !deferredEvaluator;
        if (firstSweep)
            firstCoefficients = begin(work.coefficientHalo);
        // Post both transfers before queueing core work. A CPU-visible MPI
        // bridge may otherwise join that work while checking producer readiness.
        if (auxiliaryPreparation && overlap)
            submit([&]
                   { auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });
        if (!deferredEvaluator)
            submit([&, variational]
                   { work.FillBoundary(state, variational && _data->settings.numerics.reconstructionBoundaryFromPolynomial, time, *evaluator); });
        const bool directGradientWork = !work.meanOnly && (numerics.reconstruction == "gradient" ||
                                                           (_data->equation.HasRANS() && numerics.reconstruction == "constant"));
        if (directGradientWork && overlap && !deferredEvaluator)
            submit([&]
                   { gradient(CFV::ReconstructionRegion::Core); });
        if (firstSweep)
            submit([&]
                   { sweep(CFV::ReconstructionRegion::Core); });
        wait(meanOperations);
        if (auxiliaryPreparation)
        {
            submit([&]
                   {
                auxiliaryPreparation->Apply(overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);
                auxiliaryPreparation->Finish(*auxiliary); });
            local([&]
                  { stream.Record().WaitHost(); });
        }
        else if (auxiliary)
        {
            // Two-equation transport explicitly needs mean gradients; those
            // are independent of the reconstruction being computed below.
            PrepareAuxValues(state, *auxiliary, *work.meanGradients, engine, stream, time, work.workers);
        }
        if (deferredEvaluator)
        {
            local(createEvaluator);
            submit([&, variational]
                   { work.FillBoundary(state, variational && numerics.reconstructionBoundaryFromPolynomial, time, *evaluator); });
        }
        if (_data->settings.CheckHaloFreshness())
            local([&]
                  {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                work.meanHalo[iField].RequireCurrent(state.PhysicalField(iField), work.meanEpoch); });
        if (directGradientWork)
        {
            submit([&, rest]
                   { gradient(deferredEvaluator ? CFV::ReconstructionRegion::All : rest); });
            submit([&]
                   {
                for (auto &field : work.fields)
                    if (numerics.reconstruction == "gradient")
                        _data->gradient.ConvertToCoefficients(*field.gradient, _data->reconstruction, field.reconstruction.coefficients);
                    else
                        Algebra::Fill(field.reconstruction.coefficients, 0.); });
            local([&]
                  { stream.Record().WaitHost(); });
        }
        else if (variational)
        {
            for (int iteration = 0; iteration < numerics.reconstructionIterations; ++iteration)
            {
                std::vector<R::TransferOperation> operations;
                if (iteration == 0 && firstSweep)
                    operations = std::move(firstCoefficients);
                else
                {
                    local([&]
                          { stream.Record().WaitHost(); });
                    if (numerics.reconstructionBoundaryFromPolynomial)
                        local([&]
                              { work.FillBoundary(state, true, time, *evaluator); });
                    operations = begin(work.coefficientHalo);
                    if (overlap)
                        submit([&]
                               { sweep(CFV::ReconstructionRegion::Core); });
                }
                wait(operations);
                submit([&, rest]
                       { sweep(rest); });
                local([&]
                      { stream.Record().WaitHost(); });
                bool converged = false;
                if (numerics.reconstructionTolerance > 0 && (iteration % 4 == 3 || iteration + 1 == numerics.reconstructionIterations))
                {
                    real norms[2]{};
                    local([&]
                          {
                        for (const auto &field : work.fields)
                        {
                            const auto aRead = field.reconstruction.coefficients.Owned().Read(), bRead = field.reconstruction.temporary.Owned().Read();
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
                local([&]
                      { for (auto &field : work.fields) _data->reconstruction.CommitReconstruction(field.reconstruction); });
                work.iterations = iteration + 1;
                if (converged)
                    break;
            }
        }
        else
            local([&]
                  {
                stream.Record().WaitHost();
                for (auto &field : work.fields) Algebra::Fill(field.reconstruction.coefficients, 0.); });
        if (!work.limited.empty())
        {
            local([&]
                  { EquationDofOperations::Copy(work.fluxCoefficients, work.coefficients); });
            if (numerics.limiter == "WBAP")
                for (int order = numerics.cfv.maxOrder; order >= 1; --order)
                {
                    auto operations = begin(work.fluxHalo);
                    const auto limit = [&, order](CFV::ReconstructionRegion region)
                    { for (auto &field : work.limited) _data->reconstruction.DoLimiterWBAP_3OrderTo(field.reconstruction, order, nullptr, region); };
                    if (overlap)
                        submit([limit]
                               { limit(CFV::ReconstructionRegion::Core); });
                    wait(operations);
                    submit([limit, rest]
                           { limit(rest); });
                    local([&]
                          { stream.Record().WaitHost(); });
                    local([&]
                          { for (auto &field : work.limited) _data->reconstruction.CommitReconstruction(field.reconstruction); });
                }
        }
        if (work.positivity)
            local([&]
                  { const EquationWorkspace::PhaseTime timing(work, EquationWorkspace::PPPhase);
                    work.positivity->Apply(state, work.coefficients, work.fluxCoefficients, false); });
        auto operations = begin(work.limited.empty() ? work.coefficientHalo : work.fluxHalo);
        // Explicit compatibility conversion after limiter/PP, once per row.
        // The physical-gradient RHS itself never interprets coefficient data.
        submit([&]
               { work.RefreshDirectGradients(false); });
        if (overlap && residual)
            submit([&]
                   { work.ApplyRHS(state, *residual, CFV::ReconstructionRegion::Core, *evaluator, time, auxiliary); });
        wait(operations);
        submit([&]
               { work.RefreshDirectGradients(true); });
        if (residual)
        {
            submit([&, rest]
                   { work.ApplyRHS(state, *residual, rest, *evaluator, time, auxiliary); });
            submit([&]
                   { work.PrepareVolumeSource(state, residual, *evaluator, time); });
        }
        local([&]
              { stream.Record().WaitHost(); });
        if (residual && (work.chemical || work.ransFirst >= 0) && work.sourceSampling)
            local([&]
                  { work.PublishSourceState(state, time); });
        work.boundaryMomentumCurrent = bool(residual);
        work.rhsState.emplace(state.NumericAlias());
        work.rhsCoefficients.emplace(work.fluxCoefficients.NumericAlias());
        work.rhsTime = time;
    }

    R::Array<real> FiniteVolume::BuildFaceFluxBuffer(index components) const
    {
        Metrics().Validate();
        DNDS_check_throw_info(components > 0, "face flux requires positive component count");
        return R::Array<real>(R::RowLayout::Dense(index(_data->stencilFaces.size()), components));
    }
    void FiniteVolume::EvaluateFlux(const EquationDofs &state, EquationDofs &residual, EquationWorkspace &work,
                                    CFV::ReconstructionRegion region, real time, R::Array<real> *faceOutput, const AuxiliaryValues *auxiliary) const
    {
        ValidateWorkspace(state, work);
        state.RequireCompatible(residual);
        EquationDofOperations::RequireDistinct(state, residual);
        EquationDofOperations::RequireDistinct(work.fluxCoefficients, residual);
        DNDS_check_throw_info(work.rhsState && work.rhsCoefficients && work.rhsTime == time, "flux needs a completed spatial preparation at this time");
        work.rhsState->Check(state, true);
        work.rhsCoefficients->Check(work.fluxCoefficients, true);
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                  region == CFV::ReconstructionRegion::HaloDependent,
                              "unknown flux region");
        if (faceOutput)
        {
            DNDS_check_throw_info(faceOutput->GetDevice() == R::Device::Host() && !faceOutput->Layout().IsVariable() &&
                                      faceOutput->RowCount() == index(_data->stencilFaces.size()) && faceOutput->Layout().Width() == state.ComponentNumber(),
                                  "equation face output shape/device differs");
            for (const auto *fields : std::array<const EquationDofs *, 3>{&state, &residual, &work.fluxCoefficients})
                for (index iField = 0; iField < fields->PhysicalFieldNumber(); ++iField)
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(faceOutput->Storage(), fields->PhysicalField(iField).Owned().Storage()) &&
                                              !Algebra::Detail::Overlaps(faceOutput->Storage(), fields->PhysicalField(iField).Ghost().Storage()),
                                          "face output aliases equation fields");
        }
        if (auxiliary)
            ValidateAuxValues(state, *auxiliary, time);
        auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                            {true, true, true, false, false});
        work.ApplyRHS(state, residual, region, evaluator, time, auxiliary, faceOutput);
        work.boundaryMomentumCurrent = region == CFV::ReconstructionRegion::All;
    }
    void FiniteVolume::AddSource(const EquationDofs &state, EquationDofs &residual, EquationWorkspace &work, real time) const
    {
        ValidateWorkspace(state, work);
        state.RequireCompatible(residual);
        EquationDofOperations::RequireDistinct(state, residual);
        EquationDofOperations::RequireDistinct(work.fluxCoefficients, residual);
        DNDS_check_throw_info(work.rhsState && work.rhsCoefficients && work.rhsTime == time, "source needs a completed spatial preparation at this time");
        work.rhsState->Check(state, true);
        work.rhsCoefficients->Check(work.fluxCoefficients, true);
        auto evaluator = _data->equation.MakePointEvaluator(work.workers ? work.workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{},
                                                            {true, false, false, work.chemical, false});
        work.PrepareVolumeSource(state, &residual, evaluator, time);
        work.PublishSourceState(state, time);
    }

    Vector3 FiniteVolume::LastBoundaryForce(const EquationWorkspace &work, const std::vector<int> &zones) const
    {
        Vector3 force{};
        const MPIInfo mpi(Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
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

    void FiniteVolume::ValidateEquationSourceJacobian(const EquationDofs &state, const EquationWorkspace &work, real time) const
    {
        DNDS_check_throw_info(work.owner == _data && (work.chemical || work.ransFirst >= 0), "foreign or source-free reconstruction");
        if (!work.sourceSampling)
            return;
        DNDS_check_throw_info(work.sourceState && work.sourceCoefficients && work.sourceTime == time, "source quadrature requires a current completed sample publication");
        work.sourceState->Check(state, true);
        work.sourceCoefficients->Check(work.fluxCoefficients, true);
    }
    bool FiniteVolume::AddEquationSourceJacobian(const EquationDofs &state, const EquationWorkspace &work,
                                                 const R::Array<real>::WriteLease &diagonal, DiagonalStorage storage, real scale, EquationPointEvaluator &evaluator,
                                                 const EquationFVDetail::MeanLinearizationOutput *linearization) const
    {
        return work.AddMeanSourceJacobian(state, diagonal, storage, scale, evaluator, linearization);
    }
}
