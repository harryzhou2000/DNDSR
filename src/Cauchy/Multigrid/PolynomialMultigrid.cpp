#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Solver/FAS.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <tuple>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        /// Logical rows retain their parent pitch; only descriptor metadata is
        /// assembled here. No conservative values are flattened for multigrid.
        struct EquationFASOperations : EquationDofOperations
        {
            static index ComponentNumber(const EquationDofs &state) { return state.ComponentNumber(); }
            static void CheckFrozen(const EquationDofs &state, const EquationDofs &reference, const std::vector<index> &components)
            {
                const Algebra::ComponentSelection selection(state.ComponentNumber(), components);
                const auto read = state.ReadOwned(), savedRead = reference.ReadOwned();
                const auto layout = read.ComponentLayout(), savedLayout = savedRead.ComponentLayout();
                const auto values = layout.View(), saved = savedLayout.View();
                for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                    for (index j = 0; j < values.ComponentNumber(); ++j)
                        if (!selection.Includes(j))
                            DNDS_check_throw_info(std::memcmp(&values(iRow, j), &saved(iRow, j), sizeof(real)) == 0,
                                                  "FAS callback changed a frozen equation component");
            }
            static void ProjectCorrection(EquationDofs &correction, const std::vector<index> &components)
            {
                const Algebra::ComponentSelection selection(correction.ComponentNumber(), components);
                auto write = correction.WriteOwned();
                const auto layout = write.ComponentLayout();
                const auto values = layout.View();
                for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                    for (index j = 0; j < values.ComponentNumber(); ++j)
                        if (!selection.Includes(j))
                            values(iRow, j) = 0;
            }
            static void AddSelected(EquationDofs &state, const EquationDofs &correction, const std::vector<index> &components)
            {
                if (components.empty())
                {
                    Axpy(state, 1, correction);
                    return;
                }
                const auto read = correction.ReadOwned();
                auto write = state.WriteOwned();
                const auto deltaLayout = read.ComponentLayout();
                const auto stateLayout = write.ComponentLayout();
                const auto delta = deltaLayout.View();
                const auto values = stateLayout.View();
                for (index iRow = 0; iRow < values.RowCount(); ++iRow)
                    for (const auto j : components)
                        values(iRow, j) += delta(iRow, j);
            }
        };

        struct FieldProfile
        {
            using State = CFV::FVField;
            using Operations = Multigrid::FieldFASOperations;
            using Solver = NonlinearSolver;
            using Workspace = FiniteVolume::Workspace;
            using Report = NonlinearIterationReport;
            static FlowState Norms(const State &) { return {}; }
            static State NewState(const FiniteVolume &fv, const State &prototype)
            {
                return fv.Metrics().BuildField(prototype.Owned().Layout());
            }
            static auto BuildWorkspace(const FiniteVolume &fv, State &state, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &)
            {
                AuxiliaryRequest channels;
                channels.faceSpectralRadius = channels.cellSpectralSum = true;
                return fv.BuildWorkspace(state, channels);
            }
            static auto BuildSolver(const FiniteVolume &fv, State &state, std::shared_ptr<Workspace> workspace,
                                    R::EagerEngine &engine, const R::ExecutionStream &stream, const DomainSettings &settings,
                                    const std::optional<Physics::Chemistry::ChemicalWorkerPool> &)
            {
                return std::make_unique<Solver>(fv, state, std::move(workspace), engine, stream, settings);
            }
            static void CheckTransfer(const State &a, const State &b)
            {
                a.Validate();
                b.Validate();
                DNDS_check_throw_info(a.Space().OwnedRows() == b.Space().OwnedRows() && a.Width() == b.Width(),
                                      "polynomial FAS identity transfer shape mismatch");
            }
            template <class F>
            static void ReadPair(const State &a, const State &b, F &&run)
            {
                const auto aRead = a.Owned().Read(), bRead = b.Owned().Read();
                run(aRead.View<R::UniformTableRows>(), bRead.View<R::UniformTableRows>());
            }
            template <class F>
            static void TransferRows(const State &input, State &output, F &&run)
            {
                const auto read = input.Owned().Read();
                auto write = output.Owned().Write();
                run(read.View<R::UniformTableRows>(), write.View<R::UniformTableRows>());
            }
            static void SetBoundaryInputs(const FiniteVolume &fv, Workspace &workspace, const BoundaryInputs &input)
            {
                fv.SetBoundaryInputs(workspace, input);
            }
        };

        struct EquationProfile
        {
            using State = EquationDofs;
            using Operations = EquationFASOperations;
            using Solver = EquationNonlinearSolver;
            using Workspace = FiniteVolume::EquationWorkspace;
            using Report = EquationNonlinearIterationReport;
            static std::vector<real> Norms(const State &state) { return std::vector<real>(size_t(state.ComponentNumber())); }
            static State NewState(const FiniteVolume &fv, const State &prototype)
            {
                Algebra::DofBundle fields(fv.Metrics().Space());
                for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
                    fields.Add(prototype.Fields().Name(iField), fv.Metrics().BuildField(prototype.PhysicalField(iField).Owned().Layout()));
                std::vector<EquationDofBinding> bindings;
                for (index iBlock = 0; iBlock < prototype.BlockNumber(); ++iBlock)
                {
                    const auto location = prototype.LocateBlock(iBlock);
                    bindings.push_back({prototype.Schema().blocks[iBlock].name, prototype.Fields().Name(location.physicalField), location.firstComponent});
                }
                // The new operator owns another DofSpace with the same cell
                // order. Construct that association explicitly, never relax
                // EquationDofs' ordinary same-space rebinding contract.
                return {prototype.Schema(), std::move(fields), std::move(bindings)};
            }
            static auto BuildWorkspace(const FiniteVolume &fv, State &state, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers)
            {
                return fv.BuildEquationWorkspace(state, workers);
            }
            static auto BuildSolver(const FiniteVolume &fv, State &state, std::shared_ptr<Workspace> workspace,
                                    R::EagerEngine &engine, const R::ExecutionStream &stream, const DomainSettings &settings,
                                    const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers)
            {
                const Algebra::ComponentSelection selection(state.ComponentNumber(), state.Schema().ResolveComponents(settings.implicit.residualComponents));
                return std::make_unique<Solver>(fv, state, std::move(workspace), engine, stream, settings, selection, workers);
            }
            static void CheckTransfer(const State &a, const State &b)
            {
                a.Validate();
                b.Validate();
                a.RequireSchema(b.Schema());
                DNDS_check_throw_info(a.Space().OwnedRows() == b.Space().OwnedRows(), "polynomial FAS equation identity transfer shape mismatch");
            }
            template <class F>
            static void ReadPair(const State &a, const State &b, F &&run)
            {
                const auto aRead = a.ReadOwned(), bRead = b.ReadOwned();
                const auto aLayout = aRead.ComponentLayout(), bLayout = bRead.ComponentLayout();
                run(aLayout.View(), bLayout.View());
            }
            template <class F>
            static void TransferRows(const State &input, State &output, F &&run)
            {
                const auto read = input.ReadOwned();
                auto write = output.WriteOwned();
                const auto source = read.ComponentLayout();
                const auto destination = write.ComponentLayout();
                run(source.View(), destination.View());
            }
            static void SetBoundaryInputs(const FiniteVolume &, Workspace &, const BoundaryInputs &)
            {
                DNDS_check_throw_info(false, "extended equation interface packets are not connected yet");
            }
        };

        /// One p-hierarchy implementation for both physical-state adapters.
        /// Physics, source quadrature and PP remain in the level stage solvers;
        /// time assembly and finest convergence remain outside the V-cycle.
        template <class Profile>
        struct PolynomialHierarchy
        {
            using State = typename Profile::State;
            using Operations = typename Profile::Operations;
            using Solver = typename Profile::Solver;
            using Report = typename Profile::Report;
            using Level = Multigrid::StateFASLevel<State>;
            using Transfer = Multigrid::StateFASTransfer<State>;
            using Cycle = Multigrid::StateFASCycle<State, Operations>;
            static void IdentityTransfer(const State &input, State &output, const MPIInfo &mpi, bool collectiveChecks = true)
            {
                // Only construction-validated equal cell orders may use this
                // explicit cross-space copy. Ordinary algebra stays strict.
                const auto run = [&]
                {
                    Profile::CheckTransfer(input, output);
                    Operations::RequireDistinct(input, output);
                    const index width = Operations::ComponentNumber(input);
                    Profile::TransferRows(input, output, [width](auto source, auto destination)
                                          {
                        for (index iCell = 0; iCell < source.RowCount(); ++iCell)
                            for (index j = 0; j < width; ++j) destination(iCell, j) = source(iCell, j); });
                };
                if (collectiveChecks)
                    MPI::LocalStage(mpi, run);
                else
                    run();
            }
            struct Coarse
            {
                FiniteVolume discrete;
                State state;
                std::shared_ptr<typename Profile::Workspace> workspace;
                std::unique_ptr<Solver> solver;
                Coarse(ssp<Geom::UnstructuredMesh> mesh, DomainSettings settings, const FiniteVolume &fine,
                       const State &prototype, R::EagerEngine &engine, const R::ExecutionStream &stream,
                       const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers)
                    : discrete(std::move(mesh), settings, fine.Equation()), state(Profile::NewState(discrete, prototype))
                {
                    DNDS_check_throw_info(discrete.Metrics().Cells().SameOrder(fine.Metrics().Cells()),
                                          "polynomial FAS levels changed cell identity/order");
                    // A level solver validates its physical background at
                    // construction. Restriction has not run yet: seed the
                    // owned state explicitly instead of exposing zero means.
                    IdentityTransfer(prototype, state, MPIInfo(discrete.Metrics().Communicator()));
                    workspace = Profile::BuildWorkspace(discrete, state, workers);
                    solver = Profile::BuildSolver(discrete, state, workspace, engine, stream, settings, workers);
                }
            };
            FiniteVolume fine;
            State state, value;
            Solver *fineSolver;
            ImplicitSettings settings;
            MPIInfo mpi;
            std::vector<std::unique_ptr<Coarse>> coarse;
            std::unique_ptr<Cycle> cycle;
            std::vector<Level> levelDefinitions;
            std::vector<Transfer> transferDefinitions;
            std::vector<real> smootherCFL;
            real mass = 0, rhsScale = 1, time = 0;
            int stage = 0;
            Report *report = nullptr;
            PolynomialHierarchy(const FiniteVolume &discrete, State &input, Solver &solver, const DomainSettings &controls)
                : fine(discrete), state(input), value(Operations::NewVector(input)), fineSolver(&solver),
                  settings(controls.implicit), mpi(discrete.Metrics().Communicator())
            {
                settings.ValidateComponents(Operations::ComponentNumber(input));
                DNDS_check_throw_info(settings.multigrid.levels > 0 && controls.numerics.reconstruction == "variational" &&
                                          controls.numerics.cfv.maxOrder >= 2,
                                      "polynomial FAS requires high-order variational reconstruction");
            }
            Level MakeLevel(State input, Solver &solver, index level)
            {
                auto *relax = &solver;
                const auto smooth = [this, relax, level](auto &u, const auto &source, index count, const std::vector<index> &components)
                {
                    for (index correction = 0; correction < count; ++correction)
                    {
                        const auto step = relax->CorrectOnce(u, source, mass, rhsScale, time, stage, {1, smootherCFL[level], nullptr, components});
                        NonlinearDetail::AccumulateWork(*report, step);
                    }
                    return count;
                };
                return {std::move(input),
                        [this, relax](auto &u, auto &out)
                        { relax->EvaluateEquation(u, out, mass, rhsScale, time); },
                        [smooth](auto &u, const auto &source, index count)
                        { return smooth(u, source, count, {}); },
                        [this, relax](auto &u, const auto &delta)
                        {
                            const auto [limited, alpha] = relax->ApplyCorrection(u, delta, settings.multigrid.damping, settings.multigrid.updateComponents);
                            report->incrementLimited += limited;
                            report->minimumIncrementAlpha = std::min(report->minimumIncrementAlpha, alpha);
                        },
                        smooth,
                        [this, relax](auto &u, auto &out)
                        { relax->EvaluateEquation(u, out, 0, rhsScale, time); }};
            }
            void BuildCycle()
            {
                const auto &control = settings.multigrid;
                cycle = std::make_unique<Cycle>(levelDefinitions, transferDefinitions,
                                                Multigrid::FASCycleSettings{control.preSmooth, control.postSmooth, control.coarseSmooth, control.updateComponents},
                                                [this](const auto &run)
                                                {
                                                    if (settings.collectiveChecks)
                                                        MPI::LocalStage(mpi, run);
                                                    else
                                                        run();
                                                });
            }
            void Build(const ssp<Geom::UnstructuredMesh> &mesh, R::EagerEngine &engine, const R::ExecutionStream &stream,
                       const DomainSettings &controls, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers = std::nullopt)
            {
                smootherCFL.assign(size_t(settings.multigrid.levels + 1), settings.pseudoCFL);
                levelDefinitions.push_back(MakeLevel(state, *fineSolver, 0));
                const auto identity = [mpi = mpi, checks = settings.collectiveChecks](const State &input, State &output)
                {
                    IdentityTransfer(input, output, mpi, checks);
                };
                for (int level = 0; level < settings.multigrid.levels; ++level)
                {
                    auto lower = PolynomialMultigrid::CoarseSettings(controls, level);
                    coarse.push_back(std::make_unique<Coarse>(mesh, lower, fine, state, engine, stream, workers));
                    levelDefinitions.push_back(MakeLevel(coarse.back()->state, *coarse.back()->solver, level + 1));
                    transferDefinitions.push_back({identity, identity, identity, true});
                }
                BuildCycle();
            }
            void BindEquation(const State &source, real equationMass, real equationScale, real equationTime, int equationStage, real pseudoCFL)
            {
                MPI::LocalStage(mpi, [&]
                                {
                    Operations::CheckCompatible(state, source);
                    Operations::RequireDistinct(state, source);
                    DNDS_check_throw_info(std::isfinite(equationMass) && equationMass >= 0 && std::isfinite(equationScale) && equationScale > 0 &&
                                              std::isfinite(equationTime) && (pseudoCFL == 0 || (std::isfinite(pseudoCFL) &&
                                              pseudoCFL >= settings.pseudoCFL && pseudoCFL <= settings.maxPseudoCFL)), "invalid polynomial FAS equation coefficients"); });
                std::array<real, 4> minimum{equationMass, equationScale, equationTime, pseudoCFL}, maximum = minimum;
                MPI_Allreduce(MPI_IN_PLACE, minimum.data(), 4, MPI_DOUBLE, MPI_MIN, mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, maximum.data(), 4, MPI_DOUBLE, MPI_MAX, mpi.comm);
                DNDS_check_throw_info(minimum == maximum, "polynomial FAS coefficients differ between ranks");
                mass = equationMass;
                rhsScale = equationScale;
                time = equationTime;
                stage = equationStage;
            }
            auto Measure(const State &source)
            {
                fineSolver->EvaluateEquation(state, value, mass, rhsScale, time);
                auto norms = Profile::Norms(state);
                std::vector<real> sum(norms.size() + 1, 0);
                MPI::LocalStage(mpi, [&]
                                {
                    const auto geometry = fine.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                    Profile::ReadPair(source, value, [&](auto b, auto f)
                    {
                        for (index iCell = 0; iCell < f.RowCount(); ++iCell)
                        {
                            const real volume = metric.GetCellVol(iCell); sum.back() += volume;
                            for (index j = 0; j < index(norms.size()); ++j)
                            {
                                const real difference = b(iCell, j) - f(iCell, j);
                                sum[j] += settings.normOrder == 1 ? volume * std::abs(difference) : volume * difference * difference;
                            }
                        }
                    }); });
                MPI_Allreduce(MPI_IN_PLACE, sum.data(), int(sum.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
                DNDS_check_throw_info(sum.back() > 0 && std::isfinite(sum.back()), "invalid FAS global volume");
                for (index j = 0; j < index(norms.size()); ++j)
                {
                    norms[j] = settings.normOrder == 1 ? sum[j] / sum.back() : std::sqrt(sum[j] / sum.back());
                    DNDS_check_throw_info(std::isfinite(norms[j]), "nonfinite FAS finest residual");
                }
                return norms;
            }
            Report ApplyCycle(const State &source, real mass, real rhsScale, real time, int stage, real pseudoCFL)
            {
                BindEquation(source, mass, rhsScale, time, stage, pseudoCFL);
                if (!cycle)
                    BuildCycle();
                const real cfl = pseudoCFL == 0 ? settings.pseudoCFL : pseudoCFL;
                std::fill(smootherCFL.begin(), smootherCFL.end(), cfl);
                Report work;
                work.residual = Profile::Norms(state);
                work.pseudoCFL = cfl;
                report = &work;
                try
                {
                    const auto counts = cycle->Apply(source);
                    work.multigridCycles = work.iterations = 1;
                    work.levelCorrections = counts.corrections;
                }
                catch (...)
                {
                    cycle.reset();
                    report = nullptr;
                    throw;
                }
                report = nullptr;
                return work;
            }
            Report Solve(const State &source, real mass, real rhsScale, real time, int stage)
            {
                BindEquation(source, mass, rhsScale, time, stage, 0);
                auto control = settings;
                control.iterations = settings.multigrid.cycles;
                auto residual = Profile::Norms(state);
                Report initial;
                initial.residual = residual;
                return NonlinearDetail::Iterate(control, fine.Equation().Schema(), mpi.rank, stage, {}, std::move(initial), fineSolver->ReconstructionTolerance(), [&]
                                                { residual = Measure(source); }, [&](auto &norms)
                                                { norms = residual; }, [&]
                                                { return fineSolver->ReconstructionStats(); }, [&](auto &total)
                                                { NonlinearDetail::AccumulateWork(total, ApplyCycle(source, mass, rhsScale, time, stage, total.pseudoCFL)); });
            }
            void SetBoundaryInputs(const BoundaryInputs &input)
            {
                for (auto &lower : coarse)
                    Profile::SetBoundaryInputs(lower->discrete, *lower->workspace, input);
            }
        };
    }
    struct PolynomialMultigrid::Data : PolynomialHierarchy<FieldProfile>
    {
        using PolynomialHierarchy::PolynomialHierarchy;
    };
    struct PolynomialMultigrid::EquationData : PolynomialHierarchy<EquationProfile>
    {
        using PolynomialHierarchy::PolynomialHierarchy;
    };

    DomainSettings PolynomialMultigrid::CoarseSettings(const DomainSettings &fine, index level)
    {
        DNDS_check_throw_info(level >= 0, "invalid polynomial coarse level");
        auto lower = fine;
        lower.implicit.multigrid.levels = 0;
        lower.numerics.reconstruction = "gradient";
        lower.numerics.firstOrderConvection = level > 0;
        lower.numerics.cfv.maxOrder = 1;
        if (lower.numerics.limiter == "LVR")
            lower.numerics.limiter = "WBAP";
        if (level)
            lower.numerics.limiter = "none";
        return lower;
    }

    PolynomialMultigrid::PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, CFV::FVField &state,
                                             NonlinearSolver &solver, R::EagerEngine &engine, R::ExecutionStream stream, const DomainSettings &settings)
    {
        MPI::LocalStage(MPIInfo(fine.Metrics().Communicator()), [&]
                        { _data = std::make_shared<Data>(fine, state, solver, settings); });
        _data->Build(mesh, engine, stream, settings);
    }
    PolynomialMultigrid::PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, EquationDofs &state,
                                             EquationNonlinearSolver &solver, R::EagerEngine &engine, R::ExecutionStream stream, const DomainSettings &settings,
                                             std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
    {
        MPI::LocalStage(MPIInfo(fine.Metrics().Communicator()), [&]
                        { _equationData = std::make_shared<EquationData>(fine, state, solver, settings); });
        _equationData->Build(mesh, engine, stream, settings, workers);
    }
    NonlinearIterationReport PolynomialMultigrid::Solve(const CFV::FVField &source, real mass, real rhsScale, real time, int stage)
    {
        DNDS_check_throw_info(_data, "field pMG call requires a field hierarchy");
        return _data->Solve(source, mass, rhsScale, time, stage);
    }
    EquationNonlinearIterationReport PolynomialMultigrid::Solve(const EquationDofs &source, real mass, real rhsScale, real time, int stage)
    {
        DNDS_check_throw_info(_equationData, "logical pMG call requires an equation hierarchy");
        return _equationData->Solve(source, mass, rhsScale, time, stage);
    }
    NonlinearIterationReport PolynomialMultigrid::ApplyCycle(const CFV::FVField &source, real mass, real rhsScale, real time, int stage, real pseudoCFL)
    {
        DNDS_check_throw_info(_data, "field pMG call requires a field hierarchy");
        return _data->ApplyCycle(source, mass, rhsScale, time, stage, pseudoCFL);
    }
    EquationNonlinearIterationReport PolynomialMultigrid::ApplyCycle(const EquationDofs &source, real mass, real rhsScale, real time, int stage, real pseudoCFL)
    {
        DNDS_check_throw_info(_equationData, "logical pMG call requires an equation hierarchy");
        return _equationData->ApplyCycle(source, mass, rhsScale, time, stage, pseudoCFL);
    }
    void PolynomialMultigrid::SetBoundaryInputs(const BoundaryInputs &input)
    {
        if (_data)
            _data->SetBoundaryInputs(input);
        else
            _equationData->SetBoundaryInputs(input);
    }
}
