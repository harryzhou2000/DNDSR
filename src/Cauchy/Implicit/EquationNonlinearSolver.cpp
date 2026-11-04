#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct EquationNonlinearSolver::Data
    {
        FiniteVolume discrete;
        std::shared_ptr<FiniteVolume::EquationWorkspace> workspace;
        R::EagerEngine *engine;
        R::ExecutionStream stream;
        ImplicitSettings settings;
        MPIInfo mpi;
        EquationDofOperations::Binding binding;
        EquationDofs spatial, defect, candidate;
        R::Array<real> inversePseudoTime;
        AuxiliaryValues auxiliary;
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        std::shared_ptr<LinearProblemContext> problem;
        std::unique_ptr<EquationLinearSolver> linear;
        real reconstructionTolerance, densityFloor, temperatureFloor, pressureFloor;
        index flowFirst = -1, speciesFirst = -1;

        Data(const FiniteVolume &fv, EquationDofs &state, std::shared_ptr<FiniteVolume::EquationWorkspace> work,
             R::EagerEngine &executor, R::ExecutionStream lane, const DomainSettings &controls, ImplicitSettings implicit,
             std::optional<Physics::Chemistry::ChemicalWorkerPool> pool)
            : discrete(fv), workspace(std::move(work)), engine(&executor), stream(std::move(lane)), settings(std::move(implicit)),
              mpi(fv.Metrics().Communicator()), binding(state), spatial(EquationDofOperations::NewVector(state)),
              defect(EquationDofOperations::NewVector(state)), candidate(EquationDofOperations::NewVector(state)),
              inversePseudoTime(R::RowLayout::Dense(fv.Metrics().NumCell(), 1)), workers(std::move(pool)),
              problem(fv.BuildLinearProblem(state, settings.diagonal == "block" ? DiagonalStorage::Block : DiagonalStorage::Entries,
                                            settings.jacobianAction == "semiMatrixFree" ? OffDiagonalStorage::SemiMatrixFree : OffDiagonalStorage::Assembled)),
              reconstructionTolerance(settings.requireConvergedReconstruction && controls.numerics.reconstruction == "variational" ? controls.numerics.reconstructionTolerance : 0),
              densityFloor(controls.numerics.densityFloor), temperatureFloor(controls.numerics.temperatureFloor), pressureFloor(controls.numerics.pressureFloor)
        {
            AuxiliaryRequest request;
            request.faceSpectralRadius = request.cellSpectralSum = true;
            auxiliary = fv.BuildAuxiliaryValues(request);
            index first = 0;
            for (const auto &block : state.Schema().blocks)
            {
                if (block.name == "flow")
                    flowFirst = first;
                if (block.name == "species")
                    speciesFirst = first;
                first += index(block.components.size());
            }
        }
        template <class F>
        void Stage(F &&run) const
        {
            if (settings.collectiveChecks)
                MPI::LocalStage(mpi, std::forward<F>(run));
            else
                run();
        }
        void PrepareLinear(const EquationDofs &state, bool overlap)
        {
            linear = std::make_unique<EquationLinearSolver>(problem, state, [&](auto &field)
                                                            { return discrete.Metrics().BindHalo(field); }, NonlinearDetail::ResolveLinear(settings), LinearExecution{overlap, settings.collectiveChecks});
        }
        void ValidateMeans(const EquationDofs &state) const
        {
            const auto &equation = discrete.Equation();
            auto evaluator = equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto read = state.ReadOwned();
            const auto layout = read.ComponentLayout();
            const auto values = layout.View();
            const int independent = equation.HasMixture() ? equation.Mixture().SpeciesNumber() - 1 : 0;
            auto rhoY = evaluator.SpeciesInput();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                for (index j = 0; j < values.ComponentNumber(); ++j)
                    DNDS_check_throw_info(std::isfinite(values(iCell, j)), "nonfinite implicit equation mean");
                FlowState flow;
                for (int j = 0; j < 5; ++j)
                    flow[j] = values(iCell, flowFirst + j);
                DNDS_check_throw_info(flow[0] >= densityFloor, "implicit equation density is below its floor");
                real sum = 0;
                for (int k = 0; k < independent; ++k)
                {
                    rhoY[k] = values(iCell, speciesFirst + k);
                    DNDS_check_throw_info(rhoY[k] >= 0, "negative implicit equation species mean");
                    sum += rhoY[k];
                }
                DNDS_check_throw_info(sum <= flow[0], "implicit equation species means exceed density");
                if (!equation.HasMixture())
                    DNDS_check_throw_info(ThermallyAdmissible(flow, discrete.ThermalConstraints()), "inadmissible implicit flow mean");
                const auto point = evaluator.Evaluate(flow, {independent ? rhoY.data : nullptr, independent});
                DNDS_check_throw_info(point.thermo.temperature >= temperatureFloor && point.thermo.pressure >= pressureFloor,
                                      "implicit equation mean violates temperature or pressure floor");
            }
        }
    };

    EquationNonlinearSolver::EquationNonlinearSolver(const FiniteVolume &discrete, EquationDofs &state,
                                                     std::shared_ptr<FiniteVolume::EquationWorkspace> workspace, R::EagerEngine &engine, R::ExecutionStream stream,
                                                     const DomainSettings &settings, const Algebra::ComponentSelection &residualSelection,
                                                     std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
    {
        const MPIInfo mpi(discrete.Metrics().Communicator());
        ImplicitSettings implicit = settings.implicit;
        std::string serialized;
        MPI::LocalStage(mpi, [&]
                        {
            implicit.residualComponents.clear();
            for (index j = 0; j < state.ComponentNumber(); ++j)
                if (residualSelection.Includes(j)) implicit.residualComponents.push_back(int(j));
            DNDS_check_throw_info(residualSelection.All() || residualSelection.Indices().back() < state.ComponentNumber(), "equation residual selection exceeds its dimension");
            implicit.ValidateComponents(state.ComponentNumber());
            settings.numerics.ValidatePositivity();
            DNDS_check_throw_info(!settings.numerics.positivity && !implicit.roeDissipation && implicit.multigrid.levels == 0,
                                  "equation nonlinear PP/Roe/pMG profile is not connected yet");
            DNDS_check_throw_info(!implicit.requireConvergedReconstruction || settings.numerics.reconstruction != "variational" || settings.numerics.reconstructionTolerance > 0,
                                  "strict implicit VR acceptance requires a positive reconstructionTolerance");
            DNDS_check_throw_info(workspace && stream.Context().device == R::Device::Host(), "equation implicit stage needs a CPU workspace and stream");
            stream.Record().WaitHost();
            discrete.Equation().ValidateStateSchema(state);
            discrete.ValidateWorkspace(state, *workspace);
            DNDS_check_throw_info(state.Space().Same(discrete.Metrics().Space()) && discrete.Equation().HasMixture() == bool(workers), "equation nonlinear space/worker backend differs");
            serialized = Json{{"implicit", implicit}, {"numerics", settings.numerics}, {"execution", settings.execution}}.dump();
            DNDS_check_throw_info(serialized.size() <= size_t(INT_MAX), "equation implicit controls exceed MPI string range"); });
        AgreeEquationBinding(state, mpi.comm);
        NonlinearDetail::AgreeControls(mpi, serialized);
        MPI::LocalStage(mpi, [&]
                        { _data = std::make_shared<Data>(discrete, state, std::move(workspace), engine, std::move(stream), settings, std::move(implicit), std::move(workers)); });
        MPI::LocalStage(mpi, [&]
                        { _data->ValidateMeans(state); });
        _data->PrepareLinear(state, settings.execution.overlap);
    }

    EquationNonlinearIterationReport EquationNonlinearSolver::Solve(EquationDofs &state, const EquationDofs &source,
                                                                    real mass, real rhsScale, real time, int stage, NonlinearSolveControl control)
    {
        auto &data = *_data;
        if (control.profile)
            *control.profile = {};
        const NonlinearDetail::PhaseTimer total(control.profile, NonlinearProfile::Total);
        const index nv = data.binding.snapshot.ComponentNumber();
        Algebra::ComponentSelection selection(nv);
        {
            const NonlinearDetail::PhaseTimer entry(control.profile, NonlinearProfile::Entry);
            std::string serialized;
            MPI::LocalStage(data.mpi, [&]
                            {
                data.stream.Record().WaitHost();
                data.binding.Check(state);
                data.ValidateMeans(state);
                EquationDofOperations::CheckCompatible(state, source);
                EquationDofOperations::RequireDistinct(state, source);
                selection = Algebra::ComponentSelection(nv, control.updateComponents);
                DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time) &&
                    control.correctionLimit >= -1 && control.correctionLimit <= data.settings.iterations &&
                    (control.initialPseudoCFL == 0 || (std::isfinite(control.initialPseudoCFL) && control.initialPseudoCFL >= data.settings.pseudoCFL && control.initialPseudoCFL <= data.settings.maxPseudoCFL)),
                    "invalid equation implicit stage coefficients");
                serialized = Json{{"mass", mass}, {"rhsScale", rhsScale}, {"time", time}, {"limit", control.correctionLimit},
                    {"CFL", control.initialPseudoCFL}, {"selection", selection.Indices()}}.dump();
                DNDS_check_throw_info(serialized.size() <= size_t(INT_MAX), "equation solve controls exceed MPI string range"); });
            NonlinearDetail::AgreeControls(data.mpi, serialized);
        }
        EquationNonlinearIterationReport initial;
        initial.residual.resize(size_t(nv));
        auto geometry = data.discrete.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        return NonlinearDetail::Iterate(data.settings, data.mpi.rank, stage, control, std::move(initial), data.reconstructionTolerance, [&]
                                        { data.discrete.Evaluate(state, data.spatial, *data.workspace, *data.engine, data.stream, time); }, [&](auto &norms)
                                        {
                std::vector<real> local(size_t(nv + 1));
                data.Stage([&]
                {
                    const auto u = state.ReadOwned(), r = data.spatial.ReadOwned(), b = source.ReadOwned();
                    auto write = data.defect.WriteOwned();
                    const auto ul = u.ComponentLayout(), rl = r.ComponentLayout(), bl = b.ComponentLayout();
                    const auto wl = write.ComponentLayout();
                    const auto mean = ul.View(), residual = rl.View(), constant = bl.View();
                    const auto output = wl.View();
                    for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                    {
                        const real volume = metric.GetCellVol(iCell); local[nv] += volume;
                        for (index j = 0; j < nv; ++j)
                        {
                            const real value = rhsScale * residual(iCell, j) + volume * (constant(iCell, j) - mass * mean(iCell, j));
                            DNDS_check_throw_info(std::isfinite(value), "nonfinite equation implicit defect");
                            output(iCell, j) = selection.Includes(j) ? value : 0;
                            local[j] += data.settings.normOrder == 1 ? std::abs(value) : value * value / volume;
                        }
                    }
                });
                MPI_Allreduce(MPI_IN_PLACE, local.data(), int(nv + 1), MPI_DOUBLE, MPI_SUM, data.mpi.comm);
                DNDS_check_throw_info(std::isfinite(local[nv]) && local[nv] > 0, "invalid global equation stage volume");
                for (index j = 0; j < nv; ++j)
                {
                    norms[j] = data.settings.normOrder == 1 ? local[j] / local[nv] : std::sqrt(local[j] / local[nv]);
                    DNDS_check_throw_info(std::isfinite(norms[j]), "nonfinite equation global stage norm");
                } }, [&]
                                        { return ReconstructionStats(); }, [&](auto &report)
                                        {
                {
                    const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::PseudoTimePP);
                    data.Stage([&]
                    {
                        data.discrete.PrepareAuxValues(state, data.auxiliary, time, data.workers);
                        const auto read = data.auxiliary.cellSpectralSum->Owned().Read();
                        auto write = data.inversePseudoTime.Write();
                        const auto spectral = read.View<R::UniformTableRows>();
                        const auto inverse = write.View<R::UniformTableRows>();
                        for (index iCell = 0; iCell < inverse.RowCount(); ++iCell)
                        {
                            const real value = spectral(iCell, 0) / (report.pseudoCFL * metric.GetCellVol(iCell));
                            DNDS_check_throw_info(std::isfinite(value) && value > 0, "invalid equation pseudo-time spectral sum");
                            inverse(iCell, 0) = value;
                        }
                    });
                }
                {
                    const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::Jacobian);
                    data.Stage([&] { data.discrete.PrepareJacobian(state, data.auxiliary, data.inversePseudoTime, *data.problem,
                        {mass, rhsScale, false, selection.Indices()}, time, data.workers, data.workspace.get()); });
                }
                {
                    const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::LinearSolve);
                    report.lastLinear = data.linear->Solve(data.defect);
                    report.linearIterations += report.lastLinear.iterations;
                    DNDS_check_throw_info(report.lastLinear.usable, "equation implicit linear correction is unusable");
                }
                const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::Correction);
                ApplyCorrection(state, data.linear->Solution(), data.settings.relaxation, selection.Indices()); });
    }
    std::pair<int, real> EquationNonlinearSolver::ReconstructionStats() const
    {
        return {_data->discrete.LastReconstructionIterations(*_data->workspace), _data->discrete.LastReconstructionUpdate(*_data->workspace)};
    }
    bool EquationNonlinearSolver::ReconstructionConverged() const
    {
        return _data->reconstructionTolerance <= 0 || ReconstructionStats().second <= _data->reconstructionTolerance;
    }
    std::pair<index, real> EquationNonlinearSolver::ApplyCorrection(EquationDofs &state, const EquationDofs &increment, real scale,
                                                                    const std::vector<index> &updateComponents)
    {
        auto &data = *_data;
        data.Stage([&]
                   {
            data.binding.Check(state);
            EquationDofOperations::CheckCompatible(state, increment);
            EquationDofOperations::RequireDistinct(state, increment);
            DNDS_check_throw_info(std::isfinite(scale) && scale > 0 && scale <= 1, "invalid equation correction scale");
            const Algebra::ComponentSelection selection(state.ComponentNumber(), updateComponents);
            const auto u = state.ReadOwned(), d = increment.ReadOwned();
            auto write = data.candidate.WriteOwned();
            const auto ul = u.ComponentLayout(), dl = d.ComponentLayout();
            const auto wl = write.ComponentLayout();
            const auto mean = ul.View(), delta = dl.View();
            const auto candidate = wl.View();
            for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                for (index j = 0; j < mean.ComponentNumber(); ++j)
                    candidate(iCell, j) = selection.Includes(j) ? mean(iCell, j) + scale * delta(iCell, j) : mean(iCell, j); });
        // Validate globally before acquiring any input-state writer. This is
        // admissibility checking, not a new line search or rollback policy.
        MPI::LocalStage(data.mpi, [&]
                        { data.ValidateMeans(data.candidate); });
        data.Stage([&]
                   { EquationDofOperations::Copy(state, data.candidate); });
        return {0, 1};
    }
    void EquationNonlinearSolver::EvaluateEquation(EquationDofs &state, EquationDofs &value, real mass, real rhsScale, real time)
    {
        auto &data = *_data;
        MPI::LocalStage(data.mpi, [&]
                        {
            data.binding.Check(state);
            EquationDofOperations::CheckCompatible(state, value);
            EquationDofOperations::RequireDistinct(state, value);
            DNDS_check_throw_info(std::isfinite(mass) && mass >= 0 && std::isfinite(rhsScale) && rhsScale > 0 && std::isfinite(time), "invalid equation coefficients");
            auto write = value.WriteOwned(); write.CancelUnmodified(); });
        data.discrete.Evaluate(state, data.spatial, *data.workspace, *data.engine, data.stream, time);
        data.Stage([&]
                   {
            const auto geometry = data.discrete.Metrics().AcquireRead(); const auto metric = geometry.HostView();
            const auto u = state.ReadOwned(), r = data.spatial.ReadOwned(); auto write = value.WriteOwned();
            const auto ul = u.ComponentLayout(), rl = r.ComponentLayout(); const auto wl = write.ComponentLayout();
            const auto mean = ul.View(), residual = rl.View(); const auto output = wl.View();
            for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
                for (index j = 0; j < mean.ComponentNumber(); ++j)
                    output(iCell, j) = mass * mean(iCell, j) - rhsScale * residual(iCell, j) / metric.GetCellVol(iCell); });
    }
}
