#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void Domain::SetTimeStage(int stage)
    {
        const bool coupled = _state.timeStepper && _state.timeStepper->Method() == "ditr";
        _state.trialEpoch.stage = coupled ? 0 : stage;
        _state.trialEpoch.coupledStage = coupled ? stage : 0;
    }
    NonlinearSolveControl Domain::ProgressControl() const
    {
        NonlinearSolveControl control;
        if (_progressObserver)
            control.progress = [observer = _progressObserver](const NonlinearProgress &progress)
            { observer(progress); };
        if (_quiescentObserver)
            control.quiescent = [observer = _quiescentObserver](const NonlinearSolveState &state)
            { observer(state); };
        return control;
    }
    Json Domain::UpdatePolicyMetadata() const
    {
        if (!_execution.nonlinearController)
            return {{"name", "explicit"},
                    {"coupled_stages", 0},
                    {"residual_observation", "none"},
                    {"method_stopping_residual", false}};
        return _execution.nonlinearController->Policy().Metadata();
    }
    void Domain::InitializeImplicit(const DomainSettings &settings, const TimeSettings &time, const ssp<Geom::UnstructuredMesh> &mesh)
    {
        _steady = time.steady;
        _coupledReconstructionTolerance = settings.implicit.requireConvergedReconstruction &&
                                                  settings.numerics.reconstruction == "variational"
                                              ? settings.numerics.reconstructionTolerance
                                              : 0;
        _rejectUnconvergedStages = settings.implicit.rejectUnconvergedStages;
        if (time.method == "ssprk3" && !_steady)
        {
            DNDS_check_throw_info(!settings.implicit.multigrid.levels, "polynomial FAS requires an implicit time method");
            return;
        }
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { time.ditr.Validate();
                          _implicitControls = time.method == "ditr" ? time.ditr.ResolveControls(settings.implicit) : settings.implicit;
                          _ditrComponents = _problem.equation.Schema().ResolveComponents(settings.implicit.residualComponents);
                          const bool ditr = time.method == "ditr";
                          _execution.nonlinearController = std::make_unique<NonlinearController>(
                              _implicitControls, _problem.equation.Schema(), mpi.rank,
                              _implicitControls.updatePolicy,
                              UpdatePolicyRequirements{
                                  ditr ? 2 : 1,
                                  ditr ? NonlinearDetail::ResidualObservation::AfterUpdate
                                       : NonlinearDetail::ResidualObservation::BeforeUpdate,
                                  ditr}); });
        _execution.coupledFactory = [this, mesh, settings, mpi](EquationDofs &middle)
        {
            const auto reconstruction = _problem.finiteVolume.BuildReconstructionState(middle);
            // Endpoint and midpoint solves are serialized by the coupled DITR
            // loop, so they reuse one allocation-independent scratch workspace.
            // Their means, halos and durable reconstruction states stay distinct.
            const auto workspace = _execution.workspace;
            const auto solver = std::make_shared<NonlinearSolver>(
                _problem.finiteVolume, middle, reconstruction, workspace,
                _execution.Context(), settings,
                Algebra::ComponentSelection(middle.ComponentNumber(), _ditrComponents));
            std::shared_ptr<PolynomialMultigrid> multigrid;
            if (settings.implicit.multigrid.levels)
                multigrid = std::make_shared<PolynomialMultigrid>(
                    mesh, _problem.finiteVolume, middle, *solver, _execution.Context(), settings);
            return DomainCoupledStageOperator{
                [this, solver](auto &state, auto &rate, real time, int stage)
                { SetTimeStage(stage); solver->EvaluateRate(state, rate, time); },
                [this, solver, multigrid](const auto &equation, const auto &rate, real cfl)
                {
                    const auto &c = equation.coefficients;
                    _state.progressTime = c.time;
                    SetTimeStage(equation.coupledStage);
                    return multigrid ? multigrid->ApplyCycle(equation.resOther, c.inverseDt, c.rhsScale, c.time, equation.coupledStage, cfl, std::cref(rate))
                                     : solver->CorrectOnce(equation.state, equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                           equation.coupledStage, {1, cfl}, std::cref(rate));
                },
                [solver]
                { return solver->ReconstructionStats(); },
                reconstruction};
        };
        MPI::LocalStage(mpi, [&]
                        { _state.timeStepper = std::make_unique<EquationTimeStepper>(_state.trial, _steady ? "bdf1" : time.method, [mpi, checks = settings.implicit.collectiveChecks](const auto &run)
                                                                                     { if (checks) MPI::LocalStage(mpi, run); else run(); }, ODE::DITRSettings{time.ditr.fraction, time.ditr.theta, time.ditr.variant == "U2R1" ? ODE::DITRVariant::U2R1 : ODE::DITRVariant::U2R2}); });
        _execution.implicit = std::make_unique<NonlinearSolver>(
            _problem.finiteVolume, _state.trial, _state.reconstruction, _execution.workspace,
            _execution.Context(), settings,
            Algebra::ComponentSelection(_state.trial.ComponentNumber(), _ditrComponents));
        if (settings.implicit.multigrid.levels)
            _execution.multigrid = std::make_unique<PolynomialMultigrid>(
                mesh, _problem.finiteVolume, _state.trial, *_execution.implicit, _execution.Context(), settings);
    }
    void Domain::EvaluateImplicitRate(EquationDofs &state, EquationDofs &rate, FiniteVolume::Workspace &workspace, real time, int stage)
    {
        SetTimeStage(stage);
        DNDS_check_throw_info(_execution.implicit && _execution.workspace.get() == &workspace,
                              "implicit equation rate belongs to another workspace");
        _execution.implicit->EvaluateRate(state, rate, time);
    }
    void Domain::NormalizeRate(EquationDofs &rate)
    {
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            const auto geometry = _problem.finiteVolume.Metrics().AcquireRead(); const auto metrics = geometry.HostView();
            const auto read = _state.residual.ReadOwned(); auto write = rate.WriteOwned();
            for (index iBlock = 0; iBlock < rate.BlockNumber(); ++iBlock)
            {
                const auto input = read.Block(iBlock); const auto output = write.Block(iBlock);
                for (index iCell = 0; iCell < input.owned.RowCount(); ++iCell)
                    for (index j = 0; j < input.ComponentNumber(); ++j) output(iCell, j) = input(iCell, j) / metrics.GetCellVol(iCell);
            } });
    }
    bool Domain::RecordImplicitStage(NonlinearIterationReport report)
    {
        _lastAlphaCells += report.incrementLimited;
        _lastAlpha = std::min(_lastAlpha, report.minimumIncrementAlpha);
        const bool converged = report.converged;
        _lastImplicit.push_back(std::move(report));
        // A cap is an iteration-control decision, not a numerical failure.
        // Preserve the actual convergence flag even when Euler-style policy
        // accepts the final iterate. Invalid corrections/states still throw.
        return converged || !_rejectUnconvergedStages;
    }
    void Domain::PreserveReconstructionGuess()
    {
        if (!_problem.finiteVolume.PolynomialReconstruction())
            return;
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            if (_state.restoreReconstruction)
                _problem.finiteVolume.RestoreReconstructionGuess(*_state.reconstruction, *_state.reconstructionBackup,
                                               _state.limitedVRBackup
                                                   ? OptionalRef<const Runtime::Array<real>>(*_state.limitedVRBackup)
                                                   : OptionalRef<const Runtime::Array<real>>{});
            else
            {
                const auto &guess = _problem.finiteVolume.ReconstructionGuess(*_state.reconstruction);
                if (!_state.reconstructionBackup) _state.reconstructionBackup.emplace(EquationDofOperations::NewVector(guess));
                EquationDofOperations::Copy(*_state.reconstructionBackup, guess);
                if (const auto history = _problem.finiteVolume.LimitedVRHistory(*_state.reconstruction))
                {
                    if (!_state.limitedVRBackup) _state.limitedVRBackup.emplace(history->Layout());
                    _state.limitedVRBackup->CopyFrom(*history).Wait();
                }
                _state.restoreReconstruction = true;
            } });
    }
    bool Domain::SolveCoupledImplicit(EquationTimeStepper::DITR &method, EquationDofs &endpoint, bool resume)
    {
        const auto owner = method.IntermediateState().PhysicalField(0).Owned().SharedStorage();
        if (!_coupledMiddle || _execution.coupledMiddleOwner != owner)
        {
            _coupledMiddle = _execution.coupledFactory(method.IntermediateState());
            _execution.coupledMiddleOwner = owner;
        }
        if (!resume)
            _problem.finiteVolume.CopyReconstructionState(*_coupledMiddle->state, *_state.reconstruction);
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            if (!_execution.coupledResidual)
                _execution.coupledResidual.emplace(EquationDofOperations::NewVector(endpoint)); });
        const auto evaluate = [&](EquationDofs &state, EquationDofs &rate, real time, int stage)
        {
            if (stage == 0)
                _coupledMiddle->evaluate(state, rate, time, stage);
            else
                EvaluateImplicitRate(state, rate, *_execution.workspace, time, stage);
        };
        const auto &settings = _implicitControls;
        NonlinearIterationReport initial;
        initial.residual.resize(size_t(endpoint.ComponentNumber()));
        auto coupledControl = ProgressControl();
        coupledControl.state.Emplace(_state.coupledNonlinear);
        coupledControl.resume = resume;
        if (!resume)
        {
            ++_state.coupledNonlinear.attempt;
            _state.coupledNonlinear.convergenceComponents = settings.residualComponents;
        }
        else
            DNDS_check_throw_info(_state.coupledNonlinear.convergenceComponents == settings.residualComponents,
                                  "restored DITR convergence components differ");
        DNDS_check_throw_info(_execution.nonlinearController,
                              "DITR update policy was not resolved");
        const auto report = _execution.nonlinearController->Execute(
            0, {std::move(coupledControl), std::move(initial),
                _coupledReconstructionTolerance, [&]
                { method.AssembleStoppingResidual(endpoint, *_execution.coupledResidual); }, [&](auto &norms)
                {
            std::vector<real> sum(norms.size() + 1, 0);
            MPI::LocalStage(mpi, [&]
            {
                const auto geometry = _problem.finiteVolume.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                const auto read = _execution.coupledResidual->ReadOwned(); const auto layout = read.ComponentLayout(); const auto values = layout.View();
                for (index iCell = 0; iCell < values.RowCount(); ++iCell)
                {
                    const real volume = metric.GetCellVol(iCell); sum.back() += volume;
                    for (index j = 0; j < index(norms.size()); ++j)
                    {
                        const real value = values(iCell, j);
                        DNDS_check_throw_info(std::isfinite(value), "nonfinite DITR stopping defect");
                        sum[j] += volume * (settings.normOrder == 1 ? std::abs(value) : value * value);
                    }
                }
            });
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<real>{sum.data(), sum.size()}, mpi.comm);
            DNDS_check_throw_info(std::isfinite(sum.back()) && sum.back() > 0, "invalid DITR global volume");
            for (index j = 0; j < index(norms.size()); ++j)
                norms[j] = settings.normOrder == 1 ? sum[j] : std::sqrt(sum[j]); }, [&]
                {
            const auto a = _coupledMiddle->reconstruction();
            const auto b = _execution.implicit->ReconstructionStats();
            return std::make_pair(std::max(a.first, b.first), std::max(a.second, b.second)); }, [&](int coupledStage, auto &iteration)
                {
                SetTimeStage(coupledStage);
                const auto equation = method.PrepareCoupledStage(endpoint, coupledStage);
                const auto &c = equation.coefficients;
                _state.progressTime = c.time;
                const auto &rate = method.CurrentRate();
                NonlinearIterationReport work;
                if (coupledStage == 0)
                    work = _coupledMiddle->correct(equation, rate, iteration.pseudoCFL);
                else
                    work = _execution.multigrid ? _execution.multigrid->ApplyCycle(equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                               coupledStage, iteration.pseudoCFL, std::cref(rate))
                                      : _execution.implicit->CorrectOnce(endpoint, equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                               coupledStage, {1, iteration.pseudoCFL}, std::cref(rate));
                NonlinearDetail::AccumulateWork(iteration, work);
                method.CompleteCoupledStage(endpoint, evaluate); }});
        _lastCoupledIterations = report.iterations;
        _lastCoupledTargetRatio = report.residualTargetRatio;
        return RecordImplicitStage(report);
    }
    StepResult Domain::PrepareImplicitStep(real dt)
    {
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
        const auto evaluate = [&](EquationDofs &state, EquationDofs &rate, real time, int stage)
        {
            if (_state.timeStepper->Method() == "ditr" && stage == 0)
            {
                DNDS_check_throw_info(_coupledMiddle, "DITR intermediate operator is not bound");
                _coupledMiddle->evaluate(state, rate, time, stage);
            }
            else
                EvaluateImplicitRate(state, rate, *_execution.workspace, time, stage);
        };
        const auto solve = [&](EquationDofs &state, const EquationDofs &source, real mass, real scale, real time, int stage)
        {
            _state.progressTime = time;
            SetTimeStage(stage);
            const auto progress = ProgressControl();
            return RecordImplicitStage(_execution.multigrid ? _execution.multigrid->Solve(source, mass, scale, time, stage, progress)
                                                            : _execution.implicit->Solve(state, source, mass, scale, time, stage, progress));
        };
        const auto coupled = [&](EquationTimeStepper::DITR &method, EquationDofs &endpoint)
        { return SolveCoupledImplicit(method, endpoint, false); };
        const bool prepared = _steady ? _state.timeStepper->PrepareSteady(_state.initial, _state.trial, _state.time, solve)
                                      : _state.timeStepper->Prepare(_state.initial, _state.trial, _state.time, dt, evaluate, solve, coupled);
        if (!prepared)
        {
            // Failure-only diagnostics keep capped inner solves distinct from
            // accepted stages and make timestep retries auditable. No logging
            // or extra reduction is added to the successful numerical path.
            Json details{{"method", _state.timeStepper->Method()}, {"steady", _steady}, {"stage", _state.trialEpoch.stage}, {"dt", _steady ? 0 : dt}, {"coupled_iterations", _lastCoupledIterations}, {"coupled_target_ratio", _lastCoupledTargetRatio}};
            if (!_lastImplicit.empty())
            {
                const auto &last = _lastImplicit.back();
                details["last_solve_converged"] = last.converged;
                details["nonlinear_iterations"] = last.iterations;
                details["linear_iterations"] = last.linearIterations;
                details["pseudo_cfl"] = last.pseudoCFL;
                details["initial_residual"] = last.initialResidual;
                details["reference_residual"] = last.referenceResidual;
                details["residual_target_ratio"] = last.residualTargetRatio;
                details["last_residual"] = last.residual;
                details["vr_sweeps"] = last.reconstructionIterations;
                details["vr_relative_update"] = last.reconstructionRelativeUpdate;
            }
            return StepResult::Rejected("Cauchy implicit stage reached its correction cap; accepted state/history unchanged; " + details.dump());
        }
        if (!Admissible(_state.trial))
            return StepResult::Rejected("Cauchy implicit candidate is inadmissible");
        return {};
    }
    StepResult Domain::ResumeImplicitStep()
    {
        DNDS_check_throw_info(_state.internalResume && _state.timeStepper && !_state.timeStepper->HasPendingTrial(),
                              "Cauchy domain has no exact internal step to resume");
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
        const auto evaluate = [&](EquationDofs &state, EquationDofs &rate, real time, int stage)
        {
            if (_state.timeStepper->Method() == "ditr" && stage == 0)
            {
                DNDS_check_throw_info(_coupledMiddle, "restored DITR intermediate operator is not bound");
                _coupledMiddle->evaluate(state, rate, time, stage);
            }
            else
                EvaluateImplicitRate(state, rate, *_execution.workspace, time, stage);
        };
        bool prepared = false;
        if (_state.timeStepper->Method() == "ditr")
            prepared = _state.timeStepper->ResumeCoupled(_state.trial, [&](auto &method, auto &endpoint)
                                                         { return SolveCoupledImplicit(method, endpoint, true); });
        else
        {
            const auto restored = _state.timeStepper->RestoredStageEquation();
            // The method owns the frozen source. Use its validated prepared
            // ticket instead of reconstructing resOther in Domain.
            const auto resumeSolve = [&](EquationDofs &state, const EquationDofs &source, real mass, real scale, real time, int stage)
            {
                _state.progressTime = time;
                SetTimeStage(stage);
                auto control = ProgressControl();
                control.resume = true;
                return RecordImplicitStage(_execution.multigrid ? _execution.multigrid->Solve(source, mass, scale, time, stage, control)
                                                                : _execution.implicit->Solve(state, source, mass, scale, time, stage, control));
            };
            prepared = _state.timeStepper->SolvePreparedStage(restored, _state.trial, resumeSolve);
            if (prepared)
            {
                _state.timeStepper->CompleteStage(_state.trial, evaluate);
                while (!_state.timeStepper->HasPendingTrial())
                {
                    const auto equation = _state.timeStepper->PrepareStage(_state.trial);
                    bool accepted = true;
                    if (equation.implicit)
                    {
                        const auto normalSolve = [&](EquationDofs &state, const EquationDofs &source, real mass, real scale, real time, int stage)
                        {
                            _state.progressTime = time;
                            SetTimeStage(stage);
                            const auto control = ProgressControl();
                            return RecordImplicitStage(_execution.multigrid ? _execution.multigrid->Solve(source, mass, scale, time, stage, control)
                                                                            : _execution.implicit->Solve(state, source, mass, scale, time, stage, control));
                        };
                        accepted = _state.timeStepper->SolvePreparedStage(equation, _state.trial, normalSolve);
                    }
                    if (!accepted)
                    {
                        prepared = false;
                        break;
                    }
                    _state.timeStepper->CompleteStage(_state.trial, evaluate);
                }
            }
        }
        if (!prepared)
            return StepResult::Rejected("resumed Cauchy implicit stage reached its correction cap; accepted state/history unchanged");
        if (!Admissible(_state.trial))
            return StepResult::Rejected("resumed Cauchy implicit candidate is inadmissible");
        return {};
    }
}
