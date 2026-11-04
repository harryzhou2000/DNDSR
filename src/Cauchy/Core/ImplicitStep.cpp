#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void Domain::SetTimeStage(int stage)
    {
        const bool coupled = _timeStepper && _timeStepper->Method() == "ditr";
        _trialEpoch.stage = coupled ? 0 : stage;
        _trialEpoch.coupledStage = coupled ? stage : 0;
    }
    void Domain::InitializeImplicit(const DomainSettings &settings, const TimeSettings &time, const ssp<Geom::UnstructuredMesh> &mesh)
    {
        _steady = time.steady;
        _implicitControls = settings.implicit;
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
        const MPIInfo mpi(_fv.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { time.ditr.Validate(); _ditrControl = time.ditr;
                          _ditrComponents = _equation.Schema().ResolveComponents(settings.implicit.residualComponents); });
        _coupledFactory = [this, mesh, settings, mpi](EquationDofs &middle)
        {
            if (_equationWorkspace)
            {
                const auto workspace = _fv.BuildEquationWorkspace(middle, _workers);
                const auto solver = std::make_shared<EquationNonlinearSolver>(_fv, middle, workspace, _engine, _stream, settings,
                                                                              Algebra::ComponentSelection(middle.ComponentNumber(), _ditrComponents), _workers);
                std::shared_ptr<PolynomialMultigrid> multigrid;
                if (settings.implicit.multigrid.levels)
                    multigrid = std::make_shared<PolynomialMultigrid>(mesh, _fv, middle, *solver, _engine, _stream, settings, _workers);
                return CoupledStageOperator{
                    [this, workspace](auto &state, auto &rate, real time, int stage)
                    { EvaluateImplicitRate(state, rate, *workspace, time, stage); },
                    [this, solver, multigrid](const auto &equation, const auto &rate, real cfl)
                    {
                        const auto &c = equation.coefficients;
                        SetTimeStage(equation.coupledStage);
                        return multigrid ? multigrid->ApplyCycle(equation.resOther, c.inverseDt, c.rhsScale, c.time, equation.coupledStage, cfl)
                                         : solver->CorrectOnce(equation.state, equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                               equation.coupledStage, {1, cfl}, &rate);
                    },
                    [solver]
                    { return solver->ReconstructionStats(); },
                    [fv = _fv, workspace, mpi]
                    { MPI::LocalStage(mpi, [&]
                                      { fv.ResetReconstruction(*workspace); }); }};
            }
            AuxiliaryRequest required;
            required.faceSpectralRadius = required.cellSpectralSum = true;
            const auto workspace = _fv.BuildWorkspace(middle, required);
            const auto solver = std::make_shared<NonlinearSolver>(_fv, middle.CoupledField(), workspace, _engine, _stream, settings);
            std::shared_ptr<PolynomialMultigrid> multigrid;
            if (settings.implicit.multigrid.levels)
                multigrid = std::make_shared<PolynomialMultigrid>(mesh, _fv, middle.CoupledField(), *solver, _engine, _stream, settings);
            return CoupledStageOperator{
                [this, workspace](auto &state, auto &rate, real time, int stage)
                { EvaluateImplicitRate(state, rate, *workspace, time, stage); },
                [this, solver, multigrid](const auto &equation, const auto &rate, real cfl)
                {
                    const auto &c = equation.coefficients;
                    SetTimeStage(equation.coupledStage);
                    return AsEquationReport(multigrid
                                                ? multigrid->ApplyCycle(equation.resOther.CoupledField(), c.inverseDt, c.rhsScale, c.time, equation.coupledStage, cfl)
                                                : solver->CorrectOnce(equation.state.CoupledField(), equation.resOther.CoupledField(), c.inverseDt, c.rhsScale, c.time,
                                                                      equation.coupledStage, {1, cfl}, &rate.CoupledField()));
                },
                [solver]
                { return solver->ReconstructionStats(); },
                [fv = _fv, workspace, mpi]
                {
                    // Intermediate polynomials are trial-local, so no second
                    // warm-history dataset is required for exact restart/retry.
                    MPI::LocalStage(mpi, [&]
                                    { fv.ResetReconstruction(*workspace); });
                }};
        };
        MPI::LocalStage(mpi, [&]
                        { _timeStepper = std::make_unique<EquationTimeStepper>(_trial, _steady ? "bdf1" : time.method, [mpi, checks = settings.implicit.collectiveChecks](const auto &run)
                                                                               { if (checks) MPI::LocalStage(mpi, run); else run(); }, ODE::DITRSettings{time.ditr.fraction, time.ditr.theta, time.ditr.variant == "U2R1" ? ODE::DITRVariant::U2R1 : ODE::DITRVariant::U2R2}); });
        if (_equationWorkspace)
            _equationImplicit = std::make_unique<EquationNonlinearSolver>(_fv, _trial, _equationWorkspace, _engine, _stream, settings,
                                                                          Algebra::ComponentSelection(_trial.ComponentNumber(), _ditrComponents), _workers);
        else
            _implicit = std::make_unique<NonlinearSolver>(_fv, _trial.CoupledField(), _workspace, _engine, _stream, settings);
        if (settings.implicit.multigrid.levels)
        {
            if (_equationImplicit)
                _multigrid = std::make_unique<PolynomialMultigrid>(mesh, _fv, _trial, *_equationImplicit, _engine, _stream, settings, _workers);
            else
                _multigrid = std::make_unique<PolynomialMultigrid>(mesh, _fv, _trial.CoupledField(), *_implicit, _engine, _stream, settings);
        }
    }
    void Domain::EvaluateImplicitRate(EquationDofs &state, EquationDofs &rate, FiniteVolume::Workspace &workspace, real time, int stage)
    {
        SetTimeStage(stage);
        _fv.Evaluate(state, _residual, workspace, _engine, _stream, time);
        _lastFallbacks += _fv.LastFaceFallbacks(workspace);
        _lastCharacteristicFallbacks += _fv.LastCharacteristicFallbacks(workspace);
        NormalizeRate(rate);
    }
    void Domain::EvaluateImplicitRate(EquationDofs &state, EquationDofs &rate, FiniteVolume::EquationWorkspace &workspace, real time, int stage)
    {
        SetTimeStage(stage);
        _fv.Evaluate(state, _residualState, workspace, _engine, _stream, time);
        NormalizeRate(rate);
    }
    void Domain::NormalizeRate(EquationDofs &rate)
    {
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            const auto geometry = _fv.Metrics().AcquireRead(); const auto metrics = geometry.HostView();
            const auto read = _residualState.ReadOwned(); auto write = rate.WriteOwned();
            for (index iBlock = 0; iBlock < rate.BlockNumber(); ++iBlock)
            {
                const auto input = read.Block(iBlock); const auto output = write.Block(iBlock);
                for (index iCell = 0; iCell < input.owned.RowCount(); ++iCell)
                    for (index j = 0; j < input.ComponentNumber(); ++j) output(iCell, j) = input(iCell, j) / metrics.GetCellVol(iCell);
            } });
    }
    bool Domain::RecordImplicitStage(NonlinearIterationReport report)
    {
        return RecordImplicitStage(AsEquationReport(std::move(report)));
    }
    EquationNonlinearIterationReport Domain::AsEquationReport(NonlinearIterationReport report)
    {
        EquationNonlinearIterationReport result;
        result.converged = report.converged;
        result.iterations = report.iterations;
        result.linearIterations = report.linearIterations;
        result.rhsLimited = report.rhsLimited;
        result.incrementLimited = report.incrementLimited;
        result.multigridCycles = report.multigridCycles;
        result.levelCorrections = std::move(report.levelCorrections);
        result.initialResidual.assign(report.initialResidual.begin(), report.initialResidual.end());
        result.referenceResidual.assign(report.referenceResidual.begin(), report.referenceResidual.end());
        result.residual.assign(report.residual.begin(), report.residual.end());
        result.residualTargetRatio = report.residualTargetRatio;
        result.minimumRHSAlpha = report.minimumRHSAlpha;
        result.minimumIncrementAlpha = report.minimumIncrementAlpha;
        result.pseudoCFL = report.pseudoCFL;
        result.reconstructionIterations = report.reconstructionIterations;
        result.reconstructionRelativeUpdate = report.reconstructionRelativeUpdate;
        result.lastLinear = std::move(report.lastLinear);
        return result;
    }
    bool Domain::RecordImplicitStage(EquationNonlinearIterationReport report)
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
        if (_equationWorkspace)
        {
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                if (_restoreReconstruction) _fv.RestoreReconstructionGuess(*_equationWorkspace, *_equationBackup);
                else
                {
                    const auto &guess = _fv.ReconstructionGuess(*_equationWorkspace);
                    if (!_equationBackup) _equationBackup.emplace(EquationDofOperations::NewVector(guess));
                    EquationDofOperations::Copy(*_equationBackup, guess);
                    _restoreReconstruction = true;
                } });
            return;
        }
        if (_restoreReconstruction)
        {
            _fv.RestoreReconstructionGuess(*_workspace, *_reconstructionBackup, _limitedVRBackup ? &*_limitedVRBackup : nullptr);
            return;
        }
        std::optional<R::Array<real>::ReadLease> guessRead, historyRead;
        std::optional<R::Array<real>::WriteLease> guessWrite, historyWrite;
        try
        {
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                const auto &guess = _fv.ReconstructionGuess(*_workspace).Owned();
                if (!_reconstructionBackup) _reconstructionBackup.emplace(guess.Layout());
                guessRead.emplace(guess.Read());
                if (const auto *history = _fv.LimitedVRHistory(*_workspace))
                {
                    if (!_limitedVRBackup) _limitedVRBackup.emplace(history->Layout());
                    historyRead.emplace(history->Read());
                }
                guessWrite.emplace(_reconstructionBackup->Write());
                if (historyRead) historyWrite.emplace(_limitedVRBackup->Write()); });
        }
        catch (...)
        {
            if (guessWrite)
                guessWrite->CancelUnmodified();
            if (historyWrite)
                historyWrite->CancelUnmodified();
            throw;
        }
        const auto source = guessRead->View<R::UniformTableRows>();
        const auto output = guessWrite->View<R::UniformTableRows>();
        for (index iCell = 0; iCell < source.RowCount(); ++iCell)
            for (index j = 0; j < source.RowSize(iCell); ++j)
                output(iCell, j) = source(iCell, j);
        if (historyRead)
        {
            const auto from = historyRead->View<R::UniformTableRows>();
            const auto to = historyWrite->View<R::UniformTableRows>();
            for (index iCell = 0; iCell < from.RowCount(); ++iCell)
                for (int j = 0; j < 2; ++j)
                    to(iCell, j) = from(iCell, j);
        }
        _restoreReconstruction = true;
    }
    void Domain::PrepareImplicitStep(real dt)
    {
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
        const auto evaluate = [&](EquationDofs &state, EquationDofs &rate, real time, int stage)
        {
            if (_timeStepper->Method() == "ditr" && stage == 0)
            {
                DNDS_check_throw_info(_coupledMiddle, "DITR intermediate operator is not bound");
                _coupledMiddle->evaluate(state, rate, time, stage);
            }
            else if (_equationWorkspace)
                EvaluateImplicitRate(state, rate, *_equationWorkspace, time, stage);
            else
                EvaluateImplicitRate(state, rate, *_workspace, time, stage);
        };
        const auto solve = [&](EquationDofs &state, const EquationDofs &source, real mass, real scale, real time, int stage)
        {
            SetTimeStage(stage);
            if (_equationImplicit)
                return RecordImplicitStage(_multigrid ? _multigrid->Solve(source, mass, scale, time, stage)
                                                      : _equationImplicit->Solve(state, source, mass, scale, time, stage));
            return RecordImplicitStage(_multigrid ? _multigrid->Solve(source.CoupledField(), mass, scale, time, stage) : _implicit->Solve(state.CoupledField(), source.CoupledField(), mass, scale, time, stage));
        };
        const auto coupled = [&](EquationTimeStepper::DITR &method, EquationDofs &endpoint)
        {
            const auto owner = method.IntermediateState().PhysicalField(0).Owned().SharedStorage();
            if (!_coupledMiddle || _coupledMiddleOwner != owner)
            {
                _coupledMiddle = _coupledFactory(method.IntermediateState());
                _coupledMiddleOwner = owner;
            }
            if (_coupledMiddle->reset)
                _coupledMiddle->reset();
            const MPIInfo mpi(_fv.Metrics().Communicator());
            MPI::LocalStage(mpi, [&]
                            {
                if (!_coupledResidual) _coupledResidual.emplace(EquationDofOperations::NewVector(endpoint)); });
            // Temporary compatibility mapping: there is only one loop/policy.
            // Configuration consolidation is a separate user decision; these
            // existing explicit DITR keys currently override the shared policy.
            auto settings = _implicitControls;
            settings.iterations = _ditrControl.iterations;
            settings.minIterations = _ditrControl.minIterations;
            settings.relativeTolerance = _ditrControl.relativeTolerance;
            settings.absoluteTolerance = _ditrControl.absoluteTolerance;
            EquationNonlinearIterationReport initial;
            initial.residual.resize(size_t(endpoint.ComponentNumber()));
            const auto report = NonlinearDetail::Iterate(settings, endpoint.Schema(), mpi.rank, 0, {}, std::move(initial), _coupledReconstructionTolerance, [&]
                                                         { method.AssembleStoppingResidual(endpoint, *_coupledResidual); }, [&](auto &norms)
                                                         {
                    std::vector<real> sum(norms.size() + 1, 0);
                    MPI::LocalStage(mpi, [&]
                    {
                        const auto geometry = _fv.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                        const auto read = _coupledResidual->ReadOwned(); const auto layout = read.ComponentLayout(); const auto values = layout.View();
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
                    MPI_Allreduce(MPI_IN_PLACE, sum.data(), int(sum.size()), MPI_DOUBLE, MPI_SUM, mpi.comm);
                    DNDS_check_throw_info(std::isfinite(sum.back()) && sum.back() > 0, "invalid DITR global volume");
                    for (index j = 0; j < index(norms.size()); ++j)
                        norms[j] = settings.normOrder == 1 ? sum[j] / sum.back() : std::sqrt(sum[j] / sum.back()); }, [&]
                                                         {
                    const auto a = _coupledMiddle->reconstruction();
                    const auto b = _equationImplicit ? _equationImplicit->ReconstructionStats() : _implicit->ReconstructionStats();
                    return std::make_pair(std::max(a.first, b.first), std::max(a.second, b.second)); }, [&](auto &iteration)
                                                         {
                    for (int coupledStage = 0; coupledStage < method.CoupledStageNumber(); ++coupledStage)
                    {
                        SetTimeStage(coupledStage);
                        const auto equation = method.PrepareCoupledStage(endpoint, coupledStage);
                        const auto &c = equation.coefficients;
                        const auto &rate = method.CurrentRate();
                        EquationNonlinearIterationReport work;
                        if (coupledStage == 0)
                            work = _coupledMiddle->correct(equation, rate, iteration.pseudoCFL);
                        else if (_equationImplicit)
                            work = _multigrid ? _multigrid->ApplyCycle(equation.resOther, c.inverseDt, c.rhsScale, c.time, coupledStage, iteration.pseudoCFL)
                                             : _equationImplicit->CorrectOnce(endpoint, equation.resOther, c.inverseDt, c.rhsScale, c.time,
                                                                               coupledStage, {1, iteration.pseudoCFL}, &rate);
                        else
                            work = AsEquationReport(_multigrid
                                ? _multigrid->ApplyCycle(equation.resOther.CoupledField(), c.inverseDt, c.rhsScale, c.time, coupledStage, iteration.pseudoCFL)
                                : _implicit->CorrectOnce(endpoint.CoupledField(), equation.resOther.CoupledField(), c.inverseDt, c.rhsScale, c.time,
                                                         coupledStage, {1, iteration.pseudoCFL}, &rate.CoupledField()));
                        NonlinearDetail::AccumulateWork(iteration, work);
                        method.CompleteCoupledStage(endpoint, evaluate);
                    } }, NonlinearDetail::ResidualObservation::AfterUpdate);
            _lastCoupledIterations = report.iterations;
            _lastCoupledTargetRatio = report.residualTargetRatio;
            return RecordImplicitStage(report);
        };
        const bool prepared = _steady ? _timeStepper->PrepareSteady(_initial, _trial, _time, solve)
                                      : _timeStepper->Prepare(_initial, _trial, _time, dt, evaluate, solve, coupled);
        if (!prepared)
        {
            // Failure-only diagnostics keep capped inner solves distinct from
            // accepted stages and make timestep retries auditable. No logging
            // or extra reduction is added to the successful numerical path.
            Json details{{"method", _timeStepper->Method()}, {"steady", _steady}, {"stage", _trialEpoch.stage}, {"dt", _steady ? 0 : dt}, {"coupled_iterations", _lastCoupledIterations}, {"coupled_target_ratio", _lastCoupledTargetRatio}};
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
            DNDS_check_throw_info(false, "Cauchy implicit stage reached its correction cap; accepted state/history unchanged; " + details.dump());
        }
        DNDS_check_throw_info(Admissible(_trial), "Cauchy implicit candidate is inadmissible");
    }
}
