#include "Cauchy/Core/Domain.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void Domain::InitializeImplicit(const DomainSettings &settings, const TimeSettings &time, const ssp<Geom::UnstructuredMesh> &mesh)
    {
        _steady = time.steady;
        if (time.method == "ssprk3" && !_steady)
        {
            DNDS_check_throw_info(!settings.implicit.multigrid.levels, "polynomial FAS requires an implicit time method");
            return;
        }
        const MPIInfo mpi(_fv.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { time.ditr.Validate(); _ditrControl = time.ditr; });
        const ODE::FieldDITR::StageFactory factory = [this, mesh, settings, mpi](CFV::FVField &middle)
        {
            AuxiliaryRequest required;
            required.faceSpectralRadius = required.cellSpectralSum = true;
            const auto workspace = _fv.BuildWorkspace(middle, required);
            const auto solver = std::make_shared<NonlinearSolver>(_fv, middle, workspace, _engine, _stream, settings);
            std::shared_ptr<PolynomialMultigrid> multigrid;
            if (settings.implicit.multigrid.levels)
                multigrid = std::make_shared<PolynomialMultigrid>(mesh, _fv, middle, *solver, _engine, _stream, settings);
            return ODE::FieldDITR::StageOperator{
                [this, workspace](auto &state, auto &rate, real time, int stage)
                { EvaluateImplicitRate(state, rate, *workspace, time, stage); },
                [this, solver, multigrid](auto &state, const auto &source, real mass, real scale, real time, int stage)
                {
                    _trialEpoch.stage = stage;
                    return RecordImplicitStage(multigrid ? multigrid->Solve(source, mass, scale, time, stage) : solver->Solve(state, source, mass, scale, time, stage));
                },
                [fv = _fv, workspace, mpi]
                {
                    // Intermediate polynomials are trial-local, so no second
                    // warm-history dataset is required for exact restart/retry.
                    MPI::LocalStage(mpi, [&]
                                    { fv.ResetReconstruction(*workspace); });
                }};
        };
        MPI::LocalStage(mpi, [&]
                        { _timeStepper = std::make_unique<ODE::FieldTimeStepper>(_trial.Block("flow"), _steady ? "bdf1" : time.method, [mpi, checks = settings.implicit.collectiveChecks](const auto &run)
                                                                                 { if (checks) MPI::LocalStage(mpi, run); else run(); }, ODE::DITRSettings{time.ditr.fraction, time.ditr.theta, time.ditr.iterations}, factory); });
        _implicit = std::make_unique<NonlinearSolver>(_fv, _trial.Block("flow"), _workspace, _engine, _stream, settings);
        if (settings.implicit.multigrid.levels)
            _multigrid = std::make_unique<PolynomialMultigrid>(mesh, _fv, _trial.Block("flow"), *_implicit, _engine, _stream, settings);
    }
    void Domain::EvaluateImplicitRate(CFV::FVField &state, CFV::FVField &rate, FiniteVolume::Workspace &workspace, real time, int stage)
    {
        _trialEpoch.stage = stage;
        _fv.Evaluate(state, _residual, workspace, _engine, _stream, time);
        _lastFallbacks += _fv.LastFaceFallbacks(workspace);
        _lastCharacteristicFallbacks += _fv.LastCharacteristicFallbacks(workspace);
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            const auto geometry = _fv.Metrics().AcquireRead(); const auto metrics = geometry.HostView();
            const auto read = _residual.Owned().Read(); auto write = rate.Owned().Write();
            const auto input = read.View<R::UniformTableRows>(); const auto output = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < input.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j) output(iCell, j) = input(iCell, j) / metrics.GetCellVol(iCell); });
    }
    bool Domain::RecordImplicitStage(NonlinearIterationReport report)
    {
        _lastAlphaCells += report.incrementLimited;
        _lastAlpha = std::min(_lastAlpha, report.minimumIncrementAlpha);
        const bool converged = report.converged;
        _lastImplicit.push_back(std::move(report));
        return converged;
    }
    void Domain::PreserveReconstructionGuess()
    {
        if (_restoreReconstruction)
        {
            _fv.RestoreReconstructionGuess(*_workspace, *_reconstructionBackup);
            return;
        }
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            const auto &guess = _fv.ReconstructionGuess(*_workspace).Owned();
            if (!_reconstructionBackup) _reconstructionBackup.emplace(guess.Layout());
            const auto read = guess.Read(); auto write = _reconstructionBackup->Write();
            const auto source = read.View<R::UniformTableRows>(); const auto output = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < source.RowCount(); ++iCell)
                for (index j = 0; j < source.RowSize(iCell); ++j) output(iCell, j) = source(iCell, j); });
        _restoreReconstruction = true;
    }
    void Domain::PrepareImplicitStep(real dt)
    {
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
        const auto evaluate = [&](CFV::FVField &state, CFV::FVField &rate, real time, int stage)
        {
            EvaluateImplicitRate(state, rate, *_workspace, time, stage);
        };
        const auto solve = [&](CFV::FVField &state, const CFV::FVField &source, real mass, real scale, real time, int stage)
        {
            _trialEpoch.stage = stage;
            return RecordImplicitStage(_multigrid ? _multigrid->Solve(source, mass, scale, time, stage) : _implicit->Solve(state, source, mass, scale, time, stage));
        };
        std::array<real, 10> initialNorm{};
        const auto coupled = [&](const CFV::FVField &hermite, const CFV::FVField &integral, index iteration)
        {
            std::array<real, 11> norms{};
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                _fv.Metrics().ValidateField(hermite, 5); _fv.Metrics().ValidateField(integral, 5);
                const auto geometry = _fv.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                const auto hRead = hermite.Owned().Read(), iRead = integral.Owned().Read();
                const auto h = hRead.View<R::UniformTableRows>(), i = iRead.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < h.RowCount(); ++iCell)
                {
                    const real volume = metric.GetCellVol(iCell); norms[10] += volume;
                    for (int j = 0; j < 5; ++j)
                    {
                        norms[j] += volume * h(iCell, j) * h(iCell, j);
                        norms[5 + j] += volume * i(iCell, j) * i(iCell, j);
                    }
                } });
            MPI_Allreduce(MPI_IN_PLACE, norms.data(), 11, MPI_DOUBLE, MPI_SUM, _fv.Metrics().Communicator());
            DNDS_check_throw_info(norms[10] > 0 && std::isfinite(norms[10]), "invalid coupled stage volume");
            const auto &control = _ditrControl;
            _lastCoupledTargetRatio = 0;
            _lastCoupledIterations = iteration;
            for (int j = 0; j < 10; ++j)
            {
                const real norm = std::sqrt(norms[j] / norms[10]);
                DNDS_check_throw_info(std::isfinite(norm), "nonfinite DITR coupled stage residual");
                if (!iteration)
                    initialNorm[j] = norm;
                const real target = std::max({control.absoluteTolerance, control.relativeTolerance * initialNorm[j], verySmallReal});
                _lastCoupledTargetRatio = std::max(_lastCoupledTargetRatio, norm / target);
            }
            return iteration >= control.minIterations && _lastCoupledTargetRatio <= 1;
        };
        const bool prepared = _steady ? _timeStepper->PrepareSteady(_initial, _trial.Block("flow"), _time, solve)
                                      : _timeStepper->Prepare(_initial, _trial.Block("flow"), _time, dt, evaluate, solve, coupled);
        if (!prepared)
        {
            // Failure-only diagnostics keep capped inner solves distinct from
            // accepted stages and make timestep retries auditable. No logging
            // or extra reduction is added to the successful numerical path.
            Json details{{"method", _timeStepper->Method()}, {"steady", _steady}, {"stage", _trialEpoch.stage}, {"dt", _steady ? 0 : dt}, {"coupled_iterations", _lastCoupledIterations}, {"coupled_target_ratio", _lastCoupledTargetRatio}};
            if (!_lastImplicit.empty())
            {
                const auto &last = _lastImplicit.back();
                details["last_inner_converged"] = last.converged;
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
        DNDS_check_throw_info(_fv.Admissible(_trial.Block("flow")), "Cauchy implicit candidate is inadmissible");
    }
}
