#include "Cauchy/Core/Domain.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void Domain::InitializeImplicit(const DomainSettings &settings, const TimeSettings &time)
    {
        if (time.method == "ssprk3")
            return;
        const MPIInfo mpi(_fv.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { _timeStepper = std::make_unique<ODE::FieldTimeStepper>(_trial.Block("flow"), time.method,
                                                                                 [mpi, checks = settings.implicit.collectiveChecks](const auto &run)
                                                                                 { if (checks) MPI::LocalStage(mpi, run); else run(); }); });
        _implicit = std::make_unique<NonlinearSolver>(_fv, _trial.Block("flow"), _workspace, _engine, _stream, settings);
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
        PreserveReconstructionGuess();
        _lastImplicit.clear();
        const auto evaluate = [&](CFV::FVField &state, CFV::FVField &rate, real time, int stage)
        {
            _trialEpoch.stage = stage;
            _fv.Evaluate(state, _residual, *_workspace, _engine, _stream, time);
            _lastFallbacks += _fv.LastFaceFallbacks(*_workspace);
            _lastCharacteristicFallbacks += _fv.LastCharacteristicFallbacks(*_workspace);
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                const auto geometry = _fv.Metrics().AcquireRead(); const auto metrics = geometry.HostView();
                const auto read = _residual.Owned().Read(); auto write = rate.Owned().Write();
                const auto input = read.View<R::UniformTableRows>(); const auto output = write.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < input.RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j) output(iCell, j) = input(iCell, j) / metrics.GetCellVol(iCell); });
        };
        const auto solve = [&](CFV::FVField &state, const CFV::FVField &source, real mass, real scale, real time, int stage)
        {
            _trialEpoch.stage = stage;
            auto report = _implicit->Solve(state, source, mass, scale, time, stage);
            _lastAlphaCells += report.incrementLimited;
            _lastAlpha = std::min(_lastAlpha, report.minimumIncrementAlpha);
            const bool converged = report.converged;
            _lastImplicit.push_back(std::move(report));
            return converged;
        };
        DNDS_check_throw_info(_timeStepper->Prepare(_initial, _trial.Block("flow"), _time, dt, evaluate, solve),
                              "Cauchy implicit stage reached its correction cap; accepted state/history unchanged");
        DNDS_check_throw_info(_fv.Admissible(_trial.Block("flow")), "Cauchy implicit candidate is inadmissible");
    }
}
