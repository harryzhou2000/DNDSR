#include "Cauchy/Core/Domain.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        R::RowLayout FlowLayout(const DomainSettings &settings)
        {
            if (settings.execution.layout == "soa")
                return R::RowLayout::ColMajor(0, 5);
            if (settings.execution.layout == "aosoa32")
                return R::RowLayout::TiledColMajor(0, 5, 32);
            return R::RowLayout::Dense(0, 5);
        }
        EquationDofs BuildState(const FiniteVolume &discrete, const EquationSystem &equation, const DomainSettings &settings)
        {
            const auto &fv = discrete.Metrics();
            auto field = fv.BuildField(FlowLayout(settings));
            discrete.Initialize(field, settings.initialProfile, settings.initialPrimitive);
            return EquationDofs::Coupled(equation.Schema(), std::move(field));
        }
        EquationSystem ResolveEquation(const std::string &name, const DomainSettings &settings)
        {
            settings.implicit.Validate();
            ResolvedExecutionPlan::Resolve({{name, {0}}, settings});
            return EquationSystem::Resolve(settings.equation);
        }
        AuxiliaryRequest TimeAuxiliary(const TimeSettings &time)
        {
            AuxiliaryRequest request;
            request.faceSpectralRadius = request.cellSpectralSum = time.steady || time.method != "ssprk3";
            return request;
        }
    }
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time)
        : _name(std::move(name)), _equation(ResolveEquation(_name, settings)), _fv(mesh, settings, _equation),
          _accepted(BuildState(_fv, _equation, settings)), _trial(_accepted.Clone()),
          _residual(_fv.Metrics().BuildField(FlowLayout(settings))), _initial(EquationDofOperations::NewVector(_accepted)),
          _workspace(_fv.BuildWorkspace(_trial, TimeAuxiliary(time))),
          _engine(size_t(settings.execution.hostThreads), settings.execution.checkAccess ? std::make_shared<R::AccessTrackingContext>() : nullptr),
          _stream(_engine.CreateCPUStream())
    {
        _positivity = settings.numerics.positivity;
        _equation.ValidateStateSchema(_accepted);
        DNDS_check_throw_info(_fv.Metrics().Dimension() == settings.mesh.dimension, "Cauchy config/mesh dimension differs");
        DNDS_check_throw_info(_fv.Admissible(Flow()), "Cauchy initial means violate density or minimum-temperature internal-energy bounds");
        ConfigureTimeStep(time, settings.implicit.positivity.compressionPower);
        InitializeImplicit(settings, time, mesh);
    }
    real Domain::StableStep(real cfl) const
    {
        if (!_timeAuxiliary)
        {
            AuxiliaryRequest request;
            request.meanThermodynamics = request.cellSpectralSum = true;
            _timeAuxiliary.emplace(_fv.BuildAuxiliaryValues(request));
            auto input = Flow();
            _timeMeanHalo.emplace(_fv.Metrics().BindHalo(input));
        }
        auto transfer = _engine.ChecksAccess() ? _timeMeanHalo->BeginChecked(_stream)
                                               : _timeMeanHalo->BeginScheduled(_stream.Context());
        transfer.Wait();
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        { _fv.PrepareAuxValues(Flow(), *_timeAuxiliary, _time, _interfaceInputs ? &*_interfaceInputs : nullptr); });
        return _fv.StableStep(Flow(), *_timeAuxiliary, cfl, _time);
    }
    void Domain::UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime)
    {
        _fv.Evaluate(_trial, _residual, *_workspace, _engine, _stream, stageTime);
        _lastFallbacks += _fv.LastFaceFallbacks(*_workspace);
        _lastCharacteristicFallbacks += _fv.LastCharacteristicFallbacks(*_workspace);
        auto geometry = _fv.Metrics().AcquireRead();
        const auto metrics = geometry.HostView();
        real alpha = 1;
        int safe[2]{1, 1}; // high candidate, low candidate
        if (_positivity)
        {
            const auto reconstruction = _fv.PositivityReconstructionStats(*_workspace);
            _lastBetaCells += reconstruction.first;
            _lastBeta = std::min(_lastBeta, reconstruction.second);
            index limited = 0;
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                const auto initialRead = _initial.CoupledField().Owned().Read(), stateRead = _trial.CoupledField().Owned().Read();
                const auto highRead = _residual.Owned().Read(), lowRead = _fv.LowOrderResidual(*_workspace).Owned().Read();
                const auto initial = initialRead.View<R::UniformTableRows>(), state = stateRead.View<R::UniformTableRows>();
                const auto high = highRead.View<R::UniformTableRows>(), low = lowRead.View<R::UniformTableRows>();
                for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
                {
                    FlowState base, target;
                    for (int j = 0; j < 5; ++j)
                    {
                        const real constant = historyWeight * initial(iCell, j) + stageWeight * state(iCell, j);
                        base[j] = constant + stageWeight * dt * low(iCell, j) / metrics.GetCellVol(iCell);
                        target[j] = constant + stageWeight * dt * high(iCell, j) / metrics.GetCellVol(iCell);
                        DNDS_check_throw_info(std::isfinite(base[j]) && std::isfinite(target[j]), "nonfinite stage input cannot be repaired by PP");
                    }
                    safe[0] &= ThermallyAdmissible(target, _fv.ThermalConstraints());
                    const bool lowSafe = ThermallyAdmissible(base, _fv.ThermalConstraints());
                    safe[1] &= lowSafe;
                    DNDS_check_throw_info(lowSafe || ThermallyAdmissible(target, _fv.ThermalConstraints()), "both low/high PP stage candidates are inadmissible; reduce dt");
                    const auto fraction = lowSafe ? AdmissibleFraction(base, target, _fv.ThermalConstraints()) : real(1);
                    if (fraction < 1) { ++limited; alpha = std::min(alpha, fraction); }
                } });
            MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, MPI_DOUBLE, MPI_MIN, _fv.Metrics().Communicator());
            MPI_Allreduce(MPI_IN_PLACE, safe, 2, MPI_INT, MPI_MIN, _fv.Metrics().Communicator());
            if (safe[0])
            {
                alpha = 1;
                limited = 0; // No correction was applied to this high stage.
            }
            else
            {
                // A shared flux blend uses the low-order endpoint everywhere;
                // local success of the high endpoint cannot excuse an invalid
                // low endpoint when another cell actually requires blending.
                DNDS_check_throw_info(safe[1], "global conservative PP blend requires admissible low-order stage; reduce dt");
            }
            MPI_Allreduce(MPI_IN_PLACE, &limited, 1, MPI_INT64_T, MPI_SUM, _fv.Metrics().Communicator());
            _lastAlpha = std::min(_lastAlpha, alpha);
            _lastAlphaCells += limited;
        }
        std::optional<R::Array<real>::ReadLease> lowRead;
        R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows> low;
        if (_positivity)
        {
            lowRead.emplace(_fv.LowOrderResidual(*_workspace).Owned().Read());
            low = lowRead->View<R::UniformTableRows>();
        }
        R::SubmitArrays(_engine, _stream, [dt, historyWeight, stageWeight, metrics, alpha, low](auto, auto initial, auto rhs, auto stage)
                        {
            for (index iCell = 0; iCell < stage.rows; ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    const real residual = alpha == 1 ? rhs(iCell, j) : (1 - alpha) * low(iCell, j) + alpha * rhs(iCell, j);
                    stage(iCell, j) = historyWeight * initial(iCell, j) + stageWeight * (stage(iCell, j) + dt * residual / metrics.GetCellVol(iCell));
                } }, R::ReadAs<R::UniformTableRows>(_initial.CoupledField().Owned()), R::ReadAs<R::UniformTableRows>(_residual.Owned()), R::ReadWriteAs<R::UniformTableRows>(_trial.CoupledField().Owned()));
        _stream.Record().WaitHost();
        DNDS_check_throw_info(_fv.Admissible(_trial.CoupledField()), "Cauchy SSPRK3 stage is inadmissible; accepted state unchanged");
    }
    void Domain::BeginTrial(real dt)
    {
        _trialReady = false;
        int valid = std::isfinite(dt) && dt > 0 && (_steady || (std::isfinite(_time + dt) && _time + dt > _time));
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _fv.Metrics().Communicator());
        DNDS_check_throw_info(valid, "invalid Cauchy physical timestep");
        real min = dt, max = dt;
        MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, _fv.Metrics().Communicator());
        MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, _fv.Metrics().Communicator());
        DNDS_check_throw_info(min == max, "Cauchy timestep differs between domain ranks");
        DNDS_check_throw_info(_fv.Admissible(Flow()), "accepted Cauchy mean is inadmissible");
        // Owned read scopes may coexist with a retained accepted-state reader.
        // Drain earlier engine work, then copy every logical segment through
        // the same synchronous CPU state adapter used by the time driver.
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            _stream.Record().WaitHost();
            EquationDofOperations::Copy(_trial, _accepted);
            EquationDofOperations::Copy(_initial, _accepted); });
        _lastFallbacks = 0;
        _lastAlpha = _lastBeta = 1;
        _lastAlphaCells = _lastBetaCells = 0;
        _lastCharacteristicFallbacks = 0;
        _trialEpoch = _epoch;
        _trialDt = _steady ? 0 : dt;
        _trialEpoch.trial = ++_nextTrial;
        // The polynomial guess is numerical state for explicit fixed-sweep VR
        // too. Preserve it before either time driver can overwrite the trial.
        PreserveReconstructionGuess();
    }
    void Domain::PrepareStep(real dt)
    {
        DNDS_check_throw_info(!_fv.HasInterfaces(), "interface domains must advance through their coordinated Simulation");
        BeginTrial(dt);
        if (_timeStepper)
        {
            PrepareImplicitStep(dt);
            _trialTime = _steady ? _time : _time + dt;
            _trialReady = true;
            return;
        }
        // Shu-Osher SSPRK(3,3), including its non-monotone physical stage times.
        _trialEpoch.stage = 0;
        UpdateStage(dt, 0, 1, _time);
        _trialEpoch.stage = 1;
        UpdateStage(dt, .75, .25, _time + dt);
        _trialEpoch.stage = 2;
        UpdateStage(dt, 1. / 3, 2. / 3, _time + .5 * dt);
        _trialTime = _time + dt;
        _trialReady = true;
    }
    int Domain::CoordinatedStageNumber() const { return _timeStepper ? _timeStepper->StageNumber() : 3; }
    void Domain::BindInterfaceInputs(const BoundaryInputs &input)
    {
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            _fv.SetBoundaryInputs(*_workspace,input);
            if(_observationWorkspace) _fv.SetBoundaryInputs(*_observationWorkspace,input);
            if(_timePPWorkspace) _fv.SetBoundaryInputs(*_timePPWorkspace,input);
            _interfaceInputs=input; });
        if (_multigrid)
            _multigrid->SetBoundaryInputs(input);
    }
    void Domain::BeginCoordinatedStep(real dt)
    {
        DNDS_check_throw_info(!_timeStepper || _timeStepper->SupportsStageSequence(), "multi-domain DITR needs an external coupled midpoint/end iteration controller");
        BeginTrial(dt);
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
        _coordinatedStage = 0;
        _coordinatedPrepared = false;
        _stageEquation.reset();
        if (_timeStepper)
            _timeStepper->BeginStages(_initial, _trial, _time, dt, _steady);
    }
    real Domain::PrepareCoordinatedStage(int stage)
    {
        DNDS_check_throw_info(stage == _coordinatedStage && stage < CoordinatedStageNumber() && !_coordinatedPrepared,
                              "coordinated domain stage is out of order");
        _trialEpoch.stage = stage;
        _coordinatedPrepared = true;
        if (_timeStepper)
        {
            _stageEquation = _timeStepper->PrepareStage(_trial);
            return _stageEquation->time;
        }
        static constexpr real c[]{0, 1, .5};
        return _time + c[stage] * _trialDt;
    }
    void Domain::SolveCoordinatedStage()
    {
        DNDS_check_throw_info(_coordinatedPrepared, "coordinated domain stage was not prepared");
        if (_stageEquation)
        {
            const auto &equation = *_stageEquation;
            if (equation.implicit)
            {
                const auto report = _multigrid ? _multigrid->Solve(equation.source->CoupledField(), equation.mass, equation.rhsScale, equation.time, equation.stage)
                                               : _implicit->Solve(_trial.CoupledField(), equation.source->CoupledField(), equation.mass, equation.rhsScale, equation.time, equation.stage);
                DNDS_check_throw_info(RecordImplicitStage(report), "coordinated implicit stage reached its correction cap");
            }
        }
        else
        {
            static constexpr real history[]{0, .75, 1. / 3}, weight[]{1, .25, 2. / 3}, c[]{0, 1, .5};
            UpdateStage(_trialDt, history[_coordinatedStage], weight[_coordinatedStage], _time + c[_coordinatedStage] * _trialDt);
        }
    }
    void Domain::CompleteCoordinatedStage()
    {
        DNDS_check_throw_info(_coordinatedPrepared, "no coordinated stage to complete");
        if (_timeStepper)
            _timeStepper->CompleteStage(_trial, [this](auto &state, auto &rate, real time, int stage)
                                        { EvaluateImplicitRate(state, rate, *_workspace, time, stage); });
        _coordinatedPrepared = false;
        _stageEquation.reset();
        if (++_coordinatedStage == CoordinatedStageNumber())
        {
            DNDS_check_throw_info(_fv.Admissible(_trial.CoupledField()), "coordinated candidate is inadmissible");
            _trialTime = _steady ? _time : _time + _trialDt;
            _trialReady = true;
        }
    }
    void Domain::CommitStep()
    {
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        { DNDS_check_throw_info(_trialReady && (!_timeStepper || _timeStepper->HasPendingTrial()), "no prepared Cauchy candidate"); });
        auto epoch = _trialEpoch;
        ++epoch.acceptedStep;
        epoch.stage = 0;
        PublishTrial(_trialTime, epoch);
        if (_timeStepper)
            _timeStepper->Commit();
        _lastAcceptedDt = _trialDt;
    }
    void Domain::CommitReserved(R::Array<real>::WriteLease &destination, R::Array<real>::ReadLease &source)
    {
        // Simulation reserved every domain's destination/source and agreed
        // readiness globally before entering this allocation-free publication.
        const auto input = source.View<R::UniformTableRows>();
        const auto output = destination.View<R::UniformTableRows>();
        for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                output(iCell, j) = input(iCell, j);
        _time = _trialTime;
        _epoch = _trialEpoch;
        ++_epoch.acceptedStep;
        _epoch.stage = 0;
        _trialReady = _restoreReconstruction = false;
        if (_timeStepper)
            _timeStepper->Commit();
        _lastAcceptedDt = _trialDt;
    }
    void Domain::PublishTrial(real time, StateEpoch epoch, const R::Array<real> *reconstructionGuess, const R::Array<real> *limitedVRHistory)
    {
        // Agree local immediate-access availability before any accepted values
        // change. A retained debug/Python view must not allow a partial commit
        // on only the other ranks. Keep this lease through the actual copy.
        std::optional<R::Array<real>::WriteLease> destination;
        std::optional<R::Array<real>::ReadLease> source;
        const auto comm = _fv.Metrics().Communicator();
        MPIInfo mpi(comm);
        try
        {
            MPI::LocalStage(mpi, [&]
                            { destination.emplace(Flow().Owned().Write()); source.emplace(_trial.CoupledField().Owned().Read()); });
            // Restore only after accepted-state publication is known to be
            // available. A retained debug reader must not change the future
            // polynomial guess when it rejects the accompanying state load.
            if (reconstructionGuess)
                _fv.RestoreReconstructionGuess(*_workspace, *reconstructionGuess, limitedVRHistory);
        }
        catch (...)
        {
            // Successful ranks acquired but never used their destination.
            // Cancel before unwinding; a normal exceptional write destructor
            // deliberately poisons possibly partial writes.
            if (destination)
                destination->CancelUnmodified();
            if (source)
                source->CancelUnmodified();
            throw;
        }
        const auto input = source->View<R::UniformTableRows>();
        const auto output = destination->View<R::UniformTableRows>();
        for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
            for (int component = 0; component < 5; ++component)
                output(iCell, component) = input(iCell, component);
        _time = time;
        _epoch = epoch;
        _trialReady = false;
        _restoreReconstruction = false;
    }
    const CFV::FVField &Domain::Residual()
    {
        _trialReady = false;
        DNDS_check_throw_info(_fv.Admissible(Flow()), "accepted Cauchy mean is inadmissible");
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            Algebra::Copy(_engine, _stream, _trial.CoupledField(), Flow());
            _stream.Record().WaitHost(); });
        _fv.Evaluate(_trial, _residual, *_workspace, _engine, _stream, _time);
        _lastFallbacks = _fv.LastFaceFallbacks(*_workspace);
        return _residual;
    }
    std::array<real, 14> Domain::Diagnostics(const DomainSettings &settings, const std::vector<int> &forceZones)
    {
        EvaluateObservation();
        std::array<real, 14> result{};
        result[13] = std::numeric_limits<real>::max();
        result[2] = result[3] = std::numeric_limits<real>::max();
        {
            const auto geometry = _fv.Metrics().AcquireRead();
            const auto metrics = geometry.HostView();
            auto state = Flow().Owned().Read(), rhs = _observationResidual->Owned().Read();
            auto u = state.View<R::UniformTableRows>(), r = rhs.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < u.RowCount(); ++iCell)
            {
                FlowState value{};
                for (int j = 0; j < 5; ++j)
                    value[j] = u(iCell, j);
                const real volume = metrics.GetCellVol(iCell);
                result[0] += volume * value[0];
                result[1] += volume * value[4];
                result[2] = std::min(result[2], value[0]);
                result[3] = std::min(result[3], _equation.Gas().Pressure(value));
                result[13] = std::min(result[13], _equation.Gas().Primitive(value).temperature);
                for (int j = 0; j < 5; ++j)
                    result[4] += r(iCell, j) * r(iCell, j) / volume;
            }
        }
        const auto comm = _fv.Metrics().Communicator();
        MPI_Allreduce(MPI_IN_PLACE, result.data(), 2, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, result.data() + 2, 2, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, result.data() + 4, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, result.data() + 13, 1, MPI_DOUBLE, MPI_MIN, comm);
        result[4] = std::sqrt(result[4]);
        const auto force = _fv.LastBoundaryForce(*_observationWorkspace, forceZones);
        for (int d = 0; d < 3; ++d)
            result[5 + d] = force[d];
        result[8] = _fv.LastReconstructionIterations(*_observationWorkspace);
        result[9] = _fv.LastReconstructionUpdate(*_observationWorkspace);
        if (settings.initialProfile.kind == "isentropicVortex")
        {
            const auto error = _fv.Error(Flow(), settings.initialProfile, settings.initialPrimitive, _time);
            for (int i = 0; i < 3; ++i)
                result[10 + i] = error[i][0];
        }
        return result;
    }
    void Domain::WriteVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        EvaluateObservation();
        _fv.WriteVisualization(Flow(), *_observationWorkspace, prefix, _time, hdfDeflateLevel);
    }
    void Domain::EvaluateObservation()
    {
        // Output cadence must not change the warm-started numerical trajectory.
        // Observation owns its own persistent reconstruction scratch and RHS.
        if (!_observationWorkspace)
        {
            _observationWorkspace = _fv.BuildWorkspace(_accepted);
            _observationResidual.emplace(_fv.Metrics().BuildField(Flow().Owned().Layout()));
            if (_interfaceInputs)
                _fv.SetBoundaryInputs(*_observationWorkspace, *_interfaceInputs);
        }
        // Start every observation from the accepted numerical guess, not the
        // last observation and not a cold zero polynomial. This preserves
        // output-cadence independence while avoiding a much less converged
        // boundary reconstruction than the one used by the implicit stage.
        // A pending/rejected trial owns a different polynomial; its backup is
        // the accepted guess until publication, just as for checkpoint writing.
        const auto &guess = _restoreReconstruction ? *_reconstructionBackup : _fv.ReconstructionGuess(*_workspace).Owned();
        const auto *history = _restoreReconstruction ? (_limitedVRBackup ? &*_limitedVRBackup : nullptr) : _fv.LimitedVRHistory(*_workspace);
        _fv.RestoreReconstructionGuess(*_observationWorkspace, guess, history);
        _fv.Evaluate(_accepted, *_observationResidual, *_observationWorkspace, _engine, _stream, _time);
    }
}
