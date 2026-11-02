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
        Algebra::DofBundle BuildState(const FiniteVolume &discrete, const DomainSettings &settings)
        {
            const auto &fv = discrete.Metrics();
            auto field = fv.BuildField(FlowLayout(settings));
            Algebra::DofBundle result(field.Space());
            discrete.Initialize(field, settings.initialProfile, settings.initialPrimitive);
            result.Add("flow", std::move(field));
            return result;
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
          _accepted(BuildState(_fv, settings)), _trial(_accepted.Clone()),
          _residual(_fv.Metrics().BuildField(FlowLayout(settings))), _initial(_fv.Metrics().BuildField(FlowLayout(settings))),
          _workspace(_fv.BuildWorkspace(_trial.Block("flow"), TimeAuxiliary(time))),
          _engine(size_t(settings.execution.hostThreads), settings.execution.checkAccess ? std::make_shared<R::AccessTrackingContext>() : nullptr),
          _stream(_engine.CreateCPUStream())
    {
        _positivity = settings.numerics.positivity;
        _equation.ValidateStateSchema(_accepted);
        DNDS_check_throw_info(_fv.Metrics().Dimension() == settings.mesh.dimension, "Cauchy config/mesh dimension differs");
        DNDS_check_throw_info(_fv.Admissible(Flow()), "Cauchy initial means violate density or minimum-temperature internal-energy bounds");
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
                        { _fv.PrepareAuxValues(Flow(), *_timeAuxiliary, _time); });
        return _fv.StableStep(Flow(), *_timeAuxiliary, cfl, _time);
    }
    void Domain::UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime)
    {
        _fv.Evaluate(_trial.Block("flow"), _residual, *_workspace, _engine, _stream, stageTime);
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
                const auto initialRead = _initial.Owned().Read(), stateRead = _trial.Block("flow").Owned().Read();
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
                } }, R::ReadAs<R::UniformTableRows>(_initial.Owned()), R::ReadAs<R::UniformTableRows>(_residual.Owned()), R::ReadWriteAs<R::UniformTableRows>(_trial.Block("flow").Owned()));
        _stream.Record().WaitHost();
        DNDS_check_throw_info(_fv.Admissible(_trial.Block("flow")), "Cauchy SSPRK3 stage is inadmissible; accepted state unchanged");
    }
    void Domain::PrepareStep(real dt)
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
        // Scheduled reads may coexist with a retained accepted-state reader.
        // Immediate algebra's stronger idle-input contract would reject only
        // the retaining rank and strand peers at the next domain collective.
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            Algebra::Copy(_engine, _stream, _trial.Block("flow"), Flow());
            Algebra::Copy(_engine, _stream, _initial, Flow());
            _stream.Record().WaitHost(); });
        _lastFallbacks = 0;
        _lastAlpha = _lastBeta = 1;
        _lastAlphaCells = _lastBetaCells = 0;
        _lastCharacteristicFallbacks = 0;
        _trialEpoch = _epoch;
        _trialEpoch.trial = ++_nextTrial;
        // The polynomial guess is numerical state for explicit fixed-sweep VR
        // too. Preserve it before either time driver can overwrite the trial.
        PreserveReconstructionGuess();
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
    }
    void Domain::PublishTrial(real time, StateEpoch epoch, const R::Array<real> *reconstructionGuess)
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
                            { destination.emplace(Flow().Owned().Write()); source.emplace(_trial.Block("flow").Owned().Read()); });
            // Restore only after accepted-state publication is known to be
            // available. A retained debug reader must not change the future
            // polynomial guess when it rejects the accompanying state load.
            if (reconstructionGuess)
                _fv.RestoreReconstructionGuess(*_workspace, *reconstructionGuess);
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
            Algebra::Copy(_engine, _stream, _trial.Block("flow"), Flow());
            _stream.Record().WaitHost(); });
        _fv.Evaluate(_trial.Block("flow"), _residual, *_workspace, _engine, _stream, _time);
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
            _observationWorkspace = _fv.BuildWorkspace(Flow());
            _observationResidual.emplace(_fv.Metrics().BuildField(Flow().Owned().Layout()));
        }
        // Start every observation from the accepted numerical guess, not the
        // last observation and not a cold zero polynomial. This preserves
        // output-cadence independence while avoiding a much less converged
        // boundary reconstruction than the one used by the implicit stage.
        // A pending/rejected trial owns a different polynomial; its backup is
        // the accepted guess until publication, just as for checkpoint writing.
        const auto &guess = _restoreReconstruction ? *_reconstructionBackup : _fv.ReconstructionGuess(*_workspace).Owned();
        _fv.RestoreReconstructionGuess(*_observationWorkspace, guess);
        _fv.Evaluate(Flow(), *_observationResidual, *_observationWorkspace, _engine, _stream, _time);
    }
}
