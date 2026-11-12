#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/State/CPU/FieldKernels.hpp"
#include "Cauchy/State/CPU/StatePropertiesKernels.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/StateInput.hpp"
#include "Cauchy/Physics/EquationSource.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        DomainSettings SpatialSettings(const DomainSettings &settings, const TimeSettings &time)
        {
            time.ValidateSourceIntegration();
            auto result = settings;
            if (time.sourceSplitting == "strang" && settings.equation.source == "chemical")
                result.equation.source = "none";
            return result;
        }
        EquationSetup ResolveEquation(const std::string &name, const DomainSettings &settings, const MPIInfo &mpi)
        {
            std::optional<EquationSetup> setup;
            MPI::LocalStage(mpi, [&]
                            {
                ResolvedExecutionPlan::Resolve({{name, {0}}, settings});
                // Current EOS phases share one ordered CPU lane, even when
                // its launcher runs on worker threads. Reserve actual required
                // capacity rather than cloning once per engine worker.
                setup.emplace(EquationSetup::Resolve(settings.equation, 1));
                index width = 0;
                for (const auto &block : setup->equation.Schema().blocks) width += index(block.components.size());
                settings.implicit.ValidateComponents(width);
                setup->equation.Schema().ResolveComponents(settings.implicit.residualComponents); });
            return std::move(*setup);
        }
        ResolvedPhysicalInputs ResolveInputs(const DomainSettings &settings, const EquationSetup &setup, const MPIInfo &mpi)
        {
            std::optional<ResolvedPhysicalInputs> result;
            MPI::LocalStage(mpi, [&]
                            {
                settings.execution.ValidateMPIThreadSupport(MPI::GetMPIThreadLevel());
                result = ResolvePhysicalStateInputs(settings, setup.equation, setup.workers); });
            return std::move(*result);
        }
    }
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time)
        : Domain(name, mesh, settings, time, ResolveEquation(name, settings, mesh->getMPI())) {}
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time, EquationSetup setup)
        : Domain(name, mesh, ResolveInputs(settings, setup, mesh->getMPI()), time, setup) {}
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const ResolvedPhysicalInputs &inputs, const TimeSettings &time, EquationSetup setup)
        : _problem(std::move(name), mesh, inputs.settings, time,
                   std::move(setup.equation)),
          _state(_problem, inputs.settings, setup.workers,
                 inputs.initialConservative),
          _execution(_problem, _state, inputs.settings.execution,
                     std::move(setup.workers))
    {
        // All spatial operators, including coarse FAS and approximate
        // Jacobians, see the same source-free transport profile when splitting.
        // Physical config/material metadata still describe the chemical PDE.
        const auto settings = SpatialSettings(inputs.settings, time);
        _sourceStrang = time.sourceSplitting == "strang" && inputs.settings.equation.source == "chemical";
        _reactorControls = time.reactor;
        _positivity = settings.numerics.positivity;
        _problem.equation.ValidateStateSchema(_state.accepted);
        DNDS_check_throw_info(_problem.finiteVolume.Metrics().Dimension() == settings.mesh.dimension, "Cauchy config/mesh dimension differs");
        _problem.admissibility.emplace(_problem.finiteVolume.PrepareAdmissibility(_execution.ChemicalWorkers()));
        DNDS_check_throw_info(Admissible(_state.accepted), "Cauchy initial means violate density or minimum-temperature internal-energy bounds");
        ConfigureTimeStep(time, settings.implicit.positivity.compressionPower);
        InitializeImplicit(settings, time, mesh);
    }
    bool Domain::Admissible(const EquationDofs &state) const
    {
        try
        {
            DNDS_assert(_problem.admissibility);
            return _problem.finiteVolume.Admissible(state, *_problem.admissibility);
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy domain operation failed");
            throw;
        }
    }
    bool Domain::RefreshDerived(DerivedFields &fields, bool onlyIfStale) const
    {
        const auto &workers = _execution.ChemicalWorkers();
        return workers ? RefreshDerivedFields(_state.accepted, _problem.equation, fields, workers->Acquire(), onlyIfStale)
                       : RefreshDerivedFields(_state.accepted, _problem.equation, fields, onlyIfStale);
    }
    real Domain::StableStep(real cfl) const
    {
        AuxiliaryInputs inputs;
        if (_execution.interfaceInputs)
            inputs.boundary.Emplace(*_execution.interfaceInputs);
        if (!_execution.timeAuxiliary)
        {
            AuxiliaryRequest request;
            request.meanThermodynamics = request.cellSpectralSum = true;
            std::optional<AuxiliaryValues> prepared;
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            { prepared.emplace(_problem.finiteVolume.BuildAuxiliaryValues(request)); });
            std::vector<R::TransferBinding> halos;
            for (index iField = 0; iField < _state.accepted.PhysicalFieldNumber(); ++iField)
            {
                auto input = _state.accepted.PhysicalField(iField);
                halos.push_back(_problem.finiteVolume.BindHalo(input));
            }
            _execution.timeMeanHalo = std::move(halos);
            _execution.timeAuxiliary = std::move(prepared);
        }
        if (_problem.equation.HasRANS() && _problem.equation.RANS().ComponentNumber() == 2)
        {
            auto state = _state.accepted.NumericAlias();
            if (!_execution.timeMeanGradients)
                _execution.timeMeanGradients.emplace(_problem.finiteVolume.BuildMeanGradients(state));
            _problem.finiteVolume.PrepareAuxValues(state, *_execution.timeAuxiliary, *_execution.timeMeanGradients,
                                                   _execution.Engine(), _execution.PrimaryStream(), _state.time,
                                                   _execution.ChemicalWorkers(), inputs);
            return _problem.finiteVolume.StableStep(state, *_execution.timeAuxiliary, cfl, _state.time);
        }
        std::vector<R::TransferOperation> transfers;
        for (const auto &halo : _execution.timeMeanHalo)
            transfers.push_back(_execution.Engine().ChecksAccess()
                                    ? halo.BeginChecked(_execution.PrimaryStream())
                                    : halo.BeginScheduled(_execution.PrimaryStream()));
        for (auto &transfer : transfers)
            transfer.Wait();
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        { _problem.finiteVolume.PrepareAuxValues(_state.accepted, *_execution.timeAuxiliary, _state.time,
                                                                 _execution.ChemicalWorkers(), inputs); });
        return _problem.finiteVolume.StableStep(_state.accepted, *_execution.timeAuxiliary, cfl, _state.time);
    }
    void Domain::BeginTrial(real dt)
    {
        _state.trialReady = false;
        int valid = std::isfinite(dt) && dt > 0 && (_steady || (std::isfinite(_state.time + dt) && _state.time + dt > _state.time));
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _problem.finiteVolume.Metrics().Communicator());
        DNDS_check_throw_info(valid, "invalid Cauchy physical timestep");
        real min = dt, max = dt;
        min = MPI::AllreduceValue<MPI::Reduction::Min>(min, _problem.finiteVolume.Metrics().Communicator());
        max = MPI::AllreduceValue<MPI::Reduction::Max>(max, _problem.finiteVolume.Metrics().Communicator());
        DNDS_check_throw_info(min == max, "Cauchy timestep differs between domain ranks");
        DNDS_check_throw_info(Admissible(_state.accepted), "accepted Cauchy mean is inadmissible");
        // Owned read scopes may coexist with a retained accepted-state reader.
        // Drain earlier engine work, then copy every logical segment through
        // the same synchronous CPU state adapter used by the time driver.
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary();
            EquationDofOperations::Copy(_state.trial, _state.accepted);
            EquationDofOperations::Copy(_state.initial, _state.accepted); });
        _lastFallbacks = 0;
        _lastAlpha = _lastBeta = 1;
        _lastAlphaCells = _lastBetaCells = 0;
        _lastCharacteristicFallbacks = 0;
        _state.trialEpoch = _state.epoch;
        _state.trialDt = _steady ? 0 : dt;
        _state.trialEpoch.trial = ++_state.nextTrial;
        // The polynomial guess is numerical state for explicit fixed-sweep VR
        // too. Preserve it before either time driver can overwrite the trial.
        PreserveReconstructionGuess();
        if (_sourceStrang)
        {
            ChemicalHalfStep(_state.accepted, _state.trial, .5 * dt);
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            { EquationDofOperations::Copy(_state.initial, _state.trial); });
            // Euler's physical split does not reuse multistep transport
            // history across source jumps. Accepted split history is empty.
            if (_state.timeStepper)
                _state.timeStepper->ResetHistory();
        }
    }
    void Domain::ChemicalHalfStep(const EquationDofs &input, EquationDofs &output, real dt)
    {
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary();
            const auto &workers = _execution.ChemicalWorkers();
            DNDS_check_throw_info(_problem.equation.HasMixture() && workers, "chemical source step requires its explicit mixture worker");
            AdvanceChemicalSource(_problem.equation, input, output, workers->Acquire(), dt, _reactorControls); });
    }
    void Domain::FinishChemicalStep()
    {
        if (!_sourceStrang)
            return;
        // The transport method has finished using its initial state. Reuse
        // that allocation as a distinct source output, then copy into trial.
        ChemicalHalfStep(_state.trial, _state.initial, .5 * _state.trialDt);
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        { EquationDofOperations::Copy(_state.trial, _state.initial); });
        DNDS_check_throw_info(Admissible(_state.trial), "chemical source candidate is inadmissible; accepted state unchanged");
    }
    StepResult Domain::PrepareStep(real dt)
    {
        DNDS_check_throw_info(!_problem.finiteVolume.HasInterfaces(), "interface domains must advance through their coordinated Simulation");
        if (_state.internalResume)
        {
            int valid = _steady || dt == _state.trialDt;
            valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _problem.finiteVolume.Metrics().Communicator());
            DNDS_check_throw_info(valid, "resumed exact internal step requires its stored physical dt");
            if (auto result = ResumeImplicitStep(); !result)
                return result;
            FinishChemicalStep();
            _state.trialTime = _steady ? _state.time : _state.time + _state.trialDt;
            _state.trialReady = true;
            _state.internalResume = false;
            return {};
        }
        BeginTrial(dt);
        if (_state.timeStepper)
        {
            if (auto result = PrepareImplicitStep(dt); !result)
                return result;
        }
        else
        {
            // Shu-Osher SSPRK(3,3), including its non-monotone physical stage times.
            _state.trialEpoch.stage = 0;
            if (auto result = UpdateStage(dt, 0, 1, _state.time); !result)
                return result;
            _state.trialEpoch.stage = 1;
            if (auto result = UpdateStage(dt, .75, .25, _state.time + dt); !result)
                return result;
            _state.trialEpoch.stage = 2;
            if (auto result = UpdateStage(dt, 1. / 3, 2. / 3, _state.time + .5 * dt); !result)
                return result;
        }
        FinishChemicalStep();
        _state.trialTime = _steady ? _state.time : _state.time + dt;
        _state.trialReady = true;
        return {};
    }
    int Domain::CoordinatedStageNumber() const { return _state.timeStepper ? _state.timeStepper->StageNumber() : 3; }
    void Domain::BindInterfaceInputs(const BoundaryInputs &input)
    {
        // Local validated aliases only. This must stay free of MPI so cleanup
        // can restore an already accepted packet without another collective.
        _problem.finiteVolume.SetBoundaryInputs(*_execution.workspace, input);
        if (_execution.observationWorkspace)
            _problem.finiteVolume.SetBoundaryInputs(*_execution.observationWorkspace, input);
        if (_execution.timePPWorkspace)
            _problem.finiteVolume.SetBoundaryInputs(*_execution.timePPWorkspace, input);
        _execution.interfaceInputs = input;
        if (_execution.multigrid)
            _execution.multigrid->SetBoundaryInputs(input);
    }
    void Domain::BeginCoordinatedStep(real dt)
    {
        BeginTrial(dt);
        _lastImplicit.clear();
        _lastCoupledIterations = 0;
        _lastCoupledTargetRatio = 0;
        _state.coordinatedStage = 0;
        _state.coordinatedPrepared = false;
        _state.stageEquation.reset();
        for (auto &input : _execution.coupledInterfaceInputs)
            input.reset();
        if (_state.timeStepper)
        {
            if (_state.timeStepper->SupportsStageSequence())
                _state.timeStepper->BeginStages(
                    _state.initial, _state.trial, _state.time, dt, _steady);
            else
            {
                const auto evaluate = [this](EquationDofs &state,
                                             EquationDofs &rate, real time,
                                             int stage)
                {
                    EvaluateImplicitRate(state, rate, *_execution.workspace,
                                         time, stage);
                };
                _state.timeStepper->BeginCoupled(
                    _state.initial, _state.trial, _state.time, dt, evaluate);
                PrepareCoupledImplicit(_state.timeStepper->CoupledMethod(),
                                       false);
            }
        }
    }
    bool Domain::CoordinatedStepIsCoupled() const
    {
        return _state.timeStepper &&
               !_state.timeStepper->SupportsStageSequence();
    }
    real Domain::CoupledStageTime(int coupledStage) const
    {
        DNDS_check_throw_info(CoordinatedStepIsCoupled(),
                              "domain has no active coupled stage group");
        return _state.timeStepper->CoupledMethod().CoupledStageTime(
            coupledStage);
    }
    void Domain::PrepareCoupledInterface(int coupledStage, bool reconstructed)
    {
        DNDS_check_throw_info(CoordinatedStepIsCoupled() &&
                                  (coupledStage == 0 || coupledStage == 1) &&
                                  _coupledMiddle,
                              "invalid coupled interface preparation");
        auto &method = _state.timeStepper->CoupledMethod();
        auto &state = coupledStage == 0 ? method.IntermediateState()
                                        : _state.trial;
        auto &reconstruction = coupledStage == 0 ? *_coupledMiddle->state
                                                 : *_state.reconstruction;
        _problem.finiteVolume.Execute(
            state, reconstruction, *_execution.workspace,
            _execution.Context(),
            reconstructed
                ? FiniteVolumeEvaluation::Reconstruction(
                      method.CoupledStageTime(coupledStage))
                : FiniteVolumeEvaluation::Sampling());
    }
    FiniteVolume::InterfaceDonorPublication
    Domain::PublishCoupledInterfaceDonor(int coupledStage, bool reconstructed)
    {
        DNDS_check_throw_info(CoordinatedStepIsCoupled() &&
                                  (coupledStage == 0 || coupledStage == 1) &&
                                  _coupledMiddle,
                              "invalid coupled interface publication");
        auto &method = _state.timeStepper->CoupledMethod();
        auto &state = coupledStage == 0 ? method.IntermediateState()
                                        : _state.trial;
        auto &reconstruction = coupledStage == 0 ? *_coupledMiddle->state
                                                 : *_state.reconstruction;
        return _problem.finiteVolume.PublishInterfaceDonor(
            state, reconstruction, *_execution.workspace, reconstructed);
    }
    void Domain::BindCoupledInterfaceInputs(int coupledStage,
                                            const BoundaryInputs &input)
    {
        DNDS_check_throw_info(coupledStage == 0 || coupledStage == 1,
                              "invalid coupled interface slot");
        _execution.coupledInterfaceInputs[size_t(coupledStage)] =
            _problem.finiteVolume.FreezeBoundaryInputs(input);
        ActivateCoupledInterface(coupledStage);
    }
    void Domain::ActivateCoupledInterface(int coupledStage)
    {
        DNDS_check_throw_info(coupledStage == 0 || coupledStage == 1,
                              "invalid coupled interface activation");
        const auto &input =
            _execution.coupledInterfaceInputs[size_t(coupledStage)];
        if (_problem.finiteVolume.HasInterfaces())
            DNDS_check_throw_info(input,
                                  "coupled stage has no frozen interface packet");
        if (!input)
            return;
        BindInterfaceInputs(*input);
        if (coupledStage == 0 && _coupledMiddle &&
            _coupledMiddle->bindBoundary)
            _coupledMiddle->bindBoundary(*input);
    }
    StepResult Domain::SolveCoordinatedCoupledStep()
    {
        DNDS_check_throw_info(CoordinatedStepIsCoupled(),
                              "domain has no begun coupled stage group");
        const bool accepted = _state.timeStepper->RunCoupled(
            _state.trial, [this](auto &method, auto &endpoint)
            { return SolveCoupledImplicit(method, endpoint, false); });
        if (!accepted)
            return StepResult::Rejected(
                "coordinated DITR stage group reached its correction cap");
        FinishChemicalStep();
        if (!Admissible(_state.trial))
            return StepResult::Rejected(
                "coordinated DITR candidate is inadmissible");
        _state.trialTime = _state.time + _state.trialDt;
        _state.trialReady = true;
        return {};
    }
    real Domain::PrepareCoordinatedStage(int stage)
    {
        DNDS_check_throw_info(stage == _state.coordinatedStage && stage < CoordinatedStageNumber() && !_state.coordinatedPrepared,
                              "coordinated domain stage is out of order");
        _state.trialEpoch.stage = stage;
        _state.coordinatedPrepared = true;
        if (_state.timeStepper)
        {
            _state.stageEquation.emplace(_state.timeStepper->PrepareStage(_state.trial));
            return _state.stageEquation->time;
        }
        static constexpr real c[]{0, 1, .5};
        return _state.time + c[stage] * _state.trialDt;
    }
    StepResult Domain::SolveCoordinatedStage()
    {
        DNDS_check_throw_info(_state.coordinatedPrepared, "coordinated domain stage was not prepared");
        if (_state.stageEquation)
        {
            const auto &equation = *_state.stageEquation;
            if (equation.implicit)
            {
                const bool accepted = _state.timeStepper->SolvePreparedStage(equation, _state.trial,
                                                                             [&](EquationDofs &trial, const EquationDofs &source, real mass, real rhsScale, real time, int stage)
                                                                             {
                                                                                 _state.progressTime = time;
                                                                                 SetTimeStage(stage);
                                                                                 return RecordImplicitStage(
                                                                                     _execution.multigrid
                                                                                         ? _execution.multigrid->SolveWithController(
                                                                                               *_execution.nonlinearController,
                                                                                               source, mass, rhsScale, time,
                                                                                               stage, ProgressControl())
                                                                                         : _execution.implicit->SolveWithController(
                                                                                               *_execution.nonlinearController,
                                                                                               trial, source, mass, rhsScale,
                                                                                               time, stage, ProgressControl()));
                                                                             });
                if (!accepted)
                    return StepResult::Rejected("coordinated implicit stage reached its correction cap");
            }
        }
        else
        {
            static constexpr real history[]{0, .75, 1. / 3}, weight[]{1, .25, 2. / 3}, c[]{0, 1, .5};
            return UpdateStage(_state.trialDt, history[_state.coordinatedStage], weight[_state.coordinatedStage], _state.time + c[_state.coordinatedStage] * _state.trialDt);
        }
        return {};
    }
    StepResult Domain::CompleteCoordinatedStage()
    {
        DNDS_check_throw_info(_state.coordinatedPrepared, "no coordinated stage to complete");
        if (_state.timeStepper)
            _state.timeStepper->CompleteStage(_state.trial, [this](auto &state, auto &rate, real time, int stage)
                                              { EvaluateImplicitRate(state, rate, *_execution.workspace, time, stage); });
        _state.coordinatedPrepared = false;
        _state.stageEquation.reset();
        if (++_state.coordinatedStage == CoordinatedStageNumber())
        {
            FinishChemicalStep();
            if (!Admissible(_state.trial))
                return StepResult::Rejected("coordinated candidate is inadmissible");
            _state.trialTime = _steady ? _state.time : _state.time + _state.trialDt;
            _state.trialReady = true;
        }
        return {};
    }
    void Domain::CommitStep()
    {
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        { DNDS_check_throw_info(_state.trialReady && (!_state.timeStepper || _state.timeStepper->HasPendingTrial()), "no prepared Cauchy candidate"); });
        auto epoch = _state.trialEpoch;
        ++epoch.acceptedStep;
        epoch.stage = 0;
        epoch.coupledStage = 0;
        PublishTrial(_state.trialTime, epoch);
        if (_state.timeStepper)
        {
            _state.timeStepper->Commit();
            if (_sourceStrang)
                _state.timeStepper->ResetHistory();
        }
        _state.lastAcceptedDt = _state.trialDt;
    }
    void Domain::CommitReserved(EquationDofs::WriteScope &destination, EquationDofs::ReadScope &source)
    {
        // Simulation reserved every domain's destination/source and agreed
        // readiness globally before entering this allocation-free publication.
        for (index iBlock = 0; iBlock < _state.accepted.BlockNumber(); ++iBlock)
        {
            const auto input = source.Block(iBlock);
            const auto output = destination.Block(iBlock);
            CPU::CopyRows(input.owned, output, output.ComponentNumber());
        }
        _state.time = _state.trialTime;
        _state.epoch = _state.trialEpoch;
        ++_state.epoch.acceptedStep;
        _state.epoch.stage = 0;
        _state.epoch.coupledStage = 0;
        _state.trialReady = _state.restoreReconstruction = false;
        if (_state.timeStepper)
        {
            _state.timeStepper->Commit();
            if (_sourceStrang)
                _state.timeStepper->ResetHistory();
        }
        _state.lastAcceptedDt = _state.trialDt;
    }
    void Domain::PublishTrial(real time, StateEpoch epoch)
    {
        // Agree local immediate-access availability before any accepted values
        // change. A retained debug/Python view must not allow a partial commit
        // on only the other ranks. Keep this lease through the actual copy.
        std::optional<EquationDofs::WriteScope> destination;
        std::optional<EquationDofs::ReadScope> source;
        const auto comm = _problem.finiteVolume.Metrics().Communicator();
        MPIInfo mpi(comm);
        try
        {
            MPI::LocalStage(mpi, [&]
                            { destination.emplace(_state.accepted.WriteOwned()); source.emplace(_state.trial.ReadOwned()); });
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy domain operation failed");
            // Successful ranks acquired but never used their destination.
            // Cancel before unwinding; a normal exceptional write destructor
            // deliberately poisons possibly partial writes.
            if (destination)
                destination->CancelUnmodified();
            throw;
        }
        for (index iBlock = 0; iBlock < _state.accepted.BlockNumber(); ++iBlock)
        {
            const auto input = source->Block(iBlock);
            const auto output = destination->Block(iBlock);
            CPU::CopyRows(input.owned, output, output.ComponentNumber());
        }
        _state.time = time;
        _state.epoch = epoch;
        _state.trialReady = false;
        _state.restoreReconstruction = false;
    }
    const EquationDofs &Domain::Residual()
    {
        _state.trialReady = false;
        DNDS_check_throw_info(Admissible(_state.accepted), "accepted Cauchy mean is inadmissible");
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary(); EquationDofOperations::Copy(_state.trial, _state.accepted); });
        _problem.finiteVolume.Execute(_state.trial, *_state.reconstruction, *_execution.workspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(_state.residual, _state.time));
        _lastFallbacks = _problem.finiteVolume.LastFaceFallbacks(*_execution.workspace);
        _lastCharacteristicFallbacks = _problem.finiteVolume.LastCharacteristicFallbacks(*_execution.workspace);
        return _state.residual;
    }
    std::array<real, 14> Domain::Diagnostics(const DomainSettings &settings, const std::vector<int> &forceZones)
    {
        const auto snapshot = CaptureStateSnapshot(true);
        EvaluateObservation(snapshot);
        std::array<real, 14> result{};
        result[13] = std::numeric_limits<real>::max();
        result[2] = result[3] = std::numeric_limits<real>::max();
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            const auto &workers = _execution.ChemicalWorkers();
            auto evaluator = _problem.equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto geometry = _problem.finiteVolume.Metrics().AcquireRead(); const auto metric = geometry.HostView();
            const auto read = snapshot.Means().ReadOwned(), residual = _execution.observationState->ReadOwned();
            const auto flow = read.Block("flow");
            const auto layout = residual.ComponentLayout(); const auto rhs = layout.View();
            const int independent = _problem.equation.HasMixture() ? _problem.equation.Mixture().SpeciesNumber() - 1 : 0;
            std::optional<EquationBlockReadView> species;
            if (independent) species.emplace(read.Block("species"));
            const index width = snapshot.Means().ComponentNumber();
            RequireNumericalSuccess(CPU::AccumulateStateDiagnostics(flow, species, rhs, metric, evaluator, {result.data(), index(result.size())}), "AccumulateStateDiagnostics"); });
        const auto comm = _problem.finiteVolume.Metrics().Communicator();
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(
            MPI::ContiguousView<real>{result.data(), 2}, comm);
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{result.data() + 2, 2}, comm);
        result[4] = MPI::AllreduceValue<MPI::Reduction::Sum>(result[4], comm);
        result[13] = MPI::AllreduceValue<MPI::Reduction::Min>(result[13], comm);
        // Report the volume-integrated L2 defect, matching Euler's res0
        // convention rather than a volume-averaged residual.
        result[4] = std::sqrt(result[4]);
        const auto force = _problem.finiteVolume.LastBoundaryForce(*_execution.observationWorkspace, forceZones);
        for (int d = 0; d < 3; ++d)
            result[5 + d] = force[d];
        result[8] = _problem.finiteVolume.LastReconstructionIterations(*_execution.observationWorkspace);
        result[9] = _problem.finiteVolume.LastReconstructionUpdate(*_execution.observationWorkspace);
        if (!_problem.equation.HasMixture() && !_problem.equation.HasRANS() && settings.initialProfile.kind == "isentropicVortex")
        {
            const auto error = _problem.finiteVolume.Error(snapshot.Means(), settings.initialProfile, settings.initialPrimitive, snapshot.Time());
            for (int i = 0; i < 3; ++i)
                result[10 + i] = error[i][0];
        }
        return result;
    }
    void Domain::WriteVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        const auto snapshot = CaptureStateSnapshot(true);
        EvaluateObservation(snapshot);
        _problem.finiteVolume.WriteVisualization(*_execution.observationInput, *_execution.observationReconstruction, *_execution.observationWorkspace, prefix, snapshot.Time(), hdfDeflateLevel, _execution.ChemicalWorkers());
    }
    void Domain::WriteInternalVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        const MPIInfo mpi(_problem.finiteVolume.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(_state.timeStepper && _state.trialEpoch.trial &&
                                                    (!_problem.finiteVolume.PolynomialReconstruction() || _state.restoreReconstruction) &&
                                                    std::isfinite(_state.progressTime),
                                                "Cauchy internal visualization has no active implicit trial"); });
        if (!_execution.observationWorkspace)
        {
            MPI::LocalStage(mpi, [&]
                            {
                _execution.observationInput.emplace(EquationDofOperations::NewVector(_state.trial));
                _execution.observationState.emplace(EquationDofOperations::NewVector(_state.trial)); });
            _execution.observationWorkspace = _problem.finiteVolume.BuildWorkspace(*_execution.observationInput, _execution.ChemicalWorkers());
            _execution.observationReconstruction = _problem.finiteVolume.BuildReconstructionState(*_execution.observationInput);
            if (_execution.interfaceInputs)
                _problem.finiteVolume.SetBoundaryInputs(*_execution.observationWorkspace, *_execution.interfaceInputs);
        }
        MPI::LocalStage(mpi, [&]
                        {
            _execution.JoinPrimary();
            EquationDofOperations::Copy(*_execution.observationInput, _state.trial); });
        if (_problem.finiteVolume.PolynomialReconstruction())
            _problem.finiteVolume.RestoreReconstructionGuess(*_execution.observationReconstruction,
                                                             _problem.finiteVolume.ReconstructionGuess(*_state.reconstruction));
        _problem.finiteVolume.Execute(*_execution.observationInput, *_execution.observationReconstruction,
                                      *_execution.observationWorkspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(*_execution.observationState, _state.progressTime));
        _problem.finiteVolume.WriteVisualization(*_execution.observationInput, *_execution.observationReconstruction, *_execution.observationWorkspace, prefix, _state.progressTime,
                                                 hdfDeflateLevel, _execution.ChemicalWorkers());
    }
    DomainStateSnapshot Domain::CaptureStateSnapshot(
        bool includeReconstruction) const
    {
        return DomainStateSnapshot(*this, includeReconstruction);
    }
    void Domain::EvaluateObservation(const DomainStateSnapshot &snapshot)
    {
        if (!_execution.observationWorkspace)
        {
            std::optional<EquationDofs> input;
            std::optional<EquationDofs> residual;
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            {
                input.emplace(EquationDofOperations::NewVector(snapshot.Means()));
                residual.emplace(EquationDofOperations::NewVector(snapshot.Means())); });
            auto workspace = _problem.finiteVolume.BuildWorkspace(*input, _execution.ChemicalWorkers());
            auto reconstruction = _problem.finiteVolume.BuildReconstructionState(*input);
            _execution.observationInput = std::move(input);
            _execution.observationState = std::move(residual);
            _execution.observationReconstruction = std::move(reconstruction);
            _execution.observationWorkspace = std::move(workspace);
            if (_execution.interfaceInputs)
                _problem.finiteVolume.SetBoundaryInputs(*_execution.observationWorkspace, *_execution.interfaceInputs);
        }
        // Start every observation from the accepted numerical guess, not the
        // last observation and not a cold zero polynomial. This preserves
        // output-cadence independence while avoiding a much less converged
        // boundary reconstruction than the one used by the implicit stage.
        // A pending/rejected trial owns a different polynomial; its backup is
        // the accepted guess until publication, just as for checkpoint writing.
        MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary();
            EquationDofOperations::Copy(*_execution.observationInput, snapshot.Means()); });
        if (_problem.finiteVolume.PolynomialReconstruction())
        {
            DNDS_check_throw_info(snapshot.Reconstruction(),
                                  "observation snapshot lacks polynomial reconstruction");
            _problem.finiteVolume.RestoreReconstructionGuess(
                *_execution.observationReconstruction,
                *snapshot.Reconstruction());
        }
        _problem.finiteVolume.Execute(*_execution.observationInput, *_execution.observationReconstruction,
                                      *_execution.observationWorkspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(*_execution.observationState, snapshot.Time()));
    }
}
