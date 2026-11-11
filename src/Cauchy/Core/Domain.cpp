#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/StepResult.hpp"
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
        ssp<Geom::UnstructuredMesh> PrepareWallField(ssp<Geom::UnstructuredMesh> mesh, const EquationSystem &equation,
                                                     const DomainSettings &settings)
        {
            if (!equation.HasRANS())
                return mesh;
            const auto mpi = mesh->getMPI();
            int available = mesh->HasNodeWallDist();
            available = MPI::AllreduceValue<MPI::Reduction::Sum>(available, mpi.comm);
            DNDS_check_throw_info(available == 0 || available == mpi.size, "RANS nodal wall-field presence differs between ranks");
            if (!available)
                mesh->BuildNodeWallDist([&](auto zone)
                                        {
                    const auto found = settings.boundaries.find(std::to_string(zone));
                    return found != settings.boundaries.end() && found->second.kind == "noSlipWall"; });
            return mesh;
        }
        R::RowLayout FlowLayout(const DomainSettings &settings, index width = 5)
        {
            if (settings.execution.layout == "soa")
                return R::RowLayout::ColMajor(0, width);
            if (settings.execution.layout == "aosoa32")
                return R::RowLayout::TiledColMajor(0, width, 32);
            return R::RowLayout::Dense(0, width);
        }
        EquationDofs BuildState(const FiniteVolume &discrete, const EquationSystem &equation, const DomainSettings &settings,
                                const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers, const std::vector<real> &initialConservative)
        {
            const auto &fv = discrete.Metrics();
            if (equation.HasMixture() || equation.HasRANS())
            {
                const auto allocate = [&]
                {
                    if (settings.execution.fieldGrouping == "coupled")
                    {
                        index components = 0;
                        for (const auto &block : equation.Schema().blocks)
                            components += index(block.components.size());
                        return EquationDofs::Coupled(equation.Schema(), fv.BuildField(FlowLayout(settings, components)), "equations");
                    }
                    Algebra::DofBundle fields(fv.Space());
                    for (const auto &block : equation.Schema().blocks)
                        fields.Add(block.name, fv.BuildField(FlowLayout(settings, index(block.components.size()))));
                    return EquationDofs::Separate(equation.Schema(), std::move(fields));
                };
                auto state = allocate();
                if (!initialConservative.empty())
                    discrete.InitializeConservative(state, initialConservative);
                else
                    discrete.InitializeConstant(state, settings.initialPrimitive, settings.initialMassFractions, settings.initialTemperature, workers, settings.initialRANS);
                discrete.InitializeField(state, workers);
                return state;
            }
            auto field = fv.BuildField(FlowLayout(settings));
            if (!initialConservative.empty())
            {
                auto state = EquationDofs::Coupled(equation.Schema(), std::move(field));
                discrete.InitializeConservative(state, initialConservative);
                discrete.InitializeField(state, workers);
                return state;
            }
            auto primitive = settings.initialPrimitive;
            if (settings.initialTemperature > 0)
                primitive[4] = primitive[0] * equation.Gas().gasConstant * settings.initialTemperature;
            auto state = EquationDofs::Coupled(equation.Schema(), std::move(field));
            discrete.Initialize(state, settings.initialProfile, primitive);
            discrete.InitializeField(state, workers);
            return state;
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
        : _name(std::move(name)), _equation(std::move(setup.equation)),
          _execution(inputs.settings.execution, std::move(setup.workers)),
          _fv(PrepareWallField(mesh, _equation, inputs.settings), SpatialSettings(inputs.settings, time), _equation),
          _accepted(BuildState(_fv, _equation, inputs.settings, _execution.ChemicalWorkers(), inputs.initialConservative)), _trial(_accepted.Clone()),
          _residualState(EquationDofOperations::NewVector(_accepted)), _initial(EquationDofOperations::NewVector(_accepted)),
          _reconstruction(_fv.BuildReconstructionState(_trial)),
          _workspace(_fv.BuildWorkspace(_trial, _execution.ChemicalWorkers()))
    {
        // All spatial operators, including coarse FAS and approximate
        // Jacobians, see the same source-free transport profile when splitting.
        // Physical config/material metadata still describe the chemical PDE.
        const auto settings = SpatialSettings(inputs.settings, time);
        _sourceStrang = time.sourceSplitting == "strang" && inputs.settings.equation.source == "chemical";
        _reactorControls = time.reactor;
        _positivity = settings.numerics.positivity;
        _equation.ValidateStateSchema(_accepted);
        DNDS_check_throw_info(_fv.Metrics().Dimension() == settings.mesh.dimension, "Cauchy config/mesh dimension differs");
        _admissibility.emplace(_fv.PrepareAdmissibility(_execution.ChemicalWorkers()));
        DNDS_check_throw_info(Admissible(_accepted), "Cauchy initial means violate density or minimum-temperature internal-energy bounds");
        ConfigureTimeStep(time, settings.implicit.positivity.compressionPower);
        InitializeImplicit(settings, time, mesh);
    }
    bool Domain::Admissible(const EquationDofs &state) const
    {
        try
        {
            DNDS_assert(_admissibility);
            return _fv.Admissible(state, *_admissibility);
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
        return workers ? RefreshDerivedFields(_accepted, _equation, fields, workers->Acquire(), onlyIfStale)
                       : RefreshDerivedFields(_accepted, _equation, fields, onlyIfStale);
    }
    real Domain::StableStep(real cfl) const
    {
        AuxiliaryInputs inputs;
        if (_interfaceInputs)
            inputs.boundary.Emplace(*_interfaceInputs);
        if (!_timeAuxiliary)
        {
            AuxiliaryRequest request;
            request.meanThermodynamics = request.cellSpectralSum = true;
            std::optional<AuxiliaryValues> prepared;
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            { prepared.emplace(_fv.BuildAuxiliaryValues(request)); });
            std::vector<R::TransferBinding> halos;
            for (index iField = 0; iField < _accepted.PhysicalFieldNumber(); ++iField)
            {
                auto input = _accepted.PhysicalField(iField);
                halos.push_back(_fv.BindHalo(input));
            }
            _timeMeanHalo = std::move(halos);
            _timeAuxiliary = std::move(prepared);
        }
        if (_equation.HasRANS() && _equation.RANS().ComponentNumber() == 2)
        {
            auto state = _accepted.NumericAlias();
            if (!_timeMeanGradients)
                _timeMeanGradients.emplace(_fv.BuildMeanGradients(state));
            _fv.PrepareAuxValues(state, *_timeAuxiliary, *_timeMeanGradients,
                                 _execution.Engine(), _execution.PrimaryStream(), _time,
                                 _execution.ChemicalWorkers(), inputs);
            return _fv.StableStep(state, *_timeAuxiliary, cfl, _time);
        }
        std::vector<R::TransferOperation> transfers;
        for (const auto &halo : _timeMeanHalo)
            transfers.push_back(_execution.Engine().ChecksAccess()
                                    ? halo.BeginChecked(_execution.PrimaryStream())
                                    : halo.BeginScheduled(_execution.PrimaryStream()));
        for (auto &transfer : transfers)
            transfer.Wait();
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        { _fv.PrepareAuxValues(_accepted, *_timeAuxiliary, _time,
                                               _execution.ChemicalWorkers(), inputs); });
        return _fv.StableStep(_accepted, *_timeAuxiliary, cfl, _time);
    }
    StepResult Domain::UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime)
    {
        _fv.Execute(_trial, *_reconstruction, *_workspace, _execution,
                    FiniteVolumeEvaluation::Spatial(_residualState, stageTime));
        _lastFallbacks += _fv.LastFaceFallbacks(*_workspace);
        _lastCharacteristicFallbacks += _fv.LastCharacteristicFallbacks(*_workspace);
        if (_equation.HasMixture() || _equation.HasRANS())
        {
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                const auto geometry = _fv.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                const auto initial = _initial.ReadOwned(), residual = _residualState.ReadOwned();
                auto output = _trial.WriteOwned();
                const auto flow = output.Block("flow");
                for (index iBlock = 0; iBlock < _trial.BlockNumber(); ++iBlock)
                {
                    const auto base = initial.Block(iBlock), rhs = residual.Block(iBlock); const auto state = output.Block(iBlock);
                    const auto &ransState = _admissibility->RANS();
                    const bool rans = ransState.components &&
                                      _trial.Schema().blocks[iBlock].name == "rans";
                    for (index iCell = 0; iCell < state.RowCount(); ++iCell)
                    {
                        if (rans)
                        {
                            // PP POLICY -- explicit extended equations retain
                            // raw flow/species stage increments and reject the
                            // whole step when their accepted mean is invalid.
                            // Euler's scalar-only RANS repair is permitted here.
                            // It must never rescale the flow5/species update.
                            std::array<real, 2> old{}, change{}, next{};
                            for (int j = 0; j < ransState.components; ++j)
                            {
                                old[j] = state(iCell, j);
                                change[j] = historyWeight * base(iCell, j) + stageWeight * (old[j] + dt * rhs(iCell, j) / metric.GetCellVol(iCell)) - old[j];
                                DNDS_check_throw_info(std::isfinite(change[j]), "nonfinite explicit RANS increment");
                            }
                            ransState.RepairIncrement(old, change);
                            for (int j = 0; j < ransState.components; ++j) next[j] = old[j] + change[j];
                            ransState.RepairMean(flow(iCell, 0), next);
                            for (int j = 0; j < ransState.components; ++j) state(iCell, j) = next[j];
                            continue;
                        }
                        // PP POLICY -- no cellwise thermal increment compression:
                        // flow5
                        // and species retain their conservative residuals.
                        // Do not clip/normalize species means here either;
                        // the full raw candidate must pass Admissible below.
                        for (index j = 0; j < state.ComponentNumber(); ++j)
                            state(iCell, j) = historyWeight * base(iCell, j) + stageWeight * (state(iCell, j) + dt * rhs(iCell, j) / metric.GetCellVol(iCell));
                    }
                } });
            if (!Admissible(_trial))
                return StepResult::Rejected("Cauchy explicit extended-equation stage is inadmissible; accepted state unchanged");
            return {};
        }
        auto &residual = _residualState.CoupledField();
        auto geometry = _fv.Metrics().AcquireRead();
        const auto metrics = geometry.HostView();
        real alpha = 1;
        int safe[2]{1, 1}; // high candidate, low candidate
        if (_positivity)
        {
            // PP POLICY -- pure-flow5 compatibility baseline: one domain-global
            // conservative high/low face-flux blend. This is the only explicit
            // thermal compression currently allowed; extended equations retry
            // the step instead of compressing their pointwise increments.
            // Deferred explicit PP work should limit shared face-flux
            // corrections using both adjacent cells, including MPI faces.
            // A cell-local increment factor is not a conservative substitute.
            const auto reconstruction = _fv.PositivityReconstructionStats(*_workspace);
            _lastBetaCells += reconstruction.first;
            _lastBeta = std::min(_lastBeta, reconstruction.second);
            index limited = 0;
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                const auto initialRead = _initial.CoupledField().Owned().Read(), stateRead = _trial.CoupledField().Owned().Read();
                const auto highRead = residual.Owned().Read(), lowRead = _fv.LowOrderResidual(*_workspace).CoupledField().Owned().Read();
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
                    const auto fraction = lowSafe ? AdmissibleFraction(base, target, _fv.ThermalConstraints()) : real(1);
                    if (fraction < 1) { ++limited; alpha = std::min(alpha, fraction); }
                } });
            alpha = MPI::AllreduceValue<MPI::Reduction::Min>(alpha, _fv.Metrics().Communicator());
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<int>{safe, 2}, _fv.Metrics().Communicator());
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
                if (!safe[1])
                    return StepResult::Rejected("global conservative PP blend requires admissible low-order stage; reduce dt");
            }
            limited = MPI::AllreduceValue<MPI::Reduction::Sum>(limited, _fv.Metrics().Communicator());
            _lastAlpha = std::min(_lastAlpha, alpha);
            _lastAlphaCells += limited;
        }
        std::optional<R::Array<real>::ReadLease> lowRead;
        R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows> low;
        if (_positivity)
        {
            lowRead.emplace(_fv.LowOrderResidual(*_workspace).CoupledField().Owned().Read());
            low = lowRead->View<R::UniformTableRows>();
        }
        R::SubmitArrays(_execution.Engine(), _execution.PrimaryStream(), [dt, historyWeight, stageWeight, metrics, alpha, low](auto, auto initial, auto rhs, auto stage)
                        {
            for (index iCell = 0; iCell < stage.rows; ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    const real residual = alpha == 1 ? rhs(iCell, j) : (1 - alpha) * low(iCell, j) + alpha * rhs(iCell, j);
                    stage(iCell, j) = historyWeight * initial(iCell, j) + stageWeight * (stage(iCell, j) + dt * residual / metrics.GetCellVol(iCell));
                } }, R::ReadAs<R::UniformTableRows>(_initial.CoupledField().Owned()), R::ReadAs<R::UniformTableRows>(residual.Owned()), R::ReadWriteAs<R::UniformTableRows>(_trial.CoupledField().Owned()));
        _execution.JoinPrimary();
        if (!Admissible(_trial))
            return StepResult::Rejected("Cauchy SSPRK3 stage is inadmissible; accepted state unchanged");
        return {};
    }
    void Domain::BeginTrial(real dt)
    {
        _trialReady = false;
        int valid = std::isfinite(dt) && dt > 0 && (_steady || (std::isfinite(_time + dt) && _time + dt > _time));
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _fv.Metrics().Communicator());
        DNDS_check_throw_info(valid, "invalid Cauchy physical timestep");
        real min = dt, max = dt;
        min = MPI::AllreduceValue<MPI::Reduction::Min>(min, _fv.Metrics().Communicator());
        max = MPI::AllreduceValue<MPI::Reduction::Max>(max, _fv.Metrics().Communicator());
        DNDS_check_throw_info(min == max, "Cauchy timestep differs between domain ranks");
        DNDS_check_throw_info(Admissible(_accepted), "accepted Cauchy mean is inadmissible");
        // Owned read scopes may coexist with a retained accepted-state reader.
        // Drain earlier engine work, then copy every logical segment through
        // the same synchronous CPU state adapter used by the time driver.
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary();
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
        if (_sourceStrang)
        {
            ChemicalHalfStep(_accepted, _trial, .5 * dt);
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            { EquationDofOperations::Copy(_initial, _trial); });
            // Euler's physical split does not reuse multistep transport
            // history across source jumps. Accepted split history is empty.
            if (_timeStepper)
                _timeStepper->ResetHistory();
        }
    }
    void Domain::ChemicalHalfStep(const EquationDofs &input, EquationDofs &output, real dt)
    {
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary();
            const auto &workers = _execution.ChemicalWorkers();
            DNDS_check_throw_info(_equation.HasMixture() && workers, "chemical source step requires its explicit mixture worker");
            AdvanceChemicalSource(_equation, input, output, workers->Acquire(), dt, _reactorControls); });
    }
    void Domain::FinishChemicalStep()
    {
        if (!_sourceStrang)
            return;
        // The transport method has finished using its initial state. Reuse
        // that allocation as a distinct source output, then copy into trial.
        ChemicalHalfStep(_trial, _initial, .5 * _trialDt);
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        { EquationDofOperations::Copy(_trial, _initial); });
        DNDS_check_throw_info(Admissible(_trial), "chemical source candidate is inadmissible; accepted state unchanged");
    }
    StepResult Domain::PrepareStep(real dt)
    {
        DNDS_check_throw_info(!_fv.HasInterfaces(), "interface domains must advance through their coordinated Simulation");
        if (_internalResume)
        {
            int valid = _steady || dt == _trialDt;
            valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _fv.Metrics().Communicator());
            DNDS_check_throw_info(valid, "resumed exact internal step requires its stored physical dt");
            if (auto result = ResumeImplicitStep(); !result)
                return result;
            FinishChemicalStep();
            _trialTime = _steady ? _time : _time + _trialDt;
            _trialReady = true;
            _internalResume = false;
            return {};
        }
        BeginTrial(dt);
        if (_timeStepper)
        {
            if (auto result = PrepareImplicitStep(dt); !result)
                return result;
        }
        else
        {
            // Shu-Osher SSPRK(3,3), including its non-monotone physical stage times.
            _trialEpoch.stage = 0;
            if (auto result = UpdateStage(dt, 0, 1, _time); !result)
                return result;
            _trialEpoch.stage = 1;
            if (auto result = UpdateStage(dt, .75, .25, _time + dt); !result)
                return result;
            _trialEpoch.stage = 2;
            if (auto result = UpdateStage(dt, 1. / 3, 2. / 3, _time + .5 * dt); !result)
                return result;
        }
        FinishChemicalStep();
        _trialTime = _steady ? _time : _time + dt;
        _trialReady = true;
        return {};
    }
    int Domain::CoordinatedStageNumber() const { return _timeStepper ? _timeStepper->StageNumber() : 3; }
    void Domain::BindInterfaceInputs(const BoundaryInputs &input)
    {
        // Local validated aliases only. This must stay free of MPI so cleanup
        // can restore an already accepted packet without another collective.
        _fv.SetBoundaryInputs(*_workspace, input);
        if (_observationWorkspace)
            _fv.SetBoundaryInputs(*_observationWorkspace, input);
        if (_timePPWorkspace)
            _fv.SetBoundaryInputs(*_timePPWorkspace, input);
        _interfaceInputs = input;
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
            _stageEquation.emplace(_timeStepper->PrepareStage(_trial));
            return _stageEquation->time;
        }
        static constexpr real c[]{0, 1, .5};
        return _time + c[stage] * _trialDt;
    }
    StepResult Domain::SolveCoordinatedStage()
    {
        DNDS_check_throw_info(_coordinatedPrepared, "coordinated domain stage was not prepared");
        if (_stageEquation)
        {
            const auto &equation = *_stageEquation;
            if (equation.implicit)
            {
                const bool accepted = _timeStepper->SolvePreparedStage(equation, _trial,
                                                                       [&](EquationDofs &trial, const EquationDofs &source, real mass, real rhsScale, real time, int stage)
                                                                       {
                                                                           _progressTime = time;
                                                                           SetTimeStage(stage);
                                                                           return RecordImplicitStage(_multigrid ? _multigrid->Solve(source, mass, rhsScale, time, stage, ProgressControl())
                                                                                                                 : _implicit->Solve(trial, source, mass, rhsScale, time, stage, ProgressControl()));
                                                                       });
                if (!accepted)
                    return StepResult::Rejected("coordinated implicit stage reached its correction cap");
            }
        }
        else
        {
            static constexpr real history[]{0, .75, 1. / 3}, weight[]{1, .25, 2. / 3}, c[]{0, 1, .5};
            return UpdateStage(_trialDt, history[_coordinatedStage], weight[_coordinatedStage], _time + c[_coordinatedStage] * _trialDt);
        }
        return {};
    }
    StepResult Domain::CompleteCoordinatedStage()
    {
        DNDS_check_throw_info(_coordinatedPrepared, "no coordinated stage to complete");
        if (_timeStepper)
            _timeStepper->CompleteStage(_trial, [this](auto &state, auto &rate, real time, int stage)
                                        { EvaluateImplicitRate(state, rate, *_workspace, time, stage); });
        _coordinatedPrepared = false;
        _stageEquation.reset();
        if (++_coordinatedStage == CoordinatedStageNumber())
        {
            FinishChemicalStep();
            if (!Admissible(_trial))
                return StepResult::Rejected("coordinated candidate is inadmissible");
            _trialTime = _steady ? _time : _time + _trialDt;
            _trialReady = true;
        }
        return {};
    }
    void Domain::CommitStep()
    {
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        { DNDS_check_throw_info(_trialReady && (!_timeStepper || _timeStepper->HasPendingTrial()), "no prepared Cauchy candidate"); });
        auto epoch = _trialEpoch;
        ++epoch.acceptedStep;
        epoch.stage = 0;
        epoch.coupledStage = 0;
        PublishTrial(_trialTime, epoch);
        if (_timeStepper)
        {
            _timeStepper->Commit();
            if (_sourceStrang)
                _timeStepper->ResetHistory();
        }
        _lastAcceptedDt = _trialDt;
    }
    void Domain::CommitReserved(EquationDofs::WriteScope &destination, EquationDofs::ReadScope &source)
    {
        // Simulation reserved every domain's destination/source and agreed
        // readiness globally before entering this allocation-free publication.
        for (index iBlock = 0; iBlock < _accepted.BlockNumber(); ++iBlock)
        {
            const auto input = source.Block(iBlock);
            const auto output = destination.Block(iBlock);
            for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
                for (index j = 0; j < output.ComponentNumber(); ++j)
                    output(iCell, j) = input(iCell, j);
        }
        _time = _trialTime;
        _epoch = _trialEpoch;
        ++_epoch.acceptedStep;
        _epoch.stage = 0;
        _epoch.coupledStage = 0;
        _trialReady = _restoreReconstruction = false;
        if (_timeStepper)
        {
            _timeStepper->Commit();
            if (_sourceStrang)
                _timeStepper->ResetHistory();
        }
        _lastAcceptedDt = _trialDt;
    }
    void Domain::PublishTrial(real time, StateEpoch epoch)
    {
        // Agree local immediate-access availability before any accepted values
        // change. A retained debug/Python view must not allow a partial commit
        // on only the other ranks. Keep this lease through the actual copy.
        std::optional<EquationDofs::WriteScope> destination;
        std::optional<EquationDofs::ReadScope> source;
        const auto comm = _fv.Metrics().Communicator();
        MPIInfo mpi(comm);
        try
        {
            MPI::LocalStage(mpi, [&]
                            { destination.emplace(_accepted.WriteOwned()); source.emplace(_trial.ReadOwned()); });
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
        for (index iBlock = 0; iBlock < _accepted.BlockNumber(); ++iBlock)
        {
            const auto input = source->Block(iBlock);
            const auto output = destination->Block(iBlock);
            for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
                for (index j = 0; j < output.ComponentNumber(); ++j)
                    output(iCell, j) = input(iCell, j);
        }
        _time = time;
        _epoch = epoch;
        _trialReady = false;
        _restoreReconstruction = false;
    }
    const EquationDofs &Domain::Residual()
    {
        _trialReady = false;
        DNDS_check_throw_info(Admissible(_accepted), "accepted Cauchy mean is inadmissible");
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary(); EquationDofOperations::Copy(_trial, _accepted); });
        _fv.Execute(_trial, *_reconstruction, *_workspace, _execution,
                    FiniteVolumeEvaluation::Spatial(_residualState, _time));
        _lastFallbacks = _fv.LastFaceFallbacks(*_workspace);
        _lastCharacteristicFallbacks = _fv.LastCharacteristicFallbacks(*_workspace);
        return _residualState;
    }
    std::array<real, 14> Domain::Diagnostics(const DomainSettings &settings, const std::vector<int> &forceZones)
    {
        EvaluateObservation();
        std::array<real, 14> result{};
        result[13] = std::numeric_limits<real>::max();
        result[2] = result[3] = std::numeric_limits<real>::max();
        const MPIInfo mpi(_fv.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        {
            const auto &workers = _execution.ChemicalWorkers();
            auto evaluator = _equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto geometry = _fv.Metrics().AcquireRead(); const auto metric = geometry.HostView();
            const auto read = _accepted.ReadOwned(), residual = _observationState->ReadOwned();
            const auto flow = read.Block("flow");
            const auto layout = residual.ComponentLayout(); const auto rhs = layout.View();
            const int independent = _equation.HasMixture() ? _equation.Mixture().SpeciesNumber() - 1 : 0;
            std::optional<EquationBlockReadView> species;
            if (independent) species.emplace(read.Block("species"));
            const index width = _accepted.ComponentNumber();
            auto rhoY = evaluator.SpeciesInput();
            for (index iCell = 0; iCell < _fv.Metrics().NumCell(); ++iCell)
            {
                FlowState value;
                for (int j = 0; j < 5; ++j) value[j] = flow(iCell, j);
                if (independent)
                {
                    for (int k = 0; k < independent; ++k) rhoY[k] = (*species)(iCell, k);
                }
                const auto point = evaluator.Evaluate(value, rhoY);
                const real volume = metric.GetCellVol(iCell);
                result[0] += volume * value[0]; result[1] += volume * value[4];
                result[2] = std::min(result[2], value[0]); result[3] = std::min(result[3], point.thermo.pressure);
                result[13] = std::min(result[13], point.thermo.temperature);
                for (index j = 0; j < width; ++j) result[4] += rhs(iCell, j) * rhs(iCell, j) / volume;
        } });
        const auto comm = _fv.Metrics().Communicator();
        MPI::AllreduceInPlace<MPI::Reduction::Sum>(
            MPI::ContiguousView<real>{result.data(), 2}, comm);
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{result.data() + 2, 2}, comm);
        result[4] = MPI::AllreduceValue<MPI::Reduction::Sum>(result[4], comm);
        result[13] = MPI::AllreduceValue<MPI::Reduction::Min>(result[13], comm);
        // Report the volume-integrated L2 defect, matching Euler's res0
        // convention rather than a volume-averaged residual.
        result[4] = std::sqrt(result[4]);
        const auto force = _fv.LastBoundaryForce(*_observationWorkspace, forceZones);
        for (int d = 0; d < 3; ++d)
            result[5 + d] = force[d];
        result[8] = _fv.LastReconstructionIterations(*_observationWorkspace);
        result[9] = _fv.LastReconstructionUpdate(*_observationWorkspace);
        if (!_equation.HasMixture() && !_equation.HasRANS() && settings.initialProfile.kind == "isentropicVortex")
        {
            const auto error = _fv.Error(_accepted, settings.initialProfile, settings.initialPrimitive, _time);
            for (int i = 0; i < 3; ++i)
                result[10 + i] = error[i][0];
        }
        return result;
    }
    void Domain::WriteVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        EvaluateObservation();
        _fv.WriteVisualization(*_observationInput, *_observationReconstruction, *_observationWorkspace, prefix, _time, hdfDeflateLevel, _execution.ChemicalWorkers());
    }
    void Domain::WriteInternalVisualization(const std::string &prefix, int hdfDeflateLevel)
    {
        const MPIInfo mpi(_fv.Metrics().Communicator());
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(_timeStepper && _trialEpoch.trial &&
                                                    (!_fv.PolynomialReconstruction() || _restoreReconstruction) &&
                                                    std::isfinite(_progressTime),
                                                "Cauchy internal visualization has no active implicit trial"); });
        if (!_observationWorkspace)
        {
            MPI::LocalStage(mpi, [&]
                            {
                _observationInput.emplace(EquationDofOperations::NewVector(_trial));
                _observationState.emplace(EquationDofOperations::NewVector(_trial)); });
            _observationWorkspace = _fv.BuildWorkspace(*_observationInput, _execution.ChemicalWorkers());
            _observationReconstruction = _fv.BuildReconstructionState(*_observationInput);
            if (_interfaceInputs)
                _fv.SetBoundaryInputs(*_observationWorkspace, *_interfaceInputs);
        }
        MPI::LocalStage(mpi, [&]
                        {
            _execution.JoinPrimary();
            EquationDofOperations::Copy(*_observationInput, _trial); });
        if (_fv.PolynomialReconstruction())
            _fv.RestoreReconstructionGuess(*_observationReconstruction,
                                           _fv.ReconstructionGuess(*_reconstruction));
        _fv.Execute(*_observationInput, *_observationReconstruction,
                    *_observationWorkspace, _execution,
                    FiniteVolumeEvaluation::Spatial(*_observationState, _progressTime));
        _fv.WriteVisualization(*_observationInput, *_observationReconstruction, *_observationWorkspace, prefix, _progressTime,
                               hdfDeflateLevel, _execution.ChemicalWorkers());
    }
    void Domain::EvaluateObservation()
    {
        if (!_observationWorkspace)
        {
            std::optional<EquationDofs> input;
            std::optional<EquationDofs> residual;
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                input.emplace(EquationDofOperations::NewVector(_accepted));
                residual.emplace(EquationDofOperations::NewVector(_accepted)); });
            auto workspace = _fv.BuildWorkspace(*input, _execution.ChemicalWorkers());
            auto reconstruction = _fv.BuildReconstructionState(*input);
            _observationInput = std::move(input);
            _observationState = std::move(residual);
            _observationReconstruction = std::move(reconstruction);
            _observationWorkspace = std::move(workspace);
            if (_interfaceInputs)
                _fv.SetBoundaryInputs(*_observationWorkspace, *_interfaceInputs);
        }
        // Start every observation from the accepted numerical guess, not the
        // last observation and not a cold zero polynomial. This preserves
        // output-cadence independence while avoiding a much less converged
        // boundary reconstruction than the one used by the implicit stage.
        // A pending/rejected trial owns a different polynomial; its backup is
        // the accepted guess until publication, just as for checkpoint writing.
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            _execution.JoinPrimary();
            EquationDofOperations::Copy(*_observationInput, _accepted); });
        if (_fv.PolynomialReconstruction())
        {
            const auto &guess = _restoreReconstruction ? *_reconstructionBackup
                                                       : _fv.ReconstructionGuess(*_reconstruction);
            _fv.RestoreReconstructionGuess(*_observationReconstruction, guess);
        }
        _fv.Execute(*_observationInput, *_observationReconstruction,
                    *_observationWorkspace, _execution,
                    FiniteVolumeEvaluation::Spatial(*_observationState, _time));
    }
}
