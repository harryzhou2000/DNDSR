#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/StepRejected.hpp"
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
            MPI_Allreduce(MPI_IN_PLACE, &available, 1, MPI_INT, MPI_SUM, mpi.comm);
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
            discrete.Initialize(field, settings.initialProfile, primitive);
            auto state = EquationDofs::Coupled(equation.Schema(), std::move(field));
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
        AuxiliaryRequest TimeAuxiliary(const TimeSettings &time)
        {
            AuxiliaryRequest request;
            request.faceSpectralRadius = request.cellSpectralSum = time.steady || time.method != "ssprk3";
            return request;
        }
        ResolvedPhysicalInputs ResolveInputs(const DomainSettings &settings, const EquationSetup &setup, const MPIInfo &mpi)
        {
            std::optional<ResolvedPhysicalInputs> result;
            MPI::LocalStage(mpi, [&]
                            { result = ResolvePhysicalStateInputs(settings, setup.equation, setup.workers); });
            return std::move(*result);
        }
    }
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time)
        : Domain(name, mesh, settings, time, ResolveEquation(name, settings, mesh->getMPI())) {}
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const TimeSettings &time, EquationSetup setup)
        : Domain(name, mesh, ResolveInputs(settings, setup, mesh->getMPI()), time, setup) {}
    Domain::Domain(std::string name, ssp<Geom::UnstructuredMesh> mesh, const ResolvedPhysicalInputs &inputs, const TimeSettings &time, EquationSetup setup)
        : _name(std::move(name)), _equation(std::move(setup.equation)), _workers(std::move(setup.workers)), _fv(PrepareWallField(mesh, _equation, inputs.settings), SpatialSettings(inputs.settings, time), _equation),
          _accepted(BuildState(_fv, _equation, inputs.settings, _workers, inputs.initialConservative)), _trial(_accepted.Clone()),
          _residualState(EquationDofOperations::NewVector(_accepted)), _initial(EquationDofOperations::NewVector(_accepted)),
          _workspace(_equation.HasMixture() || _equation.HasRANS() ? nullptr : _fv.BuildWorkspace(_trial, TimeAuxiliary(time))),
          _equationWorkspace(_equation.HasMixture() || _equation.HasRANS() ? _fv.BuildEquationWorkspace(_trial, _workers) : nullptr),
          _engine(size_t(inputs.settings.execution.hostThreads), inputs.settings.execution.checkAccess ? std::make_shared<R::AccessTrackingContext>() : nullptr),
          _stream(_engine.CreateCPUStream())
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
        if (_equationWorkspace)
        {
            const auto &n = settings.numerics;
            _equationThermal.emplace(_equation, n.densityFloor, n.temperatureFloor, n.sensibleEnergyFloor, n.pressureFloor,
                                     _workers ? _workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
        }
        if (_equation.HasRANS())
            _ransState.emplace(ResolveRANSStatePolicy(_equation, settings.numerics.ransState));
        DNDS_check_throw_info(Admissible(_accepted), "Cauchy initial means violate density or minimum-temperature internal-energy bounds");
        ConfigureTimeStep(time, settings.implicit.positivity.compressionPower);
        InitializeImplicit(settings, time, mesh);
    }
    CFV::FVField &Domain::FlowField(EquationDofs &state)
    {
        const auto location = state.LocateBlock("flow");
        auto &field = state.PhysicalField(location.physicalField);
        DNDS_check_throw_info(location.firstComponent == 0 && field.Width() == 5, "Domain flow debug field requires a dedicated flow5 allocation");
        return field;
    }
    const CFV::FVField &Domain::FlowField(const EquationDofs &state)
    {
        const auto location = state.LocateBlock("flow");
        const auto &field = state.PhysicalField(location.physicalField);
        DNDS_check_throw_info(location.firstComponent == 0 && field.Width() == 5, "Domain flow debug field requires a dedicated flow5 allocation");
        return field;
    }
    bool Domain::Admissible(const EquationDofs &state) const
    {
        if (!_equationWorkspace)
            return _fv.Admissible(FlowField(state));
        int valid = 1;
        std::exception_ptr failure;
        try
        {
            auto evaluator = _equation.MakePointEvaluator(_workers ? _workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto read = state.ReadOwned();
            const auto layout = read.ComponentLayout();
            const auto values = layout.View();
            const auto bounds = _equationThermal->View();
            const auto scalars = _ransState ? read.Block("rans") : EquationBlockReadView{};
            auto rhoY = evaluator.SpeciesInput();
            for (index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const EquationPositivityDetail::PointRow<decltype(values)> row{values, iCell};
                if (!bounds.Admissible(row))
                {
                    valid = 0;
                    break;
                }
                if (_ransState && !_ransState->Admissible(scalars[iCell]))
                {
                    valid = 0;
                    break;
                }
                for (int k = 0; k < rhoY.nSpecies; ++k)
                    rhoY[k] = values(iCell, bounds.speciesFirst + k);
                const auto point = evaluator.Evaluate(bounds.Flow(row), rhoY);
                valid &= point.thermo.pressure > 0 && point.thermo.temperature > 0;
            }
        }
        catch (...)
        {
            // Purely local thermodynamic work above. Preserve backend/access
            // errors rather than converting them into numerical rejection.
            failure = std::current_exception();
        }
        int status[]{valid, !failure};
        MPI_Allreduce(MPI_IN_PLACE, status, 2, MPI_INT, MPI_MIN, _fv.Metrics().Communicator());
        if (failure)
            std::rethrow_exception(failure);
        DNDS_check_throw_info(status[1], "Cauchy admissibility evaluation failed on another rank");
        return bool(status[0]);
    }
    bool Domain::RefreshDerived(DerivedFields &fields, bool onlyIfStale) const
    {
        return _workers ? RefreshDerivedFields(_accepted, _equation, fields, _workers->Acquire(), onlyIfStale)
                        : RefreshDerivedFields(_accepted, _equation, fields, onlyIfStale);
    }
    real Domain::StableStep(real cfl) const
    {
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
            _fv.PrepareAuxValues(state, *_timeAuxiliary, *_timeMeanGradients, _engine, _stream, _time, _workers);
            return _fv.StableStep(state, *_timeAuxiliary, cfl, _time);
        }
        std::vector<R::TransferOperation> transfers;
        for (const auto &halo : _timeMeanHalo)
            transfers.push_back(_engine.ChecksAccess() ? halo.BeginChecked(_stream) : halo.BeginScheduled(_stream.Context()));
        for (auto &transfer : transfers)
            transfer.Wait();
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        { if (_equationWorkspace) _fv.PrepareAuxValues(_accepted, *_timeAuxiliary, _time, _workers);
                          else _fv.PrepareAuxValues(Flow(), *_timeAuxiliary, _time, _interfaceInputs ? &*_interfaceInputs : nullptr); });
        return _equationWorkspace ? _fv.StableStep(_accepted, *_timeAuxiliary, cfl, _time) : _fv.StableStep(Flow(), *_timeAuxiliary, cfl, _time);
    }
    void Domain::UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime)
    {
        if (_equationWorkspace)
        {
            _fv.Evaluate(_trial, _residualState, *_equationWorkspace, _engine, _stream, stageTime);
            MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                            {
                const auto geometry = _fv.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                const auto initial = _initial.ReadOwned(), residual = _residualState.ReadOwned();
                auto output = _trial.WriteOwned();
                const auto flow = output.Block("flow");
                for (index iBlock = 0; iBlock < _trial.BlockNumber(); ++iBlock)
                {
                    const auto base = initial.Block(iBlock), rhs = residual.Block(iBlock); const auto state = output.Block(iBlock);
                    const bool rans = _ransState && _trial.Schema().blocks[iBlock].name == "rans";
                    for (index iCell = 0; iCell < state.RowCount(); ++iCell)
                    {
                        if (rans)
                        {
                            // Euler's scalar-only repair is permitted here.
                            // It must never rescale the flow5/species update.
                            std::array<real, 2> old{}, change{}, next{};
                            for (int j = 0; j < _ransState->components; ++j)
                            {
                                old[j] = state(iCell, j);
                                change[j] = historyWeight * base(iCell, j) + stageWeight * (old[j] + dt * rhs(iCell, j) / metric.GetCellVol(iCell)) - old[j];
                                DNDS_check_throw_info(std::isfinite(change[j]), "nonfinite explicit RANS increment");
                            }
                            _ransState->RepairIncrement(old, change);
                            for (int j = 0; j < _ransState->components; ++j) next[j] = old[j] + change[j];
                            _ransState->RepairMean(flow(iCell, 0), next);
                            for (int j = 0; j < _ransState->components; ++j) state(iCell, j) = next[j];
                            continue;
                        }
                        // No cellwise thermal increment compression: flow5
                        // and species retain their conservative residuals.
                        // Do not clip/normalize species means here either;
                        // the full raw candidate must pass Admissible below.
                        for (index j = 0; j < state.ComponentNumber(); ++j)
                            state(iCell, j) = historyWeight * base(iCell, j) + stageWeight * (state(iCell, j) + dt * rhs(iCell, j) / metric.GetCellVol(iCell));
                    }
                } });
            if (!Admissible(_trial))
                throw StepRejected("Cauchy explicit extended-equation stage is inadmissible; accepted state unchanged");
            return;
        }
        auto &residual = _residualState.CoupledField();
        _fv.Evaluate(_trial, residual, *_workspace, _engine, _stream, stageTime);
        _lastFallbacks += _fv.LastFaceFallbacks(*_workspace);
        _lastCharacteristicFallbacks += _fv.LastCharacteristicFallbacks(*_workspace);
        auto geometry = _fv.Metrics().AcquireRead();
        const auto metrics = geometry.HostView();
        real alpha = 1;
        int safe[2]{1, 1}; // high candidate, low candidate
        if (_positivity)
        {
            // Compatibility baseline: one domain-global conservative blend.
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
                const auto highRead = residual.Owned().Read(), lowRead = _fv.LowOrderResidual(*_workspace).Owned().Read();
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
                if (!safe[1])
                    throw StepRejected("global conservative PP blend requires admissible low-order stage; reduce dt");
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
                } }, R::ReadAs<R::UniformTableRows>(_initial.CoupledField().Owned()), R::ReadAs<R::UniformTableRows>(residual.Owned()), R::ReadWriteAs<R::UniformTableRows>(_trial.CoupledField().Owned()));
        _stream.Record().WaitHost();
        if (!_fv.Admissible(_trial.CoupledField()))
            throw StepRejected("Cauchy SSPRK3 stage is inadmissible; accepted state unchanged");
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
        DNDS_check_throw_info(Admissible(_accepted), "accepted Cauchy mean is inadmissible");
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
            _stream.Record().WaitHost();
            DNDS_check_throw_info(_equation.HasMixture() && _workers, "chemical source step requires its explicit mixture worker");
            AdvanceChemicalSource(_equation, input, output, _workers->Acquire(), dt, _reactorControls); });
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
    void Domain::PrepareStep(real dt)
    {
        DNDS_check_throw_info(!_fv.HasInterfaces(), "interface domains must advance through their coordinated Simulation");
        BeginTrial(dt);
        if (_timeStepper)
        {
            PrepareImplicitStep(dt);
        }
        else
        {
            // Shu-Osher SSPRK(3,3), including its non-monotone physical stage times.
            _trialEpoch.stage = 0;
            UpdateStage(dt, 0, 1, _time);
            _trialEpoch.stage = 1;
            UpdateStage(dt, .75, .25, _time + dt);
            _trialEpoch.stage = 2;
            UpdateStage(dt, 1. / 3, 2. / 3, _time + .5 * dt);
        }
        FinishChemicalStep();
        _trialTime = _steady ? _time : _time + dt;
        _trialReady = true;
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
                const bool accepted = _equationImplicit ? RecordImplicitStage(_multigrid ? _multigrid->Solve(*equation.source, equation.mass, equation.rhsScale, equation.time, equation.stage)
                                                                                         : _equationImplicit->Solve(_trial, *equation.source, equation.mass, equation.rhsScale, equation.time, equation.stage))
                                                        : RecordImplicitStage(_multigrid ? _multigrid->Solve(equation.source->CoupledField(), equation.mass, equation.rhsScale, equation.time, equation.stage)
                                                                                         : _implicit->Solve(_trial.CoupledField(), equation.source->CoupledField(), equation.mass, equation.rhsScale, equation.time, equation.stage));
                if (!accepted)
                    throw StepRejected("coordinated implicit stage reached its correction cap");
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
                                        { if (_equationWorkspace) EvaluateImplicitRate(state, rate, *_equationWorkspace, time, stage);
                                          else EvaluateImplicitRate(state, rate, *_workspace, time, stage); });
        _coordinatedPrepared = false;
        _stageEquation.reset();
        if (++_coordinatedStage == CoordinatedStageNumber())
        {
            FinishChemicalStep();
            if (!Admissible(_trial))
                throw StepRejected("coordinated candidate is inadmissible");
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
    const CFV::FVField &Domain::Residual()
    {
        ResidualState();
        return FlowField(_residualState);
    }
    const EquationDofs &Domain::ResidualState()
    {
        _trialReady = false;
        DNDS_check_throw_info(Admissible(_accepted), "accepted Cauchy mean is inadmissible");
        MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                        {
            _stream.Record().WaitHost(); EquationDofOperations::Copy(_trial, _accepted); });
        if (_equationWorkspace)
            _fv.Evaluate(_trial, _residualState, *_equationWorkspace, _engine, _stream, _time);
        else
        {
            _fv.Evaluate(_trial, _residualState.CoupledField(), *_workspace, _engine, _stream, _time);
            _lastFallbacks = _fv.LastFaceFallbacks(*_workspace);
        }
        return _residualState;
    }
    std::array<real, 14> Domain::Diagnostics(const DomainSettings &settings, const std::vector<int> &forceZones)
    {
        EvaluateObservation();
        std::array<real, 14> result{};
        result[13] = std::numeric_limits<real>::max();
        result[2] = result[3] = std::numeric_limits<real>::max();
        if (_equationWorkspace)
        {
            const MPIInfo mpi(_fv.Metrics().Communicator());
            MPI::LocalStage(mpi, [&]
                            {
            auto evaluator = _equation.MakePointEvaluator(_workers ? _workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
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
            MPI_Allreduce(MPI_IN_PLACE, result.data(), 2, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(MPI_IN_PLACE, result.data() + 2, 2, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(MPI_IN_PLACE, result.data() + 4, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(MPI_IN_PLACE, result.data() + 13, 1, MPI_DOUBLE, MPI_MIN, comm);
            result[4] = std::sqrt(result[4]);
            const auto force = _fv.LastBoundaryForce(*_equationObservationWorkspace, forceZones);
            for (int d = 0; d < 3; ++d)
                result[5 + d] = force[d];
            result[8] = _fv.LastReconstructionIterations(*_equationObservationWorkspace);
            result[9] = _fv.LastReconstructionUpdate(*_equationObservationWorkspace);
            return result;
        }
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
        if (_equationWorkspace)
            _fv.WriteVisualization(_accepted, *_equationObservationWorkspace, prefix, _time, hdfDeflateLevel, _workers);
        else
            _fv.WriteVisualization(Flow(), *_observationWorkspace, prefix, _time, hdfDeflateLevel);
    }
    void Domain::EvaluateObservation()
    {
        if (_equationWorkspace)
        {
            if (!_equationObservationWorkspace)
            {
                std::optional<EquationDofs> residual;
                MPI::LocalStage(MPIInfo(_fv.Metrics().Communicator()), [&]
                                { residual.emplace(EquationDofOperations::NewVector(_accepted)); });
                auto workspace = _fv.BuildEquationWorkspace(_accepted, _workers);
                _observationState = std::move(residual);
                _equationObservationWorkspace = std::move(workspace);
            }
            const auto &guess = _restoreReconstruction ? *_equationBackup : _fv.ReconstructionGuess(*_equationWorkspace);
            _fv.RestoreReconstructionGuess(*_equationObservationWorkspace, guess);
            _fv.Evaluate(_accepted, *_observationState, *_equationObservationWorkspace, _engine, _stream, _time);
            return;
        }
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
