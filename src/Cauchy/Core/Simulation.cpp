#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Cauchy/Core/StepResult.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::Cauchy
{
    namespace
    {
        ssp<Geom::UnstructuredMesh> ReadMesh(const DomainConfig &domain, const MPIInfo &mpi,
                                             const std::shared_ptr<Runtime::CommunicationContext> &domainCommunication)
        {
            DNDS_check_throw_info(!domain.settings.mesh.file.empty(), "Cauchy mesh path is empty");
            Geom::PartitionOptions partition;
            const auto mapper = [ids = domain.settings.mesh.boundaryIds](const std::string &name)
            {
                const auto found = ids.find(name);
                DNDS_check_throw_info(found != ids.end(), "CGNS boundary name is absent from mesh.boundaryIds: " + name);
                return found->second;
            };
            std::optional<Geom::NativePeriodicImportOptions> periodic;
            if (domain.settings.mesh.periodicLengths != std::array<real, 3>{0, 0, 0})
            {
                periodic.emplace();
                for (int d = 0; d < 3; ++d)
                {
                    periodic->frames.translation[d + 1].map().setZero();
                    periodic->frames.translation[d + 1].map()[d] = domain.settings.mesh.periodicLengths[d];
                }
            }
            auto communication = domainCommunication->WithPreflight(domain.settings.implicit.collectiveChecks);
            auto owned = Geom::ReadAndDeriveNativeMeshFromCGNSOwned(mpi, domain.settings.mesh.dimension, domain.settings.mesh.file,
                                                                    partition, mapper, periodic, domain.settings.mesh.elevation, domain.settings.mesh.bisect, 0, Geom::MeshAuditOptions{}, communication);
            Geom::PrepareMeshOptions prepare;
            prepare.buildSerialOut = false;
            prepare.reorderCells = 1;
            return Geom::UnstructuredMesh::FromNative(mpi, Geom::PrepareNativeMeshForSolverOwned(std::move(owned), mpi, prepare));
        }
    }
    Simulation::Simulation(SimulationConfig config, MPI_Comm comm, const MeshFactory &factory)
        : _config(std::move(config)), _placement(comm, _config.RankGroups()), _domains(size_t(_placement.DomainNumber()))
    {
        for (auto iDomain : _placement.LocalDomains())
        {
            const auto &domain = _config.Domains()[iDomain];
            const MPIInfo mpi(_placement.Context(iDomain)->Communicator());
            auto mesh = factory ? factory(iDomain, domain, mpi) : ReadMesh(domain, mpi, _placement.Context(iDomain));
            _domains[iDomain] = std::make_unique<Domain>(domain.placement.name, std::move(mesh), domain.settings, _config.Time());
        }
        if (!_config.Interfaces().empty())
        {
            std::vector<std::optional<FiniteVolume>> local(_domains.size());
            std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>>
                workers(_domains.size());
            for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                if (_domains[iDomain])
                {
                    local[iDomain] = _domains[iDomain]->_fv;
                    workers[iDomain] = _domains[iDomain]->ChemicalWorkers();
                }
            _interfaces = std::make_unique<InterfaceCoupling>(
                _config, _placement.ControlContext(), local, workers);
            // Establish accepted-time traces before CFL/diagnostics or the
            // first trial. Domain trial fields initially equal accepted means.
            // Separate packet allocations preserve these traces on rejection.
            const auto initialize = [&](bool prepared)
            {
                for (auto &domain : _domains)
                    if (domain)
                    {
                        if (prepared)
                            domain->_fv.PrepareReconstruction(domain->_trial, *domain->_reconstruction, *domain->_workspace, domain->_engine, domain->_stream, 0);
                        else
                            domain->_fv.RefreshSamplingInputs(domain->_trial, *domain->_reconstruction, *domain->_workspace, domain->_engine, domain->_stream);
                    }
                std::vector<InterfaceDomainInput> ports(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        ports[iDomain] = {domain->_fv.PublishInterfaceDonor(
                            domain->_trial, *domain->_reconstruction,
                            *domain->_workspace, prepared)};
                _interfaces->Refresh(ports, 0, {});
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                    {
                        if (const auto input = _interfaces->Inputs(iDomain))
                            _domains[iDomain]->BindInterfaceInputs(*input);
                    }
            };
            initialize(false);
            initialize(true);
            _interfaces->Commit();
        }
    }
    Simulation::~Simulation() = default;
    InterfaceDiagnostics Simulation::GetInterfaceDiagnostics() const
    {
        return _interfaces ? _interfaces->Diagnostics() : InterfaceDiagnostics{};
    }
    bool Simulation::Local(index iDomain) const
    {
        DNDS_check_throw_info(iDomain >= 0 && iDomain < index(_domains.size()), "invalid Cauchy domain index");
        return bool(_domains[iDomain]);
    }
    Domain &Simulation::GetDomain(index iDomain)
    {
        DNDS_check_throw_info(Local(iDomain), "Cauchy domain is remote on this rank");
        return *_domains[iDomain];
    }
    const Domain &Simulation::GetDomain(index iDomain) const
    {
        DNDS_check_throw_info(Local(iDomain), "Cauchy domain is remote on this rank");
        return *_domains[iDomain];
    }
    StepResult Simulation::Advance(real dt, std::optional<real> acceptedTime)
    {
        real limits[2]{dt, -dt};
        int valid = std::isfinite(dt) && dt > 0;
        valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _placement.Control());
        DNDS_check_throw_info(valid, "invalid coordinated Cauchy timestep");
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<real>{limits, 2}, _placement.Control());
        DNDS_check_throw_info(limits[0] == -limits[1], "simulation ranks requested different timesteps");
        int targetFlags[2]{int(acceptedTime.has_value()), -int(acceptedTime.has_value())};
        MPI::AllreduceInPlace<MPI::Reduction::Min>(
            MPI::ContiguousView<int>{targetFlags, 2}, _placement.Control());
        DNDS_check_throw_info(targetFlags[0] == -targetFlags[1], "simulation ranks disagree on exact accepted-time snapping");
        real acceptedTarget = acceptedTime.value_or(0);
        if (acceptedTime)
        {
            real targets[2]{acceptedTarget, -acceptedTarget};
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<real>{targets, 2}, _placement.Control());
            DNDS_check_throw_info(targets[0] == -targets[1] && std::isfinite(acceptedTarget) && acceptedTarget > _time,
                                  "simulation ranks requested different or invalid exact accepted times");
            const real expected = _time + dt;
            const real tolerance = 8 * std::numeric_limits<real>::epsilon() *
                                   std::max({real(1), std::abs(expected), std::abs(acceptedTarget)});
            DNDS_check_throw_info(std::abs(expected - acceptedTarget) <= tolerance,
                                  "exact accepted time is inconsistent with the requested timestep");
        }
        if (_placement.DomainNumber() != 1 || _interfaces || _placement.Group(0).ranks.size() != size_t(_placement.Size()))
        {
            uint64_t trialIdentifier = 0;
            const auto localPhase = [&](const auto &action, const std::string &name)
            {
                // Numerical rejection is a normal result, already agreed on
                // each domain. Unexpected exceptions abort before this control
                // agreement; they must never masquerade as a rejected step.
                int ready = 1;
                std::string error;
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                    {
                        if constexpr (std::is_same_v<decltype(action(iDomain, *_domains[iDomain])), StepResult>)
                        {
                            auto result = action(iDomain, *_domains[iDomain]);
                            if (!result)
                            {
                                ready = 0;
                                error = std::move(result.reason);
                            }
                        }
                        else
                            action(iDomain, *_domains[iDomain]);
                    }
                ready = MPI::AllreduceValue<MPI::Reduction::Min>(ready, _placement.Control());
                return ready ? StepResult{} : StepResult::Rejected("coordinated " + name + " rejected; accepted states unchanged: " + error);
            };
            const auto ports = [&](bool prepared)
            {
                std::vector<InterfaceDomainInput> inputs(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        inputs[iDomain] = {domain->_fv.PublishInterfaceDonor(
                            domain->_trial, *domain->_reconstruction,
                            *domain->_workspace, prepared)};
                return inputs;
            };
            const auto refresh = [&](real time, int stage, bool prepared, bool acceptedSnapshot = false)
            {
                StateEpoch epoch;
                epoch.acceptedStep = _step + uint64_t(acceptedSnapshot);
                epoch.trial = trialIdentifier;
                epoch.stage = stage;
                epoch.couplingIteration = prepared ? 1 : 0;
                _interfaces->Refresh(ports(prepared), time, epoch);
                localPhase([&](index iDomain, Domain &domain)
                           {
                    if(const auto input=_interfaces->Inputs(iDomain)) domain.BindInterfaceInputs(*input); }, "interface binding");
            };
            std::vector<std::optional<EquationDofs::WriteScope>> destinations(_domains.size());
            std::vector<std::optional<EquationDofs::ReadScope>> sources(_domains.size());
            const auto prepareAndCommit = [&]() -> StepResult
            {
                localPhase([&](index, Domain &domain)
                           {
                    MPI::LocalStage(MPIInfo(domain._fv.Metrics().Communicator()),[&]
                    {
                        DNDS_check_throw_info(domain.Time()==_time&&domain.Epoch().acceptedStep==_step,"domain advanced outside its Simulation");
                    });
                    domain.BeginCoordinatedStep(dt); }, "step preparation");
                if (Local(0))
                    trialIdentifier = _domains[0]->_trialEpoch.trial;
                MPI_Bcast(&trialIdentifier, 1, MPI_UINT64_T, _placement.SimulationRank(0, 0), _placement.Control());
                int stages = Local(0) ? _domains[0]->CoordinatedStageNumber() : 0;
                MPI_Bcast(&stages, 1, MPI_INT, _placement.SimulationRank(0, 0), _placement.Control());
                for (int stage = 0; stage < stages; ++stage)
                {
                    real stageTime = 0;
                    std::vector<real> localTimes(_domains.size(), 0);
                    localPhase([&](index iDomain, Domain &domain)
                               {
                        DNDS_check_throw_info(domain.CoordinatedStageNumber()==stages,"domain stage counts differ");
                        localTimes[iDomain]=domain.PrepareCoordinatedStage(stage); }, "stage preparation");
                    if (Local(0))
                        stageTime = localTimes[0];
                    MPI_Bcast(&stageTime, 1, MPI_DOUBLE, _placement.SimulationRank(0, 0), _placement.Control());
                    localPhase([&](index iDomain, Domain &)
                               { DNDS_check_throw_info(localTimes[iDomain] == stageTime, "domain stage times differ"); }, "stage-time agreement");
                    if (_interfaces)
                    {
                        localPhase([](index, Domain &domain)
                                   { domain._fv.RefreshSamplingInputs(domain._trial, *domain._reconstruction, *domain._workspace, domain._engine, domain._stream); }, "predictor halos");
                        refresh(stageTime, stage, false);
                        localPhase([&](index, Domain &domain)
                                   { domain._fv.PrepareReconstruction(domain._trial, *domain._reconstruction, *domain._workspace, domain._engine, domain._stream, stageTime); }, "interface reconstruction");
                        refresh(stageTime, stage, true);
                    }
                    if (auto result = localPhase([](index, Domain &domain)
                                                 { return domain.SolveCoordinatedStage(); }, "stage solve");
                        !result)
                        return result;
                    if (auto result = localPhase([](index, Domain &domain)
                                                 { return domain.CompleteCoordinatedStage(); }, "stage completion");
                        !result)
                        return result;
                }
                if (_interfaces)
                {
                    localPhase([](index, Domain &domain)
                               { domain._fv.RefreshSamplingInputs(domain._trial, *domain._reconstruction, *domain._workspace, domain._engine, domain._stream); }, "accepted-time predictor halos");
                    refresh(_config.Time().steady ? _time : _time + dt, 0, false, true);
                }
                // Only local lease acquisition belongs in this control-stage
                // catch. Domain solves above contain MPI and must not be caught
                // and followed by an unrelated control reduction on local failure.
                try
                {
                    MPI::LocalStage(MPIInfo(_placement.Control()), [&]
                                    {
                    for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                        if (auto &domain = _domains[iDomain])
                        {
                            DNDS_check_throw_info(domain->_trialReady&&(!domain->_timeStepper||domain->_timeStepper->HasPendingTrial()),"domain has no complete candidate");
                            destinations[iDomain].emplace(domain->_accepted.WriteOwned());
                            sources[iDomain].emplace(domain->_trial.ReadOwned());
                        } });
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("Cauchy coordinated state reservation failed");
                    for (auto &destination : destinations)
                        if (destination)
                            destination->CancelUnmodified();
                    for (auto &domain : _domains)
                        if (domain)
                            domain->DiscardStep();
                    if (_interfaces)
                    {
                        _interfaces->Discard();
                        for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                            if (_domains[iDomain])
                                if (const auto input = _interfaces->Inputs(iDomain))
                                    _domains[iDomain]->BindInterfaceInputs(*input);
                    }
                    throw;
                }
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                        _domains[iDomain]->CommitReserved(*destinations[iDomain], *sources[iDomain]);
                if (_interfaces)
                    _interfaces->Commit();
                return {};
            };
            const auto result = prepareAndCommit();
            if (!result)
            {
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                {
                    if (destinations[iDomain])
                        destinations[iDomain]->CancelUnmodified();
                    if (_domains[iDomain])
                        _domains[iDomain]->DiscardStep();
                }
                if (_interfaces)
                {
                    _interfaces->Discard();
                    // Restore only ready accepted packets. No donor search,
                    // payload transfer or physical recomputation is needed.
                    for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                        if (_domains[iDomain])
                            if (const auto input = _interfaces->Inputs(iDomain))
                                _domains[iDomain]->BindInterfaceInputs(*input);
                }
                return result;
            }
            if (!_config.Time().steady)
            {
                _time = acceptedTime.value_or(_time + dt);
                for (auto &domain : _domains)
                    if (domain)
                        domain->_time = domain->_trialTime = _time;
            }
            ++_step;
            return {};
        }
        for (auto &domain : _domains)
            if (domain)
            {
                MPI::LocalStage(MPIInfo(domain->Discretization().Metrics().Communicator()), [&]
                                { DNDS_check_throw_info(domain->Time() == _time && domain->Epoch().acceptedStep == _step,
                                                        "domain was advanced outside its Simulation; use the coordinated frontend"); });
                if (auto result = domain->PrepareStep(dt); !result)
                {
                    domain->DiscardStep();
                    return result;
                }
            }
        // Commit is serialized and collectively preflighted inside each domain.
        // External debug writers/aliases must be released before this call.
        for (auto &domain : _domains)
            if (domain)
                domain->CommitStep();
        if (!_config.Time().steady)
        {
            _time = acceptedTime.value_or(_time + dt);
            for (auto &domain : _domains)
                if (domain)
                    domain->_time = domain->_trialTime = _time;
        }
        ++_step;
        return {};
    }
}
