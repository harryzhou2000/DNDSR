#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        ssp<Geom::UnstructuredMesh> ReadMesh(const DomainConfig &domain, const MPIInfo &mpi)
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
            auto owned = Geom::ReadAndDeriveNativeMeshFromCGNSOwned(mpi, domain.settings.mesh.dimension, domain.settings.mesh.file, partition, mapper, periodic);
            Geom::PrepareMeshOptions prepare;
            prepare.buildSerialOut = false;
            prepare.reorderCells = 1;
            return Geom::UnstructuredMesh::FromNative(mpi, Geom::PrepareNativeMeshForSolverOwned(std::move(owned), mpi, prepare));
        }
    }
    Simulation::Simulation(SimulationConfig config, MPI_Comm comm, const MeshFactory &factory)
        : _config(std::move(config)), _placement(comm, _config.RankGroups()), _domains(size_t(_placement.DomainNumber()))
    {
        int valid = 1;
        std::string error;
        for (auto iDomain : _placement.LocalDomains())
            try
            {
                const auto &domain = _config.Domains()[iDomain];
                const MPIInfo mpi(_placement.Context(iDomain)->Communicator());
                auto mesh = factory ? factory(iDomain, domain, mpi) : ReadMesh(domain, mpi);
                _domains[iDomain] = std::make_unique<Domain>(domain.placement.name, std::move(mesh), domain.settings, _config.Time());
            }
            catch (const std::exception &exception)
            {
                valid = 0;
                error = exception.what();
            }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
        DNDS_check_throw_info(valid, "Cauchy domain construction failed: " + error);
        if (!_config.Interfaces().empty())
        {
            std::vector<const FiniteVolume *> local(_domains.size(), nullptr);
            for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                if (_domains[iDomain])
                    local[iDomain] = &_domains[iDomain]->_fv;
            _interfaces = std::make_unique<InterfaceCoupling>(_config, _placement.Control(), local);
            // Establish accepted-time traces before CFL/diagnostics or the
            // first trial. Domain trial fields initially equal accepted means.
            // Separate packet allocations preserve these traces on rejection.
            const auto initialize = [&](bool prepared)
            {
                int ready = 1;
                std::string failure;
                for (auto &domain : _domains)
                    if (domain)
                        try
                        {
                            if (prepared)
                                domain->_fv.PrepareReconstruction(domain->_trial.CoupledField(), *domain->_workspace, domain->_engine, domain->_stream, 0);
                            else
                                domain->_fv.RefreshSamplingInputs(domain->_trial.CoupledField(), *domain->_workspace, domain->_engine, domain->_stream);
                        }
                        catch (const std::exception &exception)
                        {
                            ready = 0;
                            failure = exception.what();
                        }
                MPI_Allreduce(MPI_IN_PLACE, &ready, 1, MPI_INT, MPI_MIN, _placement.Control());
                DNDS_check_throw_info(ready, "initial interface preparation failed: " + failure);
                std::vector<InterfaceDomainInput> ports(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        ports[iDomain] = {&domain->_fv, &domain->_trial.CoupledField(),
                                          prepared ? &domain->_fv.FluxReconstruction(*domain->_workspace) : &domain->_fv.ReconstructionGuess(*domain->_workspace)};
                _interfaces->Refresh(ports, 0, {});
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                        try
                        {
                            if (const auto *input = _interfaces->Inputs(iDomain))
                                _domains[iDomain]->BindInterfaceInputs(*input);
                        }
                        catch (const std::exception &exception)
                        {
                            ready = 0;
                            failure = exception.what();
                        }
                MPI_Allreduce(MPI_IN_PLACE, &ready, 1, MPI_INT, MPI_MIN, _placement.Control());
                DNDS_check_throw_info(ready, "initial interface binding failed: " + failure);
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
    void Simulation::Advance(real dt)
    {
        real limits[2]{dt, -dt};
        int valid = std::isfinite(dt) && dt > 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
        DNDS_check_throw_info(valid, "invalid coordinated Cauchy timestep");
        MPI_Allreduce(MPI_IN_PLACE, limits, 2, MPI_DOUBLE, MPI_MIN, _placement.Control());
        DNDS_check_throw_info(limits[0] == -limits[1], "simulation ranks requested different timesteps");
        if (_placement.DomainNumber() != 1 || _interfaces || _placement.Group(0).ranks.size() != size_t(_placement.Size()))
        {
            uint64_t trialIdentifier = 0;
            const auto localPhase = [&](const auto &action, const std::string &name)
            {
                int ready = 1;
                std::string error;
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                        try
                        {
                            action(iDomain, *_domains[iDomain]);
                        }
                        catch (const std::exception &exception)
                        {
                            ready = 0;
                            error = exception.what();
                        }
                MPI_Allreduce(MPI_IN_PLACE, &ready, 1, MPI_INT, MPI_MIN, _placement.Control());
                DNDS_check_throw_info(ready, "coordinated " + name + " failed; accepted states unchanged: " + error);
            };
            const auto ports = [&](bool prepared)
            {
                std::vector<InterfaceDomainInput> inputs(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        inputs[iDomain] = {&domain->_fv, &domain->_trial.CoupledField(),
                                           prepared ? &domain->_fv.FluxReconstruction(*domain->_workspace) : &domain->_fv.ReconstructionGuess(*domain->_workspace)};
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
                    if(const auto *input=_interfaces->Inputs(iDomain)) domain.BindInterfaceInputs(*input); }, "interface binding");
            };
            std::vector<std::optional<EquationDofs::WriteScope>> destinations(_domains.size());
            std::vector<std::optional<EquationDofs::ReadScope>> sources(_domains.size());
            try
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
                                   { domain._fv.RefreshSamplingInputs(domain._trial.CoupledField(), *domain._workspace, domain._engine, domain._stream); }, "predictor halos");
                        refresh(stageTime, stage, false);
                        localPhase([&](index, Domain &domain)
                                   { domain._fv.PrepareReconstruction(domain._trial.CoupledField(), *domain._workspace, domain._engine, domain._stream, stageTime); }, "interface reconstruction");
                        refresh(stageTime, stage, true);
                    }
                    localPhase([](index, Domain &domain)
                               { domain.SolveCoordinatedStage(); }, "stage solve");
                    localPhase([](index, Domain &domain)
                               { domain.CompleteCoordinatedStage(); }, "stage completion");
                }
                if (_interfaces)
                {
                    localPhase([](index, Domain &domain)
                               { domain._fv.RefreshSamplingInputs(domain._trial.CoupledField(), *domain._workspace, domain._engine, domain._stream); }, "accepted-time predictor halos");
                    refresh(_config.Time().steady ? _time : _time + dt, 0, false, true);
                }
                localPhase([&](index iDomain, Domain &domain)
                           { MPI::LocalStage(MPIInfo(domain._fv.Metrics().Communicator()), [&]
                                             {
                        DNDS_check_throw_info(domain._trialReady&&(!domain._timeStepper||domain._timeStepper->HasPendingTrial()),"domain has no complete candidate");
                        destinations[iDomain].emplace(domain._accepted.WriteOwned());
                        sources[iDomain].emplace(domain._trial.ReadOwned()); }); }, "publication reservation");
                for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                    if (_domains[iDomain])
                        _domains[iDomain]->CommitReserved(*destinations[iDomain], *sources[iDomain]);
                if (_interfaces)
                    _interfaces->Commit();
            }
            catch (...)
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
                    localPhase([&](index iDomain, Domain &domain)
                               {if(const auto *input=_interfaces->Inputs(iDomain)) domain.BindInterfaceInputs(*input); }, "accepted trace restoration");
                }
                throw;
            }
            if (!_config.Time().steady)
                _time += dt;
            ++_step;
            return;
        }
        std::string failure;
        for (auto &domain : _domains)
            if (domain)
                try
                {
                    MPI::LocalStage(MPIInfo(domain->Discretization().Metrics().Communicator()), [&]
                                    { DNDS_check_throw_info(domain->Time() == _time && domain->Epoch().acceptedStep == _step,
                                                            "domain was advanced outside its Simulation; use the coordinated frontend"); });
                    domain->PrepareStep(dt);
                }
                catch (const std::exception &exception)
                {
                    valid = 0;
                    failure = exception.what();
                }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
        if (!valid)
        {
            for (auto &domain : _domains)
                if (domain)
                    domain->DiscardStep();
            DNDS_check_throw_info(false, "Cauchy candidate rejected; simulation clock unchanged: " + failure);
        }
        // Commit is serialized and collectively preflighted inside each domain.
        // External debug writers/aliases must be released before this call.
        for (auto &domain : _domains)
            if (domain)
                domain->CommitStep();
        if (!_config.Time().steady)
            _time += dt;
        ++_step;
    }
    void Simulation::ReadCheckpoint(const std::string &path)
    {
        DNDS_check_throw_info(_placement.DomainNumber() == 1 && _placement.Group(0).ranks.size() == size_t(_placement.Size()),
                              "initial Simulation restart requires one domain spanning all control ranks");
        auto &domain = GetDomain(0);
        domain.ReadCheckpoint(path, _config.Root());
        _time = domain.Time();
        _step = domain.Epoch().acceptedStep;
    }
}
