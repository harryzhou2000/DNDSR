#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Interface/InterfaceCoupling.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        ssp<Geom::UnstructuredMesh> ReadMesh(
            const DomainConfig &domain, const MPIInfo &mpi,
            const std::shared_ptr<Runtime::CommunicationContext> &domainCommunication)
        {
            DNDS_check_throw_info(!domain.settings.mesh.file.empty(),
                                  "Cauchy mesh path is empty");
            Geom::PartitionOptions partition;
            const auto mapper = [ids = domain.settings.mesh.boundaryIds](
                                    const std::string &name)
            {
                const auto found = ids.find(name);
                DNDS_check_throw_info(
                    found != ids.end(),
                    "CGNS boundary name is absent from mesh.boundaryIds: " + name);
                return found->second;
            };
            std::optional<Geom::NativePeriodicImportOptions> periodic;
            if (domain.settings.mesh.periodicLengths !=
                std::array<real, 3>{0, 0, 0})
            {
                periodic.emplace();
                for (int d = 0; d < 3; ++d)
                {
                    periodic->frames.translation[d + 1].map().setZero();
                    periodic->frames.translation[d + 1].map()[d] =
                        domain.settings.mesh.periodicLengths[d];
                }
            }
            auto communication = domainCommunication->WithPreflight(
                domain.settings.implicit.collectiveChecks);
            auto owned = Geom::ReadAndDeriveNativeMeshFromCGNSOwned(
                mpi, domain.settings.mesh.dimension, domain.settings.mesh.file,
                partition, mapper, periodic, domain.settings.mesh.elevation,
                domain.settings.mesh.bisect, 0, Geom::MeshAuditOptions{},
                communication);
            Geom::PrepareMeshOptions prepare;
            prepare.buildSerialOut = false;
            prepare.reorderCells = 1;
            return Geom::UnstructuredMesh::FromNative(
                mpi, Geom::PrepareNativeMeshForSolverOwned(
                         std::move(owned), mpi, prepare));
        }
    }

    Simulation::Simulation(SimulationConfig config, MPI_Comm comm,
                           const MeshFactory &factory)
        : _config(std::move(config)),
          _placement(comm, _config.RankGroups()),
          _domains(size_t(_placement.DomainNumber()))
    {
        if (_config.Time().steady || _config.Time().method != "ssprk3")
        {
            const bool ditr = !_config.Time().steady && _config.Time().method == "ditr";
            const UpdatePolicyRequirements requirements{
                ditr ? 2 : 1,
                ditr ? NonlinearDetail::ResidualObservation::AfterUpdate
                     : NonlinearDetail::ResidualObservation::BeforeUpdate,
                ditr};
            // Resolve over every configured domain on every control rank,
            // including ranks with no local mesh. Membership must not decide
            // whether an optional control-communicator agreement is executed.
            for (const auto &domain : _config.Domains())
            {
                const auto controls = ditr
                                          ? _config.Time().ditr.ResolveControls(domain.settings.implicit)
                                          : domain.settings.implicit;
                auto policy = UpdatePolicyRegistry::Resolve(controls.updatePolicy, requirements);
                if (_updatePolicy)
                    DNDS_check_throw_info(
                        policy->Descriptor().Metadata() == _updatePolicy->Descriptor().Metadata(),
                        "coordinated domains require the same nonlinear update policy");
                else
                    _updatePolicy = std::move(policy);
                _nonlinearCollectiveChecks |= controls.collectiveChecks;
            }
        }
        for (auto iDomain : _placement.LocalDomains())
        {
            const auto &domain = _config.Domains()[iDomain];
            const MPIInfo mpi(_placement.Context(iDomain)->Communicator());
            auto mesh = factory ? factory(iDomain, domain, mpi)
                                : ReadMesh(domain, mpi,
                                           _placement.Context(iDomain));
            _domains[iDomain] = std::make_unique<Domain>(
                domain.placement.name, std::move(mesh), domain.settings,
                _config.Time());
        }
        if (!_config.Interfaces().empty())
        {
            std::vector<std::optional<FiniteVolume>> local(_domains.size());
            std::vector<std::optional<Physics::Chemistry::ChemicalWorkerPool>>
                workers(_domains.size());
            for (index iDomain = 0; iDomain < index(_domains.size()); ++iDomain)
                if (_domains[iDomain])
                {
                    local[iDomain] = _domains[iDomain]->Discretization();
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
                        auto &control = *domain;
                        control.PrepareInterface(0, prepared, 0);
                    }
                std::vector<InterfaceDomainInput> ports(_domains.size());
                for (index iDomain = 0; iDomain < index(_domains.size());
                     ++iDomain)
                    if (auto &domain = _domains[iDomain])
                        ports[iDomain] = {
                            domain->PublishInterfaceDonor(0, prepared)};
                _interfaces->Refresh(ports, 0, {});
                for (index iDomain = 0; iDomain < index(_domains.size());
                     ++iDomain)
                    if (_domains[iDomain])
                        if (const auto input = _interfaces->Inputs(iDomain))
                            _domains[iDomain]->BindInterfaceInputs(
                                *input);
            };
            initialize(false);
            initialize(true);
            _interfaces->Commit();
        }
    }

    Simulation::~Simulation() = default;

    InterfaceDiagnostics Simulation::GetInterfaceDiagnostics() const
    {
        return _interfaces ? _interfaces->Diagnostics()
                           : InterfaceDiagnostics{};
    }

    bool Simulation::Local(index iDomain) const
    {
        DNDS_check_throw_info(iDomain >= 0 &&
                                  iDomain < index(_domains.size()),
                              "invalid Cauchy domain index");
        return bool(_domains[iDomain]);
    }

    Domain &Simulation::GetDomain(index iDomain)
    {
        DNDS_check_throw_info(Local(iDomain),
                              "Cauchy domain is remote on this rank");
        return *_domains[iDomain];
    }

    const Domain &Simulation::GetDomain(index iDomain) const
    {
        DNDS_check_throw_info(Local(iDomain),
                              "Cauchy domain is remote on this rank");
        return *_domains[iDomain];
    }

}
