#include "Cauchy/Core/Simulation.hpp"
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
            auto owned = Geom::ReadAndDeriveNativeMeshFromCGNSOwned(mpi, domain.settings.mesh.dimension, domain.settings.mesh.file, partition, mapper);
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
                _domains[iDomain] = std::make_unique<Domain>(domain.placement.name, std::move(mesh), domain.settings);
            }
            catch (const std::exception &exception)
            {
                valid = 0;
                error = exception.what();
            }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
        DNDS_check_throw_info(valid, "Cauchy domain construction failed: " + error);
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
        DNDS_check_throw_info(_placement.DomainNumber() == 1, "physical multi-domain advancement requires the later coupling/commit package; use CoordinatedSchedule for placement mocks");
        DNDS_check_throw_info(_placement.Group(0).ranks.size() == size_t(_placement.Size()),
                              "initial physical Simulation requires all control ranks in its domain");
        real limits[2]{dt, -dt};
        int valid = std::isfinite(dt) && dt > 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
        DNDS_check_throw_info(valid, "invalid coordinated Cauchy timestep");
        MPI_Allreduce(MPI_IN_PLACE, limits, 2, MPI_DOUBLE, MPI_MIN, _placement.Control());
        DNDS_check_throw_info(limits[0] == -limits[1], "simulation ranks requested different timesteps");
        for (auto &domain : _domains)
            if (domain)
                try
                {
                    MPI::LocalStage(MPIInfo(domain->Discretization().Metrics().Communicator()), [&]
                                    { DNDS_check_throw_info(domain->Time() == _time && domain->Epoch().acceptedStep == _step,
                                                            "domain was advanced outside its Simulation; use the coordinated frontend"); });
                    domain->PrepareStep(dt);
                }
                catch (...)
                {
                    valid = 0;
                }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
        if (!valid)
        {
            for (auto &domain : _domains)
                if (domain)
                    domain->DiscardStep();
            DNDS_check_throw_info(false, "Cauchy candidate rejected; simulation clock unchanged");
        }
        // Commit is serialized and collectively preflighted inside each domain.
        // External debug writers/aliases must be released before this call.
        for (auto &domain : _domains)
            if (domain)
                domain->CommitStep();
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
