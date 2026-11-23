#include "Cauchy/Core/DomainMesh.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "DNDS/Core/OMP.hpp"
#include <filesystem>
#include <set>

namespace DNDS::Cauchy
{
    namespace
    {
        constexpr std::string_view archiveSuffix = ".dnds.h5";
        bool IsArchive(std::string_view path)
        {
            return path.size() >= archiveSuffix.size() && path.substr(path.size() - archiveSuffix.size()) == archiveSuffix;
        }
        std::string ArchiveBase(const std::string &path)
        {
            return IsArchive(path) ? path.substr(0, path.size() - archiveSuffix.size()) : path;
        }
        ssp<Geom::UnstructuredMesh> ReadPartition(
            const DomainConfig &domain, const MPIInfo &mpi,
            const std::shared_ptr<Runtime::CommunicationContext> &domainCommunication, DomainSetupTimings &timings,
            Diagnostics::SetupProgress progress)
        {
            // A preceding colocated domain may already have entered the
            // per-rank context. Restore the reader-owned team for this domain's
            // import, serial partitioning and distribution.
            OMP::set_serial_ingress_OMP();
            DNDS_check_throw_info(!domain.settings.mesh.file.empty(),
                                  "Cauchy mesh path is empty");
            const auto &partition = domain.settings.mesh.partition;
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
                periodic->progress = progress;
                for (int d = 0; d < 3; ++d)
                {
                    periodic->frames.translation[d + 1].map().setZero();
                    periodic->frames.translation[d + 1].map()[d] =
                        domain.settings.mesh.periodicLengths[d];
                }
            }
            auto communication = domainCommunication->WithPreflight(
                domain.settings.implicit.collectiveChecks);
            const real ingressStarted = MPI_Wtime();
            const auto &settings = domain.settings.mesh;
            const bool archived = IsArchive(settings.file);
            Geom::NativeOwnedMeshData owned;
            if (archived)
            {
                OMP::set_full_parallel_OMP();
                auto stage = progress.Stage("mesh/read-archive");
                // Euler partition archives already contain the derived cells and periodic
                // correspondences. Elevation and bisection apply only to CGNS ingress.
                owned = Geom::ReadNativeMeshFromArchiveOwned(mpi, Serializer::SerializerFactory("H5"),
                                                             ArchiveBase(settings.file), settings.repartition, partition);
                DNDS_check_throw_info(owned.dimension == settings.dimension, "distributed mesh dimension differs from mesh.dimension");
                owned.communication = communication;
                for (auto &[kind, space] : owned.spaces)
                    space = space.WithCommunication(communication);
            }
            else
            {
                DNDS_check_throw_info(!settings.repartition, "mesh.repartition requires a distributed .dnds.h5 input");
                owned = Geom::ReadAndDeriveNativeMeshFromCGNSOwned(
                    mpi, settings.dimension, settings.file, partition, mapper, periodic,
                    settings.elevation, settings.bisect, 0, Geom::MeshAuditOptions{}, communication, progress);
            }
            timings.meshIngress = MPI_Wtime() - ingressStarted;
            // Native topology, reorder, CFV setup and solver work now belong to
            // distributed ranks and must not inherit the node-wide reader team.
            OMP::set_full_parallel_OMP();
            const real prepareStarted = MPI_Wtime();
            Geom::PrepareMeshOptions prepare;
            prepare.buildSerialOut = false;
            prepare.reorderCells = 1;
            prepare.coreFirst = domain.settings.execution.coreFirstCellOrdering.value_or(domain.settings.execution.overlap);
            prepare.reorderParts = OMP::DistributedThreads();
            prepare.progress = progress;
            auto mesh = Geom::UnstructuredMesh::FromNative(
                mpi, Geom::PrepareNativeMeshForSolverOwned(
                         std::move(owned), mpi, prepare));
            timings.meshPreparation = MPI_Wtime() - prepareStarted;
            return mesh;
        }
    }

    ssp<Geom::UnstructuredMesh> ReadDomainMesh(const DomainConfig &domain, const MPIInfo &mpi,
                                               const std::shared_ptr<Runtime::CommunicationContext> &communication, DomainSetupTimings &timings,
                                               Diagnostics::SetupProgress progress)
    {
        auto mesh = ReadPartition(domain, mpi, communication, timings, progress);
        const real started = MPI_Wtime();
        const auto &settings = domain.settings.mesh;
        if (settings.scale != 1 || settings.rotationZDegrees != 0)
        {
            auto setupStage = progress.Stage("mesh/geometry");
            // Import/matching uses the file frame. The ordinary Geom
            // publication transforms coordinates AND periodic frames
            // together, exactly once before metrics/initial-state sampling.
            const real angle = settings.rotationZDegrees * std::acos(real(-1)) / 180;
            const real c = std::cos(angle), s = std::sin(angle);
            Eigen::Matrix3d rotation;
            rotation << c, -s, 0, s, c, 0, 0, 0, 1;
            mesh->ApplyGeometryTransform(rotation, settings.scale);
        }
        timings.meshPreparation += MPI_Wtime() - started;
        return mesh;
    }

    ssp<Geom::UnstructuredMesh> EnsureDomainWallDistance(
        ssp<Geom::UnstructuredMesh> mesh,
        const DomainSettings &settings, Diagnostics::SetupProgress progress)
    {
        const auto mpi = mesh->getMPI();
        int available = mesh->HasNodeWallDist();
        available = MPI::AllreduceValue<MPI::Reduction::Sum>(available,
                                                             mpi.comm);
        DNDS_check_throw_info(
            available == 0 || available == mpi.size,
            "nodal wall-field presence differs between ranks");
        auto setupStage = progress.Stage("mesh/wall-distance", available ? "reuse published nodal vectors" : "compute from no-slip zones");
        if (!available)
            mesh->BuildNodeWallDist(
                [&](auto zone)
                {
                    const auto found = settings.boundaries.find(
                        std::to_string(zone));
                    return found != settings.boundaries.end() &&
                           found->second.kind == "noSlipWall";
                });
        return mesh;
    }

    void WritePartitionedMeshes(const SimulationConfig &config, MPI_Comm control)
    {
        DomainPlacement placement(control, config.RankGroups());
        std::vector<std::string> bases;
        std::set<std::filesystem::path> destinations;
        // Check every destination on every control rank before disjoint groups open files.
        for (const auto &domain : config.Domains())
        {
            const auto &mesh = domain.settings.mesh;
            DNDS_check_throw_info(!mesh.file.empty() && !IsArchive(mesh.file),
                                  "partitionMeshOnly requires a CGNS input mesh for each domain");
            const auto base = mesh.partitionOutput.empty()
                                  ? Geom::MeshH5Path(mesh.file, int(domain.placement.ranks.size()), mesh.elevation, mesh.bisect)
                                  : ArchiveBase(mesh.partitionOutput);
            DNDS_check_throw_info(!base.empty() && destinations.insert(std::filesystem::weakly_canonical(base + std::string(archiveSuffix))).second,
                                  "partition archive paths must be distinct; set mesh.partitionOutput for domains sharing an input mesh");
            bases.push_back(base);
        }
        for (const auto id : placement.LocalDomains())
        {
            const auto &domain = config.Domains()[id];
            const auto context = placement.Context(id);
            const MPIInfo mpi(context->Communicator());
            DomainSetupTimings timings;
            auto mesh = ReadPartition(domain, mpi, context, timings, {});
            if (domain.settings.mesh.partitionWallDistance)
                EnsureDomainWallDistance(mesh, domain.settings, {});
            Serializer::SerializerFactory writer("H5");
            Geom::SerializeMesh(*mesh, bases[id], writer);
            if (mpi.rank == 0)
                log() << "Partitioned mesh " << domain.placement.name << ": " << bases[id] << archiveSuffix << std::endl;
        }
    }
}
