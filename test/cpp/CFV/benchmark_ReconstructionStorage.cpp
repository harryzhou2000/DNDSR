// Isolated-process capability/storage probe, deliberately outside CTest.
// One profile per process prevents a previous full plan's allocator high-water
// mark from being attributed to a later reduced profile. Report plan bytes,
// known Runtime scratch, current RSS and whole-process high-water separately.
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "ReconstructionFixture.hpp"
#include "../Cauchy/CartesianFixture.hpp"
#include <sys/resource.h>
#include <sched.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <limits>

namespace C = DNDS::CFV;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;

namespace
{
    uint64_t ResidentBytes()
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line))
            if (line.compare(0, 6, "VmRSS:") == 0)
                return std::stoull(line.substr(6)) * 1024;
        DNDS_check_throw_info(false, "storage probe requires Linux VmRSS");
        return 0;
    }

    uint64_t PeakResidentBytes()
    {
        rusage usage{};
        DNDS_check_throw_info(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
        return uint64_t(usage.ru_maxrss) * 1024; // Linux reports KiB.
    }

    uint64_t Digest(const C::FVField &field)
    {
        auto read = field.Owned().Read();
        const auto view = read.View();
        uint64_t hash = 14695981039346656037ULL;
        for (I iCell = 0; iCell < I(field.Owned().RowCount()); ++iCell)
            for (I component = 0; component < field.Width(); ++component)
            {
                const auto value = view(iCell, component);
                DNDS_check_throw_info(std::isfinite(value), "nonfinite reconstruction result");
                uint64_t bits;
                static_assert(sizeof(bits) == sizeof(value));
                std::memcpy(&bits, &value, sizeof(bits));
                hash = (hash ^ bits) * 1099511628211ULL;
            }
        return hash;
    }

    void Fill(C::FVField &field, const DNDS::Geom::UnstructuredMesh &mesh)
    {
        auto write = field.Owned().Write();
        const auto view = write.View();
        for (I iCell = 0; iCell < I(field.Owned().RowCount()); ++iCell)
            for (I component = 0; component < field.Width(); ++component)
                view(iCell, component) = std::sin(.031 * mesh.OriginalEntityId(DNDS::Geom::EntityKind::Cell, iCell) + .17 * component);
    }

    void Report(const J &rankRecord, const DNDS::MPIInfo &mpi)
    {
        const auto encoded = rankRecord.dump();
        DNDS_check_throw_info(encoded.size() < size_t(INT_MAX), "storage record exceeds MPI range");
        int length = int(encoded.size());
        std::vector<int> lengths(mpi.size), offsets(mpi.size);
        MPI_Gather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, mpi.comm);
        std::string gathered;
        if (mpi.rank == 0)
        {
            int total = 0;
            for (int rank = 0; rank < mpi.size; ++rank)
            {
                DNDS_check_throw_info(lengths[rank] <= INT_MAX - total, "gathered record exceeds MPI range");
                offsets[rank] = total;
                total += lengths[rank];
            }
            gathered.resize(total);
        }
        MPI_Gatherv(encoded.data(), length, MPI_CHAR, gathered.data(), lengths.data(), offsets.data(), MPI_CHAR, 0, mpi.comm);
        if (mpi.rank == 0)
        {
            J rows = J::array();
            for (int rank = 0; rank < mpi.size; ++rank)
                rows.push_back(J::parse(gathered.substr(offsets[rank], lengths[rank])));
            std::cout << "PLAN53_RESULT " << rows.dump() << std::endl;
        }
    }

    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 5, "usage: storage_probe dimension width degree full|sweeps|wbap");
        const int dimension = std::stoi(argv[1]), width = std::stoi(argv[2]), degree = std::stoi(argv[3]);
        const std::string profile = argv[4];
        DNDS_check_throw_info((dimension == 2 || dimension == 3) && width >= 2 && width <= 128 && degree >= 1 && degree <= 3,
                              "storage probe geometry/order outside bounds");
        DNDS_check_throw_info(profile == "full" || profile == "sweeps" || profile == "wbap", "unknown capability profile");
        DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        auto mesh = dimension == 2 ? CauchyTest::PeriodicQuads(mpi, width) : CFVTest::CartesianHexes(mpi, width);
        C::ReconstructionSettings settings;
        settings.maxOrder = degree;
        settings.intOrder = degree + 2;
        settings.cacheDiffBase = true;
        settings.cacheDiffBaseOrder = 1;
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, settings.intOrder, false, false);
        const auto capabilities = profile == "full" ? C::ReconstructionCapabilities{} : C::ReconstructionCapabilities::ForSweeps(profile == "wbap");
        const auto before = ResidentBytes(), peakBefore = PeakResidentBytes();
        MPI_Barrier(mpi.comm);
        const double start = MPI_Wtime();
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, capabilities);
        const double prepareSeconds = MPI_Wtime() - start;
        const auto after = ResidentBytes(), peakAfter = PeakResidentBytes();
        const auto storage = plan.StorageReport();
        // Workspace and validation allocations are deliberately after the RSS
        // observations. Their numerical digest checks profile equivalence across
        // separate processes; small independent UTs remain the numerical oracle.
        auto mean = metrics.BuildField(5);
        auto work = plan.BuildWorkspace(5);
        Fill(mean, *mesh);
        Fill(work.coefficients, *mesh);
        auto boundary = plan.BuildBoundaryValues(5);
        {
            auto write = boundary.values.Write();
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
                for (I component = 0; component < 5; ++component)
                    write.View()(iPoint, component) = std::cos(.11 * iPoint + .17 * component);
        }
        metrics.BindHalo(mean).Begin().Wait();
        metrics.BindHalo(work.coefficients).Begin().Wait();
        plan.DoReconstructionIterTo(mean, work.coefficients, work, boundary);
        const auto sweepDigest = Digest(work.temporary);
        J limiterDigest = nullptr;
        if (capabilities.wbap)
        {
            plan.DoLimiterWBAP_3OrderTo(work, degree);
            limiterDigest = Digest(work.temporary);
        }
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        DNDS_check_throw_info(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "rank affinity query failed");
        std::vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &affinity))
                cpus.push_back(cpu);
        J tables = J::array();
        for (const auto &table : storage.tables)
            tables.push_back({{"name", table.name}, {"allocated", table.allocated}, {"payload_bytes", table.payloadBytes}, {"allocation_bytes", table.allocationBytes}});
        Report({{"rank", mpi.rank}, {"np", mpi.size}, {"dimension", dimension}, {"width", width}, {"degree", degree}, {"profile", profile}, {"capability_mask", capabilities.Mask()}, {"owned_cells", metrics.NumCell()}, {"modes", plan.NRecDOF()}, {"cpu_affinity", cpus}, {"mpi_provided_thread_level", provided}, {"prepare_seconds", prepareSeconds}, {"rss_before_bytes", before}, {"rss_after_bytes", after}, {"hwm_before_bytes", peakBefore}, {"hwm_after_bytes", peakAfter}, {"retained_payload_bytes", storage.retainedPayloadBytes}, {"retained_allocation_bytes", storage.retainedAllocationBytes}, {"runtime_scratch_bytes", storage.preparationRuntimeScratchBytes}, {"runtime_peak_bytes", storage.preparationRuntimePeakBytes}, {"sweep_digest", sweepDigest}, {"limiter_digest", limiterDigest}, {"tables", tables}}, mpi);
    }
}

int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    try
    {
        Run(argc, argv); // All MPI-owning objects die before finalization.
    }
    catch (const std::exception &error)
    {
        std::cerr << "storage probe: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
