#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include <hdf5.h>
#include <filesystem>
#include <future>
#include <map>
#include <unistd.h>

using namespace DNDS;
using namespace DNDS::Geom;

// Observe the communicator contract without creating intentionally mismatched
// collectives. Each writer executes the same sequence on every rank; a bad
// source-communicator call is counted and fails after the worker joins.
static thread_local MPI_Comm watchedSource = MPI_COMM_NULL;
static thread_local int sourceReductions = 0;
extern "C" int MPI_Allreduce(const void *send, void *receive, int count,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    if (watchedSource != MPI_COMM_NULL && comm == watchedSource)
        ++sourceReductions;
    return PMPI_Allreduce(send, receive, count, datatype, op, comm);
}

static ssp<UnstructuredMesh> CurrentOutputFixture(const MPIInfo &mpi)
{
    auto mesh = make_ssp<UnstructuredMesh>(mpi, 2);
    tPoint zero{0, 0, 0};
    mesh->SetPeriodicGeometry({10, 0, 0}, zero, zero, {0, 10, 0}, zero, zero, zero, zero, zero);
    auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial((root / "data/mesh/IV10_10.cgns").string());
    reader.Deduplicate1to1Periodic(1e-8);
    reader.BuildCell2Cell();
    UnstructuredMeshSerialRW::PartitionOptions options;
    options.metisType = "KWAY";
    options.metisSeed = 42;
    options.metisNcuts = 1;
    reader.MeshPartitionCell2Cell(options);
    reader.PartitionReorderToMeshCell2Cell();
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary(3);
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    return mesh;
}

static std::vector<double> ReadCurrentOutputDataset(hid_t file, const char *name)
{
    auto dataset = H5Dopen2(file, name, H5P_DEFAULT);
    DNDS_check_throw_info(dataset >= 0, "current output dataset missing");
    auto space = H5Dget_space(dataset);
    auto size = H5Sget_simple_extent_npoints(space);
    DNDS_check_throw_info(size >= 0, "current output dataset shape invalid");
    std::vector<double> result(static_cast<size_t>(size));
    double dummy = 0;
    auto error = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                         result.empty() ? &dummy : result.data());
    H5Sclose(space);
    H5Dclose(dataset);
    DNDS_check_throw_info(error >= 0, "current output dataset read failed");
    return result;
}

TEST_CASE("Current periodic VTKHDF maps point callbacks from detached numbering to source rows")
{
    MPIInfo mpi;
    mpi.setWorld();
    auto mesh = CurrentOutputFixture(mpi);
    auto requirements = MeshReadRequirements::CellFlux();
    mesh->PublishNative(requirements);
    auto retained = mesh->AcquireRead(requirements);
    auto view = retained.HostView();
    auto stamp = mesh->versions.Stamp();
    auto nodes = mesh->Entities(EntityKind::Node);
    std::vector<double> ids(size_t(view.NumNodeProc())), coordinates(size_t(view.NumNodeProc()) * 3);
    for (DNDS::index node = 0; node < view.NumNodeProc(); ++node)
    {
        ids[size_t(node)] = double(nodes.LocalToGlobal(node));
        for (int component = 0; component < 3; ++component)
            coordinates[size_t(node) * 3 + size_t(component)] = view.Coord(node)[component];
    }
    // The writer must use native query geometry even when raw workspace rows differ.
    if (mesh->coords.father->Size())
        mesh->coords[0] += tPoint{1000, 1000, 1000};

    std::vector<double> local;
    for (DNDS::index node = 0; node < view.NumNode(); ++node)
    {
        local.push_back(ids[size_t(node)]);
        for (int component = 0; component < 3; ++component)
            local.push_back(view.Coord(node)[component]);
    }
    int count = int(local.size());
    std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size + 1), 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
    for (int rank = 0; rank < mpi.size; ++rank)
        offsets[size_t(rank + 1)] = offsets[size_t(rank)] + counts[size_t(rank)];
    std::vector<double> all(size_t(offsets.back()));
    MPI_Allgatherv(local.data(), count, MPI_DOUBLE, all.data(), counts.data(), offsets.data(), MPI_DOUBLE, mpi.comm);
    std::map<DNDS::index, tPoint> expected;
    for (size_t i = 0; i < all.size(); i += 4)
        expected.emplace(DNDS::index(all[i]), tPoint{all[i + 1], all[i + 2], all[i + 3]});

    int rootPid = int(getpid());
    MPI_Bcast(&rootPid, 1, MPI_INT, 0, mpi.comm);
    auto base = (std::filesystem::temp_directory_path() /
                 ("dndsr-current-output-" + std::to_string(rootPid) + "-np" + std::to_string(mpi.size)))
                    .string();
    MPI_Comm outputComm = MPI_COMM_NULL;
    MPI_Comm_dup(mpi.comm, &outputComm);
    mesh->PrintParallelVTKHDFDataArray(base, "", 0, 0, 1, 1, {}, {}, {}, {}, [](int)
                                       { return "entity_global"; }, [&](int, DNDS::index row)
                                       { return ids.at(size_t(row)); }, [](int)
                                       { return "source_coord"; }, [&](int, DNDS::index row, DNDS::rowsize component)
                                       { return coordinates.at(size_t(row) * 3 + size_t(component)); }, 0, outputComm);
    MPI_Comm_free(&outputComm);
    CHECK(stamp.Valid());
    CHECK(mesh->HasNativeState());
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        auto file = H5Fopen((base + ".vtkhdf").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file >= 0);
        auto points = ReadCurrentOutputDataset(file, "/VTKHDF/Points");
        auto outputIds = ReadCurrentOutputDataset(file, "/VTKHDF/PointData/entity_global");
        auto outputCoordinates = ReadCurrentOutputDataset(file, "/VTKHDF/PointData/source_coord");
        H5Fclose(file);
        REQUIRE(points.size() == outputIds.size() * 3);
        REQUIRE(outputCoordinates.size() == points.size());
        for (size_t row = 0; row < outputIds.size(); ++row)
        {
            const auto &coordinate = expected.at(DNDS::index(outputIds[row]));
            for (int component = 0; component < 3; ++component)
            {
                CHECK(outputCoordinates[row * 3 + size_t(component)] == doctest::Approx(coordinate[component]));
                auto delta = points[row * 3 + size_t(component)] - coordinate[component];
                if (component < 2)
                    CHECK(delta / 10 == doctest::Approx(std::round(delta / 10)).epsilon(1e-10));
                else
                    CHECK(delta == doctest::Approx(0).epsilon(1e-10));
            }
        }
        std::filesystem::remove(base + ".vtkhdf");
    }
    MPI_Barrier(mpi.comm);
}

TEST_CASE("Prepared VTKHDF worker keeps reductions on its output communicator")
{
    MPIInfo mpi;
    mpi.setWorld();
    auto source = CurrentOutputFixture(mpi);
    source->PublishNative(MeshReadRequirements::CellFlux());
    // Complete all source-communicator setup before submitting the writer.
    auto workspace = CloneCurrentMeshForSetup(*source);
    int rootPid = int(getpid());
    MPI_Bcast(&rootPid, 1, MPI_INT, 0, mpi.comm);
    const auto base = (std::filesystem::temp_directory_path() /
                       ("dndsr-worker-output-" + std::to_string(rootPid) + "-np" + std::to_string(mpi.size)))
                          .string();
    MPI_Comm output = MPI_COMM_NULL;
    MPI_Comm_dup(mpi.comm, &output);
    auto worker = std::async(std::launch::async, [workspace, output, base, sourceComm = mpi.comm]
                             {
        watchedSource = sourceComm;
        sourceReductions = 0;
        workspace->PrintParallelVTKHDFDataArray(base, "", 0, 0, 0, 0, {}, {}, {}, {}, {}, {}, {}, {}, 0, output);
        watchedSource = MPI_COMM_NULL;
        return sourceReductions; });
    CHECK(worker.get() == 0);
    MPI_Comm_free(&output);
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        auto file = H5Fopen((base + ".vtkhdf").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file >= 0);
        CHECK_FALSE(ReadCurrentOutputDataset(file, "/VTKHDF/Points").empty());
        H5Fclose(file);
        std::filesystem::remove(base + ".vtkhdf");
    }
    MPI_Barrier(mpi.comm);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    DNDS_check_throw_info(provided >= MPI_THREAD_MULTIPLE, "output worker test requires MPI_THREAD_MULTIPLE");
    auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
