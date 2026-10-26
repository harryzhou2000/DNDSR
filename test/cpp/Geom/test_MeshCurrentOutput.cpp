#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "LegacyMeshFixtureCodec.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/NativeMeshOutput.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "MeshFacadeOverlapFixture.hpp"
#include "NativePeriodicFixtures.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <hdf5.h>
#include <filesystem>
#include <future>
#include <map>
#include <fstream>
#include <sstream>
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
    bool native = false;
    SUBCASE("prepared legacy output") { native = false; }
    SUBCASE("native output buffers") { native = true; }
    // The native writer builds its output-only buffers directly using output
    // comm; the baseline retains its separately prepared legacy workspace.
    auto workspace = native ? source : TestFixtureCodec::DecodeLegacy(*source, true);
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

TEST_CASE("Native VTKHDF owns output buffers without legacy mesh conversion")
{
    MPIInfo mpi;
    mpi.setWorld();
    auto fixture = OverlapTest::BuildNativeFixture(mpi);
    auto mesh = fixture.mesh;
    auto stamp = mesh->versions.Stamp();
    auto source = mesh->CurrentNativeState();
    auto read = mesh->AcquireRead(source->Requirements());
    const auto view = read.HostView();
    auto layout = NativeMeshOutput::Build(*source, mpi);
    CHECK_FALSE(mesh->coords.father);
    CHECK_FALSE(mesh->cell2node.father);
    CHECK(layout.globalNodes == 121);
    CHECK(layout.globalCells == 100);
    CHECK(layout.globalConnectivity == 400);
    auto coordinates = layout.coordinates.Read();
    for (size_t row = 0; row < layout.coordinates.Rows(); ++row)
        for (size_t component = 0; component < 3; ++component)
            CHECK(coordinates.View()(row, component) == view.Coord(DNDS::index(row))[component]);
    for (DNDS::index cell = 0; cell < mesh->NumCell(); ++cell)
    {
        const auto id = mesh->OriginalEntityId(EntityKind::Cell, cell);
        const auto x = id % 10, y = id / 10;
        const std::array<DNDS::index, 4> expected{11 * y + x, 11 * y + x + 1, 11 * (y + 1) + x + 1, 11 * (y + 1) + x};
        CHECK(layout.types.at(size_t(cell)) == 9);
        for (size_t slot = 0; slot < 4; ++slot)
            CHECK(layout.connectivity.at(size_t(cell) * 4 + slot) == expected[slot]);
    }
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto base = (std::filesystem::temp_directory_path() / ("native-output-" + std::to_string(pid))).string();
    MPI_Comm output;
    MPI_Comm_dup(mpi.comm, &output);
    mesh->PrintParallelVTKHDFDataArray(base, "", 1, 0, 1, 0, [](int)
                                       { return "cell_id"; }, [&](int, DNDS::index row)
                                       { return double(mesh->OriginalEntityId(EntityKind::Cell, row)); }, {}, {}, [](int)
                                       { return "node_id"; }, [&](int, DNDS::index row)
                                       { return double(mesh->OriginalEntityId(EntityKind::Node, row)); }, {}, {}, 0, output);
    MPI_Comm_free(&output);
    CHECK(stamp.Valid());
    CHECK_FALSE(mesh->coords.father);
    CHECK_FALSE(mesh->cell2node.father);
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        auto file = H5Fopen((base + ".vtkhdf").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file >= 0);
        auto cellIds = ReadCurrentOutputDataset(file, "/VTKHDF/CellData/cell_id");
        auto nodeIds = ReadCurrentOutputDataset(file, "/VTKHDF/PointData/node_id");
        auto connectivity = ReadCurrentOutputDataset(file, "/VTKHDF/Connectivity");
        auto points = ReadCurrentOutputDataset(file, "/VTKHDF/Points");
        H5Fclose(file);
        REQUIRE(cellIds.size() == 100);
        REQUIRE(nodeIds.size() == 121);
        REQUIRE(connectivity.size() == 400);
        REQUIRE(points.size() == 363);
        for (size_t cell = 0; cell < 100; ++cell)
        {
            CHECK(cellIds[cell] == double(cell));
            const size_t x = cell % 10, y = cell / 10;
            const std::array<size_t, 4> expected{11 * y + x, 11 * y + x + 1, 11 * (y + 1) + x + 1, 11 * (y + 1) + x};
            for (size_t slot = 0; slot < 4; ++slot)
                CHECK(connectivity[cell * 4 + slot] == double(expected[slot]));
        }
        for (size_t node = 0; node < 121; ++node)
        {
            CHECK(nodeIds[node] == double(node));
            CHECK(points[node * 3] == doctest::Approx(double(node % 11) / 10));
            CHECK(points[node * 3 + 1] == doctest::Approx(double(node / 11) / 10));
            CHECK(points[node * 3 + 2] == 0);
        }
        std::filesystem::remove(base + ".vtkhdf");
    }
    MPI_Barrier(mpi.comm);
}

TEST_CASE("Native serial and distributed VTU and Tecplot preserve rows and periodic images")
{
    const MPIInfo mpi(MPI_COMM_WORLD);
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        const bool periodic = scenario % 2;
        const int flag = scenario / 2;
        const int readerRank = mpi.size - 1;
        auto mesh = periodic ? UnstructuredMesh::FromNative(mpi, NativeMeshBuilder::Build(NativePeriodicFixtures::TwoCells(mpi), {1, true}))
                             : OverlapTest::BuildNativeFixture(mpi).mesh;
        const auto stamp = mesh->versions.Stamp();
        auto read = mesh->AcquireRead();
        const auto view = read.HostView();
        // The public raw members are not the native semantics.
        mesh->dim = 3;
        mesh->isPeriodic = !periodic;
        int pid = int(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
        const auto base = (std::filesystem::temp_directory_path() /
                           ("native-piece-output-" + std::to_string(pid) + "-" + std::to_string(scenario)))
                              .string();
        UnstructuredMeshSerialRW writer(mesh, readerRank);
        if (flag == 0)
            writer.BuildSerialOut();
        auto cellName = [](int)
        { return "cell_id"; };
        auto nodeName = [](int)
        { return "node_id"; };
        auto cellData = [&](int, DNDS::index row)
        { return double(flag == 0 ? row : mesh->OriginalEntityId(EntityKind::Cell, row)); };
        auto nodeData = [&](int, DNDS::index row)
        { return double(flag == 0 ? row : mesh->OriginalEntityId(EntityKind::Node, row)); };
        writer.PrintSerialPartVTKDataArray(base, "", 1, 0, 1, 0, cellName, cellData, {}, {}, nodeName, nodeData, {}, {}, 0, flag);
        writer.PrintSerialPartPltBinaryDataArray(base, 1, 1, cellName, cellData, nodeName, nodeData, 0, flag);
        if (flag == 1 || mpi.rank == readerRank)
        {
            auto load = [](const std::string &path)
            {
                std::ifstream input(path, std::ios::binary);
                REQUIRE(input.good());
                return std::string(std::istreambuf_iterator<char>(input), {});
            };
            auto xml = load(base + (flag == 0 ? ".vtu" : fmt::format(".vtu.dir/{:04d}.vtu", mpi.rank)));
            auto array = [&](const std::string &marker)
            {
                auto start = xml.find(marker);
                REQUIRE(start != std::string::npos);
                if (marker == "<Points")
                    start = xml.find("<DataArray", start);
                start = xml.find('>', start) + 1;
                const auto end = xml.find("</DataArray>", start);
                REQUIRE(end != std::string::npos);
                std::istringstream stream(xml.substr(start, end - start));
                std::vector<double> values;
                for (double value; stream >> value;)
                    values.push_back(value);
                return values;
            };
            const auto coordinates = array("<Points"), nodes = array("Name=\"node_id\""), cells = array("Name=\"cell_id\"");
            const auto connectivity = array("Name=\"connectivity\""), types = array("Name=\"types\"");
            REQUIRE(coordinates.size() == nodes.size() * 3);
            REQUIRE(cells.size() == size_t(flag == 0 ? mesh->Entities(EntityKind::Cell).NumGlobal() : view.NumCell()));
            REQUIRE(connectivity.size() == cells.size() * 4);
            REQUIRE(types.size() == cells.size());
            for (size_t cell = 0; cell < cells.size(); ++cell)
            {
                const auto id = DNDS::index(cells[cell]);
                CHECK(cells[cell] == cellData(0, DNDS::index(cell)));
                CHECK(types[cell] == 9);
                for (size_t slot = 0; slot < 4; ++slot)
                {
                    const auto point = size_t(connectivity[cell * 4 + slot]);
                    REQUIRE(point < nodes.size());
                    const double x = periodic ? .5 * (double(id) + double(slot == 1 || slot == 2))
                                              : .1 * (double(id % 10) + double(slot == 1 || slot == 2));
                    const double y = periodic ? double(slot >= 2) : .1 * (double(id / 10) + double(slot >= 2));
                    CHECK(coordinates[point * 3] == doctest::Approx(x));
                    CHECK(coordinates[point * 3 + 1] == doctest::Approx(y));
                    CHECK(coordinates[point * 3 + 2] == 0);
                    const DNDS::index node = periodic ? DNDS::index((id + DNDS::index(slot == 1 || slot == 2)) % 2 + 2 * DNDS::index(slot >= 2))
                                                      : 11 * (id / 10 + DNDS::index(slot >= 2)) + id % 10 + DNDS::index(slot == 1 || slot == 2);
                    CHECK(nodes[point] == double(node));
                }
            }
            // Tecplot's final data block is XYZ, node field, cell field, rank, then
            // zero-based quad connectivity. Decode bytes without alignment casts.
            auto binary = load(base + (flag == 0 ? ".plt" : fmt::format(".dir/{:06d}.plt", mpi.rank)));
            CHECK(binary.substr(0, 8) == "#!TDV112");
            const auto doubles = 4 * nodes.size() + 2 * cells.size();
            const auto bytes = doubles * sizeof(double) + connectivity.size() * sizeof(int32_t);
            REQUIRE(binary.size() >= bytes);
            size_t cursor = binary.size() - bytes;
            auto scalar = [&]
            {
                double value;
                std::memcpy(&value, binary.data() + cursor, sizeof(value));
                cursor += sizeof(value);
                return value;
            };
            for (size_t component = 0; component < 3; ++component)
                for (size_t point = 0; point < nodes.size(); ++point)
                    CHECK(scalar() == doctest::Approx(coordinates[point * 3 + component]));
            for (double value : nodes)
                CHECK(scalar() == value);
            for (double value : cells)
                CHECK(scalar() == value);
            for (size_t cell = 0; cell < cells.size(); ++cell)
                CHECK(scalar() == double(flag == 0 ? mesh->Entities(EntityKind::Cell).Owner(DNDS::index(cell)) : mpi.rank));
            for (double expected : connectivity)
            {
                int32_t value;
                std::memcpy(&value, binary.data() + cursor, sizeof(value));
                cursor += sizeof(value);
                CHECK(value == int32_t(expected));
            }
            CHECK(cursor == binary.size());
        }
        CHECK(stamp.Valid());
        CHECK_FALSE(mesh->coords.father);
        CHECK_FALSE(mesh->cell2node.father);
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0)
        {
            std::filesystem::remove_all(base + ".vtu.dir");
            std::filesystem::remove_all(base + ".dir");
            std::filesystem::remove(base + ".pvtu");
            std::filesystem::remove(base + ".vtu");
            std::filesystem::remove(base + ".plt");
        }
        MPI_Barrier(mpi.comm);
    }
}

TEST_CASE("Native output rejects cross-rank publication and periodic schedule mismatches")
{
    MPIInfo mpi;
    mpi.setWorld();
    if (mpi.size == 1)
        return;
    auto first = OverlapTest::BuildNativeFixture(mpi).mesh;
    auto second = OverlapTest::BuildNativeFixture(mpi).mesh;
    CHECK_THROWS(NativeMeshOutput::Build(*(mpi.rank == 0 ? first : second)->CurrentNativeState(), mpi));
    auto periodic = CurrentOutputFixture(mpi);
    periodic->PublishNative(MeshReadRequirements::CellFlux());
    CHECK_THROWS(NativeMeshOutput::Build(*(mpi.rank == 0 ? first : periodic)->CurrentNativeState(), mpi));
    CHECK_NOTHROW(NativeMeshOutput::Build(*first->CurrentNativeState(), mpi));
}

TEST_CASE("CGNS output rejects a globally empty native mesh before file creation")
{
    namespace R = DNDS::Runtime;
    MPIInfo mpi;
    mpi.setWorld();
    auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, 0);
    auto cells = nodes.CreateSibling(EntityKind::Cell, 0);
    auto boundaries = nodes.CreateSibling(EntityKind::Bnd, 0);
    auto mesh = UnstructuredMesh::FromNative(mpi, NativeMeshBuilder::Build(
                                                      {mpi, 2, cells, nodes, boundaries, R::Array<DNDS::index>(R::RowLayout::Dense(0, 4)), R::Array<DNDS::index>(R::RowLayout::Dense(0, 2)), R::Array<DNDS::real>(R::RowLayout::Dense(0, 3)), R::Array<ElemInfo>(R::RowLayout::Dense(0, 1)), R::Array<ElemInfo>(R::RowLayout::Dense(0, 1)), {}}));
    const auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->PrintMeshCGNS("/unused/native-empty.cgns", {}, {}));
    CHECK(stamp.Valid());
}

TEST_CASE("Native CGNS output round trips independently owned nodes and empty ranks")
{
    namespace R = DNDS::Runtime;
    MPIInfo mpi;
    mpi.setWorld();
    const bool cellOwner = mpi.rank == 0, nodeOwner = mpi.rank == mpi.size - 1;
    auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, nodeOwner ? 4 : 0);
    auto cells = nodes.CreateSibling(EntityKind::Cell, cellOwner ? 1 : 0);
    auto boundaries = nodes.CreateSibling(EntityKind::Bnd, cellOwner ? 4 : 0);
    R::Array<DNDS::real> coords(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
    R::Array<DNDS::index> cellNodes(R::RowLayout::Dense(size_t(cells.NumOwned()), 4));
    R::Array<DNDS::index> boundaryNodes(R::RowLayout::Dense(size_t(boundaries.NumOwned()), 2));
    R::Array<ElemInfo> cellInfo(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
    R::Array<ElemInfo> boundaryInfo(R::RowLayout::Dense(size_t(boundaries.NumOwned()), 1));
    const std::array<tPoint, 4> points{tPoint{0, 0, 0}, tPoint{2, 0, 0}, tPoint{2, 1, 0}, tPoint{0, 1, 0}};
    {
        auto c = coords.Write();
        auto cn = cellNodes.Write(), bn = boundaryNodes.Write();
        auto ci = cellInfo.Write(), bi = boundaryInfo.Write();
        if (nodeOwner)
            for (size_t row = 0; row < 4; ++row)
                for (size_t j = 0; j < 3; ++j)
                    c.View()(row, j) = points[row][j];
        if (cellOwner)
        {
            ci.View()(0, 0) = ElemInfo{Elem::Quad4, INTERNAL_ZONE};
            for (size_t row = 0; row < 4; ++row)
            {
                cn.View()(0, row) = DNDS::index(row);
                bn.View()(row, 0) = DNDS::index(row);
                bn.View()(row, 1) = DNDS::index((row + 1) % 4);
                bi.View()(row, 0) = ElemInfo{Elem::Line2, row == 0 ? BC_ID_DEFAULT_WALL : BC_ID_DEFAULT_FAR};
            }
        }
    }
    auto mesh = UnstructuredMesh::FromNative(mpi, NativeMeshBuilder::Build(
                                                      {mpi, 2, cells, nodes, boundaries, cellNodes, boundaryNodes, coords, cellInfo, boundaryInfo, {}}, {1, true}));
    auto stamp = mesh->versions.Stamp();
    auto held = mesh->AcquireRead();
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("native-cgns-output-" + std::to_string(pid) + ".cgns")).string();
    auto names = [](t_index zone)
    { return zone == BC_ID_DEFAULT_WALL ? "WALL" : "FAR"; };
    CHECK_THROWS(mesh->PrintMeshCGNS(path, names, {"WALL"}));
    CHECK_THROWS(mesh->PrintMeshCGNS(path, [](t_index) -> std::string
                                     { throw std::runtime_error("boundary callback failed"); }, {"WALL", "FAR"}));
    if (mpi.size > 1)
        CHECK_THROWS(mesh->PrintMeshCGNS(path, names, mpi.rank == 0 ? std::vector<std::string>{"WALL", "FAR"} : std::vector<std::string>{"FAR", "WALL"}));
    CHECK(stamp.Valid());
    mesh->dim = 3; // Raw legacy fields cannot override native query geometry.
    mesh->PrintMeshCGNS(path, names, {"WALL", "FAR"});
    CHECK(stamp.Valid());
    CHECK_FALSE(mesh->coords.father);
    CHECK_FALSE(mesh->cell2node.father);
    auto restored = UnstructuredMesh::FromNative(mpi, ReadNativeCGNS(mpi, path, 2,
                                                                     [](const std::string &name)
                                                                     { return name == "WALL" ? BC_ID_DEFAULT_WALL : BC_ID_DEFAULT_FAR; }, mpi.size - 1, {1, true}));
    CHECK(restored->NumCellGlobal() == 1);
    CHECK(restored->NumNodeGlobal() == 4);
    CHECK(restored->NumBndGlobal() == 4);
    auto read = restored->AcquireRead();
    const auto view = read.HostView();
    const auto nodeSpace = restored->Entities(EntityKind::Node);
    for (DNDS::index row = 0; row < view.NumNodeProc(); ++row)
        CHECK(view.Coord(row) == points.at(size_t(nodeSpace.LocalToGlobal(row))));
    for (DNDS::index row = 0; row < view.NumCell(); ++row)
        for (size_t slot = 0; slot < 4; ++slot)
            CHECK(view.Coord(view.CellNodes(row)[slot]) == points[slot]);
    for (DNDS::index row = 0; row < view.NumBnd(); ++row)
    {
        const auto cone = view.BoundaryNodes(row);
        const auto first = nodeSpace.LocalToGlobal(cone[0]);
        CHECK(nodeSpace.LocalToGlobal(cone[1]) == (first + 1) % 4);
        CHECK(view.GetBndZone(row) == (first == 0 ? BC_ID_DEFAULT_WALL : BC_ID_DEFAULT_FAR));
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::filesystem::remove(path);
}

TEST_CASE("Native primary compatibility archive streams payloads including boundary curves")
{
    const MPIInfo mpi(MPI_COMM_WORLD);
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        auto mesh = scenario == 1 ? UnstructuredMesh::FromNative(mpi, NativeMeshBuilder::Build(NativePeriodicFixtures::TwoCells(mpi), {2, true}))
                                  : OverlapTest::BuildNativeFixture(mpi).mesh;
        if (scenario == 2)
            mesh = mesh->ExtractBoundary(); // 1D native mesh with durable ancestry.
        const auto stamp = mesh->versions.Stamp();
        auto retained = mesh->AcquireRead();
        const auto view = retained.HostView();
        auto archive = make_ssp<Serializer::SerializerJSON>();
        archive->CreatePath("/");
        const auto cwd = archive->GetCurrentPath();
        WriteCurrentPrimaryMeshSnapshot(*mesh, archive, "primary");
        CHECK(archive->GetCurrentPath() == cwd);
        CHECK(stamp.Valid());
        CHECK_FALSE(mesh->coords.father);
        CHECK_FALSE(mesh->cell2node.father);
        auto restored = make_ssp<UnstructuredMesh>(mpi, view.Dimension());
        restored->ReadSerialize(archive, "primary");
        REQUIRE(restored->NumCell() == view.NumCell());
        REQUIRE(restored->NumNode() == view.NumNode());
        REQUIRE(restored->NumBnd() == view.NumBnd());
        CHECK(restored->isPeriodic == view.IsPeriodic());
        const auto nodes = mesh->Entities(EntityKind::Node);
        for (DNDS::index node = 0; node < view.NumNode(); ++node)
        {
            CHECK(restored->coords[node] == view.Coord(node));
            CHECK(restored->node2nodeOrig(node, 0) == mesh->OriginalEntityId(EntityKind::Node, node));
        }
        for (DNDS::index cell = 0; cell < view.NumCell(); ++cell)
        {
            const auto cone = view.CellNodes(cell);
            REQUIRE(restored->cell2node.RowSize(cell) == DNDS::rowsize(cone.size()));
            CHECK(restored->cell2cellOrig(cell, 0) == mesh->OriginalEntityId(EntityKind::Cell, cell));
            CHECK(restored->cellElemInfo(cell, 0).zone == view.GetCellZone(cell));
            CHECK(restored->cellElemInfo(cell, 0).getElemType() == view.GetCellElement(cell).type);
            for (size_t slot = 0; slot < cone.size(); ++slot)
            {
                CHECK(restored->cell2node(cell, DNDS::rowsize(slot)) == nodes.LocalToGlobal(cone[slot]));
                if (view.IsPeriodic())
                    CHECK(uint8_t(restored->cell2nodePbi(cell, DNDS::rowsize(slot))) == uint8_t(view.CellNodePeriodicBits(cell)[slot]));
            }
        }
        for (DNDS::index boundary = 0; boundary < view.NumBnd(); ++boundary)
        {
            const auto cone = view.BoundaryNodes(boundary);
            REQUIRE(restored->bnd2node.RowSize(boundary) == DNDS::rowsize(cone.size()));
            CHECK(restored->bnd2bndOrig(boundary, 0) == mesh->OriginalEntityId(EntityKind::Bnd, boundary));
            CHECK(restored->bndElemInfo(boundary, 0).zone == view.GetBndZone(boundary));
            for (size_t slot = 0; slot < cone.size(); ++slot)
            {
                CHECK(restored->bnd2node(boundary, DNDS::rowsize(slot)) == nodes.LocalToGlobal(cone[slot]));
                if (view.IsPeriodic())
                    CHECK(uint8_t(restored->bnd2nodePbi(boundary, DNDS::rowsize(slot))) == uint8_t(view.BoundaryNodePeriodicBits(boundary)[slot]));
            }
        }
    }
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
