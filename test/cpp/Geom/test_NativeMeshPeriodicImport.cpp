#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshPeriodicImport.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativeMeshArchive.hpp"
#include "Geom/Mesh/RelationAccess.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include "Geom/CGNS.hpp"
#include <filesystem>
#include <unistd.h>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    template <class T>
    R::Array<T> Rows(const std::vector<std::vector<T>> &rows, size_t width = 0)
    {
        auto layout = width ? R::RowLayout::Dense(rows.size(), width) : R::RowLayout::Ragged([&]
                                                                                             { std::vector<size_t> sizes; for (const auto &row : rows) sizes.push_back(row.size()); return sizes; }());
        R::Array<T> result(layout);
        auto write = result.Write();
        for (size_t row = 0; row < rows.size(); ++row)
            std::copy(rows[row].begin(), rows[row].end(), write.View()[row].data());
        return result;
    }
    DNDS::MPIInfo MPI()
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        return mpi;
    }
    G::NativeSerialMeshData Fixture(bool quadratic = true, bool volume = false)
    {
        G::NativeSerialMeshData data;
        const std::vector<std::vector<DNDS::real>> quad{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {.5, 0, 0}, {1, .5, 0}, {.5, 1, 0}, {0, .5, 0}, {.5, .5, 0}};
        const std::vector<std::vector<DNDS::real>> hex{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
        auto points = volume ? hex : quad;
        if (!volume && !quadratic)
            points.resize(4);
        data.coordinates = Rows(points, 3);
        std::vector<I> cell;
        for (size_t node = 0; node < points.size(); ++node)
            cell.push_back(I(node));
        data.cellNodes = Rows<I>({cell});
        data.cellInfo = Rows<G::ElemInfo>({{G::ElemInfo{volume ? G::Elem::Hex8 : quadratic ? G::Elem::Quad9
                                                                                           : G::Elem::Quad4,
                                                        G::INTERNAL_ZONE}}},
                                          1);
        const std::vector<std::vector<I>> boundaries = volume ? std::vector<std::vector<I>>{{0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {0, 4, 7, 3}, {4, 5, 6, 7}} : quadratic ? std::vector<std::vector<I>>{{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}}
                                                                                                                                                                                            : std::vector<std::vector<I>>{{0, 1}, {1, 2}, {2, 3}, {3, 0}};
        data.boundaryNodes = Rows(boundaries);
        const std::vector<G::t_index> zones = volume ? std::vector<G::t_index>{12, 12, -4, 12, -1, 12} : std::vector<G::t_index>{-2, -4, -5, -1};
        std::vector<std::vector<G::ElemInfo>> info;
        std::vector<std::vector<I>> ids;
        for (size_t row = 0; row < boundaries.size(); ++row)
        {
            info.push_back({G::ElemInfo{volume ? G::Elem::Quad4 : quadratic ? G::Elem::Line3
                                                                            : G::Elem::Line2,
                                        zones[row]}});
            ids.push_back({2000 + I(row)});
        }
        data.boundaryInfo = Rows(info, 1);
        data.boundaryIds = Rows(ids, 1);
        data.cellIds = Rows<I>({{3000}}, 1);
        ids.clear();
        for (size_t node = 0; node < points.size(); ++node)
            ids.push_back({1000 + 7 * I(node)});
        data.nodeIds = Rows(ids, 1);
        return data;
    }
    void CG(int result)
    {
        DNDS_check_throw_info(result == CG_OK, cg_get_error());
    }
    void Write(const std::string &path, const G::NativeSerialMeshData &data, bool periodicProperty = false)
    {
        int file, base, zone, coordinate, section, bc;
        CG(cg_open(path.c_str(), CG_MODE_WRITE, &file));
        struct Guard
        {
            int file;
            ~Guard() { cg_close(file); }
        } guard{file};
        CG(cg_base_write(file, "Base", 2, 2, &base));
        cgsize_t sizes[]{cgsize_t(data.coordinates.RowCount()), 1, 0};
        CG(cg_zone_write(file, base, "Zone", sizes, Unstructured, &zone));
        auto coords = data.coordinates.Read();
        for (size_t component = 0; component < 2; ++component)
        {
            std::vector<double> values(data.coordinates.RowCount());
            for (size_t row = 0; row < values.size(); ++row)
                values[row] = coords.View()(row, component);
            CG(cg_coord_write(file, base, zone, RealDouble, component == 0 ? "CoordinateX" : "CoordinateY", values.data(), &coordinate));
        }
        auto cellNodes = data.cellNodes.Read(), boundaryNodes = data.boundaryNodes.Read();
        std::vector<cgsize_t> nodes;
        for (auto node : cellNodes.View()[0])
            nodes.push_back(cgsize_t(node + 1));
        CG(cg_section_write(file, base, zone, "Cell", QUAD_9, 1, 1, 0, nodes.data(), &section));
        nodes.clear();
        for (size_t row = 0; row < data.boundaryNodes.RowCount(); ++row)
            for (auto node : boundaryNodes.View()[row])
                nodes.push_back(cgsize_t(node + 1));
        CG(cg_section_write(file, base, zone, "Boundary", BAR_3, 2, 5, 0, nodes.data(), &section));
        const char *names[]{"P2", "P1_DONOR", "P2_DONOR", "P1"};
        for (size_t row = 0; row < 4; ++row)
        {
            cgsize_t element = cgsize_t(2 + row);
            CG(cg_boco_write(file, base, zone, names[row], BCGeneral, PointList, 1, &element, &bc));
            CG(cg_boco_gridlocation_write(file, base, zone, bc, EdgeCenter));
        }
        if (periodicProperty)
        {
            int connection;
            const cgsize_t main[]{1, 4}, donor[]{2, 3};
            CG(cg_conn_write(file, base, zone, "TransformedSeam", Vertex, Abutting1to1, PointList, 2, main,
                             "Zone", Unstructured, PointListDonor, LongInteger, 2, donor, &connection));
            const float center[3]{0, 0, 0}, angle[3]{0, 0, 0}, translation[3]{1, 0, 0};
            CG(cg_conn_periodic_write(file, base, zone, connection, center, angle, translation));
        }
    }
    G::t_index Zone(const std::string &name)
    {
        if (name == "P1")
            return -1;
        if (name == "P2")
            return -2;
        if (name == "P1_DONOR")
            return -4;
        if (name == "P2_DONOR")
            return -5;
        throw std::runtime_error("unexpected fixture boundary name");
    }
}

TEST_CASE("Native serial periodic canonicalization preserves high-order image geometry with compact canonical IDs")
{
    bool quadratic = true, volume = false;
    SUBCASE("quad9 doubly periodic") {}
    SUBCASE("quad4 doubly periodic") { quadratic = false; }
    SUBCASE("hex8 periodic") { volume = true; }
    auto input = Fixture(quadratic, volume);
    auto original = input.coordinates.Read();
    auto result = G::CanonicalizeNativePeriodic(input, {});
    REQUIRE(result.periodic);
    CHECK(result.coordinates.RowCount() == (volume ? 4 : quadratic ? 4
                                                                   : 1));
    CHECK(result.identity.nextIds.at(G::EntityKind::Node) == I(result.coordinates.RowCount()));
    CHECK(input.coordinates.RowCount() == (volume ? 8 : quadratic ? 9
                                                                  : 4));
    {
        auto compactIds = result.nodeIds.Read();
        for (size_t row = 0; row < result.nodeIds.RowCount(); ++row)
            CHECK(compactIds.View()(row, 0) == I(row));
    }
    auto nodes = result.cellNodes.Read();
    auto bits = result.cellNodesPbi->Read();
    auto coords = result.coordinates.Read();
    for (size_t slot = 0; slot < input.coordinates.RowCount(); ++slot)
    {
        const auto node = size_t(nodes.View()(0, slot));
        auto point = result.periodicInfo.GetCoordByBits(G::tPoint{coords.View()(node, 0), coords.View()(node, 1), coords.View()(node, 2)}, bits.View()(0, slot));
        for (size_t component = 0; component < 3; ++component)
            CHECK(point[component] == original.View()(slot, component));
    }
    if (quadratic && !volume)
    {
        CHECK(std::vector<I>(nodes.View()[0].begin(), nodes.View()[0].end()) == std::vector<I>{0, 0, 0, 0, 1, 2, 1, 2, 3});
        const uint8_t expected[]{0, 1, 3, 2, 0, 1, 2, 0, 0};
        for (size_t slot = 0; slot < 9; ++slot)
            CHECK(uint8_t(bits.View()(0, slot)) == expected[slot]);
        auto ids = result.nodeIds.Read();
        CHECK(ids.View()(0, 0) == 0);
        CHECK(ids.View()(1, 0) == 1);
        CHECK(ids.View()(2, 0) == 2);
        CHECK(ids.View()(3, 0) == 3);
    }
}

TEST_CASE("Periodic ingress preserves an explicit ID reservation without reserving discarded source labels")
{
    auto source = Fixture();
    source.identity.nextIds[G::EntityKind::Node] = 4096;
    auto result = G::CanonicalizeNativePeriodic(source, {});
    CHECK(result.identity.nextIds.at(G::EntityKind::Node) == 4096);
    auto ids = result.nodeIds.Read();
    for (size_t row = 0; row < result.nodeIds.RowCount(); ++row)
        CHECK(ids.View()(row, 0) == I(row));
    CHECK(source.nodeIds.Read().View()(0, 0) == 1000);
}

TEST_CASE("Native serial periodic identification rejects incomplete ambiguous and fixed-point pairings without changing source")
{
    auto source = Fixture();
    auto bad = source;
    bad.boundaryInfo = Rows<G::ElemInfo>({{{G::Elem::Line3, -2}}, {{G::Elem::Line3, 12}}, {{G::Elem::Line3, -5}}, {{G::Elem::Line3, -1}}}, 1);
    CHECK_THROWS(G::CanonicalizeNativePeriodic(bad, {}));
    bad = source;
    auto points = R::Array<DNDS::real>(R::RowLayout::Dense(9, 3));
    {
        auto read = source.coordinates.Read();
        auto write = points.Write();
        for (size_t row = 0; row < 9; ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = read.View()(row, component);
        write.View()(5, 0) = 1.1;
    }
    bad.coordinates = std::move(points);
    CHECK_THROWS(G::CanonicalizeNativePeriodic(bad, {}));
    G::NativePeriodicImportOptions options;
    options.tolerance = -1;
    CHECK_THROWS(G::CanonicalizeNativePeriodic(source, options));
    auto fixed = Fixture(false);
    fixed.boundaryNodes = Rows<I>({{0, 1}, {0, 1}, {1, 2}, {2, 3}});
    fixed.boundaryInfo = Rows<G::ElemInfo>({{{G::Elem::Line2, -1}}, {{G::Elem::Line2, -4}}, {{G::Elem::Line2, 12}}, {{G::Elem::Line2, 12}}}, 1);
    options = {};
    options.frames.translation[1].map().setZero();
    CHECK_THROWS(G::CanonicalizeNativePeriodic(fixed, options));
    auto ambiguous = source;
    ambiguous.boundaryNodes = Rows<I>({{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}, {1, 2, 5}, {3, 0, 7}});
    ambiguous.boundaryInfo = Rows<G::ElemInfo>({{{G::Elem::Line3, -2}}, {{G::Elem::Line3, -4}}, {{G::Elem::Line3, -5}}, {{G::Elem::Line3, -1}}, {{G::Elem::Line3, -4}}, {{G::Elem::Line3, -1}}}, 1);
    CHECK_THROWS(G::CanonicalizeNativePeriodic(ambiguous, {}));
    auto nonbijective = Fixture(false);
    nonbijective.coordinates = Rows<DNDS::real>({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}, {0, 2, 0}, {1, 2, 0}}, 3);
    nonbijective.nodeIds = Rows<I>({{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}}, 1);
    nonbijective.boundaryNodes = Rows<I>({{0, 3}, {1, 2}, {3, 6}, {5, 7}});
    nonbijective.boundaryInfo = Rows<G::ElemInfo>({{{G::Elem::Line2, -1}}, {{G::Elem::Line2, -4}}, {{G::Elem::Line2, -1}}, {{G::Elem::Line2, -4}}}, 1);
    CHECK_THROWS(G::CanonicalizeNativePeriodic(nonbijective, {}));
    CHECK_NOTHROW(G::CanonicalizeNativePeriodic(source, {}));
}

TEST_CASE("Native periodic canonicalization uses rigid rotations and labels all original node occurrences")
{
    auto source = Fixture(false);
    source.coordinates = Rows<DNDS::real>({{1, 0, 0}, {2, 0, 0}, {0, 2, 0}, {0, 1, 0}}, 3);
    source.boundaryInfo = Rows<G::ElemInfo>({{{G::Elem::Line2, -1}}, {{G::Elem::Line2, 12}}, {{G::Elem::Line2, -4}}, {{G::Elem::Line2, 12}}}, 1);
    G::NativePeriodicImportOptions options;
    options.frames.translation[1].map().setZero();
    options.frames.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    const auto result = G::CanonicalizeNativePeriodic(source, options);
    CHECK(result.coordinates.RowCount() == 2);
    auto bits = result.cellNodesPbi->Read();
    const uint8_t expected[]{0, 0, 1, 1};
    for (size_t slot = 0; slot < 4; ++slot)
        CHECK(uint8_t(bits.View()(0, slot)) == expected[slot]);
    // A half-turn can reconstruct either side geometrically. The source main
    // side remains canonical even when donor input rows precede its rows.
    source.coordinates = Rows<DNDS::real>({{-1, 0, 0}, {-2, 0, 0}, {2, 0, 0}, {1, 0, 0}}, 3);
    source.boundaryInfo = Rows<G::ElemInfo>({{{G::Elem::Line2, -4}}, {{G::Elem::Line2, 12}}, {{G::Elem::Line2, -1}}, {{G::Elem::Line2, 12}}}, 1);
    options.frames.rotation[1].map() << -1, 0, 0, 0, -1, 0, 0, 0, 1;
    auto halfTurn = G::CanonicalizeNativePeriodic(source, options);
    auto ids = halfTurn.nodeIds.Read();
    CHECK(ids.View()(0, 0) == 0);
    CHECK(ids.View()(1, 0) == 1);
    auto mainCoords = halfTurn.coordinates.Read();
    CHECK(mainCoords.View()(0, 0) == 2);
    CHECK(mainCoords.View()(1, 0) == 1);
    auto halfTurnBits = halfTurn.cellNodesPbi->Read();
    const uint8_t expectedHalfTurn[]{1, 1, 0, 0};
    for (size_t slot = 0; slot < 4; ++slot)
        CHECK(uint8_t(halfTurnBits.View()(0, slot)) == expectedHalfTurn[slot]);
}

TEST_CASE("Native periodic CGNS import builds directly on a nonzero reader and retains frames through transition and archive")
{
    auto mpi = MPI();
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("native-periodic-import-" + std::to_string(pid) + ".cgns")).string();
    if (mpi.rank == 0)
        Write(path, Fixture());
    MPI_Barrier(mpi.comm);
    const int reader = mpi.size - 1;
    auto state = G::ReadNativeCGNS(mpi, path, 2, Zone, reader, {}, G::NativePeriodicImportOptions{});
    CHECK(state->NumGlobal(G::EntityKind::Node) == 4);
    CHECK(state->NextEntityId(G::EntityKind::Node) == 4);
    CHECK(state->BorrowHostView().IsPeriodic());
    state = G::PrepareNativeMeshTopology(*state, mpi);
    CHECK(state->NumGlobal(G::EntityKind::Face) == 2);
    G::ReorderInput reorder;
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd, G::EntityKind::Face})
        reorder.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(state->NumOwned(kind)), (mpi.rank + 1) % mpi.size)});
    state = G::NativeMeshTransition::Build(*state, mpi, reorder).state;
    const auto name = state->MeshNamespace();
    const auto archive = path + "-r" + std::to_string(mpi.rank) + ".json";
    auto serializer = std::make_shared<DNDS::Serializer::SerializerJSON>();
    serializer->OpenFile(archive, false);
    G::WriteNativeMeshArchive(*state, mpi, serializer, "mesh");
    serializer->CloseFile();
    state.reset();
    serializer->OpenFile(archive, true);
    state = G::ReadNativeMeshArchive(mpi, serializer, "mesh");
    serializer->CloseFile();
    CHECK(state->MeshNamespace() == name);
    CHECK(state->NextEntityId(G::EntityKind::Node) == 4);
    CHECK(state->NumGlobal(G::EntityKind::Face) == 2);
    auto bad = G::NativePeriodicImportOptions{};
    if (mpi.rank == reader)
        bad.frames.translation[1].map()[0] = 2;
    CHECK_THROWS(G::ReadNativeCGNS(mpi, path, 2, Zone, reader, {}, bad));
    CHECK_NOTHROW(G::ReadNativeCGNS(mpi, path, 2, Zone, reader, {}, G::NativePeriodicImportOptions{}));
    const auto properties = path + "-property.cgns";
    if (mpi.rank == 0)
        Write(properties, Fixture(), true);
    MPI_Barrier(mpi.comm);
    CHECK_THROWS(G::ReadNativeCGNS(mpi, properties, 2, Zone, reader, {}, G::NativePeriodicImportOptions{}));
    std::filesystem::remove(archive);
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
    {
        std::filesystem::remove(path);
        std::filesystem::remove(properties);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
