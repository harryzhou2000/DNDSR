#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeRelationSynthetic.hpp"
#include "Geom/Mesh/MeshConnectivity.hpp"
#include "Geom/Mesh/RelationAccessLegacy.hpp"
#include "NativeStructuredInterpolation.hpp"
#include "NativeConnectivityLegacyConditions.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include <array>
#include <cmath>
#include <filesystem>

namespace G = DNDS::Geom;
namespace T = NativeRelationSynthetic;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Inverse: 4-quad serial correctness")
{
    auto cone = T::FourQuadCones(mpi);
    CHECK(cone.Source().NumOwned() == (mpi.size == 1 ? 4 : mpi.rank < 2 ? 2
                                                                        : 0));
    CHECK(cone.Target().NumOwned() == (mpi.size == 1 ? 9 : mpi.rank == 0 ? 5
                                                       : mpi.rank == 1   ? 4
                                                                         : 0));
    auto inverse = G::NativeConnectivity::Inverse(cone.Read(), mpi);
    const std::vector<std::set<I>> expected{{0}, {0, 1}, {1}, {0, 2}, {0, 1, 2, 3}, {1, 3}, {2}, {2, 3}, {3}};
    const auto actual = T::GatherTiny(inverse.Read(), mpi);
    for (I node = 0; node < 9; ++node)
        CHECK(actual[size_t(node)] == expected[size_t(node)]);
}

TEST_CASE("Inverse: round-trip covers original cell2node")
{
    auto cone = T::FourQuadCones(mpi);
    auto inverse = G::NativeConnectivity::Inverse(cone.Read(), mpi);
    auto round = G::NativeConnectivity::Inverse(inverse.Read(), mpi);
    const auto source = cone.Read(), result = round.Read();
    for (I cell = 0; cell < source.NumOwned(); ++cell)
    {
        const auto actual = result.GlobalRow(cell);
        for (const I node : source.GlobalRow(cell))
            CHECK(std::count(actual.begin(), actual.end(), node) == 1);
    }
}

TEST_CASE("ComposeFiltered: 4-quad cell2cell via node-neighbor")
{
    auto cone = T::FourQuadCones(mpi);
    auto inverse = G::NativeConnectivity::Inverse(cone.Read(), mpi);
    auto support = inverse.WithGhostRows(G::RelationGhostRoute(inverse.Source()));
    auto result = G::NativeConnectivity::Compose(cone.Read(), support.Read(), true);
    const auto global = T::GatherTiny(result.Read(), mpi);
    for (I cell = 0; cell < 4; ++cell)
    {
        CHECK(global[size_t(cell)].size() == 3);
        for (I other = 0; other < 4; ++other)
            if (other != cell)
                CHECK(global[size_t(cell)].count(other) == 1);
    }
}

TEST_CASE("ComposeFiltered: face-share filter (minShared=dim)")
{
    auto cone = T::FourQuadCones(mpi);
    auto inverse = G::NativeConnectivity::Inverse(cone.Read(), mpi);
    auto support = inverse.WithGhostRows(G::RelationGhostRoute(inverse.Source()));
    const auto left = cone.Read(), right = support.Read();
    const auto key = G::AdjKind::Composition(left.Kind(), right.Kind(), G::RelationSelection::Custom, true, "shared-two", 1);
    auto result = G::NativeConnectivity::ComposeFiltered(left, right, key,
                                                         [](I a, I b, int count)
                                                         { return a != b && count >= 2; });
    const std::vector<std::set<I>> expected{{1, 2}, {0, 3}, {0, 3}, {1, 2}};
    const auto actual = T::GatherTiny(result.Read(), mpi);
    for (I cell = 0; cell < 4; ++cell)
        CHECK(actual[size_t(cell)] == expected[size_t(cell)]);
}

static std::shared_ptr<const G::NativeMeshState> importedGrid(bool legacyPartition)
{
    namespace L = NativeConnectivityLegacyConditions;
    return L::UniformSquare10(mpi, legacyPartition ? L::Placement::LegacyReaderMetis : L::Placement::NativeEven);
}

static std::array<I, 100> gridCells(const G::NativeMeshState &state)
{
    std::array<I, 100> globals;
    globals.fill(-1);
    auto access = state.Snapshot(DNDS::Runtime::Device::Host()).AcquireRead();
    const auto view = access.HostView();
    const auto cells = state.Entities(G::EntityKind::Cell);
    for (I cell = 0; cell < cells.NumOwned(); ++cell)
    {
        G::MeshQueryCoords points;
        view.GetCoordsOnCell(cell, points);
        const int x = int(std::lround(10 * points.row(0).minCoeff()));
        const int y = int(std::lround(10 * points.row(1).minCoeff()));
        REQUIRE(x >= 0);
        REQUIRE(x < 10);
        REQUIRE(y >= 0);
        REQUIRE(y < 10);
        globals[size_t(10 * y + x)] = cells.LocalToGlobal(cell);
    }
    MPI_Allreduce(MPI_IN_PLACE, globals.data(), 100, DNDS::DNDS_MPI_INDEX, MPI_MAX, mpi.comm);
    for (const auto global : globals)
        CHECK(global >= 0);
    return globals;
}

TEST_CASE("Regression: Inverse matches independent lattice node stars on UniformSquare_10")
{
    bool legacyPartition = true;
    SUBCASE("legacy reader METIS KWAY seed42 ufactor30 ncuts1") { legacyPartition = true; }
    SUBCASE("additional native direct even partition") { legacyPartition = false; }
    auto state = importedGrid(legacyPartition);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto cone = G::RelationReadAccess::Current(*mesh, G::Adj::Cell2Node, G::MeshReadRequirements::Primary());
    auto inverse = G::NativeConnectivity::Inverse(cone, mpi);
    const auto actual = inverse.Read();
    const auto globals = gridCells(*state);
    auto access = mesh->AcquireRead();
    for (I node = 0; node < actual.NumOwned(); ++node)
    {
        const auto point = access.HostView().Coord(node);
        const int x = int(std::lround(10 * point[0])), y = int(std::lround(10 * point[1]));
        std::set<I> expected;
        for (const int cx : {x - 1, x})
            for (const int cy : {y - 1, y})
                if (cx >= 0 && cx < 10 && cy >= 0 && cy < 10)
                    expected.insert(globals[size_t(10 * cy + cx)]);
        const auto row = actual.GlobalRow(node);
        CHECK(std::set<I>(row.begin(), row.end()) == expected);
    }
}

TEST_CASE("Regression: ComposeFiltered matches independent lattice neighbors on UniformSquare_10")
{
    bool legacyPartition = true;
    SUBCASE("legacy reader METIS KWAY seed42 ufactor30 ncuts1") { legacyPartition = true; }
    SUBCASE("additional native direct even partition") { legacyPartition = false; }
    auto state = importedGrid(legacyPartition);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    auto cone = G::RelationReadAccess::Current(*mesh, G::Adj::Cell2Node, G::MeshReadRequirements::Primary());
    auto inverse = G::NativeConnectivity::Inverse(cone, mpi);
    auto support = inverse.WithGhostRows(G::RelationGhostRoute(inverse.Source()));
    auto neighbors = G::NativeConnectivity::Compose(cone, support.Read(), true);
    const auto actual = neighbors.Read();
    const auto globals = gridCells(*state);
    auto access = mesh->AcquireRead();
    for (I cell = 0; cell < actual.NumOwned(); ++cell)
    {
        G::MeshQueryCoords points;
        access.HostView().GetCoordsOnCell(cell, points);
        const int x = int(std::lround(10 * points.row(0).minCoeff()));
        const int y = int(std::lround(10 * points.row(1).minCoeff()));
        std::set<I> expected;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if ((dx || dy) && x + dx >= 0 && x + dx < 10 && y + dy >= 0 && y + dy < 10)
                    expected.insert(globals[size_t(10 * (y + dy) + x + dx)]);
        const auto row = actual.GlobalRow(cell);
        CHECK(std::set<I>(row.begin(), row.end()) == expected);
    }
}

TEST_CASE("Native registry cone management rejects duplicate and omitted definitions")
{
    auto cone = T::FourQuadCones(mpi);
    G::NativeConnectivity registry(mpi);
    CHECK(registry.RegisteredCount() == 0);
    CHECK_FALSE(registry.Find(G::Adj::Cell2Node));
    CHECK_FALSE(registry.Find(G::Adj::Node2Cell));
    registry.Register(cone.Read());
    auto found = registry.Find(G::Adj::Cell2Node);
    REQUIRE(found);
    CHECK(found->Kind().from == G::EntityKind::Cell);
    CHECK(found->Kind().to == G::EntityKind::Node);
    CHECK(registry.Find(G::Adj::Cell2Node).get() == found.get());
    auto faces = cone.Source().CreateSibling(G::EntityKind::Face, 0);
    G::NativeRelation other(G::Adj::Face2Node, faces, cone.Target(),
                            DNDS::Runtime::Array<I>(DNDS::Runtime::RowLayout::Ragged(std::vector<size_t>{})));
    registry.Register(other.Read());
    CHECK(registry.Find(G::Adj::Face2Node));
    CHECK(registry.RegisteredCount() == 2);
    CHECK_THROWS(registry.Register(cone.Read()));
    auto direct = G::CompiledGhostTree::compile(G::GhostSpec{{{G::EntityKind::Cell, {G::Adj::Cell2Node}, G::EntityKind::Node}}});
    CHECK_NOTHROW(registry.Evaluate(direct));
    auto missing = G::CompiledGhostTree::compile(G::GhostSpec{{{G::EntityKind::Cell, {G::Adj::Cell2Face}, G::EntityKind::Face}}});
    CHECK_THROWS(registry.Evaluate(missing));
}

TEST_CASE("Native registry support management preserves semantic inverse identity")
{
    auto cone = T::FourQuadCones(mpi);
    auto inverse = G::NativeConnectivity::Inverse(cone.Read(), mpi);
    CHECK(inverse.Read().Kind() == cone.Read().Kind().Inverted());
    CHECK(inverse.Read().Kind() != G::Adj::Node2Cell);
    G::NativeConnectivity registry(mpi);
    registry.Register(cone.Read());
    registry.Register(inverse.Read());
    auto found = registry.Find(inverse.Read().Kind());
    REQUIRE(found);
    CHECK(found->Kind().from == G::EntityKind::Node);
    CHECK(found->Kind().to == G::EntityKind::Cell);
    CHECK(registry.Find(inverse.Read().Kind()).get() == found.get());
    CHECK(registry.RegisteredCount() == 2);
    CHECK_FALSE(registry.Find(G::Adj::Bnd2Node));
    CHECK_THROWS(registry.Register(inverse.Read()));
    auto spec = G::CompiledGhostTree::compile(G::GhostSpec{{{G::EntityKind::Node, {inverse.Read().Kind()}, G::EntityKind::Cell}}});
    CHECK_NOTHROW(registry.Evaluate(spec));
}

TEST_CASE("Native cone typed access validates family state and ordered companions")
{
    static_assert(!std::is_default_constructible_v<G::NativeRelation>);
    static_assert(!std::is_default_constructible_v<G::RelationReadAccess>);
    auto cone = T::FourQuadCones(mpi);
    auto read = cone.Read();
    std::vector<std::vector<G::NodePeriodicBits>> labels(size_t(read.NumOwned()), std::vector<G::NodePeriodicBits>(4));
    for (size_t row = 0; row < labels.size(); ++row)
        for (size_t slot = 0; slot < 4; ++slot)
            labels[row][slot] = G::NodePeriodicBits{uint8_t((row + slot) % 8)};
    G::RelationPeriodicBits valid(read, NativePeriodicFixtures::Rows(labels));
    valid.Validate(read);
    for (I row = 0; row < read.NumOwned(); ++row)
        for (size_t slot = 0; slot < 4; ++slot)
            CHECK(valid(row, slot) == labels[size_t(row)][slot]);
    // Fixed-width and variable-width payloads expose the same typed relation
    // contract; variant instantiation is not an alternate storage path.
    auto faces = cone.Source().CreateSibling(G::EntityKind::Face, mpi.rank == 0 ? 1 : 0);
    DNDS::Runtime::Array<I> fixed(DNDS::Runtime::RowLayout::Dense(size_t(faces.NumOwned()), 2));
    if (faces.NumOwned())
    {
        auto write = fixed.Write();
        write.View()(0, 0) = 0;
        write.View()(0, 1) = DNDS::UnInitIndex;
    }
    G::NativeRelation fixedSupport(G::Adj::Face2Cell, faces, cone.Source(), fixed);
    auto fixedRead = fixedSupport.Read();
    if (faces.NumOwned())
    {
        CHECK(fixedRead.RowSize(0) == 2);
        CHECK(fixedRead.ReadEntry(0, 0).global == 0);
        CHECK(fixedRead.ReadEntry(0, 1).absent);
    }
    CHECK_THROWS(G::RelationReadAccess(G::Adj::Face2Node, cone.Source(), cone.Target(), G::Adj_PointToGlobal, cone.Owned()));
    CHECK_THROWS(G::RelationReadAccess(G::Adj::Cell2Node, cone.Source(), cone.Target(), G::Adj_Unknown, cone.Owned()));
    DNDS::Runtime::Array<G::NodePeriodicBits> wrong(DNDS::Runtime::RowLayout::Dense(size_t(cone.Source().NumOwned()) + 1, 3));
    CHECK_THROWS(G::RelationPeriodicBits(read, wrong));
    auto moved = std::move(read);
    CHECK_THROWS(read.Kind());
    CHECK(moved.Kind() == G::Adj::Cell2Node);
}

TEST_CASE("Native registry lookup retains the exact typed row lease after registry destruction")
{
    auto cone = T::FourQuadCones(mpi);
    auto alias = cone.Owned();
    std::shared_ptr<const G::RelationReadAccess> retained;
    {
        G::NativeConnectivity registry(mpi);
        registry.Register(cone.Read());
        retained = registry.Find(G::Adj::Cell2Node);
        REQUIRE(retained);
    }
    if (alias.Rows())
        CHECK_THROWS(alias.Write());
    for (I row = 0; row < retained->NumOwned(); ++row)
    {
        CHECK(retained->RowSize(row) == 4);
        CHECK(retained->GlobalRow(row).size() == 4);
    }
    retained.reset();
    CHECK_NOTHROW(alias.Write());
}

TEST_CASE("Native support has explicit sorted entity semantics independent of node image companions")
{
    auto cone = T::FourQuadCones(mpi);
    auto inverse = G::NativeConnectivity::Inverse(cone.Read(), mpi);
    auto read = inverse.Read();
    CHECK(read.Kind().order == G::RelationOrder::SortedUnique);
    for (I row = 0; row < read.NumOwned(); ++row)
    {
        const auto globals = read.GlobalRow(row);
        CHECK(std::is_sorted(globals.begin(), globals.end()));
        CHECK(std::adjacent_find(globals.begin(), globals.end()) == globals.end());
    }
}

TEST_CASE("Inverse: fixed-width face2cell input keeps exact sentinel semantics")
{
    auto faces = G::EntitySpace::CreateOwned(G::EntityKind::Face, mpi, mpi.rank == 0 ? 5 : 0);
    auto cells = faces.CreateSibling(G::EntityKind::Cell, mpi.rank == 0 ? 2 : 0);
    DNDS::Runtime::Array<I> values(DNDS::Runtime::RowLayout::Dense(size_t(faces.NumOwned()), 2, 4));
    if (mpi.rank == 0)
    {
        auto write = values.Write();
        const I expected[5][2]{{0, DNDS::UnInitIndex}, {0, 1}, {0, DNDS::UnInitIndex}, {1, DNDS::UnInitIndex}, {1, DNDS::UnInitIndex}};
        for (size_t row = 0; row < 5; ++row)
            for (size_t slot = 0; slot < 2; ++slot)
                write.View()(row, slot) = expected[row][slot];
    }
    G::NativeRelation physical(G::Adj::Face2Cell, faces, cells, values);
    auto inverse = G::NativeConnectivity::Inverse(physical.Read(), mpi);
    if (mpi.rank == 0)
    {
        const auto a = inverse.Read().GlobalRow(0), b = inverse.Read().GlobalRow(1);
        CHECK(std::set<I>(a.begin(), a.end()) == std::set<I>{0, 1, 2});
        CHECK(std::set<I>(b.begin(), b.end()) == std::set<I>{1, 3, 4});
    }
}

TEST_CASE("Periodic 2x2x2: node-count composition cannot replace physical face neighbors")
{
    auto mesh = NativeStructuredInterpolation::BuildHex(mpi, 2, 7, true);
    auto cone = mesh.cone.Read();
    auto inverse = G::NativeConnectivity::Inverse(cone, mpi);
    auto support = inverse.WithGhostRows(G::RelationGhostRoute(inverse.Source()));
    const auto key = G::AdjKind::Composition(cone.Kind(), support.Read().Kind(), G::RelationSelection::Custom, true, "three-shared", 1);
    auto wrong = G::NativeConnectivity::ComposeFiltered(cone, support.Read(), key, [](I a, I b, int shared)
                                                        { return a != b && shared >= 3; });
    for (I cell = 0; cell < cone.NumOwned(); ++cell)
        CHECK(wrong.Read().RowSize(cell) == 7);
    // The independent geometric oracle for 2x2x2 torus is only 3 distinct
    // face neighbors, two opposite physical faces to each neighbor.
    CHECK(key != G::Adj::Cell2CellFace);
}

// Explicit generic-pair compatibility unit: this shape adapter remains public.
// No legacy UnstructuredMesh allocation or production setup participates.
using namespace DNDS;
using namespace DNDS::Geom;
TEST_CASE("narrowAdjToFixed: variable-to-fixed-2 conversion")
{
    // Build a variable-width adjacency with rows of sizes 1 and 2.
    tAdjPair source;
    source.InitPair("source", mpi);
    source.father->Resize(3);
    source.father->ResizeRow(0, 2);
    source.father->operator()(0, 0) = 10;
    source.father->operator()(0, 1) = 20;
    source.father->ResizeRow(1, 1);
    source.father->operator()(1, 0) = 30;
    source.father->ResizeRow(2, 2);
    source.father->operator()(2, 0) = 40;
    source.father->operator()(2, 1) = 50;

    tAdj2Pair target;
    target.InitPair("target", mpi);
    target.father->Resize(3);

    narrowAdjToFixed<2>(source, target, 3);

    CHECK(target.father->operator()(0, 0) == 10);
    CHECK(target.father->operator()(0, 1) == 20);
    CHECK(target.father->operator()(1, 0) == 30);
    CHECK(target.father->operator()(1, 1) == UnInitIndex);
    CHECK(target.father->operator()(2, 0) == 40);
    CHECK(target.father->operator()(2, 1) == 50);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
