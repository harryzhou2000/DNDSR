#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "NativePeriodicFixtures.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

static void CheckRoundTrip(const G::MeshAdjacencyRead &source)
{
    auto global = G::MeshAdjacency::Convert(source, G::Adj_PointToGlobal);
    auto local = G::MeshAdjacency::Convert(global.Read(), G::Adj_PointToLocal);
    const auto globalRead = global.Read(), localRead = local.Read();
    CHECK(globalRead.State() == G::Adj_PointToGlobal);
    CHECK(localRead.State() == G::Adj_PointToLocal);
    CHECK(source.Kind() == globalRead.Kind());
    CHECK(source.Source().SameOrder(localRead.Source()));
    CHECK(source.Target().SameOrder(localRead.Target()));
    REQUIRE(source.NumOwned() == globalRead.NumOwned());
    REQUIRE(source.RowCount() == globalRead.RowCount());
    REQUIRE(source.RowCount() == localRead.RowCount());
    for (I row = 0; row < source.RowCount(); ++row)
    {
        REQUIRE(source.RowSize(row) == globalRead.RowSize(row));
        REQUIRE(source.RowSize(row) == localRead.RowSize(row));
        for (size_t slot = 0; slot < source.RowSize(row); ++slot)
        {
            const auto entry = source.ReadEntry(row, slot);
            CHECK(globalRead.Raw(row, slot) == (entry.absent ? DNDS::UnInitIndex : entry.global));
            CHECK(globalRead.ReadEntry(row, slot).global == entry.global);
            CHECK(localRead.ReadEntry(row, slot).global == entry.global);
            if (source.State() == G::Adj_PointToLocal)
                CHECK(localRead.Raw(row, slot) == source.Raw(row, slot));
            if (!entry.absent)
            {
                const auto target = source.Target().GlobalToLocal(entry.global);
                CHECK(localRead.Raw(row, slot) == (target ? *target : -1 - entry.global));
            }
        }
    }
    CHECK_THROWS(G::MeshAdjacency::Convert(source, G::Adj_Unknown));
}

TEST_CASE("Native conversion preserves owned and ghost shape slots and unavailable negative encodings")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 3);
    auto ownedCells = nodes.CreateSibling(G::EntityKind::Cell, 2);
    const I next = 2 * ((mpi.rank + 1) % mpi.size);
    auto cells = ownedCells.WithGhosts(mpi.size > 1 ? std::vector<I>{next} : std::vector<I>{});
    const I last = nodes.NumGlobal() - 1;
    const auto owned = NativePeriodicFixtures::Rows<I>({{last, DNDS::UnInitIndex, 0, last}, {1}});
    const auto ghost = NativePeriodicFixtures::Rows<I>(mpi.size > 1 ? std::vector<std::vector<I>>{{0, last, DNDS::UnInitIndex}} : std::vector<std::vector<I>>{});
    G::MeshAdjacencyRead input(G::Adj::Cell2Node, cells, nodes, G::Adj_PointToGlobal, owned, &ghost);
    CheckRoundTrip(input);
    auto local = G::MeshAdjacency::Convert(input, G::Adj_PointToLocal);
    auto localRead = local.Read();
    CHECK(localRead.Raw(0, 1) == DNDS::UnInitIndex);
    CHECK(input.Raw(0, 0) == last);
    if (mpi.size > 1)
    {
        CHECK_THROWS(G::MeshAdjacency::Convert(input, G::Adj_PointToLocal, false));
        const I missing = mpi.rank == 0 ? last : 0;
        const auto missingLocal = nodes.GlobalToLocal(missing);
        REQUIRE_FALSE(missingLocal.has_value());
        CHECK(localRead.ReadEntry(0, mpi.rank == 0 ? 0 : 2).unavailable);
        CHECK_THROWS(localRead.ConvertedIndex(0, mpi.rank == 0 ? 0 : 2, G::Adj_PointToLocal, false));
    }
    CHECK_THROWS(G::MeshAdjacencyRead(G::Adj::Cell2Node, cells, nodes, G::Adj_Unknown, owned, &ghost));
    const auto wrongGlobal = NativePeriodicFixtures::Rows<I>({{-1}, {0}});
    CHECK_THROWS(G::MeshAdjacencyRead(G::Adj::Cell2Node, ownedCells, nodes, G::Adj_PointToGlobal, wrongGlobal));
    const I present = nodes.LocalToGlobal(0);
    const auto wrongUnavailable = NativePeriodicFixtures::Rows<I>({{-1 - present}, {0}});
    CHECK_THROWS(G::MeshAdjacencyRead(G::Adj::Cell2Node, ownedCells, nodes, G::Adj_PointToLocal, wrongUnavailable));
    const auto outOfRange = NativePeriodicFixtures::Rows<I>({{nodes.NumLocal()}, {0}});
    CHECK_THROWS(G::MeshAdjacencyRead(G::Adj::Cell2Node, ownedCells, nodes, G::Adj_PointToLocal, outOfRange));
}

TEST_CASE("Local relation validation checks ghost bounds global ordering and ghost source self incidences")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto ownedNodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 3);
    auto ownedCells = ownedNodes.CreateSibling(G::EntityKind::Cell, 2);
    const I remoteNode = 3 * ((mpi.rank + mpi.size - 1) % mpi.size);
    const I remoteCell = 2 * ((mpi.rank + 1) % mpi.size);
    auto nodes = ownedNodes.WithGhosts(mpi.size > 1 ? std::vector<I>{remoteNode} : std::vector<I>{});
    auto cells = ownedCells.WithGhosts(mpi.size > 1 ? std::vector<I>{remoteCell} : std::vector<I>{});
    const I own = nodes.LocalToGlobal(0);
    const I other = mpi.size > 1 ? remoteNode : nodes.LocalToGlobal(1);
    const I lower = std::min(own, other), upper = std::max(own, other);
    const I lowerLocal = *nodes.GlobalToLocal(lower), upperLocal = *nodes.GlobalToLocal(upper);
    const auto sortedOwned = NativePeriodicFixtures::Rows<I>({{lowerLocal, upperLocal}, {}});
    const auto sortedGhost = NativePeriodicFixtures::Rows<I>(mpi.size > 1 ? std::vector<std::vector<I>>{{lowerLocal, upperLocal}} : std::vector<std::vector<I>>{});
    G::AdjacencyDescriptor sorted{G::EntityKind::Cell, G::EntityKind::Node};
    sorted.path = {G::EntityKind::Cell, G::EntityKind::Node};
    sorted.selection = G::RelationSelection::Incidence;
    sorted.order = G::RelationOrder::SortedUnique;
    sorted.periodic = G::RelationPeriodicPolicy::EntityIdentity;
    G::MeshAdjacencyRead ordered(sorted, cells, nodes, G::Adj_PointToLocal, sortedOwned, &sortedGhost);
    CHECK(ordered.GlobalRow(0) == std::vector<I>{lower, upper});
    const auto reversed = NativePeriodicFixtures::Rows<I>({{upperLocal, lowerLocal}, {}});
    CHECK_THROWS(G::MeshAdjacencyRead(sorted, cells, nodes, G::Adj_PointToLocal, reversed, &sortedGhost));
    const auto duplicate = NativePeriodicFixtures::Rows<I>({{lowerLocal, lowerLocal}, {}});
    CHECK_THROWS(G::MeshAdjacencyRead(sorted, cells, nodes, G::Adj_PointToLocal, duplicate, &sortedGhost));
    const auto absent = NativePeriodicFixtures::Rows<I>({{DNDS::UnInitIndex}, {}});
    CHECK_THROWS(G::MeshAdjacencyRead(sorted, cells, nodes, G::Adj_PointToLocal, absent, &sortedGhost));
    const auto unordered = NativePeriodicFixtures::Rows<I>({{upperLocal, DNDS::UnInitIndex, lowerLocal, upperLocal}, {}});
    G::MeshAdjacencyRead slots(G::Adj::Cell2Node, cells, nodes, G::Adj_PointToLocal, unordered, &sortedGhost);
    CHECK(slots.GlobalRow(0) == std::vector<I>{upper, DNDS::UnInitIndex, lower, upper});
    CHECK(slots.ReadEntry(0, 1).absent);

    const auto noSelf = G::AdjacencyDescriptor::Composition(G::Adj::Cell2Node, G::Adj::Cell2Node.Inverted(), G::RelationSelection::Any, true);
    const auto distinctOwned = NativePeriodicFixtures::Rows<I>({{1}, {0}});
    const auto distinctGhost = NativePeriodicFixtures::Rows<I>(mpi.size > 1 ? std::vector<std::vector<I>>{{0}} : std::vector<std::vector<I>>{});
    G::MeshAdjacencyRead distinct(noSelf, cells, cells, G::Adj_PointToLocal, distinctOwned, &distinctGhost);
    CHECK(distinct.GlobalRow(0) == std::vector<I>{cells.LocalToGlobal(1)});
    const auto selfOwned = NativePeriodicFixtures::Rows<I>({{0}, {}});
    CHECK_THROWS(G::MeshAdjacencyRead(noSelf, cells, cells, G::Adj_PointToLocal, selfOwned, &distinctGhost));
    if (mpi.size > 1)
    {
        CHECK(ordered.GlobalRow(2) == std::vector<I>{lower, upper});
        if (mpi.rank > 0)
            CHECK(ordered.Raw(2, 0) > ordered.Raw(2, 1));
        const auto reversedGhost = NativePeriodicFixtures::Rows<I>({{upperLocal, lowerLocal}});
        CHECK_THROWS(G::MeshAdjacencyRead(sorted, cells, nodes, G::Adj_PointToLocal, sortedOwned, &reversedGhost));
        for (I invalid : {nodes.NumLocal(), -1 - remoteNode, -1 - nodes.NumGlobal()})
        {
            const auto invalidGhost = NativePeriodicFixtures::Rows<I>({{invalid}});
            CHECK_THROWS(G::MeshAdjacencyRead(G::Adj::Cell2Node, cells, nodes, G::Adj_PointToLocal, unordered, &invalidGhost));
        }
        // A different remote node remains genuinely unavailable; the ghost
        // mapping contains only remoteNode, and absence has its own sentinel.
        const I missing = remoteNode + 1;
        REQUIRE_FALSE(nodes.GlobalToLocal(missing).has_value());
        const auto unavailableGhost = NativePeriodicFixtures::Rows<I>({{-1 - missing, DNDS::UnInitIndex}});
        G::MeshAdjacencyRead unavailable(G::Adj::Cell2Node, cells, nodes, G::Adj_PointToLocal, unordered, &unavailableGhost);
        CHECK(unavailable.ReadEntry(2, 0).unavailable);
        CHECK(unavailable.ReadEntry(2, 0).global == missing);
        CHECK(unavailable.ReadEntry(2, 1).absent);
        CHECK(distinct.GlobalRow(2) == std::vector<I>{cells.LocalToGlobal(0)});
        const auto selfGhost = NativePeriodicFixtures::Rows<I>({{cells.NumOwned()}});
        CHECK_THROWS(G::MeshAdjacencyRead(noSelf, cells, cells, G::Adj_PointToLocal, distinctOwned, &selfGhost));
    }
}

TEST_CASE("Retained relation validation follows its read scope and fresh reads validate changed values")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 2);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, 1);
    const I global = nodes.LocalToGlobal(0);
    auto values = NativePeriodicFixtures::Rows<I>({{global, DNDS::UnInitIndex}});
    G::MeshAdjacency relation(G::Adj::Cell2Node, cells, nodes, values);
    CHECK(relation.Kind() == G::Adj::Cell2Node);
    {
        auto original = relation.Read();
        auto moved = std::move(original);
        CHECK_THROWS(original.ConvertedIndex(0, 0, G::Adj_PointToGlobal));
        CHECK(moved.ConvertedIndex(0, 0, G::Adj_PointToGlobal) == global);
        CHECK(moved.ConvertedIndex(0, 0, G::Adj_PointToLocal, false) == 0);
        CHECK(moved.ConvertedIndex(0, 1, G::Adj_PointToLocal, false) == DNDS::UnInitIndex);
        CHECK_THROWS(values.Write());
        CHECK_THROWS(moved.ConvertedIndex(0, 2, G::Adj_PointToGlobal));
    }
    // MeshAdjacency reuses only a validation tied to the current storage value
    // version: an alias write after the previous scope releases forces a rescan.
    {
        auto write = values.Write();
        write.View()(0, 0) = -1;
    }
    CHECK_THROWS(relation.Read());
    {
        auto write = values.Write();
        write.View()(0, 0) = global + 1;
    }
    auto current = relation.Read();
    CHECK(current.ConvertedIndex(0, 0, G::Adj_PointToGlobal) == global + 1);
    CHECK(current.ConvertedIndex(0, 0, G::Adj_PointToLocal, false) == 1);
}

TEST_CASE("Native conversion preserves dense stride and logical padded CSR rows on empty owners")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    for (const bool emptyPeers : {false, true})
    {
        const bool empty = emptyPeers && mpi.size > 1 && mpi.rank == mpi.size - 1;
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, empty ? 0 : 3);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, empty ? 0 : 2);
        const std::vector<size_t> sizes = empty ? std::vector<size_t>{} : std::vector<size_t>{3, 0};
        for (const int shape : {0, 1, 2})
        {
            const auto layout = shape == 0   ? R::RowLayout::Dense(size_t(cells.NumOwned()), 3, 7)
                                : shape == 1 ? R::RowLayout::Padded(sizes, 6)
                                             : R::RowLayout::Ragged(sizes);
            R::Array<I> values(layout);
            {
                auto write = values.Write();
                for (I row = 0; row < cells.NumOwned(); ++row)
                    for (size_t slot = 0; slot < write.View().RowSize(size_t(row)); ++slot)
                        write.View()(size_t(row), slot) = slot == 1 ? DNDS::UnInitIndex : 0;
            }
            G::MeshAdjacencyRead read(G::Adj::Cell2Node, cells, nodes, G::Adj_PointToGlobal, values);
            CheckRoundTrip(read);
            auto converted = G::MeshAdjacency::Convert(read, G::Adj_PointToGlobal);
            CHECK(converted.Owned().Identity() != values.Identity());
            if (shape == 0)
                CHECK(converted.Owned().Layout().Stride() == 7);
            if (!empty && shape == 1)
                CHECK(converted.Owned().Layout().Stride() == 6);
        }
    }
}

static void CheckCurrent(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi)
{
    auto facade = G::UnstructuredMesh::FromNative(mpi, state);
    for (const auto &key : {G::Adj::Cell2Node, G::Adj::Cell2Face, G::Adj::Face2Node, G::Adj::Face2Cell})
    {
        auto current = G::MeshAdjacencyRead::Current(*facade, key, G::MeshReadRequirements::CellFlux());
        CHECK(current.State() == G::Adj_PointToLocal);
        CHECK(current.RowCount() == facade->Entities(key.from).NumLocal());
        CheckRoundTrip(current);
    }
    for (const auto &key : state->RelationKeys())
        CheckRoundTrip(state->Relation(key));
}

TEST_CASE("Native current numbering conversion survives local reorder redistribution reclosure and empty owners")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto state = fixture.mesh->CurrentNativeState();
    CheckCurrent(state, mpi);
    auto reordered = G::NativeMeshTransition::ReorderLocalCells(*state, mpi, 2, 1).state;
    CheckCurrent(reordered, mpi);
    G::ReorderInput placement;
    placement.explicitMaps.push_back({G::EntityKind::Cell, std::vector<DNDS::MPI_int>(size_t(reordered->NumOwned(G::EntityKind::Cell)), (mpi.rank + 1) % mpi.size)});
    auto moved = G::NativeMeshTransition::Build(*reordered, mpi, placement, 2).state;
    CheckCurrent(moved, mpi);
    auto rebuilt = G::AssembleNativeMesh(mpi, G::NativeMeshAssembler::CaptureOwned(*moved, mpi), 1);
    CheckCurrent(rebuilt, mpi);
    CHECK(state->MeshNamespace() == rebuilt->MeshNamespace());
}

static void CheckTorus(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi)
{
    auto facade = G::UnstructuredMesh::FromNative(mpi, state);
    auto read = G::MeshAdjacencyRead::Current(*facade, G::Adj::Cell2Node, G::MeshReadRequirements::CellFlux());
    CheckRoundTrip(read);
    const auto access = state->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = access.HostView();
    const I expectedIds[]{10000, 10000, 10000, 10000, 10001, 10002, 10001, 10002, 10003};
    const uint8_t expectedBits[]{0, 1, 3, 2, 0, 1, 2, 0, 0};
    const DNDS::real expectedPoints[][2]{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {.5, 0}, {1, .5}, {.5, 1}, {0, .5}, {.5, .5}};
    for (I cell = 0; cell < state->NumOwned(G::EntityKind::Cell); ++cell)
    {
        REQUIRE(read.RowSize(cell) == 9);
        REQUIRE(view.CellNodePeriodicBits(cell).size() == 9);
        G::MeshQueryCoords physical;
        view.GetCoordsOnCell(cell, physical);
        for (size_t slot = 0; slot < 9; ++slot)
        {
            const auto global = read.ReadEntry(cell, slot).global;
            CHECK(global >= 0);
            CHECK(global < state->NumGlobal(G::EntityKind::Node));
            const auto local = read.Target().GlobalToLocal(global);
            REQUIRE(local.has_value());
            CHECK(state->OriginalEntityId(G::EntityKind::Node, *local) == expectedIds[slot]);
            CHECK(global != state->OriginalEntityId(G::EntityKind::Node, *local));
            CHECK(uint8_t(view.CellNodePeriodicBits(cell)[slot]) == expectedBits[slot]);
            CHECK(physical(0, slot) == doctest::Approx(expectedPoints[slot][0]));
            CHECK(physical(1, slot) == doctest::Approx(expectedPoints[slot][1]));
        }
    }
}

TEST_CASE("Periodic conversion keeps repeated canonical IDs ordered frames and distinct persistent IDs")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto input = NativePeriodicFixtures::QuadraticTorus(mpi);
    for (const auto &space : {input.nodes, input.cells, input.boundaries})
    {
        R::Array<I> ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
        {
            auto write = ids.Write();
            for (I row = 0; row < space.NumOwned(); ++row)
                write.View()(size_t(row), 0) = 10000 + space.LocalToGlobal(row);
        }
        input.ids.emplace(space.Kind(), std::move(ids));
    }
    auto state = G::ConstructNativeMesh(std::move(input), {1, true});
    state = G::PrepareNativeMeshTopology(*state, mpi, true, false);
    CheckTorus(state, mpi);
    auto reordered = G::NativeMeshTransition::ReorderLocalCells(*state, mpi, 2, 1).state;
    CheckTorus(reordered, mpi);
    G::ReorderInput placement;
    placement.explicitMaps.push_back({G::EntityKind::Cell, std::vector<DNDS::MPI_int>(size_t(reordered->NumOwned(G::EntityKind::Cell)), (mpi.rank + 1) % mpi.size)});
    auto moved = G::NativeMeshTransition::Build(*reordered, mpi, placement, 2).state;
    CheckTorus(moved, mpi);
    auto rebuilt = G::AssembleNativeMesh(mpi, G::NativeMeshAssembler::CaptureOwned(*moved, mpi), 1);
    CheckTorus(rebuilt, mpi);
    CHECK(state->MeshNamespace() == rebuilt->MeshNamespace());
    CHECK(state->IdHighWater() == rebuilt->IdHighWater());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
