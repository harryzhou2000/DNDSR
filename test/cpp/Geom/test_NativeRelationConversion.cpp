#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "NativePeriodicFixtures.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

static void CheckRoundTrip(const G::RelationReadAccess &source)
{
    auto global = G::NativeRelation::Convert(source, G::Adj_PointToGlobal);
    auto local = G::NativeRelation::Convert(global.Read(), G::Adj_PointToLocal);
    const auto globalRead = global.Read(), localRead = local.Read();
    CHECK(globalRead.State() == G::Adj_PointToGlobal);
    CHECK(localRead.State() == G::Adj_PointToLocal);
    CHECK(source.Kind() == globalRead.Kind());
    CHECK(source.Source().SameOrder(localRead.Source()));
    CHECK(source.Target().SameOrder(localRead.Target()));
    REQUIRE(source.NumOwned() == globalRead.NumOwned());
    REQUIRE(source.Rows() == globalRead.Rows());
    REQUIRE(source.Rows() == localRead.Rows());
    for (I row = 0; row < source.Rows(); ++row)
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
    CHECK_THROWS(G::NativeRelation::Convert(source, G::Adj_Unknown));
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
    G::RelationReadAccess input(G::Adj::Cell2Node, cells, nodes, G::Adj_PointToGlobal, owned, &ghost);
    CheckRoundTrip(input);
    auto local = G::NativeRelation::Convert(input, G::Adj_PointToLocal);
    auto localRead = local.Read();
    CHECK(localRead.Raw(0, 1) == DNDS::UnInitIndex);
    CHECK(input.Raw(0, 0) == last);
    if (mpi.size > 1)
    {
        CHECK_THROWS(G::NativeRelation::Convert(input, G::Adj_PointToLocal, false));
        const I missing = mpi.rank == 0 ? last : 0;
        const auto missingLocal = nodes.GlobalToLocal(missing);
        REQUIRE_FALSE(missingLocal.has_value());
        CHECK(localRead.ReadEntry(0, mpi.rank == 0 ? 0 : 2).unavailable);
    }
    CHECK_THROWS(G::RelationReadAccess(G::Adj::Cell2Node, cells, nodes, G::Adj_Unknown, owned, &ghost));
    const auto wrongGlobal = NativePeriodicFixtures::Rows<I>({{-1}, {0}});
    CHECK_THROWS(G::RelationReadAccess(G::Adj::Cell2Node, ownedCells, nodes, G::Adj_PointToGlobal, wrongGlobal));
    const I present = nodes.LocalToGlobal(0);
    const auto wrongUnavailable = NativePeriodicFixtures::Rows<I>({{-1 - present}, {0}});
    CHECK_THROWS(G::RelationReadAccess(G::Adj::Cell2Node, ownedCells, nodes, G::Adj_PointToLocal, wrongUnavailable));
    const auto outOfRange = NativePeriodicFixtures::Rows<I>({{nodes.NumLocal()}, {0}});
    CHECK_THROWS(G::RelationReadAccess(G::Adj::Cell2Node, ownedCells, nodes, G::Adj_PointToLocal, outOfRange));
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
            G::RelationReadAccess read(G::Adj::Cell2Node, cells, nodes, G::Adj_PointToGlobal, values);
            CheckRoundTrip(read);
            auto converted = G::NativeRelation::Convert(read, G::Adj_PointToGlobal);
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
        auto current = G::RelationReadAccess::Current(*facade, key, G::MeshReadRequirements::CellFlux());
        CHECK(current.State() == G::Adj_PointToLocal);
        CHECK(current.Rows() == facade->Entities(key.from).NumLocal());
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
    auto rebuilt = G::NativeMeshAssembler::Build(mpi, G::NativeMeshAssembler::CaptureOwned(*moved, mpi), 1);
    CheckCurrent(rebuilt, mpi);
    CHECK(state->MeshNamespace() == rebuilt->MeshNamespace());
}

static void CheckTorus(const std::shared_ptr<const G::NativeMeshState> &state, const DNDS::MPIInfo &mpi)
{
    auto facade = G::UnstructuredMesh::FromNative(mpi, state);
    auto read = G::RelationReadAccess::Current(*facade, G::Adj::Cell2Node, G::MeshReadRequirements::CellFlux());
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
    auto state = G::NativeMeshBuilder::Build(std::move(input), {1, true});
    state = G::NativeMeshTopology::Build(*state, mpi, true, false);
    CheckTorus(state, mpi);
    auto reordered = G::NativeMeshTransition::ReorderLocalCells(*state, mpi, 2, 1).state;
    CheckTorus(reordered, mpi);
    G::ReorderInput placement;
    placement.explicitMaps.push_back({G::EntityKind::Cell, std::vector<DNDS::MPI_int>(size_t(reordered->NumOwned(G::EntityKind::Cell)), (mpi.rank + 1) % mpi.size)});
    auto moved = G::NativeMeshTransition::Build(*reordered, mpi, placement, 2).state;
    CheckTorus(moved, mpi);
    auto rebuilt = G::NativeMeshAssembler::Build(mpi, G::NativeMeshAssembler::CaptureOwned(*moved, mpi), 1);
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
