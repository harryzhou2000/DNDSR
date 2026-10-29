#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeMeshOverlapFixture.hpp"
#include "Geom/Mesh/MeshEdit.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshArchive.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include <limits>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;

TEST_CASE("Native geometry edits preserve detached snapshots and escaped value copies")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    auto mesh = fixture.mesh;
    auto order = mesh->Entities(G::EntityKind::Node);
    auto route = order.BuildRowTransferPlan();
    auto detached = G::NativeMeshSnapshot::Capture(*mesh, G::MeshReadRequirements::CellFlux());
    std::vector<G::tPoint> before;
    {
        auto read = mesh->AcquireRead();
        for (DNDS::index row = 0; row < mesh->NumNodeProc(); ++row)
            before.push_back(read.HostView().Coord(row));
    }
    auto edit = G::MeshEdit::Begin(mesh);
    CHECK(edit.Coord(0).isApprox(before[0]));
    auto escaped = edit.Coord(0);
    for (DNDS::index node = 0; node < edit.NumNode(); ++node)
        edit.SetCoord(node, (edit.Coord(node) + G::tPoint{0.25, 0.5, 0}).eval());
    auto values = mesh->versions.Stamp(R::Change::Values);
    edit.Commit();
    CHECK(edit.Closed());
    CHECK_THROWS(edit.Coord(0));
    CHECK_FALSE(values.Valid());
    CHECK(order.Valid());
    CHECK(route.Valid());
    CHECK(order.SameOrder(mesh->Entities(G::EntityKind::Node)));
    auto committed = mesh->versions.Stamp();
    {
        // These native operations are supported, but publication must remain
        // blocked while a managed reader holds the current query authority.
        auto held = mesh->AcquireRead();
        CHECK_THROWS(mesh->TransformCoords([](const G::tPoint &p)
                                           { return p; }));
        CHECK_THROWS(mesh->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2));
        CHECK_THROWS(mesh->RectifyBoundaryCoordinates([](G::t_index)
                                                      { return 1; }));
        CHECK_THROWS(mesh->BuildNodeWallDist([](G::t_index)
                                             { return true; }, {}));
        CHECK(held.HostView().Coord(0).isApprox(before[0] + G::tPoint{0.25, 0.5, 0}));
    }
    {
        auto held = mesh->AcquireRead();
        CHECK_THROWS(mesh->ReplaceNativeForSetup(mesh->CurrentNativeState()));
    }
    CHECK(committed.Valid());
    CHECK(mesh->HasNativeState());
    escaped = G::tPoint{-1, -2, -3};
    auto read = mesh->AcquireRead();
    for (DNDS::index node = 0; node < mesh->NumNodeProc(); ++node)
        CHECK(read.HostView().Coord(node).isApprox(before[size_t(node)] + G::tPoint{0.25, 0.5, 0}));
    CHECK(detached.AcquireRead().HostView().Coord(0).isApprox(before[0]));
    CHECK(detached.AcquireRead().HostView().Coord(0) != read.HostView().Coord(0));
}

TEST_CASE("Staged metadata and neighbor-order edits validate before publication")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    auto mesh = fixture.mesh;
    const auto oldInfo = mesh->GetCellInfo(0);
    CHECK(mesh->HasNativeState());
    {
        auto edit = G::MeshEdit::Begin(mesh);
        auto invalid = edit.CellInfo(0);
        invalid.setElemType(G::Elem::UnknownElem);
        if (mpi.rank == 0)
            edit.SetCellInfo(0, invalid);
        CHECK_THROWS(edit.Commit());
        edit.Abort();
        CHECK(mesh->GetCellInfo(0).type == oldInfo.type);
        CHECK(mesh->versions.CanMutate());
    }
    {
        auto edit = G::MeshEdit::Begin(mesh);
        const auto point = edit.Coord(0);
        if (mpi.rank == 0)
            edit.SetCoord(0, G::tPoint{std::numeric_limits<DNDS::real>::quiet_NaN(), 0, 0});
        CHECK_THROWS(edit.Commit());
        edit.SetCoord(0, point);
        auto info = edit.CellInfo(0);
        info.zone = 54321;
        edit.SetCellInfo(0, info);
        edit.Commit();
        info.zone = -999;
        CHECK(mesh->GetCellInfo(0).zone == 54321);
    }
    auto beforeOrder = mesh->Entities(G::EntityKind::Cell);
    std::vector<std::vector<DNDS::index>> expectedNeighbors;
    auto globalNeighbor = [](const G::EntitySpace &space, DNDS::index local)
    {
        if (local == DNDS::UnInitIndex)
            return local;
        return local < 0 ? -1 - local : space.LocalToGlobal(local);
    };
    {
        auto read = mesh->AcquireRead();
        for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
        {
            expectedNeighbors.emplace_back();
            for (auto local : read.HostView().CellCells(cell))
                expectedNeighbors.back().push_back(globalNeighbor(beforeOrder, local));
            std::reverse(expectedNeighbors.back().begin(), expectedNeighbors.back().end());
        }
    }
    auto graphStamp = mesh->versions.Stamp(R::Change::Topology);
    auto edit = G::MeshEdit::Begin(mesh);
    for (DNDS::index cell = 0; cell < edit.NumCell(); ++cell)
    {
        auto neighbors = edit.CellNeighbors(cell);
        for (size_t slot = 0; slot < neighbors.size(); ++slot)
            edit.SetCellNeighbor(cell, DNDS::rowsize(slot), neighbors[neighbors.size() - 1 - slot]);
    }
    edit.Commit();
    CHECK_FALSE(beforeOrder.Valid());
    CHECK(graphStamp.Valid());
    {
        auto read = mesh->AcquireRead();
        const auto currentOrder = mesh->Entities(G::EntityKind::Cell);
        for (DNDS::index cell = 0; cell < mesh->NumCellProc(); ++cell)
        {
            std::vector<DNDS::index> actual;
            for (auto local : read.HostView().CellCells(cell))
                actual.push_back(globalNeighbor(currentOrder, local));
            CHECK(actual == expectedNeighbors[size_t(cell)]);
        }
    }
    auto invalid = G::MeshEdit::Begin(mesh);
    bool changed = false;
    for (DNDS::index cell = 0; cell < invalid.NumCell() && !changed; ++cell)
        if (!invalid.CellNeighbors(cell).empty())
        {
            invalid.SetCellNeighbor(cell, 0, DNDS::UnInitIndex);
            changed = true;
        }
    CHECK_THROWS(invalid.Commit());
    invalid.Abort();
    CHECK(mesh->versions.CanMutate());
    auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
    CHECK(read.HostView().GetCellInfo(0).zone == 54321);
}

TEST_CASE("Uncommitted scopes discard copies and active reads block edit publication")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi);
    auto mesh = fixture.mesh;
    const G::tPoint before = mesh->Coord(0);
    {
        auto edit = G::MeshEdit::Begin(mesh);
        edit.SetCoord(0, G::tPoint{1, 2, 3});
        CHECK_THROWS(mesh->TransformCoords([](const G::tPoint &p)
                                           { return p; }));
    }
    CHECK(mesh->Coord(0).isApprox(before));
    CHECK(mesh->versions.CanMutate());
    {
        auto aborted = G::MeshEdit::Begin(mesh);
        if (mpi.rank == 0)
            aborted.Abort();
        CHECK_THROWS(aborted.Commit());
        aborted.Abort();
        CHECK(mesh->Coord(0).isApprox(before));
        CHECK(mesh->versions.CanMutate());
    }
    auto edit = G::MeshEdit::Begin(mesh);
    {
        auto retained = mesh->AcquireRead();
        CHECK_THROWS(edit.Commit());
        CHECK(edit.Closed());
        CHECK(retained.HostView().Coord(0).isApprox(before));
    }
    CHECK(mesh->versions.CanMutate());
}

TEST_CASE("Mesh edits require explicit native authority and recover after collective rejection")
{
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto unpublished = std::make_shared<G::UnstructuredMesh>(mpi, 2);
    const auto stamp = unpublished->versions.Stamp();
    CHECK_THROWS(G::MeshEdit::Begin(unpublished));
    CHECK(stamp.Valid());
    CHECK(unpublished->versions.CanMutate());
    CHECK_FALSE(unpublished->HasNativeState());
    auto native = G::OverlapTest::BuildNativeFixture(mpi).mesh;
    const auto point = native->Coord(0);
    if (mpi.size > 1)
    {
        auto mixed = mpi.rank == 0 ? unpublished : native;
        CHECK_THROWS(G::MeshEdit::Begin(mixed));
        CHECK(mixed->versions.CanMutate());
    }
    unpublished->ReplaceNativeForSetup(native->CurrentNativeState());
    CHECK_THROWS(G::MeshEdit::Begin(unpublished, mpi.rank == 0 ? 0 : 1));
    auto edit = G::MeshEdit::Begin(unpublished);
    const auto original = edit.Coord(0);
    edit.SetCoord(0, original);
    edit.Commit();
    CHECK(unpublished->Coord(0).isApprox(point));
    CHECK(native->HasNativeState());
}

TEST_CASE("Native-only edits preserve namespace IDs named definitions and immutable source buffers")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, 5);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, 3);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    const auto next = (mpi.rank + 1) % mpi.size;
    R::Array<DNDS::index> cone(R::RowLayout::Dense(3, 3)), emptyCone(R::RowLayout::Dense(0, 2));
    {
        auto write = cone.Write();
        for (size_t cell = 0; cell < 3; ++cell)
            for (size_t slot = 0; slot < 3; ++slot)
                write.View()(cell, slot) = 5 * next + DNDS::index(cell + slot);
    }
    R::Array<DNDS::real> coords(R::RowLayout::Dense(5, 3));
    {
        auto write = coords.Write();
        for (size_t row = 0; row < 5; ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = component == 0 ? 5 * mpi.rank + row : 0;
    }
    R::Array<G::ElemInfo> info(R::RowLayout::Dense(3, 1)), bndInfo(R::RowLayout::Dense(0, 1));
    {
        auto write = info.Write();
        for (size_t row = 0; row < 3; ++row)
            write.View()(row, 0) = G::ElemInfo{G::Elem::Tri3, G::INTERNAL_ZONE};
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, cone, emptyCone, coords, info, bndInfo, {}};
    auto state = G::ConstructNativeMesh(input, {1, true});
    auto owned = G::NativeMeshAssembler::CaptureOwned(*state, mpi);
    owned.requirements.wallDistance = true;
    owned.nodeWallDist.emplace(R::RowLayout::Dense(5, 3));
    {
        auto write = owned.nodeWallDist->Write();
        for (size_t row = 0; row < 5; ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = 1;
    }
    owned.namedRelations.emplace(G::Adj::Cell2Cell, *owned.relations[size_t(G::MeshRelation::CellCells)]);
    state = G::AssembleNativeMesh(mpi, std::move(owned));
    const auto name = state->MeshNamespace();
    const auto highWater = state->IdHighWater();
    const auto keys = state->RelationKeys();
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state);
    std::vector<std::vector<std::vector<DNDS::index>>> expected;
    for (const auto &key : keys)
    {
        auto read = state->Relation(key);
        expected.emplace_back();
        for (DNDS::index row = 0; row < read.RowCount(); ++row)
            expected.back().push_back(read.GlobalRow(row));
    }
    auto edit = G::MeshEdit::Begin(mesh);
    for (DNDS::index row = 0; row < edit.NumNode(); ++row)
        edit.SetCoord(row, (edit.Coord(row) + G::tPoint{0.5, 1, 2}).eval());
    for (DNDS::index cell = 0; cell < edit.NumCell(); ++cell)
    {
        auto neighbors = edit.CellNeighbors(cell);
        REQUIRE(neighbors.size() == 2);
        for (size_t slot = 0; slot < neighbors.size(); ++slot)
            edit.SetCellNeighbor(cell, DNDS::rowsize(slot), neighbors[neighbors.size() - 1 - slot]);
    }
    auto metadata = edit.CellInfo(0);
    metadata.zone = 12345;
    edit.SetCellInfo(0, metadata);
    edit.Commit();
    CHECK_FALSE(mesh->CurrentNativeRequirements().wallDistance);
    CHECK(mesh->NativeRelationKeys() == keys);
    for (size_t i = 0; i < keys.size(); ++i)
    {
        auto read = mesh->ReadNativeRelation(keys[i]);
        for (DNDS::index row = 0; row < read.RowCount(); ++row)
        {
            auto values = expected[i][size_t(row)];
            if (keys[i] == G::Adj::Cell2Cell)
                std::reverse(values.begin(), values.end());
            CHECK(read.GlobalRow(row) == values);
        }
    }
    {
        auto read = mesh->AcquireRead();
        CHECK(read.HostView().Coord(0)[0] == 5 * mpi.rank + 0.5);
        CHECK(read.HostView().GetCellInfo(0).zone == 12345);
    }
    CHECK(coords.Read().View()(0, 0) == 5 * mpi.rank);
    CHECK_THROWS(coords.Write());
    CHECK(mesh->OriginalEntityId(G::EntityKind::Cell, 0) == 3 * mpi.rank);
    // The old independently retained native publication is unchanged.
    CHECK(state->MeshNamespace() == name);
    CHECK(state->IdHighWater() == highWater);
    auto archive = DNDS::make_ssp<DNDS::Serializer::SerializerJSON>();
    archive->CreatePath("/");
    mesh->WriteNativeArchive(archive, "edited");
    auto restored = G::ReadNativeMeshArchive(mpi, archive, "edited");
    CHECK(restored->MeshNamespace() == name);
    CHECK(restored->IdHighWater() == highWater);
    CHECK(restored->RelationKeys() == keys);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
