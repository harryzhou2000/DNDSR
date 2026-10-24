#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Mesh/MeshEdit.hpp"
#include <limits>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;

TEST_CASE("Staged current-authority geometry edits publish owned and ghost coordinates safely")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto mesh = fixture.mesh;
    mesh->PublishNative(G::MeshReadRequirements::CellFlux());
    auto order = mesh->Entities(G::EntityKind::Node);
    auto route = order.BuildRowTransferPlan();
    auto detached = G::NativeMeshSnapshot::Capture(*mesh, G::MeshReadRequirements::CellFlux());
    std::vector<G::tPoint> before;
    {
        auto read = mesh->AcquireRead();
        for (DNDS::index row = 0; row < mesh->NumNodeProc(); ++row)
            before.push_back(read.HostView().Coord(row));
    }
    // A deliberately stale workspace must never become the edit's input.
    mesh->coords[0] = G::tPoint{900, 800, 700};
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
    CHECK_THROWS(mesh->TransformCoords([](const G::tPoint &p)
                                       { return p; }));
    CHECK_THROWS(mesh->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2));
    CHECK_THROWS(mesh->RectifyBoundaryCoordinates([](G::t_index)
                                                  { return 1; }));
    CHECK_THROWS(mesh->BuildNodeWallDist([](G::t_index)
                                         { return true; }, {}));
    CHECK_THROWS(mesh->BuildGhostPrimary(1));
    G::UnstructuredMesh legacyPeer(mpi, 2);
    auto peerStamp = legacyPeer.versions.Stamp();
    auto &mixedAuthority = mpi.rank == 0 ? *mesh : legacyPeer;
    CHECK_THROWS(mixedAuthority.BuildGhostPrimary(1));
    CHECK(peerStamp.Valid());
    CHECK_THROWS(mesh->SetPeriodicGeometry(G::tPoint::Zero(), G::tPoint::Zero(), G::tPoint::Zero(),
                                           G::tPoint::Zero(), G::tPoint::Zero(), G::tPoint::Zero(),
                                           G::tPoint::Zero(), G::tPoint::Zero(), G::tPoint::Zero()));
    CHECK(committed.Valid());
    CHECK(mesh->HasNativeState());
    escaped = G::tPoint{-1, -2, -3};
    auto read = mesh->AcquireRead();
    for (DNDS::index node = 0; node < mesh->NumNodeProc(); ++node)
        CHECK(read.HostView().Coord(node).isApprox(before[size_t(node)] + G::tPoint{0.25, 0.5, 0}));
    CHECK(detached.AcquireRead().HostView().Coord(0).isApprox(before[0]));
    CHECK(mesh->coords[0].isApprox(G::tPoint{900, 800, 700}));
}

TEST_CASE("Staged metadata and neighbor-order edits validate before publication")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto mesh = fixture.mesh;
    const auto oldInfo = mesh->GetCellInfo(0);
    CHECK_FALSE(mesh->node2cell.isLocal()); // Prepared legacy inverse setup is global.
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
    auto fixture = G::OverlapTest::BuildFixture(mpi);
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
