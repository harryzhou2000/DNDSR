#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "Geom/Mesh/NativeMeshSnapshot.hpp"
#include <optional>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;

TEST_CASE("Detached native mesh snapshots retain values metadata and independent validity")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    std::optional<G::NativeMeshSnapshot> snapshot;
    snapshot.emplace(G::NativeMeshSnapshot::Capture(*fixture.mesh, G::MeshReadRequirements::CellFlux()));
    auto first = snapshot->AcquireRead();
    auto second = snapshot->AcquireRead();
    const auto view = first.HostView(), again = second.HostView();
    REQUIRE(view.NumCell() > 0);
    const auto originalNodes = std::vector<DNDS::index>(view.CellNodes(0).begin(), view.CellNodes(0).end());
    const G::tPoint originalPoint = view.Coord(0);
    CHECK(view.CellNodes(0).data() == again.CellNodes(0).data());
    CHECK(view.Coord(0).data() == again.Coord(0).data());
    CHECK(view.CellNodes(0).data() != fixture.mesh->cell2node.father->rowPtr(0));
    CHECK(fixture.mesh->versions.CanMutate());
    fixture.mesh->ReorderLocalCells(2, 1);
    CHECK_FALSE(snapshot->SourceStamp().Valid());
    CHECK(first.Stamp().Valid());
    CHECK(snapshot->AcquireRead().Stamp().Valid());
    // Source structural replacement and deletion cannot touch snapshot shape.
    fixture.mesh->cell2node.father->Resize(0, [](DNDS::index)
                                           { return DNDS::rowsize(0); });
    fixture.mesh.reset();
    CHECK(view.NumCell() > 0);
    CHECK(std::vector<DNDS::index>(view.CellNodes(0).begin(), view.CellNodes(0).end()) == originalNodes);
    CHECK(view.Coord(0).isApprox(originalPoint));
    snapshot.reset();
    CHECK(first.HostView().Coord(0).isApprox(originalPoint));
    CHECK(second.HostView().CellNodes(0).size() == originalNodes.size());
}

TEST_CASE("Detached snapshot publication retains validation and explicit placement")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = G::OverlapTest::BuildFixture(mpi);
    auto snapshot = G::NativeMeshSnapshot::Capture(*fixture.mesh);
    CHECK_FALSE(snapshot.AcquireRead().HostView().HasRelation(G::MeshRelation::CellFaces));
    CHECK_THROWS(snapshot.AcquireRead(R::ExecutionContext::CUDA(0)));
    auto copy = snapshot.To(R::Device::Host());
    auto first = snapshot.AcquireRead(), second = copy.AcquireRead();
    CHECK(first.HostView().CellNodes(0).data() != second.HostView().CellNodes(0).data());
    fixture.mesh->cell2node(0, 0) = DNDS::index(1000000);
    CHECK_THROWS(G::NativeMeshSnapshot::Capture(*fixture.mesh));
    CHECK(first.Stamp().Valid());
    auto moved = std::move(copy);
    CHECK_THROWS(copy.GetDevice());
    CHECK_THROWS(copy.SourceStamp());
    CHECK_THROWS(copy.AcquireRead());
    CHECK_THROWS(copy.To(R::Device::Host()));
    CHECK(moved.AcquireRead().HostView().NumCell() == first.HostView().NumCell());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
