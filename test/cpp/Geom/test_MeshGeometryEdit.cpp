#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include <filesystem>
#include <set>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

static DNDS::ssp<G::UnstructuredMesh> Fixture(DNDS::MPIInfo mpi)
{
    auto mesh = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 2);
    G::UnstructuredMeshSerialRW reader(mesh, 0);
    auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    reader.ReadFromCGNSSerial((root / "data/mesh/UniformSquare_10.cgns").string());
    reader.BuildCell2Cell();
    reader.MeshPartitionCell2Cell({});
    reader.PartitionReorderToMeshCell2Cell();
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    return mesh;
}

static std::vector<G::tPoint> Points(G::UnstructuredMesh &mesh)
{
    std::vector<G::tPoint> result;
    auto read = mesh.AcquireRead();
    auto view = read.HostView();
    for (I node = 0; node < view.NumNodeProc(); ++node)
        result.emplace_back(view.Coord(node));
    return result;
}

TEST_CASE("Rigid geometry edit transforms coordinates periodic frame and wall vectors coherently")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto mesh = Fixture(mpi);
    G::tPoint zero = G::tPoint::Zero();
    mesh->SetPeriodicGeometry({2, 0, 0}, {1, 2, 0}, {0, 0, 0.25}, {0, 3, 0}, zero, zero, zero, zero, zero);
    mesh->nodeWallDist.InitPair("geometry-wall-vector", mpi);
    mesh->nodeWallDist.father->Resize(mesh->NumNode());
    mesh->nodeWallDist.son->Resize(mesh->NumNodeGhost());
    mesh->nodeWallDist.TransAttach();
    G::LegacyFieldBridge::BorrowIndexing(mesh->nodeWallDist.trans, mesh->Entities(G::EntityKind::Node));
    const G::tPoint wall{1, 2, 3};
    for (I node = 0; node < mesh->NumNodeProc(); ++node)
        mesh->nodeWallDist[node] = wall;
    auto before = Points(*mesh);
    auto oldFrame = mesh->PeriodicMetadata();
    auto stamp = mesh->versions.Stamp();
    const Eigen::Matrix3d rotation = G::RotZ(0.37);
    const DNDS::real scale = 2.5;
    {
        std::unique_ptr<G::MeshReadAccess> held;
        if (mpi.rank == 0)
            held = std::make_unique<G::MeshReadAccess>(mesh->AcquireRead());
        CHECK_THROWS(mesh->ApplyGeometryTransform(rotation, scale));
    }
    CHECK(stamp.Valid());
    CHECK_THROWS(mesh->ApplyGeometryTransform(rotation, mpi.rank == 0 ? -1.0 : 1.0));
    CHECK(stamp.Valid());
    {
        // A bridge reader can block only one rank/array after another write
        // lease has already been acquired. Rejection must leave all untouched
        // coordinate buffers usable, including on the other ranks.
        DNDS::Runtime::ReadAccess held;
        if (mpi.rank == 0)
            held = mesh->coords.son->HostReadData();
        CHECK_THROWS(mesh->ApplyGeometryTransform(rotation, scale));
    }
    CHECK(stamp.Valid());
    const auto rejected = Points(*mesh);
    CHECK(rejected == before);
    mesh->ApplyGeometryTransform(rotation, scale);
    CHECK_FALSE(stamp.Valid());
    auto after = Points(*mesh);
    for (size_t node = 0; node < after.size(); ++node)
    {
        CHECK(after[node].isApprox(scale * rotation * before[node], 1e-13));
        CHECK(mesh->nodeWallDist[I(node)].isApprox(scale * rotation * wall, 1e-13));
    }
    for (int frame = 0; frame < 4; ++frame)
    {
        CHECK(mesh->PeriodicMetadata().translation[frame].map().isApprox(scale * rotation * oldFrame.translation[frame].map(), 1e-13));
        CHECK(mesh->PeriodicMetadata().rotationCenter[frame].map().isApprox(scale * rotation * oldFrame.rotationCenter[frame].map(), 1e-13));
        CHECK(mesh->PeriodicMetadata().rotation[frame].map().isApprox(rotation * oldFrame.rotation[frame].map() * rotation.transpose(), 1e-13));
    }
}

TEST_CASE("Boundary rectification reduces remote node masks before refreshing coordinate ghosts")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto mesh = Fixture(mpi);
    auto before = Points(*mesh);
    const auto nodes = mesh->Entities(G::EntityKind::Node);
    std::vector<I> boundaryNodes;
    for (I boundary = 0; boundary < mesh->NumBnd(); ++boundary)
        for (auto node : mesh->bnd2node[boundary])
            boundaryNodes.push_back(nodes.LocalToGlobal(node));
    int count = int(boundaryNodes.size());
    std::vector<int> counts(mpi.size), starts(mpi.size + 1, 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
    for (int rank = 0; rank < mpi.size; ++rank)
        starts[rank + 1] = starts[rank] + counts[rank];
    std::vector<I> all(size_t(starts.back()));
    MPI_Allgatherv(boundaryNodes.data(), count, DNDS::DNDS_MPI_INDEX, all.data(), counts.data(), starts.data(), DNDS::DNDS_MPI_INDEX, mpi.comm);
    const std::set<I> selected(all.begin(), all.end());
    auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->RectifyBoundaryCoordinates([&](G::t_index)
                                                  {
        if (mpi.rank == 0) throw std::runtime_error("rank-local callback failure");
        return 1; }));
    CHECK(stamp.Valid());
    auto unchanged = Points(*mesh);
    for (size_t node = 0; node < unchanged.size(); ++node)
        CHECK(unchanged[node] == before[node]);
    mesh->RectifyBoundaryCoordinates([](G::t_index)
                                     { return 1; });
    auto after = Points(*mesh);
    for (size_t node = 0; node < after.size(); ++node)
    {
        auto expected = before[node];
        if (selected.count(nodes.LocalToGlobal(I(node))))
            expected[0] = 0;
        CHECK(after[node] == expected);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
