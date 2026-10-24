#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include <filesystem>

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;

static MPIInfo MutationMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

static ssp<UnstructuredMesh> MutationFixture(const MPIInfo &mpi)
{
    auto mesh = make_ssp<UnstructuredMesh>(mpi, 2);
    auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    UnstructuredMeshSerialRW reader(mesh, 0);
    reader.ReadFromCGNSSerial((root / "data/mesh/UniformSquare_10.cgns").string());
    reader.BuildCell2Cell();
    UnstructuredMeshSerialRW::PartitionOptions options;
    options.metisType = "KWAY";
    options.metisSeed = 42;
    options.metisNcuts = 1;
    reader.MeshPartitionCell2Cell(options);
    reader.PartitionReorderToMeshCell2Cell();
    mesh->RecoverNode2CellAndNode2Bnd();
    mesh->RecoverCell2CellAndBnd2Cell();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    return mesh;
}

TEST_CASE("Managed public mutations reject retained readers before changing values or generations")
{
    auto mesh = MutationFixture(MutationMPI());
    auto before = mesh->versions.Stamp();
    {
        auto read = mesh->AcquireRead();
        CHECK_THROWS(mesh->AdjLocal2GlobalPrimary());
        CHECK_THROWS(mesh->InterpolateFace());
        tPoint zero = tPoint::Zero();
        CHECK_THROWS(mesh->SetPeriodicGeometry({9, 0, 0}, zero, zero, zero, zero, zero, zero, zero, zero));
        CHECK(before.Valid());
        CHECK(mesh->adjPrimaryState == Adj_PointToLocal);
        CHECK(mesh->periodicInfo.translation[1].map()[0] == 1);
    }
    CHECK(mesh->versions.CanMutate());
    mesh->AdjLocal2GlobalPrimary();
    CHECK_FALSE(before.Valid());
}

TEST_CASE("Nested face setup and reorder publish each changed generation only once")
{
    auto mesh = MutationFixture(MutationMPI());
    auto before = mesh->versions.Stamp();
    mesh->InterpolateFace();
    mesh->AssertOnFaces();
    auto after = mesh->versions.Stamp();
    for (auto change : {R::Change::Topology, R::Change::Distribution, R::Change::Layout,
                        R::Change::Allocation, R::Change::Interface, R::Change::Values})
        CHECK(after.Generation(change) == before.Generation(change) + 1);
    CHECK(after.Generation(R::Change::Geometry) == before.Generation(R::Change::Geometry));
    mesh->BuildCell2CellFace();
    mesh->AdjGlobal2LocalC2CFace();
    mesh->AdjLocal2GlobalC2CFace();
    auto auxiliary = mesh->versions.Stamp();
    CHECK(auxiliary.Generation(R::Change::Layout) == after.Generation(R::Change::Layout));
    CHECK(auxiliary.Generation(R::Change::Topology) == after.Generation(R::Change::Topology));
    CHECK(auxiliary.Generation(R::Change::Geometry) == after.Generation(R::Change::Geometry));
    auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
    CHECK(read.HostView().NumFace() == mesh->NumFace());
}

TEST_CASE("Failed nested mutation stays failed even when an intermediate caller catches it")
{
    R::VersionTracker versions;
    int owner = 0;
    auto before = versions.Stamp();
    {
        MeshMutationToken token(&owner, versions, R::Change::Geometry);
        {
            auto outer = token.Enter(&owner, R::Change::Geometry);
            try
            {
                auto inner = token.Enter(&owner, R::Change::Layout);
                throw std::runtime_error("nested write failed");
            }
            catch (const std::runtime_error &)
            {
            }
        }
        CHECK_THROWS(token.Commit());
    }
    CHECK_FALSE(before.Valid());
    CHECK_FALSE(versions.CanMutate());
    CHECK_THROWS(versions.Read());
}

TEST_CASE("Public callback reentry cannot adopt an existing mesh writer")
{
    auto mesh = MutationFixture(MutationMPI());
    tPoint zero = tPoint::Zero();
    bool rejected = false;
    mesh->TransformCoords([&](const tPoint &point)
                          {
        try { mesh->SetPeriodicGeometry({9, 0, 0}, zero, zero, zero, zero, zero, zero, zero, zero); }
        catch (const std::runtime_error &) { rejected = true; }
        return point; });
    CHECK(rejected);
    CHECK(mesh->periodicInfo.translation[1].map()[0] == 1);
    CHECK(mesh->versions.CanMutate());
    CHECK_THROWS(mesh->TransformCoords([](const tPoint &) -> tPoint
                                       { throw std::runtime_error("coordinate edit failed"); }));
    CHECK_FALSE(mesh->versions.CanMutate());
    CHECK_THROWS(mesh->AcquireRead());
}

TEST_CASE("Bisection creates independent geometry under independent mesh mutation controllers")
{
    auto mpi = MutationMPI();
    auto linear = MutationFixture(mpi);
    auto quadratic = make_ssp<UnstructuredMesh>(mpi, 2);
    quadratic->BuildO2FromO1Elevation(*linear);
    quadratic->RecoverNode2CellAndNode2Bnd();
    quadratic->RecoverCell2CellAndBnd2Cell();
    quadratic->BuildGhostPrimary();
    quadratic->AdjGlobal2LocalPrimary();
    quadratic->AdjLocal2GlobalPrimary();
    auto bisected = make_ssp<UnstructuredMesh>(mpi, 2);
    bisected->BuildBisectO1FormO2(*quadratic);
    CHECK(bisected->coords.father != quadratic->coords.father);
    CHECK(bisected->node2nodeOrig.father != quadratic->node2nodeOrig.father);
    std::vector<tPoint> before;
    for (DNDS::index node = 0; node < quadratic->NumNode(); ++node)
        before.emplace_back(quadratic->coords[node]);
    auto sourceStamp = quadratic->versions.Stamp();
    bisected->TransformCoords([](const tPoint &point)
                              { return (point + tPoint{0, 2, 0}).eval(); });
    CHECK(sourceStamp.Valid());
    for (DNDS::index node = 0; node < quadratic->NumNode(); ++node)
    {
        CHECK(quadratic->coords[node].isApprox(before[size_t(node)]));
        CHECK(bisected->coords[node].isApprox(before[size_t(node)] + tPoint{0, 2, 0}));
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
