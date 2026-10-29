#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/NativeMeshOutput.hpp"
#include "Geom/Mesh/MeshConnectivity.hpp"
#include "Solver/Direct.hpp"
#include "NativeMeshOverlapFixture.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
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
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    return UnstructuredMesh::FromNative(mpi, ReadNativeCGNS(mpi, (root / "data/mesh/UniformSquare_10.cgns").string(), 2, FBC_Name_2_ID_Default, 0, {1, true}));
}
TEST_CASE("Managed public mutations reject retained readers while detached conversion remains pure")
{
    auto mesh = MutationFixture(MutationMPI());
    auto before = mesh->versions.Stamp();
    const auto periodicTranslation = mesh->PeriodicMetadata().translation[1].map()[0];
    {
        auto read = mesh->AcquireRead();
        CHECK_THROWS(mesh->PrepareNativeTopology());
        CHECK_THROWS(mesh->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2));
        CHECK_THROWS(mesh->ReorderLocalCells());
        auto relation = MeshAdjacencyRead::Current(*mesh, Adj::Cell2Node, MeshReadRequirements::Primary());
        auto globals = MeshAdjacency::Convert(relation, Adj_PointToGlobal);
        auto locals = MeshAdjacency::Convert(globals.Read(), Adj_PointToLocal, false);
        CHECK(locals.Read().GlobalRow(0) == relation.GlobalRow(0));
        CHECK(before.Valid());
        CHECK(read.HostView().RelationState(MeshRelation::CellNodes) == Adj_PointToLocal);
        CHECK(mesh->PeriodicMetadata().translation[1].map()[0] == periodicTranslation);
    }
    CHECK(mesh->versions.CanMutate());
    mesh->PrepareNativeTopology();
    CHECK_FALSE(before.Valid());
}
TEST_CASE("Native topology publication changes each declared generation once and auxiliary setup remains pure")
{
    auto mesh = MutationFixture(MutationMPI());
    const auto before = mesh->versions.Stamp();
    mesh->PrepareNativeTopology();
    mesh->AssertOnFaces();
    const auto after = mesh->versions.Stamp();
    for (auto change : {R::Change::Topology, R::Change::Distribution, R::Change::Layout, R::Change::Allocation})
        CHECK(after.Generation(change) == before.Generation(change) + 1);
    for (auto change : {R::Change::Geometry, R::Change::Interface, R::Change::Values})
        CHECK(after.Generation(change) == before.Generation(change));
    Direct::DirectPrecControl options;
    options.useDirectPrec = true;
    options.orderingCode = 0;
    Direct::SerialSymLUStructure graph(mesh->mpi, mesh->NumCell());
    mesh->ObtainLocalFactFillOrdering(graph, options);
    CHECK(after.Valid());
    CHECK(mesh->LocalCellNeighbors().size() == size_t(mesh->NumCell()));
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

TEST_CASE("Public callback reentry cannot adopt a writer and staged failures preserve the source")
{
    auto mesh = MutationFixture(MutationMPI());
    bool rejected = false;
    mesh->TransformCoords([&](const tPoint &point)
                          {
        try {auto write=mesh->versions.BeginMutation(R::Change::Geometry);write.Commit();}
        catch(const std::runtime_error &){rejected=true;}
        return point; });
    CHECK(rejected == (mesh->NumNode() > 0));
    CHECK(mesh->versions.CanMutate());
    const auto stamp = mesh->versions.Stamp();
    const auto state = mesh->CurrentNativeState();
    std::vector<tPoint> before;
    for (DNDS::index row = 0; row < mesh->NumNodeProc(); ++row)
        before.push_back(mesh->Coord(row));
    CHECK_THROWS(mesh->TransformCoords([](const tPoint &) -> tPoint
                                       { throw std::runtime_error("coordinate edit failed"); }));
    CHECK(stamp.Valid());
    CHECK(mesh->versions.CanMutate());
    CHECK(mesh->CurrentNativeState() == state);
    for (DNDS::index row = 0; row < mesh->NumNodeProc(); ++row)
        CHECK(mesh->Coord(row) == before[size_t(row)]);
    CHECK_NOTHROW(mesh->TransformCoords([](const tPoint &point)
                                        { return point; }));
}
TEST_CASE("Bisection creates independent geometry under independent mesh mutation controllers")
{
    auto mpi = MutationMPI();
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    auto linear = ReadNativeMeshFromCGNS(mpi, 2, (root / "data/mesh/UniformSquare_10.cgns").string(),
                                         PartitionOptions{}, FBC_Name_2_ID_Default);
    auto quadratic = make_ssp<UnstructuredMesh>(mpi, 2);
    quadratic->BuildO2FromO1Elevation(*linear);
    auto bisected = make_ssp<UnstructuredMesh>(mpi, 2);
    bisected->BuildBisectO1FormO2(*quadratic);
    CHECK(bisected->CurrentNativeState()->MeshNamespace() != quadratic->CurrentNativeState()->MeshNamespace());
    std::vector<tPoint> before;
    for (DNDS::index node = 0; node < quadratic->NumNode(); ++node)
        before.emplace_back(quadratic->Coord(node));
    auto sourceStamp = quadratic->versions.Stamp();
    bisected->TransformCoords([](const tPoint &point)
                              { return (point + tPoint{0, 2, 0}).eval(); });
    CHECK(sourceStamp.Valid());
    for (DNDS::index node = 0; node < quadratic->NumNode(); ++node)
    {
        CHECK(quadratic->Coord(node).isApprox(before[size_t(node)]));
        CHECK(bisected->Coord(node).isApprox(before[size_t(node)] + tPoint{0, 2, 0}));
    }
}

TEST_CASE("Native primary serialization shares immutable shape safely with readers and preserves the source")
{
    auto mesh = MutationFixture(MutationMPI());
    auto serializer = make_ssp<Serializer::SerializerJSON>();
    serializer->CreatePath("/");
    const auto before = mesh->versions.Stamp();
    const auto state = mesh->CurrentNativeState();
    {
        auto read = mesh->AcquireRead();
        CHECK_NOTHROW(mesh->WriteSerialize(serializer, "mesh"));
        CHECK(before.Valid());
        CHECK(mesh->CurrentNativeState() == state);
        auto restored = make_ssp<UnstructuredMesh>(mesh->mpi, mesh->getDim());
        restored->ReadSerialize(serializer, "mesh");
        CHECK(restored->NumCell() == mesh->NumCell());
        CHECK(restored->NumNode() == mesh->NumNode());
        for (DNDS::index row = 0; row < mesh->NumNode(); ++row)
        {
            CHECK(restored->OriginalEntityId(EntityKind::Node, row) == mesh->OriginalEntityId(EntityKind::Node, row));
            CHECK(restored->Coord(row) == read.HostView().Coord(row));
        }
        for (DNDS::index row = 0; row < mesh->NumCell(); ++row)
        {
            CHECK(restored->OriginalEntityId(EntityKind::Cell, row) == mesh->OriginalEntityId(EntityKind::Cell, row));
            const auto oldNodes = mesh->CellNodes(row), newNodes = restored->CellNodes(row);
            REQUIRE(newNodes.size() == oldNodes.size());
            for (size_t slot = 0; slot < oldNodes.size(); ++slot)
                CHECK(restored->OriginalEntityId(EntityKind::Node, newNodes[slot]) == mesh->OriginalEntityId(EntityKind::Node, oldNodes[slot]));
        }
    }
    CHECK(before.Valid());
    CHECK(mesh->versions.CanMutate());
}
TEST_CASE("Collective native topology setup rejects a one-rank reader while output remains readonly")
{
    auto mesh = MutationFixture(MutationMPI());
    const auto stamp = mesh->versions.Stamp();
    const auto state = mesh->CurrentNativeState();
    std::optional<R::VersionTracker::ReadLease> reader;
    if (mesh->mpi.rank == 0)
        reader.emplace(mesh->versions.Read());
    CHECK_THROWS(mesh->PrepareNativeTopology());
    CHECK(stamp.Valid());
    CHECK(mesh->CurrentNativeState() == state);
    auto output = NativeMeshOutput::Build(*state, mesh->mpi);
    CHECK(output.globalCells == mesh->NumCellGlobal());
    CHECK(stamp.Valid());
    reader.reset();
    CHECK_NOTHROW(mesh->PrepareNativeTopology());
    CHECK_FALSE(stamp.Valid());
}
TEST_CASE("Boundary destination rejection preserves both native authorities and controllers")
{
    auto mpi = MutationMPI();
    auto source = DNDS::Geom::OverlapTest::BuildNativeFixture(mpi).mesh;
    auto destination = DNDS::Geom::OverlapTest::BuildNativeFixture(mpi).mesh;
    // Native state remains dimension two; detached scalar metadata cannot make
    // it a compatible dimension-one boundary destination.
    destination->dim = 1;
    auto sourceStamp = source->versions.Stamp();
    auto destinationStamp = destination->versions.Stamp();
    const auto sourceState = source->CurrentNativeState();
    const auto destinationState = destination->CurrentNativeState();
    CHECK_THROWS(source->ConstructBndMesh(*destination));
    CHECK(sourceStamp.Valid());
    CHECK(destinationStamp.Valid());
    CHECK(source->versions.CanMutate());
    CHECK(source->CurrentNativeState() == sourceState);
    CHECK(destination->CurrentNativeState() == destinationState);
    CHECK(source->AcquireRead().HostView().NumCell() == source->NumCell());
    UnstructuredMesh freshPeer(mpi, 1);
    auto peerStamp = freshPeer.versions.Stamp();
    auto &mixedDestination = mpi.rank == 0 ? *destination : freshPeer;
    CHECK_THROWS(source->ConstructBndMesh(mixedDestination));
    // Derivation requires matching volume dimensions, so use a genuine bad
    // dimension on rank zero instead of detached scalar metadata on a native
    // two-dimensional destination. This also rejects with a single MPI rank.
    UnstructuredMesh badDerivation(mpi, 3);
    auto badStamp = badDerivation.versions.Stamp();
    auto &derivationDestination = mpi.rank == 0 ? badDerivation : *destination;
    CHECK_THROWS(derivationDestination.BuildO2FromO1Elevation(*source));
    CHECK_THROWS(derivationDestination.BuildBisectO1FormO2(*source));
    CHECK(badStamp.Valid());
    CHECK_FALSE(badDerivation.HasNativeState());
    CHECK(sourceStamp.Valid());
    CHECK(destinationStamp.Valid());
    CHECK(peerStamp.Valid());
    CHECK(source->versions.CanMutate());
    CHECK(source->CurrentNativeState() == sourceState);
    CHECK(destination->CurrentNativeState() == destinationState);
    CHECK(source->AcquireRead().HostView().NumNode() == source->NumNode());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
