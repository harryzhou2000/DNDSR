#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
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

static DNDS::ssp<G::UnstructuredMesh> NativeGeometryFixture(DNDS::MPIInfo mpi)
{
    const bool ownsNodes = mpi.rank == mpi.size - 1;
    const bool ownsCell = mpi.rank == 0;
    const bool ownsBoundary = mpi.rank == (mpi.size > 1 ? 1 : 0);
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, ownsNodes ? 3 : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, ownsCell ? 1 : 0);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, ownsBoundary ? 1 : 0);
    auto cone = [](bool owned, const std::vector<I> &values)
    {
        R::RowBuilder rows(owned ? 1 : 0, sizeof(I));
        if (owned)
        {
            rows.ResizeRow(0, values.size());
            auto row = rows.Row<I>(0);
            std::copy(values.begin(), values.end(), row.data);
        }
        return R::Array<I>::FromSharedStorage(rows.Commit());
    };
    R::Array<DNDS::real> points(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3));
    R::Array<G::ElemInfo> cellInfo(R::RowLayout::Dense(size_t(cells.NumOwned()), 1));
    R::Array<G::ElemInfo> bndInfo(R::RowLayout::Dense(size_t(boundaries.NumOwned()), 1));
    {
        auto write = points.Write();
        const G::tPoint coordinates[]{{1, 1, 1}, {2, 1, 1}, {1, 2, 1}};
        for (size_t row = 0; row < points.Rows(); ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = coordinates[row][component];
        if (ownsCell)
            cellInfo.Write().View()(0, 0) = G::ElemInfo{G::Elem::Tri3, 31};
        if (ownsBoundary)
            bndInfo.Write().View()(0, 0) = G::ElemInfo{G::Elem::Line2, 24};
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, cone(ownsCell, {0, 1, 2}), cone(ownsBoundary, {0, 1}), points, cellInfo, bndInfo, {}};
    auto source = G::NativeMeshBuilder::Build(input, {1, true});
    const auto view = source->BorrowHostView();
    auto pair = [&](bool wall)
    {
        R::ArrayPair<DNDS::real> result{R::Array<DNDS::real>(R::RowLayout::Dense(size_t(view.NumNode()), 3)),
                                        R::Array<DNDS::real>(R::RowLayout::Dense(size_t(view.NumNodeGhost()), 3))};
        auto owned = result.owned.Write(), ghost = result.ghost.Write();
        for (I node = 0; node < view.NumNodeProc(); ++node)
        {
            const G::tPoint p = wall ? G::tPoint{1, 2, 3} : G::tPoint(view.Coord(node));
            auto row = node < view.NumNode() ? owned.View()[size_t(node)] : ghost.View()[size_t(node - view.NumNode())];
            for (size_t component = 0; component < 3; ++component)
                row[component] = p[component];
        }
        return result;
    };
    G::Periodicity frame;
    frame.translation[1].map() = G::tPoint{2, 1, 0};
    frame.rotationCenter[1].map() = G::tPoint{1, 2, 0};
    frame.rotation[1].map() = G::RotZ(0.2);
    return G::UnstructuredMesh::FromNative(mpi, source->WithGeometry(pair(false), pair(true), frame));
}

TEST_CASE("Native geometry transforms preserve order identity topology and detached readers")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto mesh = NativeGeometryFixture(mpi);
    const auto nodes = mesh->Entities(G::EntityKind::Node);
    const auto namespaceBefore = mesh->CurrentNativeState()->MeshNamespace();
    auto retained = mesh->CurrentNativeState()->Snapshot(R::Device::Host()).AcquireRead();
    const auto oldView = retained.HostView();
    const auto oldFrame = mesh->PeriodicMetadata();
    auto stamp = mesh->versions.Stamp();
    const Eigen::Matrix3d rotation = G::RotZ(0.37);
    {
        std::unique_ptr<G::MeshReadAccess> held;
        if (mpi.rank == 0)
            held = std::make_unique<G::MeshReadAccess>(mesh->AcquireRead(mesh->CurrentNativeRequirements()));
        CHECK_THROWS(mesh->ApplyGeometryTransform(rotation, 2.5));
    }
    CHECK_THROWS(mesh->ApplyGeometryTransform(rotation, mpi.rank == 0 ? -1.0 : 1.0));
    if (mpi.size > 1)
        CHECK_THROWS(mesh->ApplyGeometryTransform(rotation, mpi.rank == 0 ? 2.0 : 1.0));
    CHECK(stamp.Valid());
    mesh->ApplyGeometryTransform(rotation, 2.5);
    CHECK_FALSE(stamp.Valid());
    CHECK(nodes.SameOrder(mesh->Entities(G::EntityKind::Node)));
    CHECK(mesh->CurrentNativeState()->MeshNamespace() == namespaceBefore);
    CHECK_FALSE(mesh->coords.father);
    auto current = mesh->AcquireRead(mesh->CurrentNativeRequirements());
    const auto view = current.HostView();
    for (I node = 0; node < view.NumNodeProc(); ++node)
    {
        CHECK(view.Coord(node).isApprox(2.5 * rotation * oldView.Coord(node), 1e-13));
        CHECK(view.NodeWallDist(node).isApprox(2.5 * rotation * G::tPoint{1, 2, 3}, 1e-13));
        CHECK(oldView.Coord(node)[2] == 1);
        CHECK(mesh->OriginalEntityId(G::EntityKind::Node, node) == nodes.LocalToGlobal(node));
    }
    for (I cell = 0; cell < view.NumCellProc(); ++cell)
        CHECK(view.CellNodes(cell).data() == oldView.CellNodes(cell).data());
    for (int frame = 0; frame < 4; ++frame)
    {
        CHECK(view.Periodic().rotation[frame].map().isApprox(rotation * oldFrame.rotation[frame].map() * rotation.transpose(), 1e-13));
        CHECK(view.Periodic().translation[frame].map().isApprox(2.5 * rotation * oldFrame.translation[frame].map(), 1e-13));
        CHECK(view.Periodic().rotationCenter[frame].map().isApprox(2.5 * rotation * oldFrame.rotationCenter[frame].map(), 1e-13));
    }
}

TEST_CASE("Native rectification reduces masks across independent owners and rejects callback failure")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto mesh = NativeGeometryFixture(mpi);
    const auto nodes = mesh->Entities(G::EntityKind::Node);
    auto retained = mesh->CurrentNativeState()->Snapshot(R::Device::Host()).AcquireRead();
    const auto before = retained.HostView();
    auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->RectifyBoundaryCoordinates([](G::t_index) -> int
                                                  { throw std::runtime_error("callback failed"); }));
    CHECK_THROWS(mesh->RectifyBoundaryCoordinates([](G::t_index)
                                                  { return 4; }));
    CHECK(stamp.Valid());
    CHECK_NOTHROW(mesh->RectifyBoundaryCoordinates([&](G::t_index)
                                                   {
        // Local callback reentry cannot acquire a writer while the operation
        // retains its source. Never enter a collective from selected callbacks.
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
        return 1; }));
    CHECK(nodes.SameOrder(mesh->Entities(G::EntityKind::Node)));
    CHECK_FALSE(mesh->CurrentNativeRequirements().wallDistance);
    auto read = mesh->AcquireRead(mesh->CurrentNativeRequirements());
    const auto view = read.HostView();
    for (I node = 0; node < view.NumNodeProc(); ++node)
    {
        G::tPoint expected = before.Coord(node);
        if (nodes.LocalToGlobal(node) < 2)
            expected[0] = 0;
        CHECK(view.Coord(node) == expected);
    }
}

TEST_CASE("Collective geometry operations reject mixed legacy and native authority before branching")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    if (mpi.size == 1)
        return;
    auto native = NativeGeometryFixture(mpi);
    auto legacy = Fixture(mpi);
    auto mixed = mpi.rank == 0 ? legacy : native;
    const auto stamp = mixed->versions.Stamp();
    const auto before = Points(*mixed);
    CHECK_THROWS(mixed->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2));
    CHECK_THROWS(mixed->RectifyBoundaryCoordinates([](G::t_index)
                                                   { return 1; }));
    CHECK(stamp.Valid());
    CHECK(Points(*mixed) == before);
    CHECK_NOTHROW(native->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 1));
}

TEST_CASE("Native arbitrary coordinate callbacks run on owners and refresh ghosts transactionally")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto mesh = NativeGeometryFixture(mpi);
    const auto nodes = mesh->Entities(G::EntityKind::Node);
    const auto stamp = mesh->versions.Stamp();
    auto old = mesh->CurrentNativeState()->Snapshot(R::Device::Host()).AcquireRead();
    CHECK_THROWS(mesh->TransformCoords([](const G::tPoint &) -> G::tPoint
                                       { throw std::runtime_error("owner-local failure"); }));
    CHECK(stamp.Valid());
    CHECK_THROWS(mesh->TransformCoords([](const G::tPoint &)
                                       { return G::tPoint::Constant(std::numeric_limits<double>::quiet_NaN()).eval(); }));
    CHECK(stamp.Valid());
    size_t calls = 0;
    mesh->TransformCoords([&](G::tPoint point)
                          {
        ++calls;
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
        point[0] += point[1] * point[1];
        return point; });
    CHECK(calls == size_t(nodes.NumOwned()));
    CHECK(nodes.SameOrder(mesh->Entities(G::EntityKind::Node)));
    CHECK_FALSE(mesh->CurrentNativeRequirements().wallDistance);
    auto current = mesh->AcquireRead(mesh->CurrentNativeRequirements());
    for (I node = 0; node < mesh->NumNodeProc(); ++node)
    {
        G::tPoint expected = old.HostView().Coord(node);
        expected[0] += expected[1] * expected[1];
        CHECK(current.HostView().Coord(node) == expected);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
