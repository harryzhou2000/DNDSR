#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <filesystem>
#include <set>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

static DNDS::ssp<G::UnstructuredMesh> Fixture(DNDS::MPIInfo mpi)
{
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    return G::UnstructuredMesh::FromNative(mpi, G::ReadNativeCGNS(mpi,
                                                                  (root / "data/mesh/UniformSquare_10.cgns").string(), 2,
                                                                  [](const std::string &) -> G::t_index
                                                                  { return 21; }, 0, {1, true}));
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

static DNDS::ssp<G::UnstructuredMesh> NativeGeometryFixture(DNDS::MPIInfo mpi);

TEST_CASE("Rigid geometry edit transforms coordinates periodic frame and wall vectors coherently")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto mesh = NativeGeometryFixture(mpi);
    const G::tPoint wall{1, 2, 3};
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
        // The publication epoch can be retained independently of a full row
        // view. A reader on just one rank must reject the collective edit.
        std::optional<R::VersionTracker::ReadLease> held;
        if (mpi.rank == 0)
            held.emplace(mesh->versions.Read());
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
        CHECK(mesh->NodeWallDist(I(node)).isApprox(scale * rotation * wall, 1e-13));
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
        for (auto node : mesh->BoundaryNodes(boundary))
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

TEST_CASE("Collective geometry operations reject a rank without native authority before staging")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    if (mpi.size == 1)
        return;
    auto native = NativeGeometryFixture(mpi);
    auto uninitialized = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 2);
    auto mixed = mpi.rank == 0 ? uninitialized : native;
    const auto stamp = mixed->versions.Stamp();
    const auto source = native->CurrentNativeState();
    const auto before = Points(*native);
    CHECK_THROWS(mixed->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2));
    CHECK_THROWS(mixed->RectifyBoundaryCoordinates([](G::t_index)
                                                   { return 1; }));
    CHECK(stamp.Valid());
    CHECK(Points(*native) == before);
    CHECK(native->CurrentNativeState() == source);
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

// Direct native periodic arrays with independent cell and node owners.
static DNDS::ssp<G::UnstructuredMesh> NativeAffineFixture(const DNDS::MPIInfo &mpi, int dimension)
{
    const bool ownsNodes = mpi.rank == mpi.size - 1, ownsCell = mpi.rank == 0;
    const size_t nodeCount = dimension == 2 ? 2 : 4;
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, ownsNodes ? I(nodeCount) : 0);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, ownsCell ? 1 : 0);
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    auto rows = [](auto values, bool owns)
    {
        using T = typename decltype(values)::value_type;
        R::RowBuilder result(owns ? 1 : 0, sizeof(T));
        if (owns)
        {
            result.ResizeRow(0, values.size());
            std::copy(values.begin(), values.end(), result.template Row<T>(0).data);
        }
        return R::Array<T>::FromSharedStorage(result.Commit());
    };
    std::vector<I> cone = dimension == 2 ? std::vector<I>{0, 0, 1, 1} : std::vector<I>{0, 0, 1, 1, 2, 2, 3, 3};
    std::vector<G::NodePeriodicBits> bits(cone.size());
    for (size_t slot = 0; slot < bits.size(); ++slot)
        bits[slot] = G::NodePeriodicBits{uint8_t(slot % 4 == 1 || slot % 4 == 2)};
    G::NativeMeshInput input{mpi, dimension, cells, nodes, boundaries, rows(cone, ownsCell), rows(std::vector<I>{}, false),
                             R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1))};
    input.periodic = true;
    input.cellNodesPbi = rows(bits, ownsCell);
    input.boundaryNodesPbi = rows(std::vector<G::NodePeriodicBits>{}, false);
    input.periodicInfo.rotationCenter[1].map() = G::tPoint{0.2, 0.3, 0.4};
    {
        auto write = input.coordinates.Write();
        for (size_t node = 0; node < input.coordinates.Rows(); ++node)
        {
            write.View()(node, 0) = 0;
            write.View()(node, 1) = DNDS::real(node % 2);
            write.View()(node, 2) = DNDS::real(node / 2);
        }
        if (ownsCell)
            input.cellInfo.Write().View()(0, 0) = G::ElemInfo{dimension == 2 ? G::Elem::Quad4 : G::Elem::Hex8, G::INTERNAL_ZONE};
    }
    auto state = G::NativeMeshBuilder::Build(std::move(input), {1, true});
    const auto view = state->BorrowHostView();
    auto pair = [&](bool walls)
    {
        R::ArrayPair<DNDS::real> result{R::Array<DNDS::real>(R::RowLayout::Dense(size_t(view.NumNode()), 3)),
                                        R::Array<DNDS::real>(R::RowLayout::Dense(size_t(view.NumNodeGhost()), 3))};
        auto owned = result.owned.Write(), ghost = result.ghost.Write();
        for (I node = 0; node < view.NumNodeProc(); ++node)
        {
            const G::tPoint point = walls ? G::tPoint{1, 2, 3} : G::tPoint(view.Coord(node));
            auto row = node < view.NumNode() ? owned.View()[size_t(node)] : ghost.View()[size_t(node - view.NumNode())];
            for (size_t component = 0; component < 3; ++component)
                row[component] = point[component];
        }
        return result;
    };
    return G::UnstructuredMesh::FromNative(mpi, state->WithGeometry(pair(false), pair(true), view.Periodic()));
}

TEST_CASE("Native affine geometry maps periodic physical corners and invalidates wall vectors atomically")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    int dimension = 2;
    SUBCASE("periodic quadrilateral") {}
    SUBCASE("periodic hexahedron") { dimension = 3; }
    auto mesh = NativeAffineFixture(mpi, dimension);
    const auto state = mesh->CurrentNativeState();
    const auto nodes = mesh->Entities(G::EntityKind::Node);
    const auto cells = mesh->Entities(G::EntityKind::Cell);
    auto old = state->Snapshot(R::Device::Host()).AcquireRead();
    const auto before = old.HostView();
    REQUIRE(before.HasNodeWallDist());
    const auto stamp = mesh->versions.Stamp();
    Eigen::Matrix3d linear;
    linear << 2, 0.4, 0, 0, 0.5, 0.2, 0, 0, 1.5;
    const G::tPoint offset{0.3, -0.2, 0.7};
    {
        std::unique_ptr<G::MeshReadAccess> held;
        if (mpi.rank == 0)
            held = std::make_unique<G::MeshReadAccess>(mesh->AcquireRead());
        CHECK_THROWS(mesh->ApplyAffineGeometryTransform(linear, offset));
    }
    CHECK(stamp.Valid());
    for (int invalid = 0; invalid < 3; ++invalid)
    {
        Eigen::Matrix3d bad = linear;
        if (mpi.rank == 0)
        {
            if (invalid == 0)
                bad.col(0).setZero();
            if (invalid == 1)
                bad.col(0) *= -1;
            if (invalid == 2)
                bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        CHECK_THROWS(mesh->ApplyAffineGeometryTransform(bad, offset));
        CHECK(stamp.Valid());
    }
    G::tPoint badOffset = offset;
    if (mpi.rank == 0)
        badOffset[0] = std::numeric_limits<double>::infinity();
    CHECK_THROWS(mesh->ApplyAffineGeometryTransform(linear, badOffset));
    if (mpi.size > 1)
    {
        G::tPoint different = offset;
        if (mpi.rank == 0)
            different[0] += 1;
        CHECK_THROWS(mesh->ApplyAffineGeometryTransform(linear, different));
        auto differentLinear = linear;
        if (mpi.rank == 0)
            differentLinear(0, 0) += 1;
        CHECK_THROWS(mesh->ApplyAffineGeometryTransform(differentLinear, offset));
    }
    CHECK(stamp.Valid());
    CHECK(mesh->CurrentNativeState() == state);
    mesh->ApplyAffineGeometryTransform(linear, offset);
    CHECK_FALSE(stamp.Valid());
    CHECK(nodes.SameOrder(mesh->Entities(G::EntityKind::Node)));
    CHECK(cells.SameOrder(mesh->Entities(G::EntityKind::Cell)));
    CHECK(mesh->CurrentNativeState()->MeshNamespace() == state->MeshNamespace());
    CHECK(mesh->CurrentNativeState()->IdHighWater() == state->IdHighWater());
    CHECK_FALSE(mesh->CurrentNativeRequirements().wallDistance);
    auto read = mesh->AcquireRead();
    const auto after = read.HostView();
    CHECK_FALSE(after.HasNodeWallDist());
    for (I node = 0; node < after.NumNodeProc(); ++node)
        CHECK(after.Coord(node).isApprox(linear * before.Coord(node) + offset, 1e-13));
    const G::tPoint originalCorners[]{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    for (I cell = 0; cell < after.NumCellProc(); ++cell)
    {
        CHECK(after.CellNodes(cell).data() == before.CellNodes(cell).data());
        CHECK(after.CellNodePeriodicBits(cell).data() == before.CellNodePeriodicBits(cell).data());
        std::vector<G::tPoint> corners;
        for (DNDS::rowsize slot = 0; slot < (dimension == 2 ? 4 : 8); ++slot)
        {
            const auto p = originalCorners[slot];
            const G::tPoint expected{2 * p[0] + 0.4 * p[1] + 0.3, 0.5 * p[1] + 0.2 * p[2] - 0.2, 1.5 * p[2] + 0.7};
            CHECK(before.GetCoordNodeOnCell(cell, slot).isApprox(p, 1e-13));
            corners.push_back(after.GetCoordNodeOnCell(cell, slot));
            CHECK(corners.back().isApprox(expected, 1e-13));
        }
        const G::tPoint u = corners[1] - corners[0], v = corners[3] - corners[0];
        CHECK(u.cross(v).norm() == doctest::Approx(1));
        if (dimension == 3)
        {
            const G::tPoint w = corners[4] - corners[0];
            CHECK(u.cross(v).dot(w) == doctest::Approx(1.5));
            // Hex8 uses [-1,1]^3 reference coordinates.
            CHECK((u / 2).cross(v / 2).dot(w / 2) == doctest::Approx(1.5 / 8));
        }
        else
            CHECK((u / 2).cross(v / 2).norm() == doctest::Approx(0.25));
    }
    for (size_t image = 0; image < before.Periodic().rotation.size(); ++image)
    {
        CHECK(after.Periodic().rotation[image].map().isApprox(Eigen::Matrix3d::Identity(), 1e-13));
        CHECK(after.Periodic().translation[image].map().isApprox(linear * before.Periodic().translation[image].map(), 1e-13));
        CHECK(after.Periodic().rotationCenter[image].map().isApprox(linear * before.Periodic().rotationCenter[image].map() + offset, 1e-13));
    }
    CHECK_FALSE(mesh->coords.father);
}

TEST_CASE("Native affine geometry rejects nonrigid periodic conjugation before publication")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto mesh = NativeAffineFixture(mpi, 2);
    const auto state = mesh->CurrentNativeState();
    auto read = state->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    auto copy = [&](bool wall)
    {
        R::ArrayPair<DNDS::real> result{R::Array<DNDS::real>(R::RowLayout::Dense(size_t(view.NumNode()), 3)),
                                        R::Array<DNDS::real>(R::RowLayout::Dense(size_t(view.NumNodeGhost()), 3))};
        auto owned = result.owned.Write(), ghost = result.ghost.Write();
        for (I row = 0; row < view.NumNodeProc(); ++row)
        {
            auto output = row < view.NumNode() ? owned.View()[size_t(row)] : ghost.View()[size_t(row - view.NumNode())];
            const G::tPoint point = wall ? G::tPoint{1, 2, 3} : G::tPoint(view.Coord(row));
            for (size_t component = 0; component < 3; ++component)
                output[component] = point[component];
        }
        return result;
    };
    // Synthetic proper rotation metadata isolates the conjugation guard;
    // physical image/measure oracles use the translation-only fixture above.
    auto frame = view.Periodic();
    frame.rotation[1].map() = G::RotZ(0.2);
    auto rotated = G::UnstructuredMesh::FromNative(mpi, state->WithGeometry(copy(false), copy(true), frame));
    const auto source = rotated->CurrentNativeState();
    const auto stamp = rotated->versions.Stamp();
    Eigen::Matrix3d shear = Eigen::Matrix3d::Identity();
    shear(0, 1) = 0.5;
    CHECK_THROWS(rotated->ApplyAffineGeometryTransform(shear));
    CHECK(stamp.Valid());
    CHECK(rotated->CurrentNativeState() == source);
    // A proper coordinate rotation conjugates the same periodic frame safely.
    CHECK_NOTHROW(rotated->ApplyAffineGeometryTransform(G::RotZ(0.3), G::tPoint{1, 2, 3}));
    CHECK_FALSE(stamp.Valid());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Geometry operations reject an unpublished mesh without changing its epoch")
{
    DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 2);
    const auto stamp = mesh->versions.Stamp();
    CHECK_THROWS(mesh->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 1));
    CHECK_THROWS(mesh->RectifyBoundaryCoordinates([](G::t_index)
                                                  { return 1; }));
    CHECK(stamp.Valid());
    CHECK_FALSE(mesh->HasNativeState());
}
