#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/NativeConnectivity.hpp"

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;

static MPIInfo NativeMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

template <class T>
static R::ArrayPair<T> TablePair(size_t rows, size_t width)
{
    return {R::Array<T>(R::RowLayout::Dense(rows, width)), R::Array<T>(R::RowLayout::Dense(0, width))};
}

static R::Array<DNDS::index> IndexRows(const std::vector<std::vector<DNDS::index>> &rows)
{
    R::RowBuilder builder(rows.size(), sizeof(DNDS::index));
    for (size_t row = 0; row < rows.size(); ++row)
    {
        builder.ResizeRow(row, rows[row].size());
        auto output = builder.Row<DNDS::index>(row);
        for (size_t slot = 0; slot < rows[row].size(); ++slot)
            output[slot] = rows[row][slot];
    }
    return R::Array<DNDS::index>::FromSharedStorage(builder.Commit());
}

struct NativeTriangle
{
    NativeMeshData data;
    std::map<EntityKind, EntitySpace> spaces;
    std::map<EntityKind, R::Array<DNDS::index>> ids;

    explicit NativeTriangle(const MPIInfo &mpi)
    {
        auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, 3);
        spaces.emplace(EntityKind::Node, nodes);
        spaces.emplace(EntityKind::Cell, nodes.CreateSibling(EntityKind::Cell, 1));
        spaces.emplace(EntityKind::Bnd, nodes.CreateSibling(EntityKind::Bnd, 0));
        data.dimension = 2;
        data.coords = TablePair<DNDS::real>(3, 3);
        {
            auto write = data.coords->owned.Write();
            const DNDS::real coordinates[3][3]{{DNDS::real(2 * mpi.rank), 0, 0},
                                               {DNDS::real(2 * mpi.rank + 1), 0, 0},
                                               {DNDS::real(2 * mpi.rank), 1, 0}};
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    write.View()(i, j) = coordinates[i][j];
        }
        data.cellInfo = TablePair<ElemInfo>(1, 1);
        {
            auto write = data.cellInfo->owned.Write();
            write.View()(0, 0) = ElemInfo{Elem::Tri3, INTERNAL_ZONE};
        }
        data.bndInfo = TablePair<ElemInfo>(0, 1);
        data.relations[size_t(MeshRelation::CellNodes)] = R::ArrayPair<DNDS::index>{IndexRows({{0, 1, 2}}), IndexRows({})};
        data.relations[size_t(MeshRelation::CellCells)] = R::ArrayPair<DNDS::index>{IndexRows({{}}), IndexRows({})};
        data.relations[size_t(MeshRelation::BoundaryNodes)] = R::ArrayPair<DNDS::index>{IndexRows({}), IndexRows({})};
        data.relations[size_t(MeshRelation::BoundaryCells)] = TablePair<DNDS::index>(0, 2);
        for (const auto &[kind, space] : spaces)
        {
            R::Array<DNDS::index> values(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
            auto write = values.Write();
            for (DNDS::index row = 0; row < space.NumOwned(); ++row)
                write.View()(size_t(row), 0) = 1000000000000LL + 17 * space.LocalToGlobal(row) + int(kind);
            ids.emplace(kind, std::move(values));
        }
    }
};

TEST_CASE("Native arrays publish a mesh with empty legacy storage and retained read-only aliases")
{
    const auto mpi = NativeMPI();
    NativeTriangle input(mpi);
    auto coords = input.data.coords->owned;
    auto nodes = input.data.relations[size_t(MeshRelation::CellNodes)]->owned;
    auto nodeIds = input.ids.at(EntityKind::Node);
    auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, std::move(input.ids), MeshReadRequirements::Primary());
    auto mesh = UnstructuredMesh::FromNative(mpi, state);
    CHECK(mesh->HasNativeState());
    CHECK_FALSE(mesh->coords.father);
    CHECK_FALSE(mesh->cell2node.father);
    CHECK_FALSE(mesh->node2cell.father);
    CHECK(mesh->Entities(EntityKind::Node).SameOrder(input.spaces.at(EntityKind::Node)));
    CHECK(state->MeshNamespace().size() == 36);
    // Immutable publication and repeated owner binding must not mutate the
    // shared version controller retained by factory spaces or prior states.
    R::VersionTracker secondOwner, thirdOwner;
    auto second = state->BindEpoch(secondOwner);
    auto third = second->BindEpoch(thirdOwner);
    state->Validate();
    second->Validate();
    third->Validate();
    CHECK(state->MeshNamespace().size() == 36);
    CHECK(second->MeshNamespace() == state->MeshNamespace());
    CHECK(third->MeshNamespace() == state->MeshNamespace());
    CHECK(second->IdHighWater() == state->IdHighWater());
    CHECK(third->IdHighWater() == state->IdHighWater());
    for (const auto &[kind, source] : input.spaces)
    {
        CHECK(source.Valid());
        CHECK(state->Entities(kind).SameOrder(source));
        CHECK(second->Entities(kind).SameOrder(source));
        CHECK(third->Entities(kind).SameOrder(source));
        CHECK(mesh->Entities(kind).SameOrder(source));
    }
    CHECK(mesh->OriginalEntityId(EntityKind::Node, 2) == 1000000000000LL + 17 * (3 * mpi.rank + 2) + int(EntityKind::Node));
    CHECK_THROWS(coords.Write());
    CHECK_THROWS(nodes.Write());
    CHECK_THROWS(nodeIds.Write());
    auto read = mesh->AcquireRead();
    CHECK(read.HostView().Coord(0).data() == coords.Read().View().data);
    mesh.reset();
    state.reset();
    CHECK(read.HostView().NumCell() == 1);
    CHECK(read.HostView().CellNodes(0)[2] == 2);
    CHECK(read.HostView().Coord(1)[0] == DNDS::real(2 * mpi.rank + 1));
    // Replacement of one caller handle never retargets a retained mesh scope.
    coords.Reset(R::RowLayout::Dense(0, 3));
    CHECK(read.HostView().Coord(2)[1] == 1);
}

TEST_CASE("Native publication rejects missing topology, invalid rows and aliased active writers")
{
    const auto mpi = NativeMPI();
    NativeTriangle input(mpi);
    SUBCASE("required relation missing")
    {
        input.data.relations[size_t(MeshRelation::CellCells)].reset();
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
    SUBCASE("foreign canonical space with equal size")
    {
        input.spaces.at(EntityKind::Cell) = EntitySpace::CreateOwned(EntityKind::Cell, mpi, 1);
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
    SUBCASE("owned IDs duplicated")
    {
        {
            auto write = input.ids.at(EntityKind::Node).Write();
            write.View()(1, 0) = write.View()(0, 0);
        }
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
    SUBCASE("out of closure node")
    {
        {
            auto write = input.data.relations[size_t(MeshRelation::CellNodes)]->owned.Write();
            write.View()(0, 2) = 3;
        }
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
    SUBCASE("live writer")
    {
        auto write = input.data.coords->owned.Write();
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
}

TEST_CASE("Native publication coordinates rank-local missing spaces and topology")
{
    const auto mpi = NativeMPI();
    NativeTriangle input(mpi);
    auto retainedCoords = input.data.coords->owned;
    SUBCASE("only rank zero lacks node space")
    {
        if (mpi.rank == 0)
            input.spaces.erase(EntityKind::Node);
    }
    SUBCASE("only rank zero lacks required topology")
    {
        if (mpi.rank == 0)
            input.data.relations[size_t(MeshRelation::CellNodes)].reset();
    }
    CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    // Peers that reached buffer capture before collective rejection must release
    // their preparation leases; no failed publication may seal caller inputs.
    CHECK_NOTHROW(retainedCoords.Write());
    for (const auto &[kind, space] : input.spaces)
        CHECK(space.Valid());
}

TEST_CASE("Native publication rejects persistent cell IDs owned on multiple ranks")
{
    const auto mpi = NativeMPI();
    if (mpi.size == 1)
        return;
    NativeTriangle input(mpi);
    auto cellIds = input.ids.at(EntityKind::Cell);
    {
        auto write = cellIds.Write();
        // Each rank is locally unique; only distributed validation can detect
        // the two distinct owning rows carrying this same large sparse ID.
        if (mpi.rank < 2)
            write.View()(0, 0) = 8000000000000LL;
    }
    CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    CHECK_NOTHROW(cellIds.Write());
    CHECK(input.spaces.at(EntityKind::Cell).Valid());
}

TEST_CASE("Native ghost IDs agree with their owner and failed checks retain source spaces")
{
    const auto mpi = NativeMPI();
    if (mpi.size == 1)
        return;
    NativeTriangle input(mpi);
    const DNDS::index next = (mpi.rank + 1) % mpi.size;
    const auto sourceNodes = input.spaces.at(EntityKind::Node);
    const auto ghostedNodes = sourceNodes.WithGhosts({3 * next});
    input.spaces.erase(EntityKind::Node);
    input.spaces.emplace(EntityKind::Node, ghostedNodes);
    input.data.coords->ghost = R::Array<DNDS::real>(R::RowLayout::Dense(1, 3));
    {
        auto write = input.data.coords->ghost.Write();
        write.View()(0, 0) = DNDS::real(2 * next);
        write.View()(0, 1) = 0;
        write.View()(0, 2) = 0;
    }
    auto previousIds = input.ids.at(EntityKind::Node).Read();
    R::Array<DNDS::index> combinedIds(R::RowLayout::Dense(4, 1));
    const DNDS::index ownerId = 1000000000000LL + 17 * (3 * next) + int(EntityKind::Node);
    {
        auto write = combinedIds.Write();
        for (size_t row = 0; row < 3; ++row)
            write.View()(row, 0) = previousIds.View()(row, 0);
        write.View()(3, 0) = ownerId;
    }
    input.ids.erase(EntityKind::Node);
    input.ids.emplace(EntityKind::Node, combinedIds);
    bool mismatch = false;
    SUBCASE("correct owner IDs publish with remote coordinates") {}
    SUBCASE("only rank zero has a wrong ghost ID")
    {
        mismatch = true;
        if (mpi.rank == 0)
        {
            auto write = combinedIds.Write();
            write.View()(3, 0) = ownerId + 999;
        }
    }
    auto retainedGhostCoords = input.data.coords->ghost;
    if (mismatch)
    {
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
        CHECK_NOTHROW(combinedIds.Write());
        CHECK_NOTHROW(retainedGhostCoords.Write());
    }
    else
    {
        auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary());
        auto mesh = UnstructuredMesh::FromNative(mpi, state);
        CHECK(mesh->NumNodeGhost() == 1);
        CHECK(mesh->Entities(EntityKind::Node).SameOrder(ghostedNodes));
        CHECK(mesh->OriginalEntityId(EntityKind::Node, 3) == ownerId);
        CHECK(mesh->Coord(3) == tPoint(DNDS::real(2 * next), 0, 0));
        CHECK_THROWS(combinedIds.Write());
        CHECK_THROWS(retainedGhostCoords.Write());
        auto read = mesh->AcquireRead();
        mesh.reset();
        state.reset();
        CHECK(read.HostView().Coord(3) == tPoint(DNDS::real(2 * next), 0, 0));
    }
    CHECK(sourceNodes.Valid());
    CHECK(ghostedNodes.Valid());
    for (const auto &[kind, space] : input.spaces)
        CHECK(space.Valid());
}

TEST_CASE("Native mesh namespaces and non-reuse high-water marks survive publication")
{
    const auto mpi = NativeMPI();
    NativeTriangle firstInput(mpi);
    auto first = NativeMeshState::FromNative(mpi, std::move(firstInput.data), firstInput.spaces,
                                             firstInput.ids, MeshReadRequirements::Primary());
    NativeTriangle secondInput(mpi);
    auto second = NativeMeshState::FromNative(mpi, std::move(secondInput.data), secondInput.spaces,
                                              secondInput.ids, MeshReadRequirements::Primary());
    CHECK(first->MeshNamespace().size() == 36);
    CHECK(second->MeshNamespace().size() == 36);
    CHECK(first->MeshNamespace() != second->MeshNamespace());
    auto canonical = first->MeshNamespace();
    MPI_Bcast(canonical.data(), 36, MPI_CHAR, 0, mpi.comm);
    CHECK(first->MeshNamespace() == canonical);
    CHECK(first->NextEntityId(EntityKind::Node) == 1000000000000LL + 17 * (3 * mpi.size - 1) + int(EntityKind::Node) + 1);
    CHECK(first->NextEntityId(EntityKind::Cell) == 1000000000000LL + 17 * (mpi.size - 1) + int(EntityKind::Cell) + 1);
    CHECK(first->NextEntityId(EntityKind::Bnd) == 0);
    CHECK_THROWS(first->NextEntityId(EntityKind::Face));

    NativeTriangle retainedInput(mpi);
    const std::string explicitNamespace = "01234567-89ab-4cde-8fab-0123456789ab";
    retainedInput.data.identity.meshNamespace = explicitNamespace;
    retainedInput.data.identity.nextIds[EntityKind::Node] = 9000000000000LL;
    auto retained = NativeMeshState::FromNative(mpi, std::move(retainedInput.data), retainedInput.spaces,
                                                retainedInput.ids, MeshReadRequirements::Primary());
    CHECK(retained->MeshNamespace() == explicitNamespace);
    // This supplied mark can represent IDs allocated and later deleted. It
    // remains above the surviving maximum rather than being reset to max+1.
    CHECK(retained->NextEntityId(EntityKind::Node) == 9000000000000LL);
    R::VersionTracker owner;
    auto rebound = retained->BindEpoch(owner);
    CHECK(rebound->MeshNamespace() == explicitNamespace);
    CHECK(rebound->IdHighWater() == retained->IdHighWater());
    CHECK(retainedInput.spaces.at(EntityKind::Node).Valid());
}

TEST_CASE("Native publication rejects inconsistent identity and reused or exhausted IDs")
{
    const auto mpi = NativeMPI();
    NativeTriangle input(mpi);
    SUBCASE("rank zero supplies an invalid namespace")
    {
        if (mpi.rank == 0)
            input.data.identity.meshNamespace = "invalid namespace";
    }
    SUBCASE("explicit namespaces differ across ranks")
    {
        if (mpi.size == 1)
            return;
        input.data.identity.meshNamespace = mpi.rank == 0 ? "01234567-89ab-4cde-8fab-0123456789ab" : "11234567-89ab-4cde-8fab-0123456789ab";
    }
    SUBCASE("high-water mark would reuse an existing ID")
    {
        input.data.identity.nextIds[EntityKind::Node] = 1;
    }
    SUBCASE("high-water marks differ across ranks")
    {
        if (mpi.size == 1)
            return;
        input.data.identity.nextIds[EntityKind::Node] = 9000000000000LL + mpi.rank;
    }
    SUBCASE("maximum ID has no representable next value")
    {
        if (mpi.rank == 0)
        {
            auto write = input.ids.at(EntityKind::Node).Write();
            write.View()(0, 0) = std::numeric_limits<DNDS::index>::max();
        }
    }
    auto retainedCoords = input.data.coords->owned;
    CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    CHECK_NOTHROW(retainedCoords.Write());
    for (const auto &[kind, space] : input.spaces)
        CHECK(space.Valid());
}

TEST_CASE("Native publication rejects optional values outside their validated groups")
{
    const auto mpi = NativeMPI();
    NativeTriangle input(mpi);
    auto requirements = MeshReadRequirements::Primary();
    SUBCASE("wall distance must be requested")
    {
        input.data.nodeWallDist = TablePair<DNDS::real>(3, 3);
    }
    SUBCASE("facial information must be requested")
    {
        input.data.faceInfo = TablePair<ElemInfo>(0, 1);
    }
    SUBCASE("edge information must be requested")
    {
        input.data.edgeInfo = TablePair<ElemInfo>(0, 1);
    }
    SUBCASE("nonperiodic node companions are rejected")
    {
        input.data.cellNodesPbi = TablePair<NodePeriodicBits>(1, 3);
    }
    SUBCASE("nonperiodic facial companions are rejected")
    {
        input.data.faceNodesPbi = TablePair<NodePeriodicBits>(0, 2);
    }
    SUBCASE("periodic facial companions require their group")
    {
        input.data.periodic = true;
        input.data.cellNodesPbi = TablePair<NodePeriodicBits>(1, 3);
        input.data.bndNodesPbi = TablePair<NodePeriodicBits>(0, 2);
        input.data.cellFacesPbi = TablePair<NodePeriodicBits>(1, 3);
    }
    SUBCASE("periodic edge companions require their group")
    {
        input.data.periodic = true;
        input.data.cellNodesPbi = TablePair<NodePeriodicBits>(1, 3);
        input.data.bndNodesPbi = TablePair<NodePeriodicBits>(0, 2);
        input.data.edgeNodesPbi = TablePair<NodePeriodicBits>(0, 2);
    }
    SUBCASE("requested wall distance has wrong rows")
    {
        requirements.wallDistance = true;
        input.data.nodeWallDist = TablePair<DNDS::real>(2, 3);
    }
    SUBCASE("requested wall distance has wrong width")
    {
        requirements.wallDistance = true;
        input.data.nodeWallDist = TablePair<DNDS::real>(3, 1);
    }
    CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, requirements));
}

TEST_CASE("Native element dimensions follow cell and boundary roles")
{
    const auto mpi = NativeMPI();
    NativeTriangle input(mpi);
    SUBCASE("two dimensional cell cannot be a tetrahedron")
    {
        auto write = input.data.cellInfo->owned.Write();
        write.View()(0, 0) = ElemInfo{Elem::Tet4, INTERNAL_ZONE};
        write.Wait();
        input.data.relations[size_t(MeshRelation::CellNodes)]->owned = IndexRows({{0, 1, 2, 0}});
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
    SUBCASE("two dimensional cell cannot be a line")
    {
        auto write = input.data.cellInfo->owned.Write();
        write.View()(0, 0) = ElemInfo{Elem::Line2, INTERNAL_ZONE};
        write.Wait();
        input.data.relations[size_t(MeshRelation::CellNodes)]->owned = IndexRows({{0, 1}});
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
    SUBCASE("three dimensional cell cannot be a triangle")
    {
        input.data.dimension = 3;
        CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
    }
    SUBCASE("two dimensional boundaries have codimension one")
    {
        const auto boundary = input.spaces.at(EntityKind::Node).CreateSibling(EntityKind::Bnd, 1);
        input.spaces.erase(EntityKind::Bnd);
        input.spaces.emplace(EntityKind::Bnd, boundary);
        input.data.bndInfo = TablePair<ElemInfo>(1, 1);
        {
            auto write = input.data.bndInfo->owned.Write();
            write.View()(0, 0) = ElemInfo{Elem::Line2, 1};
        }
        input.data.relations[size_t(MeshRelation::BoundaryNodes)]->owned = IndexRows({{1, 0}});
        input.data.relations[size_t(MeshRelation::BoundaryCells)]->owned = IndexRows({{0, UnInitIndex}});
        input.ids.erase(EntityKind::Bnd);
        input.ids.emplace(EntityKind::Bnd, R::Array<DNDS::index>(R::RowLayout::Dense(1, 1)));
        {
            auto write = input.ids.at(EntityKind::Bnd).Write();
            write.View()(0, 0) = 9000000000000LL + mpi.rank;
        }
        SUBCASE("line boundary remains valid")
        {
            auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary());
            auto mesh = UnstructuredMesh::FromNative(mpi, state);
            CHECK(mesh->GetBndElement(0).GetDim() == 1);
            CHECK(mesh->GetBndInfo(0).type == Elem::Line2);
        }
        SUBCASE("triangle boundary in two dimensions is rejected")
        {
            {
                auto write = input.data.bndInfo->owned.Write();
                write.View()(0, 0) = ElemInfo{Elem::Tri3, 1};
            }
            input.data.relations[size_t(MeshRelation::BoundaryNodes)]->owned = IndexRows({{0, 1, 2}});
            CHECK_THROWS(NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, MeshReadRequirements::Primary()));
        }
    }
}

TEST_CASE("Requested native wall distance is validated exposed and protected")
{
    const auto mpi = NativeMPI();
    NativeTriangle input(mpi);
    input.data.nodeWallDist = TablePair<DNDS::real>(3, 3);
    auto values = input.data.nodeWallDist->owned;
    {
        auto write = values.Write();
        for (size_t row = 0; row < 3; ++row)
            for (size_t component = 0; component < 3; ++component)
                write.View()(row, component) = DNDS::real(row * 3 + component);
    }
    auto requirements = MeshReadRequirements::Primary();
    requirements.wallDistance = true;
    auto state = NativeMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids, requirements);
    auto mesh = UnstructuredMesh::FromNative(mpi, state);
    CHECK(mesh->HasNodeWallDist());
    CHECK(mesh->NodeWallDist(2) == tPoint(6, 7, 8));
    CHECK_THROWS(values.Write());
}

TEST_CASE("Native entity spaces support remote node-star discovery without a legacy mesh")
{
    const auto mpi = NativeMPI();
    // One node and one cell per rank. Every rank's cell references the next
    // rank's node; node owners have no incident local cells when np > 1.
    const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, 1);
    const auto cells = nodes.CreateSibling(EntityKind::Cell, 1);
    const DNDS::index next = (mpi.rank + 1) % mpi.size;
    NativeRelation c2n(Adj::Cell2Node, cells, nodes, IndexRows({{next}}));
    auto n2c = MeshConnectivity::Inverse(c2n.Read(), mpi);
    CHECK(n2c.Read().GlobalRow(0) == std::vector<DNDS::index>{(mpi.rank + mpi.size - 1) % mpi.size});
    NativeConnectivity connectivity(mpi);
    connectivity.Register(c2n.Read());
    connectivity.Register(n2c.Read());
    const auto support = n2c.Read().Kind();
    CHECK(support != Adj::Node2Cell);
    GhostSpec spec{{{EntityKind::Node, {support}, EntityKind::Cell},
                    {EntityKind::Node, {support, Adj::Cell2Node}, EntityKind::Node}}};
    auto discovery = connectivity.Evaluate(CompiledGhostTree::compile(spec));
    auto ghosted = connectivity.BuildGhosted(discovery);
    const auto &newCells = ghosted.spaces.at(EntityKind::Cell);
    CHECK(newCells.NumGhost() == (mpi.size > 1 ? 1 : 0));
    CHECK(ghosted.relations.at(support).Read().GlobalRow(0) == n2c.Read().GlobalRow(0));
    CHECK(nodes.SameOrder(ghosted.spaces.at(EntityKind::Node)) == false);
    CHECK(nodes.Valid());
    const auto sourceCell = newCells.GlobalToLocal((mpi.rank + mpi.size - 1) % mpi.size);
    REQUIRE(sourceCell.has_value());
    CHECK(ghosted.relations.at(Adj::Cell2Node).Read().GlobalRow(*sourceCell) == std::vector<DNDS::index>{mpi.rank});
    // A one-rank-invalid request is rejected collectively before mapping MPI.
    CHECK_THROWS(nodes.WithGhosts(mpi.rank == 0 ? std::vector<DNDS::index>{-1} : std::vector<DNDS::index>{}));
    CHECK_THROWS(nodes.CreateSibling(EntityKind::Face, mpi.rank == 0 ? -1 : 0));
}

TEST_CASE("Replacing an entity descriptor does not mutate retained source controllers")
{
    const auto mpi = NativeMPI();
    const auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, 1);
    const auto cells = nodes.CreateSibling(EntityKind::Cell, 1);
    RelationReadAccess read(Adj::Node2Cell, nodes, cells, Adj_PointToGlobal, IndexRows({{mpi.rank}}));
    auto handle = nodes;
    const auto other = nodes.CreateSibling(EntityKind::Node, 2);
    CHECK_NOTHROW(handle = other);
    CHECK(nodes.Valid());
    CHECK(cells.Valid());
    CHECK(handle.SameOrder(other));
    CHECK_FALSE(handle.SameOrder(nodes));
    CHECK(read.GlobalRow(0) == std::vector<DNDS::index>{mpi.rank});
    CHECK_NOTHROW(handle = EntitySpace(nodes));
    CHECK(other.Valid());
    CHECK(handle.SameOrder(nodes));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
