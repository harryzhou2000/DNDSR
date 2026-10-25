#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/RelationAccessLegacy.hpp"

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;

static MPIInfo RelationMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

static std::unique_ptr<UnstructuredMesh> RelationFixture(const MPIInfo &mpi)
{
    auto mesh = std::make_unique<UnstructuredMesh>(mpi, 2);
    mesh->coords.InitPair("relation.nodes", mpi);
    mesh->coords.father->Resize(3);
    mesh->coords.TransAttach();
    mesh->coords.trans.createFatherGlobalMapping();
    const DNDS::index offset = (*mesh->coords.father->pLGlobalMapping)(mpi.rank, 0);
    const DNDS::index remote = (*mesh->coords.father->pLGlobalMapping)((mpi.rank + 1) % mpi.size, 0) + 2;
    mesh->coords.trans.createGhostMapping(std::vector<DNDS::index>{remote});
    mesh->coords.trans.createMPITypes();
    mesh->coords.trans.pullOnce();
    mesh->cell2node.InitPair("relation.cells", mpi);
    mesh->cell2node.father->Resize(3);
    const std::vector<std::vector<DNDS::index>> rows{{offset + 1, offset, offset + 1}, {}, {remote, UnInitIndex}};
    for (DNDS::index i = 0; i < 3; ++i)
    {
        mesh->cell2node.ResizeRow(i, rows[size_t(i)].size());
        for (DNDS::rowsize j = 0; j < DNDS::rowsize(rows[size_t(i)].size()); ++j)
            mesh->cell2node(i, j) = rows[size_t(i)][size_t(j)];
    }
    mesh->cell2node.father->Compress();
    // Publishing a legacy CSR pair includes its empty ghost shape. InitPair's
    // default son has not yet published compressed metadata for host sharing.
    mesh->cell2node.son->Resize(0);
    mesh->cell2node.son->Compress();
    mesh->cell2node.TransAttach();
    mesh->cell2node.trans.createFatherGlobalMapping();
    mesh->cell2node.trans.createGhostMapping(std::vector<DNDS::index>{});
    mesh->cell2node.idx.markGlobal();
    return mesh;
}

static R::Array<DNDS::index> NativeRows(const RelationReadAccess &reader)
{
    R::RowBuilder builder(size_t(reader.NumOwned()), sizeof(DNDS::index));
    for (DNDS::index i = 0; i < reader.NumOwned(); ++i)
    {
        auto entries = reader.GlobalRow(i);
        builder.ResizeRow(size_t(i), entries.size());
        for (size_t j = 0; j < entries.size(); ++j)
            builder.Row<DNDS::index>(size_t(i))[j] = entries[j];
    }
    return R::Array<DNDS::index>::FromSharedStorage(builder.Commit());
}

TEST_CASE("Storage-neutral inverse and compose preserve legacy sorted incidence and empty rows")
{
    auto mpi = RelationMPI();
    auto mesh = RelationFixture(mpi);
    auto cells = mesh->Entities(EntityKind::Cell), nodes = mesh->Entities(EntityKind::Node);
    RelationReadAccess legacy(Adj::Cell2Node, mesh->cell2node, cells, nodes);
    NativeRelation native(Adj::Cell2Node, cells, nodes, NativeRows(legacy));
    auto read = native.Read(cells, nodes);
    CHECK(read.GlobalRow(0) == legacy.GlobalRow(0));
    CHECK(read.RowSize(1) == 0);
    CHECK(read.ReadEntry(2, 1).absent);
    CHECK_FALSE(mesh->versions.CanMutate());
    CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Distribution));
    auto inverse = MeshConnectivity::Inverse(read, mpi);
    auto oldInverse = MeshConnectivity::Inverse(mesh->cell2node, 3, mpi, [&](DNDS::index i)
                                                { return cells.LocalToGlobal(i); }, [&](DNDS::index i)
                                                { return nodes.LocalToGlobal(i); }, mesh->coords.father->pLGlobalMapping);
    auto support = inverse.Read();
    CHECK(support.Kind() == Adj::Node2Cell);
    auto inverseSupport = MeshConnectivity::Inverse(support, mpi);
    auto reversedSupport = inverseSupport.Read();
    CHECK(reversedSupport.Kind() != Adj::Cell2Node);
    CHECK(reversedSupport.Kind().order == RelationOrder::SortedUnique);
    CHECK(reversedSupport.GlobalRow(0) == std::vector<DNDS::index>{nodes.LocalToGlobal(0), nodes.LocalToGlobal(1)});
    for (DNDS::index row = 0; row < 3; ++row)
    {
        std::vector<DNDS::index> expected;
        for (auto value : (*oldInverse.father)[row])
            expected.push_back(value);
        CHECK(support.GlobalRow(row) == expected);
    }
    CHECK(support.GlobalRow(0) == std::vector<DNDS::index>{cells.LocalToGlobal(0)});
    CHECK(support.GlobalRow(1) == std::vector<DNDS::index>{cells.LocalToGlobal(0)});
    const auto previousRank = (mpi.rank + mpi.size - 1) % mpi.size;
    CHECK(support.GlobalRow(2) == std::vector<DNDS::index>{3 * previousRank + 2});
    if (mpi.size > 1)
        CHECK_THROWS(MeshConnectivity::Compose(read, support));

    // Transfer only missing intermediate relation rows; numerical composition
    // is unchanged and reads native owned/ghost CSR through retained leases.
    RelationGhostRoute route(nodes);
    auto completeRelation = inverse.WithGhostRows(route);
    auto complete = completeRelation.Read();
    CHECK(complete.Rows() == nodes.NumLocal());
    CHECK(inverse.Read().Rows() == nodes.NumOwned());
    auto composed = MeshConnectivity::Compose(read, complete, true);
    auto composition = composed.Read();
    CHECK(composition.Kind().Path() == std::vector<EntityKind>{EntityKind::Cell, EntityKind::Node, EntityKind::Cell});
    CHECK(composition.Kind().removeSelf);
    CHECK(composition.Kind().order == RelationOrder::SortedUnique);
    CHECK(composition.Kind() != Adj::Cell2Cell);
    CHECK(composition.RowSize(0) == 0);
    CHECK(composition.RowSize(1) == 0);
    CHECK(composition.RowSize(2) == 0);
    auto includeSelf = MeshConnectivity::Compose(read, complete);
    auto self = includeSelf.Read();
    CHECK_FALSE(self.Kind().removeSelf);
    CHECK(self.Kind() != composition.Kind());
    auto reversed = MeshConnectivity::Inverse(composition, mpi);
    CHECK(reversed.Read().Kind().Path() == composition.Kind().Path());
    CHECK(reversed.Read().Kind().removeSelf);
    CHECK(reversed.Read().Kind().provenance == "inverse:" + composition.Kind().Signature());
    CHECK(self.GlobalRow(0) == std::vector<DNDS::index>{cells.LocalToGlobal(0)});
    CHECK(self.GlobalRow(1).empty());
    CHECK(self.GlobalRow(2) == std::vector<DNDS::index>{cells.LocalToGlobal(2)});
    // A second payload reuses the same typed route, including its row sizes.
    auto repeat = inverse.WithGhostRows(route);
    CHECK(repeat.Read().GlobalRow(nodes.NumOwned()) == complete.GlobalRow(nodes.NumOwned()));
}

TEST_CASE("Relation reader distinguishes absence, unavailable globals and wrong entity orders")
{
    auto mpi = RelationMPI();
    auto mesh = RelationFixture(mpi);
    auto other = RelationFixture(mpi);
    auto cells = mesh->Entities(EntityKind::Cell), nodes = mesh->Entities(EntityKind::Node);
    CHECK_THROWS(RelationReadAccess(Adj::Cell2Node, mesh->cell2node, nodes, cells));
    CHECK_THROWS(RelationReadAccess(Adj::Cell2Node, mesh->cell2node, cells, other->Entities(EntityKind::Node)));
    {
        RelationReadAccess source(Adj::Cell2Node, mesh->cell2node, cells, nodes);
        NativeRelation native(Adj::Cell2Node, cells, nodes, NativeRows(source));
        CHECK_THROWS(native.Read(cells, other->Entities(EntityKind::Node)));
        RelationGhostRoute wrongDomain(other->Entities(EntityKind::Cell));
        CHECK_THROWS(native.WithGhostRows(wrongDomain));
        RelationGhostRoute wrongKind(nodes);
        CHECK_THROWS(native.WithGhostRows(wrongKind));
    }
    // Encode a remote global not included in the deliberately sparse node map.
    if (mpi.size > 1)
    {
        auto remote = (*mesh->coords.father->pLGlobalMapping)((mpi.rank + 1) % mpi.size, 0);
        mesh->cell2node(2, 0) = remote;
        mesh->cell2node.idx.wireTargetMapping(mesh->coords.trans.pLGhostMapping);
        mesh->cell2node.toLocal();
        RelationReadAccess local(Adj::Cell2Node, mesh->cell2node, cells, nodes);
        CHECK(local.ReadEntry(2, 0).unavailable);
        CHECK(local.ReadEntry(2, 0).global == remote);
        CHECK(local.ReadEntry(2, 1).absent);
        CHECK(local.Raw(2, 0) != UnInitIndex);
    }
    else
    {
        mesh->cell2node.idx.wireTargetMapping(other->coords.trans.pLGhostMapping);
        mesh->cell2node.toLocal();
        CHECK_THROWS(RelationReadAccess(Adj::Cell2Node, mesh->cell2node, cells, nodes));
    }
}

TEST_CASE("Moved relation reader rejects access while destination retains values and metadata")
{
    auto mpi = RelationMPI();
    auto mesh = RelationFixture(mpi);
    auto cells = mesh->Entities(EntityKind::Cell), nodes = mesh->Entities(EntityKind::Node);
    RelationReadAccess source(Adj::Cell2Node, mesh->cell2node, cells, nodes);
    const auto expected = source.GlobalRow(0);
    {
        auto destination = std::move(source);
        CHECK(destination.GlobalRow(0) == expected);
        CHECK_THROWS(source.Raw(0, 0));
        CHECK_THROWS(source.Source());
        CHECK_FALSE(mesh->versions.CanMutate());
    }
    CHECK(mesh->versions.CanMutate());
    // A moved-from scope must not expose dangling cached pointers even after
    // its destination owner and all retained allocations have gone away.
    mesh.reset();
    CHECK_THROWS(source.GlobalRow(0));
    CHECK_THROWS(source.NumOwned());
}

TEST_CASE("Native inverse routes incidence owners with empty source and target ranks")
{
    const auto mpi = RelationMPI();
    auto mesh = std::make_unique<UnstructuredMesh>(mpi, 2);
    mesh->coords.InitPair("emptyInverse.nodes", mpi);
    mesh->coords.father->Resize(mpi.rank == 0 ? 2 : 0);
    mesh->coords.son->Resize(0);
    mesh->coords.TransAttach();
    mesh->coords.trans.createFatherGlobalMapping();
    mesh->coords.trans.createGhostMapping(std::vector<DNDS::index>{});
    mesh->cell2node.InitPair("emptyInverse.cells", mpi);
    const bool hasSource = mpi.rank == mpi.size - 1;
    mesh->cell2node.father->Resize(hasSource ? 1 : 0);
    if (hasSource)
    {
        mesh->cell2node.ResizeRow(0, 4);
        mesh->cell2node(0, 0) = 1;
        mesh->cell2node(0, 1) = 0;
        mesh->cell2node(0, 2) = 1;
        mesh->cell2node(0, 3) = UnInitIndex;
    }
    mesh->cell2node.father->Compress();
    mesh->cell2node.son->Resize(0);
    mesh->cell2node.son->Compress();
    mesh->cell2node.TransAttach();
    mesh->cell2node.trans.createFatherGlobalMapping();
    mesh->cell2node.trans.createGhostMapping(std::vector<DNDS::index>{});
    mesh->cell2node.idx.markGlobal();
    const auto cells = mesh->Entities(EntityKind::Cell), nodes = mesh->Entities(EntityKind::Node);
    RelationReadAccess cone(Adj::Cell2Node, mesh->cell2node, cells, nodes);
    auto inverse = MeshConnectivity::Inverse(cone, mpi);
    auto support = inverse.Read();
    CHECK(support.NumOwned() == (mpi.rank == 0 ? 2 : 0));
    for (DNDS::index row = 0; row < support.NumOwned(); ++row)
        CHECK(support.GlobalRow(row) == std::vector<DNDS::index>{0});
    RelationGhostRoute route(nodes);
    auto withGhosts = inverse.WithGhostRows(route);
    CHECK(withGhosts.Read().Rows() == nodes.NumLocal());
}

TEST_CASE("Inverse collectively rejects bad global payload before native communication")
{
    const auto mpi = RelationMPI();
    auto mesh = RelationFixture(mpi);
    const auto cells = mesh->Entities(EntityKind::Cell), nodes = mesh->Entities(EntityKind::Node);
    RelationReadAccess read(Adj::Cell2Node, mesh->cell2node, cells, nodes);
    const auto correct = mesh->cell2node(0, 0);
    for (DNDS::index invalid : {nodes.NumGlobal(), DNDS::index(-7)})
    {
        // Controlled serial legacy corruption exercises inverse revalidation.
        // No concurrent operation reads/writes this allocation in this test.
        if (mpi.rank == 0)
            mesh->cell2node(0, 0) = invalid;
        CHECK_THROWS(MeshConnectivity::Inverse(read, mpi));
        if (mpi.rank == 0)
            mesh->cell2node(0, 0) = correct;
    }
    CHECK_NOTHROW(MeshConnectivity::Inverse(read, mpi));
}

TEST_CASE("Typed native halo route rejects stale entity orders")
{
    const auto mpi = RelationMPI();
    auto mesh = RelationFixture(mpi);
    const auto cells = mesh->Entities(EntityKind::Cell), nodes = mesh->Entities(EntityKind::Node);
    std::optional<NativeRelation> native;
    {
        RelationReadAccess source(Adj::Cell2Node, mesh->cell2node, cells, nodes);
        native.emplace(Adj::Cell2Node, cells, nodes, NativeRows(source));
    }
    RelationGhostRoute route(cells);
    CHECK(route.Valid());
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Layout);
        mutation.Commit();
    }
    CHECK_FALSE(route.Valid());
    CHECK_THROWS(native->WithGhostRows(route));
    CHECK(mesh->versions.CanMutate());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
