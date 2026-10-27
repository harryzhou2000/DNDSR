#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeConnectivity.hpp"
#include "Geom/Mesh/MeshConnectivity.hpp"
#include "NativeMeshOverlapFixture.hpp"

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

static MPIInfo RelationMPI()
{
    MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}
static R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
{
    R::RowBuilder builder(rows.size(), sizeof(I));
    for (size_t row = 0; row < rows.size(); ++row)
    {
        builder.ResizeRow(row, rows[row].size());
        std::copy(rows[row].begin(), rows[row].end(), builder.Row<I>(row).data);
    }
    return R::Array<I>::FromSharedStorage(builder.Commit());
}
struct RelationFixture
{
    EntitySpace nodes, cells;
    R::Array<I> values;
    explicit RelationFixture(const MPIInfo &mpi)
        : nodes(EntitySpace::CreateOwned(EntityKind::Node, mpi, 3)),
          cells(nodes.CreateSibling(EntityKind::Cell, 3))
    {
        const I remote = 3 * ((mpi.rank + 1) % mpi.size) + 2;
        nodes = nodes.WithGhosts(mpi.size > 1 ? std::vector<I>{remote} : std::vector<I>{});
        values = Rows({{3 * mpi.rank, 3 * mpi.rank + 1}, {}, {remote, UnInitIndex}});
    }
    RelationReadAccess Read() const { return {Adj::Cell2Node, cells, nodes, Adj_PointToGlobal, values}; }
};

TEST_CASE("Storage-neutral inverse and compose preserve sorted incidence and empty rows")
{
    const auto mpi = RelationMPI();
    RelationFixture fixture(mpi);
    const auto &cells = fixture.cells;
    const auto &nodes = fixture.nodes;
    auto read = fixture.Read();
    CHECK(read.GlobalRow(0) == std::vector<I>{3 * mpi.rank, 3 * mpi.rank + 1});
    CHECK(read.RowSize(1) == 0);
    CHECK(read.ReadEntry(2, 1).absent);
    CHECK_THROWS(fixture.values.Write());
    auto inverse = MeshConnectivity::Inverse(read, mpi);
    auto support = inverse.Read();
    CHECK(support.Kind() == Adj::Cell2Node.Inverted());
    CHECK(support.Kind() != Adj::Node2Cell);
    CHECK(support.Kind().provenance == "inverse:" + Adj::Cell2Node.Signature());
    CHECK(support.Kind().order == RelationOrder::SortedUnique);
    const I previous = (mpi.rank + mpi.size - 1) % mpi.size;
    CHECK(support.GlobalRow(0) == std::vector<I>{3 * mpi.rank});
    CHECK(support.GlobalRow(1) == std::vector<I>{3 * mpi.rank});
    CHECK(support.GlobalRow(2) == std::vector<I>{3 * previous + 2});
    auto inverseSupport = MeshConnectivity::Inverse(support, mpi);
    auto reversed = inverseSupport.Read();
    CHECK(reversed.Kind() != Adj::Cell2Node);
    CHECK(reversed.Kind().order == RelationOrder::SortedUnique);
    CHECK(reversed.GlobalRow(0) == std::vector<I>{3 * mpi.rank, 3 * mpi.rank + 1});
    if (mpi.size > 1)
        CHECK_THROWS(MeshConnectivity::Compose(read, support));
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
    for (I row = 0; row < 3; ++row)
        CHECK(composition.RowSize(row) == 0);
    auto includeSelf = MeshConnectivity::Compose(read, complete);
    auto self = includeSelf.Read();
    CHECK_FALSE(self.Kind().removeSelf);
    CHECK(self.Kind() != composition.Kind());
    CHECK(self.GlobalRow(0) == std::vector<I>{cells.LocalToGlobal(0)});
    CHECK(self.GlobalRow(1).empty());
    CHECK(self.GlobalRow(2) == std::vector<I>{cells.LocalToGlobal(2)});
    auto reversedComposition = MeshConnectivity::Inverse(composition, mpi);
    CHECK(reversedComposition.Read().Kind().Path() == composition.Kind().Path());
    CHECK(reversedComposition.Read().Kind().removeSelf);
    CHECK(reversedComposition.Read().Kind().provenance == "inverse:" + composition.Kind().Signature());
    auto repeat = inverse.WithGhostRows(route);
    if (nodes.NumGhost())
        CHECK(repeat.Read().GlobalRow(nodes.NumOwned()) == complete.GlobalRow(nodes.NumOwned()));
}

TEST_CASE("Relation reader distinguishes absence unavailable globals and wrong entity orders")
{
    const auto mpi = RelationMPI();
    RelationFixture fixture(mpi), other(mpi);
    const auto &cells = fixture.cells;
    const auto &nodes = fixture.nodes;
    CHECK_THROWS(RelationReadAccess(Adj::Cell2Node, nodes, cells, Adj_PointToGlobal, fixture.values));
    CHECK_THROWS(RelationReadAccess(Adj::Cell2Node, cells, other.nodes, Adj_PointToGlobal, fixture.values));
    {
        auto source = fixture.Read();
        NativeRelation native(Adj::Cell2Node, cells, nodes, fixture.values);
        CHECK_THROWS(native.Read(cells, other.nodes));
        RelationGhostRoute wrongDomain(other.cells), wrongKind(nodes);
        CHECK_THROWS(native.WithGhostRows(wrongDomain));
        CHECK_THROWS(native.WithGhostRows(wrongKind));
    }
    if (mpi.size > 1)
    {
        const I remote = 3 * ((mpi.rank + 1) % mpi.size);
        {
            auto write = fixture.values.Write();
            write.View()(2, 0) = remote;
        }
        auto global = fixture.Read();
        auto converted = NativeRelation::Convert(global, Adj_PointToLocal);
        auto local = converted.Read();
        CHECK(local.ReadEntry(2, 0).unavailable);
        CHECK(local.ReadEntry(2, 0).global == remote);
        CHECK(local.ReadEntry(2, 1).absent);
        CHECK(local.Raw(2, 0) != UnInitIndex);
        CHECK(local.Raw(2, 0) == -1 - remote);
    }
    else
        CHECK_THROWS(RelationReadAccess(Adj::Cell2Node, cells, other.nodes, Adj_PointToLocal, fixture.values));
}

TEST_CASE("Moved relation reader rejects access while destination retains values and metadata")
{
    const auto mpi = RelationMPI();
    RelationFixture fixture(mpi);
    auto source = fixture.Read();
    const auto expected = source.GlobalRow(0);
    {
        auto destination = std::move(source);
        fixture.values.Reset(R::RowLayout::Ragged({}));
        CHECK(destination.GlobalRow(0) == expected);
        CHECK(destination.RowSize(1) == 0);
        CHECK(destination.ReadEntry(2, 1).absent);
        CHECK_THROWS(source.Raw(0, 0));
        CHECK_THROWS(source.Source());
    }
    CHECK_THROWS(source.GlobalRow(0));
    CHECK_THROWS(source.NumOwned());
}

TEST_CASE("Native inverse routes incidence owners with empty source and target ranks")
{
    const auto mpi = RelationMPI();
    auto ownedNodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, mpi.rank == 0 ? 2 : 0);
    const auto cells = ownedNodes.CreateSibling(EntityKind::Cell, mpi.rank == mpi.size - 1 ? 1 : 0);
    auto nodes = ownedNodes.WithGhosts(mpi.size > 1 && mpi.rank == mpi.size - 1 ? std::vector<I>{0, 1} : std::vector<I>{});
    auto values = Rows(mpi.rank == mpi.size - 1 ? std::vector<std::vector<I>>{{0, 1, 1, UnInitIndex}} : std::vector<std::vector<I>>{});
    RelationReadAccess cone(Adj::Cell2Node, cells, nodes, Adj_PointToGlobal, values);
    auto inverse = MeshConnectivity::Inverse(cone, mpi);
    auto support = inverse.Read();
    CHECK(support.NumOwned() == (mpi.rank == 0 ? 2 : 0));
    for (I row = 0; row < support.NumOwned(); ++row)
        CHECK(support.GlobalRow(row) == std::vector<I>{0});
    RelationGhostRoute route(nodes);
    auto ghosted = inverse.WithGhostRows(route);
    CHECK(ghosted.Read().Rows() == nodes.NumLocal());
    for (I row = 0; row < ghosted.Read().Rows(); ++row)
        CHECK(ghosted.Read().GlobalRow(row) == std::vector<I>{0});
}

TEST_CASE("Inverse collectively rejects bad global payload before native communication")
{
    const auto mpi = RelationMPI();
    RelationFixture fixture(mpi);
    auto read = fixture.Read();
    CHECK_THROWS(fixture.values.Write());
    // Deliberate unmanaged corruption probes the inverse's collective
    // revalidation. Managed Array writes above correctly reject this lease.
    auto *raw = static_cast<I *>(fixture.values.Storage().LegacyPointer());
    const I correct = raw[0];
    for (I invalid : {fixture.nodes.NumGlobal(), I(-7)})
    {
        if (mpi.rank == 0)
            raw[0] = invalid;
        CHECK_THROWS(MeshConnectivity::Inverse(read, mpi));
        if (mpi.rank == 0)
            raw[0] = correct;
    }
    CHECK_NOTHROW(MeshConnectivity::Inverse(read, mpi));
}

TEST_CASE("Pinned setup mapping retains bounds conversion and moved-scope errors")
{
    const auto mpi = RelationMPI();
    RelationFixture fixture(mpi);
    auto original = fixture.Read();
    auto read = std::move(original);
    CHECK_THROWS(original.ReadEntry(0, 0));
    CHECK_THROWS(read.ReadEntry(-1, 0));
    CHECK_THROWS(read.ReadEntry(read.Rows(), 0));
    CHECK_THROWS(read.ReadEntry(1, 0));
    CHECK_THROWS(read.ConvertedIndex(0, 0, Adj_Unknown));
    CHECK_THROWS(fixture.nodes.LocalToGlobal(-1));
    CHECK_FALSE(fixture.nodes.GlobalToLocal(-1));
    CHECK_FALSE(fixture.nodes.GlobalToLocal(fixture.nodes.NumGlobal()));
    auto local = NativeRelation::Convert(read, Adj_PointToLocal);
    auto localRead = local.Read();
    for (I row = 0; row < read.Rows(); ++row)
        for (size_t slot = 0; slot < read.RowSize(row); ++slot)
        {
            const auto a = read.ReadEntry(row, slot), b = localRead.ReadEntry(row, slot);
            CHECK(a.absent == b.absent);
            CHECK(a.global == b.global);
            CHECK(a.unavailable == b.unavailable);
        }
    CHECK_THROWS(fixture.values.Write());
}

TEST_CASE("Typed native halo route rejects stale entity orders")
{
    const auto mpi = RelationMPI();
    auto mesh = OverlapTest::BuildNativeFixture(mpi).mesh;
    const auto cells = mesh->Entities(EntityKind::Cell), nodes = mesh->Entities(EntityKind::Node);
    std::optional<NativeRelation> native;
    {
        auto source = RelationReadAccess::Current(*mesh, Adj::Cell2Node, MeshReadRequirements::Primary());
        native.emplace(NativeRelation::Convert(source, Adj_PointToGlobal));
        CHECK_FALSE(mesh->versions.CanMutate());
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Distribution));
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
