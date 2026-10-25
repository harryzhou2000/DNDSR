#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "Geom/Mesh/RelationAccess.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <set>

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

namespace
{
    DNDS::MPIInfo MPI()
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        return mpi;
    }
    R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
    {
        R::RowBuilder builder(rows.size(), sizeof(I));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            builder.ResizeRow(row, rows[row].size(), false);
            for (size_t slot = 0; slot < rows[row].size(); ++slot)
                builder.Row<I>(row)[slot] = rows[row][slot];
        }
        return R::Array<I>::FromSharedStorage(builder.Commit());
    }
    G::NativeMeshInput Input(const DNDS::MPIInfo &mpi, I nodeCount, I cellCount, I boundaryCount,
                             std::vector<std::vector<I>> cellNodes, std::vector<std::vector<I>> boundaryNodes)
    {
        auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, nodeCount);
        auto cells = nodes.CreateSibling(G::EntityKind::Cell, cellCount);
        auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, boundaryCount);
        G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(cellNodes), Rows(boundaryNodes), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodeCount), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cellCount), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(boundaryCount), 1)), {}};
        {
            auto coords = input.coordinates.Write();
            for (I row = 0; row < nodeCount; ++row)
            {
                const auto global = nodes.LocalToGlobal(row);
                coords.View()(size_t(row), 0) = DNDS::real(global);
                coords.View()(size_t(row), 1) = DNDS::real(global % 3 == 2);
                coords.View()(size_t(row), 2) = 0;
            }
            auto info = input.cellInfo.Write();
            for (I row = 0; row < cellCount; ++row)
                info.View()(size_t(row), 0) = G::ElemInfo{G::Elem::Tri3, G::INTERNAL_ZONE};
            auto bndInfo = input.boundaryInfo.Write();
            for (I row = 0; row < boundaryCount; ++row)
                bndInfo.View()(size_t(row), 0) = G::ElemInfo{G::Elem::Line2, G::BC_ID_DEFAULT_WALL + mpi.rank};
        }
        return input;
    }
    G::NativeMeshInput RotatedTriangles(const DNDS::MPIInfo &mpi, bool boundaries)
    {
        // Every cell's nodes belong to the next rank. An owner can have a
        // complete incident star with no incident owned cell. Boundary rows
        // belong to node owners, independently of their containing cells.
        const I next = (mpi.rank + 1) % mpi.size;
        return Input(mpi, 3, 1, boundaries ? 1 : 0,
                     {{3 * next, 3 * next + 1, 3 * next + 2}},
                     boundaries ? std::vector<std::vector<I>>{{3 * mpi.rank, 3 * mpi.rank + 1}} : std::vector<std::vector<I>>{});
    }
    std::set<I> Globals(const G::EntitySpace &space, G::MeshRowSpan<I> row)
    {
        std::set<I> result;
        for (auto local : row)
        {
            REQUIRE(local >= 0);
            result.insert(space.LocalToGlobal(local));
        }
        return result;
    }
}

TEST_CASE("Owned native capture assembles after source release with retained companions and exact descriptor rows")
{
    const auto mpi = MPI();
    auto input = RotatedTriangles(mpi, true);
    G::NativeMeshBuildOptions options;
    options.completeNodeStar = true;
    auto source = G::NativeMeshBuilder::Build(input, options);
    auto captured = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    const auto identity = captured.identity;
    const auto keys = source->RelationKeys();
    for (const auto &[kind, space] : captured.spaces)
        CHECK(space.NumGhost() == 0);
    CHECK(captured.ids.at(G::EntityKind::Cell).Rows() == 1);
    auto coordinates = *captured.coords;
    const auto nodes = captured.relations[size_t(G::MeshRelation::CellNodes)]->Read();
    CHECK(nodes.View()(0, 0) == 3 * ((mpi.rank + 1) % mpi.size));
    source.reset();
    CHECK_THROWS(coordinates.Write());
    auto rebuilt = G::NativeMeshAssembler::Build(mpi, std::move(captured), 2);
    CHECK(rebuilt->MeshNamespace() == identity.meshNamespace);
    CHECK(rebuilt->IdHighWater() == identity.nextIds);
    CHECK(rebuilt->RelationKeys() == keys);
    auto read = rebuilt->Snapshot(R::Device::Host()).AcquireRead();
    auto view = read.HostView();
    for (size_t slot = 0; slot < 3; ++slot)
    {
        const I global = 3 * ((mpi.rank + 1) % mpi.size) + I(slot);
        CHECK(rebuilt->OriginalEntityId(G::EntityKind::Node, view.CellNodes(0)[slot]) == global);
        CHECK(view.Coord(view.CellNodes(0)[slot])[0] == double(global));
    }
}

TEST_CASE("Native assembler rejects rank-local ID schema and closure schedule errors before publication")
{
    const auto mpi = MPI();
    auto input = RotatedTriangles(mpi, true);
    auto source = G::NativeMeshBuilder::Build(input);
    auto captured = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    SUBCASE("persistent ID extra columns")
    {
        if (mpi.rank == 0)
            captured.ids.at(G::EntityKind::Cell) = R::Array<I>(R::RowLayout::Dense(1, 2));
        CHECK_THROWS(G::NativeMeshAssembler::Build(mpi, std::move(captured)));
    }
    SUBCASE("different closure depth")
    {
        if (mpi.size == 1)
            return;
        CHECK_THROWS(G::NativeMeshAssembler::Build(mpi, std::move(captured), mpi.rank == 0 ? 2 : 1));
    }
    CHECK(source->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
}

TEST_CASE("Owned native capture collectively rejects mismatched source inventory before ghost-map collectives")
{
    const auto mpi = MPI();
    if (mpi.size == 1)
        return;
    auto first = G::NativeMeshBuilder::Build(RotatedTriangles(mpi, false));
    auto second = G::NativeMeshBuilder::Build(RotatedTriangles(mpi, true));
    CHECK_THROWS(G::NativeMeshAssembler::CaptureOwned(mpi.rank == 0 ? *first : *second, mpi));
    CHECK(first->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
    CHECK(second->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
}

TEST_CASE("Native transition preserves exact persistent topology and coordinate stencil across independent placements")
{
    const auto mpi = MPI();
    auto input = RotatedTriangles(mpi, true);
    G::NativeMeshBuildOptions options;
    options.completeNodeStar = true;
    auto source = G::NativeMeshBuilder::Build(input, options);
    G::ReorderInput reorder;
    reorder.explicitMaps = {{G::EntityKind::Cell, {int((mpi.rank + 1) % mpi.size)}},
                            {G::EntityKind::Node, std::vector<DNDS::MPI_int>(3, (mpi.rank + 2) % mpi.size)},
                            {G::EntityKind::Bnd, {int((mpi.rank + 3) % mpi.size)}}};
    auto before = source->Snapshot(R::Device::Host()).AcquireRead();
    const auto original = before.HostView();
    const I oldCell = (mpi.rank + mpi.size - 1) % mpi.size;
    const I oldNodeOwner = (oldCell + 1) % mpi.size;
    const double expected = double(9 * oldNodeOwner + 3);
    double originalStencil = 0;
    for (auto node : original.CellNodes(0))
        originalStencil += original.Coord(node)[0];
    CHECK(originalStencil == double(9 * ((mpi.rank + 1) % mpi.size) + 3));
    auto result = G::NativeMeshTransition::Build(*source, mpi, reorder, 2);
    REQUIRE(result.state);
    CHECK(result.transfers.size() == 3);
    CHECK(result.state->MeshNamespace() == source->MeshNamespace());
    CHECK(result.state->IdHighWater() == source->IdHighWater());
    CHECK(result.state->RelationKeys() == source->RelationKeys());
    for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
    {
        CHECK(source->Entities(kind).SameDomain(result.state->Entities(kind)));
        CHECK_FALSE(source->Entities(kind).SameOrder(result.state->Entities(kind)));
    }
    auto after = result.state->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = after.HostView();
    CHECK(result.state->OriginalEntityId(G::EntityKind::Cell, 0) == oldCell);
    const auto nodes = result.state->Entities(G::EntityKind::Node);
    double stencil = 0;
    REQUIRE(view.CellNodes(0).size() == 3);
    for (size_t slot = 0; slot < 3; ++slot)
    {
        const auto local = view.CellNodes(0)[slot];
        CHECK(result.state->OriginalEntityId(G::EntityKind::Node, local) == 3 * oldNodeOwner + I(slot));
        CHECK(view.Coord(local)[0] == double(3 * oldNodeOwner + I(slot)));
        stencil += view.Coord(local)[0];
    }
    CHECK(stencil == expected);
    for (const auto &key : result.state->RelationKeys())
    {
        auto read = result.state->Relation(key);
        for (I row = 0; row < read.Rows(); ++row)
            if (key.order == G::RelationOrder::SortedUnique)
            {
                const auto values = read.GlobalRow(row);
                CHECK(std::is_sorted(values.begin(), values.end()));
                CHECK(std::adjacent_find(values.begin(), values.end()) == values.end());
            }
    }
    // Source remains a usable independent native publication.
    CHECK(source->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
    CHECK(original.CellNodes(0)[0] >= 0);
    G::ReorderInput invalid;
    invalid.destroyKinds.insert(G::EntityKind::Face);
    CHECK_THROWS(G::NativeMeshTransition::Build(*source, mpi, invalid));
    CHECK(source->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
}

TEST_CASE("Native transition follows complete remote node stars and retains unchanged family order semantics")
{
    const auto mpi = MPI();
    auto input = RotatedTriangles(mpi, true);
    G::NativeMeshBuildOptions options;
    options.completeNodeStar = true;
    auto source = G::NativeMeshBuilder::Build(input, options);
    G::ReorderInput reorder;
    reorder.explicitMaps = {{G::EntityKind::Cell, {int((mpi.rank + 1) % mpi.size)}}};
    auto result = G::NativeMeshTransition::Build(*source, mpi, reorder);
    REQUIRE(result.state);
    CHECK(result.transfers.size() == 3);
    CHECK(result.state->MeshNamespace() == source->MeshNamespace());
    CHECK(result.state->IdHighWater() == source->IdHighWater());
    CHECK(result.state->NumOwned(G::EntityKind::Cell) == 1);
    CHECK(result.state->NumOwned(G::EntityKind::Node) == 3);
    CHECK(result.state->NumOwned(G::EntityKind::Bnd) == 1);
    CHECK(result.state->OriginalEntityId(G::EntityKind::Node, 0) == 3 * mpi.rank);
    CHECK(result.state->OriginalEntityId(G::EntityKind::Bnd, 0) == mpi.rank);
    auto read = result.state->Snapshot(R::Device::Host()).AcquireRead();
    auto view = read.HostView();
    for (size_t slot = 0; slot < 3; ++slot)
        CHECK(result.state->OriginalEntityId(G::EntityKind::Node, view.CellNodes(0)[slot]) == 3 * mpi.rank + I(slot));
}

TEST_CASE("Native transition routes empty owners and rank-skewed payload without global staging")
{
    const auto mpi = MPI();
    const bool owner = mpi.rank == 0;
    auto input = Input(mpi, owner ? 6 : 0, owner ? 2 : 0, 0,
                       owner ? std::vector<std::vector<I>>{{0, 1, 2}, {3, 4, 5}} : std::vector<std::vector<I>>{}, {});
    G::NativeMeshBuildOptions options;
    options.completeNodeStar = true;
    auto source = G::NativeMeshBuilder::Build(input, options);
    G::ReorderInput reorder;
    for (auto kind : {G::EntityKind::Cell, G::EntityKind::Node, G::EntityKind::Bnd})
        reorder.explicitMaps.push_back({kind, std::vector<DNDS::MPI_int>(size_t(source->NumOwned(kind)), mpi.size - 1)});
    auto result = G::NativeMeshTransition::Build(*source, mpi, reorder);
    const bool recipient = mpi.rank == mpi.size - 1;
    CHECK(result.state->NumOwned(G::EntityKind::Cell) == (recipient ? 2 : 0));
    CHECK(result.state->NumOwned(G::EntityKind::Node) == (recipient ? 6 : 0));
    auto read = result.state->Snapshot(R::Device::Host()).AcquireRead();
    auto view = read.HostView();
    if (recipient)
        for (I cell = 0; cell < 2; ++cell)
        {
            CHECK(result.state->OriginalEntityId(G::EntityKind::Cell, cell) == cell);
            for (size_t slot = 0; slot < 3; ++slot)
                CHECK(result.state->OriginalEntityId(G::EntityKind::Node, view.CellNodes(cell)[slot]) == 3 * cell + I(slot));
        }
}

TEST_CASE("Native prepared transition stages field histories atomically and binds entity fields only on publication")
{
    const auto mpi = MPI();
    auto input = RotatedTriangles(mpi, true);
    G::NativeMeshBuildOptions options;
    options.completeNodeStar = true;
    auto native = G::NativeMeshBuilder::Build(input, options);
    auto source = G::UnstructuredMesh::FromNative(mpi, native);
    G::MeshTransitionHandle handle(source);
    R::Field<int64_t> history({1, 4, 4}, {52, 7, 1, 0}, "cell");
    {
        auto write = history.Write();
        for (size_t component = 0; component < 4; ++component)
            write.View<int64_t>()[component] = 100 * mpi.rank + int64_t(component);
    }
    R::Array<int64_t> values(R::RowLayout::Dense(1, 1));
    values.Write().View()(0, 0) = 1000 + mpi.rank;
    G::EntityField<int64_t> field(source->Entities(G::EntityKind::Cell), values);
    const auto historyAllocation = history.Values().Identity();
    const auto valueAllocation = values.Identity();
    G::ReorderInput reorder;
    reorder.explicitMaps = {{G::EntityKind::Cell, {int((mpi.rank + 1) % mpi.size)}}};
    for (const auto *failure : {"cloned", "reordered", "rebuilt"})
    {
        CHECK_THROWS(G::PreparedMeshTransition::Prepare(handle, reorder,
                                                        {G::RegisterField(history, G::EntityKind::Cell, "history"), field.Register("bound")},
                                                        [&](const char *stage)
                                                        {
                                                            if (mpi.rank == 0 && std::string(stage) == failure)
                                                                throw std::runtime_error("injected native transition failure");
                                                        }));
        CHECK(handle.Get() == source);
        CHECK(history.Values().Identity() == historyAllocation);
        CHECK(field.UntypedValues(source->Entities(G::EntityKind::Cell)).Identity() == valueAllocation);
        CHECK(history.Read().View<int64_t>()[3] == 100 * mpi.rank + 3);
        CHECK(source->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
    }
    auto prepared = G::PreparedMeshTransition::Prepare(handle, reorder,
                                                       {G::RegisterField(history, G::EntityKind::Cell, "history"), field.Register("bound")});
    CHECK(handle.Get() == source);
    CHECK(history.Values().Identity() == historyAllocation);
    prepared.Publish();
    auto next = handle.Get();
    REQUIRE(next != source);
    CHECK(next->HasNativeState());
    CHECK_FALSE(next->coords.father);
    CHECK_FALSE(next->cell2node.father);
    CHECK_FALSE(next->node2cell.father);
    const I previous = (mpi.rank + mpi.size - 1) % mpi.size;
    CHECK(next->OriginalEntityId(G::EntityKind::Cell, 0) == previous);
    {
        auto read = history.Read();
        for (size_t component = 0; component < 4; ++component)
            CHECK(read.View<int64_t>()[component] == 100 * previous + I(component));
    }
    CHECK(field.Read(next->Entities(G::EntityKind::Cell)).View()(0, 0) == 1000 + previous);
    CHECK_THROWS(field.Validate(source->Entities(G::EntityKind::Cell)));
    CHECK(values.Read().View()(0, 0) == 1000 + mpi.rank);
    CHECK(source->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
    CHECK_THROWS(prepared.Publish());
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
