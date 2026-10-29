#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeMeshTransition.hpp"
#include "Geom/Mesh/NativeMeshAssembler.hpp"
#include "Geom/Mesh/RelationAccess.hpp"
#include "Geom/Mesh/MeshTransition.hpp"
#include "Geom/Mesh/EntityField.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "NativeMeshFacadeFixture.hpp"
#include "NativePeriodicFixtures.hpp"
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
    auto source = G::ConstructNativeMesh(input, options);
    auto captured = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    const auto identity = captured.identity;
    const auto keys = source->RelationKeys();
    for (const auto &[kind, space] : captured.spaces)
        CHECK(space.NumGhost() == 0);
    CHECK(captured.ids.at(G::EntityKind::Cell).RowCount() == 1);
    auto coordinates = *captured.coords;
    const auto nodes = captured.relations[size_t(G::MeshRelation::CellNodes)]->Read();
    CHECK(nodes.View()(0, 0) == 3 * ((mpi.rank + 1) % mpi.size));
    source.reset();
    CHECK_THROWS(coordinates.Write());
    auto rebuilt = G::AssembleNativeMesh(mpi, std::move(captured), 2);
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
    auto source = G::ConstructNativeMesh(input);
    auto captured = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    SUBCASE("persistent ID extra columns")
    {
        if (mpi.rank == 0)
            captured.ids.at(G::EntityKind::Cell) = R::Array<I>(R::RowLayout::Dense(1, 2));
        CHECK_THROWS(G::AssembleNativeMesh(mpi, std::move(captured)));
    }
    SUBCASE("different closure depth")
    {
        if (mpi.size == 1)
            return;
        CHECK_THROWS(G::AssembleNativeMesh(mpi, std::move(captured), mpi.rank == 0 ? 2 : 1));
    }
    SUBCASE("different explicit connected recipe")
    {
        if (mpi.size == 1)
            return;
        auto recipe = G::DefaultNativeMeshGhostSpec(captured);
        if (mpi.rank == 0)
            recipe.chains.push_back({G::EntityKind::Bnd,
                                     {G::Adj::Bnd2Node, G::Adj::Node2Cell, G::Adj::Cell2Node, G::Adj::Node2Bnd},
                                     G::EntityKind::Bnd});
        CHECK_THROWS(G::NativeMeshAssembler::Build(mpi, std::move(captured), recipe));
    }
    CHECK(source->OriginalEntityId(G::EntityKind::Cell, 0) == mpi.rank);
}

TEST_CASE("Owned native capture collectively rejects mismatched source inventory before ghost-map collectives")
{
    const auto mpi = MPI();
    if (mpi.size == 1)
        return;
    auto first = G::ConstructNativeMesh(RotatedTriangles(mpi, false));
    auto second = G::ConstructNativeMesh(RotatedTriangles(mpi, true));
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
    auto source = G::ConstructNativeMesh(input, options);
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
        for (I row = 0; row < read.RowCount(); ++row)
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
    auto source = G::ConstructNativeMesh(input, options);
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
    auto source = G::ConstructNativeMesh(input, options);
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
    auto native = G::ConstructNativeMesh(input, options);
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

TEST_CASE("Native transition distinguishes exact aliases from rank-local candidate shape state and order differences")
{
    const auto mpi = MPI();
    auto input = NativeFacadeFixture::Make(mpi);
    auto key = [](const std::string &name, G::EntityKind target = G::EntityKind::Node,
                  G::RelationOrder order = G::RelationOrder::Topological)
    {
        G::AdjacencyDescriptor result(G::EntityKind::Cell, target);
        result.path = {G::EntityKind::Cell, target};
        result.selection = G::RelationSelection::Custom;
        result.order = order;
        result.periodic = G::RelationPeriodicPolicy::EntityIdentity;
        result.semanticName = "transition.alias." + name;
        result.semanticVersion = 1;
        return result;
    };
    const auto a = key("a"), b = key("b"), c = key("c"), exact = key("d"), materialized = key("e"), view = key("f");
    const auto globalState = key("global"), localState = key("local"), cellTarget = key("cell", G::EntityKind::Cell);
    const auto ordered = key("ordered"), sorted = key("sorted", G::EntityKind::Node, G::RelationOrder::SortedUnique);
    const auto unavailable = key("unavailable");
    const I start = 3 * mpi.rank;
    const auto first = NativeFacadeFixture::Rows<I>({{start + 1, DNDS::UnInitIndex, start, start + 1}});
    const auto second = NativeFacadeFixture::Rows<I>({{start + 2, DNDS::UnInitIndex, start, start + 2}});
    auto add = [&](const G::AdjacencyDescriptor &descriptor, R::ArrayPair<I> rows, G::MeshAdjState state = G::Adj_PointToGlobal)
    { input.data.namedRelations.emplace(descriptor, G::NativeNamedRelation{std::move(rows), state}); };
    add(a, first);
    add(b, second); // Same row sizes, different values: never a payload alias.
    add(c, mpi.rank % 2 == 0 ? first : second);
    add(exact, first);
    add(materialized, mpi.rank == 0 ? NativeFacadeFixture::Rows<I>({{start + 1, DNDS::UnInitIndex, start, start + 1}}) : first);
    // Same allocation, different complete shape record and logical row length.
    add(view, {R::Array<I>::FromStorage(R::RowLayout::Dense(1, 3, 4), first.owned.Storage()), first.ghost});
    const auto zero = NativeFacadeFixture::Rows<I>({{0, DNDS::UnInitIndex, 0, 0}});
    add(globalState, zero);
    add(localState, zero, G::Adj_PointToLocal);
    add(cellTarget, zero, G::Adj_PointToLocal);
    const auto ascending = NativeFacadeFixture::Rows<I>({{start, start + 1}});
    add(ordered, ascending);
    add(sorted, ascending);
    const I remote = 3 * ((mpi.rank + std::max(1, mpi.size / 2)) % mpi.size) + 2;
    add(unavailable, NativeFacadeFixture::Rows<I>({{remote, DNDS::UnInitIndex}}));
    auto source = G::UnstructuredMeshState::FromNative(mpi, std::move(input.data), input.spaces, input.ids,
                                                       NativeFacadeFixture::Requirements());
    if (mpi.size > 1)
        CHECK(source->Relation(unavailable).ReadEntry(0, 0).unavailable);
    G::ReorderInput placement;
    placement.explicitMaps = {{G::EntityKind::Cell, {(mpi.rank + 1) % mpi.size}},
                              {G::EntityKind::Node, {(mpi.rank + 1) % mpi.size, mpi.rank, (mpi.rank + 1) % mpi.size}}};
    auto moved = G::NativeMeshTransition::Build(*source, mpi, placement).state;
    REQUIRE(moved);
    const I oldRank = (mpi.rank + mpi.size - 1) % mpi.size;
    // Independent oracle: gather persistent owner IDs in the destination's
    // current global order, then compare each named row to its source formula.
    auto globalIds = [&](G::EntityKind kind)
    {
        const int count = int(moved->NumOwned(kind));
        std::vector<int> counts(size_t(mpi.size)), offsets(size_t(mpi.size));
        MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
        for (int rank = 1; rank < mpi.size; ++rank)
            offsets[size_t(rank)] = offsets[size_t(rank - 1)] + counts[size_t(rank - 1)];
        std::vector<I> local(size_t(count), I(0)), result(size_t(offsets.back() + counts.back()));
        for (int row = 0; row < count; ++row)
            local[size_t(row)] = moved->OriginalEntityId(kind, row);
        MPI_Allgatherv(local.data(), count, MPI_INT64_T, result.data(), counts.data(), offsets.data(), MPI_INT64_T, mpi.comm);
        return result;
    };
    const auto nodeIds = globalIds(G::EntityKind::Node), cellIds = globalIds(G::EntityKind::Cell);
    auto check = [&](const G::AdjacencyDescriptor &descriptor, const std::vector<I> &oldGlobals)
    {
        auto read = moved->Relation(descriptor);
        const auto &ids = descriptor.to == G::EntityKind::Node ? nodeIds : cellIds;
        std::vector<I> expected;
        for (auto global : oldGlobals)
        {
            if (global == DNDS::UnInitIndex)
            {
                expected.push_back(global);
                continue;
            }
            const I persistent = 10000 + 17 * global + I(descriptor.to);
            const auto found = std::find(ids.begin(), ids.end(), persistent);
            REQUIRE(found != ids.end());
            expected.push_back(I(found - ids.begin()));
        }
        if (descriptor.order == G::RelationOrder::SortedUnique)
            std::sort(expected.begin(), expected.end());
        CHECK(read.GlobalRow(0) == expected);
        for (size_t slot = 0; slot < expected.size(); ++slot)
        {
            const auto entry = read.ReadEntry(0, slot);
            CHECK(entry.absent == (expected[slot] == DNDS::UnInitIndex));
            if (!entry.absent)
            {
                const auto local = read.Target().GlobalToLocal(expected[slot]);
                CHECK(entry.unavailable == !local.has_value());
                CHECK(read.Raw(0, slot) == (local ? *local : -1 - expected[slot]));
            }
        }
    };
    const std::vector<I> expectedA{3 * oldRank + 1, DNDS::UnInitIndex, 3 * oldRank, 3 * oldRank + 1};
    const std::vector<I> expectedB{3 * oldRank + 2, DNDS::UnInitIndex, 3 * oldRank, 3 * oldRank + 2};
    check(a, expectedA);
    check(b, expectedB);
    // Every rank has a local cache hit for c, but those hits identify different
    // earlier collectives. Reusing either output would corrupt remote rows.
    check(c, oldRank % 2 == 0 ? expectedA : expectedB);
    check(exact, expectedA);
    check(materialized, expectedA);
    check(view, std::vector<I>(expectedA.begin(), expectedA.begin() + 3));
    check(globalState, {0, DNDS::UnInitIndex, 0, 0});
    check(localState, {3 * oldRank, DNDS::UnInitIndex, 3 * oldRank, 3 * oldRank});
    check(cellTarget, {oldRank, DNDS::UnInitIndex, oldRank, oldRank});
    check(ordered, {3 * oldRank, 3 * oldRank + 1});
    check(sorted, {3 * oldRank, 3 * oldRank + 1});
    check(unavailable, {3 * ((oldRank + std::max(1, mpi.size / 2)) % mpi.size) + 2, DNDS::UnInitIndex});
    CHECK(source->Relation(a).GlobalRow(0) == std::vector<I>{start + 1, DNDS::UnInitIndex, start, start + 1});
}

TEST_CASE("Native aliased transition retains repeated periodic IDs and ordered frames")
{
    const auto mpi = MPI();
    auto source = G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi));
    auto owned = G::NativeMeshAssembler::CaptureOwned(*source, mpi);
    G::AdjacencyDescriptor ordered(G::EntityKind::Cell, G::EntityKind::Node);
    ordered.path = {G::EntityKind::Cell, G::EntityKind::Node};
    ordered.selection = G::RelationSelection::Custom;
    ordered.order = G::RelationOrder::Topological;
    ordered.periodic = G::RelationPeriodicPolicy::FrameMatched;
    ordered.semanticName = "transition.periodic.ordered";
    ordered.semanticVersion = 1;
    owned.namedRelations.emplace(ordered, *owned.relations[size_t(G::MeshRelation::CellNodes)]);
    source = G::AssembleNativeMesh(mpi, std::move(owned));
    G::ReorderInput placement;
    placement.explicitMaps.push_back({G::EntityKind::Cell, std::vector<DNDS::MPI_int>(size_t(source->NumOwned(G::EntityKind::Cell)), (mpi.rank + 1) % mpi.size)});
    auto moved = G::NativeMeshTransition::Build(*source, mpi, placement).state;
    auto read = moved->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    const I expectedNodes[]{0, 0, 0, 0, 1, 2, 1, 2, 3};
    const uint32_t expectedBits[]{0, 1, 3, 2, 0, 1, 2, 0, 0};
    auto relation = moved->Relation(ordered);
    for (I cell = 0; cell < moved->NumOwned(G::EntityKind::Cell); ++cell)
    {
        REQUIRE(view.CellNodes(cell).size() == 9);
        REQUIRE(relation.RowSize(cell) == 9);
        for (size_t slot = 0; slot < 9; ++slot)
        {
            const auto node = view.CellNodes(cell)[slot];
            CHECK(moved->OriginalEntityId(G::EntityKind::Node, node) == expectedNodes[slot]);
            CHECK(uint8_t(view.CellNodePeriodicBits(cell)[slot]) == expectedBits[slot]);
            CHECK(relation.ReadEntry(cell, slot).global == moved->Entities(G::EntityKind::Node).LocalToGlobal(node));
        }
    }
    CHECK(moved->MeshNamespace() == source->MeshNamespace());
    CHECK(moved->IdHighWater() == source->IdHighWater());
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
