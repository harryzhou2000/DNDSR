#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include "Geom/Transition/NativeMeshAssembler.hpp"
#include "Geom/Connectivity/NativeMeshTopology.hpp"
#include "Geom/Transition/NativeMeshTransition.hpp"
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
    template <class T>
    R::Array<T> VariableUniform(const R::Array<T> &source, size_t width, bool padded)
    {
        std::vector<size_t> lengths(source.Rows(), width);
        R::Array<T> result(padded ? R::RowLayout::Padded(lengths, width + 2) : R::RowLayout::Ragged(lengths));
        auto read = source.Read();
        auto write = result.Write();
        for (size_t row = 0; row < source.Rows(); ++row)
            for (size_t column = 0; column < width; ++column)
                write.View()(row, column) = read.View()(row, column);
        return result;
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

TEST_CASE("Owned primary redistribution and face construction finalize once with the supplied boundary ring")
{
    const auto mpi = MPI();
    std::vector<std::vector<I>> boundaries;
    if (mpi.size == 1)
        boundaries = {{0, 1}, {1, 2}, {2, 0}};
    else
    {
        if (mpi.rank == 0)
            boundaries.push_back({0, 1});
        if (mpi.rank == 1)
            boundaries.push_back({1, 2});
        if (mpi.rank == mpi.size - 1)
            boundaries.push_back({2, 0});
    }
    // All nodes belong to rank zero. Other boundary owners therefore need
    // remote Node2Bnd rows: an owned-Node root cannot cover their boundary ring.
    auto input = Input(mpi, mpi.rank == 0 ? 3 : 0, mpi.rank == 0 ? 1 : 0, I(boundaries.size()),
                       mpi.rank == 0 ? std::vector<std::vector<I>>{{0, 1, 2}} : std::vector<std::vector<I>>{}, boundaries);
    auto owned = G::NativeMeshBuilder::Build(input);
    for (const auto &[kind, space] : owned.spaces)
        CHECK(space.NumGhost() == 0);
    CHECK_FALSE(owned.retainedRead);
    // Deliberately omit owned-Node roots and indirect Cell→Node→Bnd paths.
    // Remote boundary owners can only obtain the other edges through the
    // explicit connected Bnd→Node→Bnd path, never by inventing another root.
    G::GhostSpec recipe;
    auto append = [&](G::EntityKind anchor, std::vector<G::AdjKind> hops)
    { recipe.chains.push_back({anchor, hops, hops.back().to}); };
    for (const auto key : {G::Adj::Cell2Node, G::Adj::Bnd2Node, G::Adj::Bnd2Cell,
                           G::Adj::Cell2Cell, G::Adj::Cell2Face, G::Adj::Face2Node,
                           G::Adj::Face2Cell, G::Adj::Face2Bnd, G::Adj::Bnd2Face})
        append(key.from, {key});
    append(G::EntityKind::Bnd, {G::Adj::Bnd2Node, G::Adj::Node2Bnd});
    append(G::EntityKind::Bnd, {G::Adj::Bnd2Node, G::Adj::Node2Bnd, G::Adj::Bnd2Node});
    owned.ghostSpec = recipe;
    G::ReorderInput placement;
    std::vector<DNDS::MPI_int> nodeRanks(size_t(input.nodes.NumOwned()));
    const int targets[3]{mpi.size - 1, 0, mpi.size > 1 ? 1 : 0};
    for (size_t row = 0; row < nodeRanks.size(); ++row)
        nodeRanks[row] = targets[row];
    placement.explicitMaps.push_back({G::EntityKind::Node, nodeRanks});
    auto redistributed = G::NativeMeshTransition::BuildOwned(owned, mpi, placement);
    owned = std::move(redistributed.owned);
    REQUIRE(owned.ghostSpec);
    CHECK(owned.ghostSpec->chains.size() == recipe.chains.size());
    for (const auto &[kind, space] : owned.spaces)
        CHECK(space.NumGhost() == 0);
    const bool hasOwnedNodes = owned.spaces.at(G::EntityKind::Node).NumOwned() != 0;
    owned = G::NativeMeshTopology::Build(std::move(owned), mpi, true, false);
    for (const auto &[kind, space] : owned.spaces)
        CHECK(space.NumGhost() == 0);
    CHECK_FALSE(owned.retainedRead);
    REQUIRE(owned.spaces.count(G::EntityKind::Face));
    CHECK(owned.spaces.at(G::EntityKind::Face).NumGlobal() == 3);
    auto state = G::NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    REQUIRE(state->GhostSpecification());
    CHECK(state->GhostSpecification()->chains.size() == recipe.chains.size());
    const auto space = state->Entities(G::EntityKind::Bnd);
    std::set<I> actual;
    for (I row = space.NumOwned(); row < space.NumOwned() + space.NumGhost(); ++row)
        actual.insert(space.LocalToGlobal(row));
    // Independent triangle oracle: every pair of distinct edges shares one
    // node. Only ranks with an owned boundary or an owned node need the star.
    std::set<I> expected;
    if (hasOwnedNodes || !boundaries.empty())
        for (I global = 0; global < 3; ++global)
            if (space.Owner(global) != mpi.rank)
                expected.insert(global);
    CHECK(actual == expected);
    const auto &view = state->BorrowHostView();
    for (I boundary = 0; boundary < space.NumOwned(); ++boundary)
        for (I node : view.BoundaryNodes(boundary))
        {
            const auto neighbors = Globals(space, view.NodeBoundaries(node));
            CHECK(neighbors.size() == 2);
            CHECK(neighbors.count(space.LocalToGlobal(boundary)) == 1);
        }
}

TEST_CASE("Native primary builder closes remote owned-node stars explicitly")
{
    const auto mpi = MPI();
    const I previous = (mpi.rank + mpi.size - 1) % mpi.size;
    for (bool complete : {false, true})
    {
        auto input = RotatedTriangles(mpi, false);
        auto coordinates = input.coordinates;
        auto metadata = input.cellInfo;
        G::NativeMeshBuildOptions options;
        options.completeNodeStar = complete;
        auto state = G::ConstructNativeMesh(input, options);
        auto read = state->Snapshot(R::Device::Host()).AcquireRead();
        const auto view = read.HostView();
        CHECK(state->NumOwned(G::EntityKind::Cell) == 1);
        CHECK(state->NumGhost(G::EntityKind::Cell) == (complete && mpi.size > 1 ? 1 : 0));
        CHECK(state->NumGhost(G::EntityKind::Node) == (mpi.size > 1 ? 3 : 0));
        CHECK(view.CellCells(0).empty());
        CHECK(view.GetCellInfo(0).getElemType() == G::Elem::Tri3);
        CHECK_THROWS(coordinates.Write());
        CHECK_THROWS(metadata.Write());
        const auto cells = state->Entities(G::EntityKind::Cell);
        const auto nodes = state->Entities(G::EntityKind::Node);
        const I next = (mpi.rank + 1) % mpi.size;
        CHECK(Globals(nodes, view.CellNodes(0)) == std::set<I>{3 * next, 3 * next + 1, 3 * next + 2});
        for (I row = 0; row < nodes.NumLocal(); ++row)
        {
            CHECK(view.Coord(row)[0] == DNDS::real(nodes.LocalToGlobal(row)));
            CHECK(state->OriginalEntityId(G::EntityKind::Node, row) == nodes.LocalToGlobal(row));
        }
        for (I node = 0; node < 3; ++node)
        {
            REQUIRE(view.NodeCells(node).size() == 1);
            if (complete || mpi.size == 1)
                CHECK(Globals(cells, view.NodeCells(node)) == std::set<I>{previous});
            else
                CHECK(view.NodeCells(node)[0] == -1 - previous);
        }
    }
}

TEST_CASE("Native primary builder matches all boundary nodes and transfers independent companions")
{
    const auto mpi = MPI();
    auto input = RotatedTriangles(mpi, true);
    for (const auto *space : {&input.cells, &input.nodes, &input.boundaries})
    {
        R::Array<I> ids(R::RowLayout::Dense(size_t(space->NumOwned()), 1));
        auto write = ids.Write();
        for (I row = 0; row < space->NumOwned(); ++row)
            write.View()(size_t(row), 0) = 1000000000000LL + 17 * space->LocalToGlobal(row) + int(space->Kind());
        input.ids.emplace(space->Kind(), std::move(ids));
    }
    auto owned = G::NativeMeshBuilder::Build(input);
    auto recipe = G::DefaultNativeMeshGhostSpec(owned, 1, false);
    // This companion-transfer fixture deliberately requests a remote boundary
    // incident to an owned cell's nodes, beyond the default owned-boundary ring.
    recipe.chains.push_back({G::EntityKind::Cell, {G::Adj::Cell2Node, G::Adj::Node2Bnd}, G::EntityKind::Bnd});
    auto state = G::NativeMeshAssembler::Build(mpi, std::move(owned), recipe);
    const auto cells = state->Entities(G::EntityKind::Cell);
    const auto boundaries = state->Entities(G::EntityKind::Bnd);
    auto read = state->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    CHECK(cells.NumGhost() == (mpi.size > 1 ? 1 : 0));
    CHECK(boundaries.NumGhost() == (mpi.size > 1 ? 1 : 0));
    REQUIRE(view.BoundaryCells(0).size() == 2);
    CHECK(cells.LocalToGlobal(view.BoundaryCells(0)[0]) == (mpi.rank + mpi.size - 1) % mpi.size);
    CHECK(view.BoundaryCells(0)[1] == DNDS::UnInitIndex);
    for (I row = 0; row < boundaries.NumLocal(); ++row)
    {
        const auto global = boundaries.LocalToGlobal(row);
        CHECK(view.GetBndInfo(row).zone == G::BC_ID_DEFAULT_WALL + global);
        CHECK(state->OriginalEntityId(G::EntityKind::Bnd, row) == 1000000000000LL + 17 * global + int(G::EntityKind::Bnd));
    }
    for (I row = 0; row < cells.NumLocal(); ++row)
        CHECK(state->OriginalEntityId(G::EntityKind::Cell, row) == 1000000000000LL + 17 * cells.LocalToGlobal(row) + int(G::EntityKind::Cell));
}

TEST_CASE("Native primary builder retains every cell layer and its node closure")
{
    const auto mpi = MPI();
    const I next = (mpi.rank + 1) % mpi.size;
    const auto triangle = mpi.size == 1 ? std::vector<I>{0, 1, 2} : std::vector<I>{3 * mpi.rank, 3 * next, 3 * mpi.rank + 1};
    for (int layers : {1, 2})
    {
        auto input = Input(mpi, 3, 1, 0, {triangle}, {});
        G::NativeMeshBuildOptions options;
        options.cellLayers = layers;
        auto state = G::ConstructNativeMesh(input, options);
        const auto cells = state->Entities(G::EntityKind::Cell);
        const auto nodes = state->Entities(G::EntityKind::Node);
        auto read = state->Snapshot(R::Device::Host()).AcquireRead();
        const auto view = read.HostView();
        std::set<I> expected;
        for (int distance = -layers; distance <= layers; ++distance)
        {
            const I global = (mpi.rank + distance + mpi.size * (layers + 1)) % mpi.size;
            if (global != mpi.rank)
                expected.insert(global);
        }
        CHECK(cells.NumGhost() == I(expected.size()));
        for (I row = cells.NumOwned(); row < cells.NumLocal(); ++row)
            CHECK(expected.count(cells.LocalToGlobal(row)) == 1);
        std::set<I> neighbors;
        if (mpi.size > 1)
        {
            neighbors.insert((mpi.rank + mpi.size - 1) % mpi.size);
            neighbors.insert(next);
        }
        CHECK(Globals(cells, view.CellCells(0)) == neighbors);
        for (I row = 0; row < cells.NumLocal(); ++row)
        {
            const I global = cells.LocalToGlobal(row), following = (global + 1) % mpi.size;
            const auto expectedNodes = mpi.size == 1 ? std::set<I>{0, 1, 2} : std::set<I>{3 * global, 3 * following, 3 * global + 1};
            CHECK(Globals(nodes, view.CellNodes(row)) == expectedNodes);
        }
    }
}

TEST_CASE("Native primary builder normalizes uniform CSR and padded companions without copying values")
{
    const auto mpi = MPI();
    for (bool padded : {false, true})
    {
        auto input = RotatedTriangles(mpi, true);
        input.coordinates = VariableUniform(input.coordinates, 3, padded);
        input.cellInfo = VariableUniform(input.cellInfo, 1, padded);
        input.boundaryInfo = VariableUniform(input.boundaryInfo, 1, padded);
        auto coordinates = input.coordinates;
        auto metadata = input.cellInfo;
        const auto coordinatePointer = coordinates.Storage().LegacyPointer();
        const auto infoPointer = metadata.Storage().LegacyPointer();
        auto state = G::ConstructNativeMesh(input);
        auto read = state->Snapshot(R::Device::Host()).AcquireRead();
        const auto view = read.HostView();
        CHECK(view.Coord(0).data() == coordinatePointer);
        CHECK(static_cast<const void *>(&view.GetCellInfo(0)) == infoPointer);
        CHECK(view.Coord(0)[0] == DNDS::real(3 * mpi.rank));
        CHECK(view.GetCellInfo(0).getElemType() == G::Elem::Tri3);
        CHECK(view.GetBndInfo(0).zone == G::BC_ID_DEFAULT_WALL + mpi.rank);
        CHECK(coordinates.Layout().IsVariable());
        CHECK(metadata.Layout().IsVariable());
        CHECK_THROWS(coordinates.Write());
        CHECK_THROWS(metadata.Write());
    }
}

TEST_CASE("Native primary builder rejects a boundary diagonal collectively then accepts a corrected face")
{
    const auto mpi = MPI();
    const I offset = 4 * mpi.rank;
    auto input = Input(mpi, 4, 1, 1, {{offset, offset + 1, offset + 2, offset + 3}},
                       {{offset, offset + (mpi.rank == 0 ? 2 : 1)}});
    {
        auto info = input.cellInfo.Write();
        info.View()(0, 0) = G::ElemInfo{G::Elem::Quad4, G::INTERNAL_ZONE};
    }
    // Every boundary is contained in exactly one cell. Only rank zero's row
    // is not an actual element edge, so containment alone cannot reject it.
    CHECK_THROWS(G::ConstructNativeMesh(input));
    {
        auto boundary = input.boundaryNodes.Write();
        if (mpi.rank == 0)
            boundary.View()(0, 1) = offset + 1;
    }
    auto state = G::ConstructNativeMesh(input);
    auto read = state->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = read.HostView();
    CHECK(view.GetCellElement(0).type == G::Elem::Quad4);
    CHECK(view.BoundaryCells(0)[0] == 0);
    CHECK(view.BoundaryCells(0)[1] == DNDS::UnInitIndex);
    CHECK(Globals(state->Entities(G::EntityKind::Node), view.BoundaryNodes(0)) == std::set<I>{offset, offset + 1});
    CHECK(input.cells.Valid());
    CHECK(input.nodes.Valid());
    CHECK(input.boundaries.Valid());
}

TEST_CASE("Native primary builder includes empty ranks and agrees on invalid input")
{
    const auto mpi = MPI();
    const bool owner = mpi.rank == 0;
    auto input = Input(mpi, owner ? 3 : 0, owner ? 1 : 0, 0,
                       owner ? std::vector<std::vector<I>>{{0, 1, 2}} : std::vector<std::vector<I>>{}, {});
    SUBCASE("empty participants")
    {
        G::NativeMeshBuildOptions options;
        options.completeNodeStar = true;
        auto state = G::ConstructNativeMesh(input, options);
        CHECK(state->NumOwned(G::EntityKind::Cell) == (owner ? 1 : 0));
        CHECK(state->NumGlobal(G::EntityKind::Cell) == 1);
        CHECK(state->NumGhost(G::EntityKind::Cell) == 0);
    }
    SUBCASE("one rank contains invalid node")
    {
        if (owner)
        {
            auto write = input.cellNodes.Write();
            write.View()[0][2] = 3;
        }
        CHECK_THROWS(G::ConstructNativeMesh(input));
    }
    SUBCASE("one rank contains a nonuniform coordinate width")
    {
        if (owner)
            input.coordinates = R::Array<DNDS::real>(R::RowLayout::Ragged({3, 2, 3}));
        CHECK_THROWS(G::ConstructNativeMesh(input));
    }
    SUBCASE("one rank requests unsupported periodic construction")
    {
        input.periodic = owner;
        CHECK_THROWS(G::ConstructNativeMesh(input));
    }
    SUBCASE("one rank has active coordinate writer")
    {
        std::optional<R::Array<DNDS::real>::WriteLease> writer;
        if (owner)
            writer.emplace(input.coordinates.Write());
        CHECK_THROWS(G::ConstructNativeMesh(input));
    }
    SUBCASE("collective options mismatch")
    {
        if (mpi.size > 1)
        {
            G::NativeMeshBuildOptions options;
            options.completeNodeStar = owner;
            CHECK_THROWS(G::ConstructNativeMesh(input, options));
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
