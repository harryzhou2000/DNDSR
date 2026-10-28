#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/NativeMeshBuilder.hpp"
#include "Geom/Mesh/NativeConnectivity.hpp"
#include "Geom/Mesh/NativePeriodicMatch.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <set>
#include <cmath>
#include <cstddef>
#include <iostream>
#include "AllocationRecorder.hpp"
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static R::Array<I> Rows(const std::vector<std::vector<I>> &rows)
{
    R::RowBuilder builder(rows.size(), sizeof(I));
    for (size_t i = 0; i < rows.size(); ++i)
    {
        builder.ResizeRow(i, rows[i].size());
        auto target = builder.Row<I>(i);
        std::copy(rows[i].begin(), rows[i].end(), target.data);
    }
    return R::Array<I>::FromSharedStorage(builder.Commit());
}
static G::NativeMeshInput Fan(const DNDS::MPIInfo &mpi, int extra = 0)
{
    // Central node belongs to rank0, which has no cells with np>1.
    // Eight disjoint wedges incident at that node; two remote owners when np>=4.
    // All remaining ranks are empty cell owners. Remote extras never touch fan.
    const int owners = mpi.size == 1 ? 1 : std::min(2, mpi.size - 1);
    auto owner = [&](int cell)
    { return mpi.size == 1 ? 0 : 1 + cell % owners; };
    std::vector<int> local;
    for (int c = 0; c < 8; ++c)
        if (owner(c) == mpi.rank)
            local.push_back(c);
    const int extras = mpi.rank == mpi.size - 1 ? extra : 0;
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, (mpi.rank == 0 ? 17 : 0) + 3 * extras);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(local.size() + extras));
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, 0);
    std::vector<std::vector<I>> cones;
    for (auto c : local)
        cones.push_back({0, 1 + 2 * c, 2 + 2 * c});
    for (int c = 0; c < extras; ++c)
    {
        auto first = nodes.LocalToGlobal((mpi.rank == 0 ? 17 : 0) + 3 * c);
        cones.push_back({first, first + 1, first + 2});
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(cones), Rows({}), R::Array<DNDS::real>(R::RowLayout::Dense(size_t(nodes.NumOwned()), 3)), R::Array<G::ElemInfo>(R::RowLayout::Dense(size_t(cells.NumOwned()), 1)), R::Array<G::ElemInfo>(R::RowLayout::Dense(0, 1)), {}};
    {
        auto w = input.coordinates.Write();
        for (I r = 0; r < nodes.NumOwned(); ++r)
        {
            auto g = nodes.LocalToGlobal(r);
            w.View()(size_t(r), 0) = double(g);
            w.View()(size_t(r), 1) = g % 2;
            w.View()(size_t(r), 2) = 0;
        }
    }
    {
        auto w = input.cellInfo.Write();
        for (size_t r = 0; r < input.cellInfo.Rows(); ++r)
            w.View()(r, 0) = G::ElemInfo{G::Elem::Tri3, G::INTERNAL_ZONE};
    }
    for (auto kind : {G::EntityKind::Node, G::EntityKind::Cell, G::EntityKind::Bnd})
    {
        const auto &space = kind == G::EntityKind::Node ? nodes : kind == G::EntityKind::Cell ? cells
                                                                                              : boundaries;
        R::Array<I> ids(R::RowLayout::Dense(size_t(space.NumOwned()), 1));
        {
            auto w = ids.Write();
            for (I r = 0; r < space.NumOwned(); ++r)
                w.View()(size_t(r), 0) = 1000000000000LL + 1000000 * I(kind) + space.LocalToGlobal(r);
        }
        input.ids.emplace(kind, std::move(ids));
    }
    return input;
}
TEST_CASE("High valence all-remote owned node has exact closure and SUM gather scatter")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto state = G::ConstructNativeMesh(Fan(mpi), {2, true});
    auto access = state->Snapshot(R::Device::Host()).AcquireRead();
    auto v = access.HostView();
    auto nodes = state->Entities(G::EntityKind::Node), cells = state->Entities(G::EntityKind::Cell);
    if (mpi.rank == 0)
    {
        CHECK(cells.NumOwned() == (mpi.size == 1 ? 8 : 0));
        REQUIRE(v.NodeCells(0).size() == 8);
        std::set<I> all;
        for (auto c : v.NodeCells(0))
        {
            REQUIRE(c >= 0);
            for (auto n : v.CellNodes(c))
            {
                REQUIRE(n >= 0);
                all.insert(nodes.LocalToGlobal(n));
            }
        }
        std::set<I> expected;
        for (I n = 0; n < 17; ++n)
            expected.insert(n);
        CHECK(all == expected);
    }
    // Each cell contributes1 to each incident node. Reverse halo SUM collides
    // remote central-node contributions at rank0; forward halo scatters totals.
    R::Array<DNDS::real> owned(R::RowLayout::Dense(size_t(nodes.NumOwned()), 1)), ghost(R::RowLayout::Dense(size_t(nodes.NumGhost()), 1));
    {
        auto a = owned.Write(), b = ghost.Write();
        for (size_t i = 0; i < owned.Rows(); ++i)
            a.View()(i, 0) = 0;
        for (size_t i = 0; i < ghost.Rows(); ++i)
            b.View()(i, 0) = 0;
        for (I c = 0; c < cells.NumOwned(); ++c)
            for (auto n : v.CellNodes(c))
            {
                if (n < nodes.NumOwned())
                    a.View()(size_t(n), 0) += 1;
                else
                    b.View()(size_t(n - nodes.NumOwned()), 0) += 1;
            }
    }
    auto route = nodes.BuildRowTransferPlan();
    R::PayloadPlan reverse(route.Reverse(), ghost.Layout(), owned.Layout(), R::ElementSchema::Arithmetic<DNDS::real>());
    reverse.Bind(ghost, owned, R::Reducer::Builtin<DNDS::real>(R::Reduction::Sum), R::TransferTransport::Direct, {R::TransferMode::Transient}).Begin().Wait();
    R::PayloadPlan(route, owned.Layout(), ghost.Layout(), R::ElementSchema::Arithmetic<DNDS::real>()).Bind(owned, ghost, R::TransferTransport::Direct, {R::TransferMode::Transient}).Begin().Wait();
    auto a = owned.Read(), b = ghost.Read();
    for (I row = 0; row < nodes.NumLocal(); ++row)
    {
        const auto value = row < nodes.NumOwned() ? a.View()(size_t(row), 0) : b.View()(size_t(row - nodes.NumOwned()), 0);
        CHECK(value == (nodes.LocalToGlobal(row) == 0 ? 8 : 1));
    }
}
TEST_CASE("Node cell node and node edge node have independent Hex8 bridge closure")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    const int host = mpi.size == 1 ? 0 : 1;
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.size == 1 ? 8 : (mpi.rank == 0 ? 1 : (mpi.rank == host ? 7 : 0)));
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, mpi.rank == host ? 1 : 0);
    auto edges = nodes.CreateSibling(G::EntityKind::Edge, mpi.rank == host ? 12 : 0);
    std::vector<std::vector<I>> cones, edgeCones;
    if (mpi.rank == host)
    {
        cones = {{0, 1, 2, 3, 4, 5, 6, 7}};
        edgeCones = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    }
    G::NativeRelation c2n(G::Adj::Cell2Node, cells, nodes, Rows(cones)), e2n(G::Adj::Edge2Node, edges, nodes, Rows(edgeCones));
    auto n2c = G::NativeConnectivity::Inverse(c2n.Read(), mpi), n2e = G::NativeConnectivity::Inverse(e2n.Read(), mpi);
    G::NativeConnectivity registry(mpi);
    registry.Register(c2n.Read());
    registry.Register(e2n.Read());
    registry.Register(n2c.Read());
    registry.Register(n2e.Read());
    for (bool viaEdge : {false, true})
    {
        const auto first = viaEdge ? n2e.Read().Kind() : n2c.Read().Kind();
        const auto second = viaEdge ? G::Adj::Edge2Node : G::Adj::Cell2Node;
        G::GhostSpec spec{{{G::EntityKind::Node, {first, second}, G::EntityKind::Node}}};
        auto discovery = registry.Evaluate(G::CompiledGhostTree::compile(spec));
        std::vector<I> expected;
        if (mpi.size > 1 && mpi.rank == 0)
            expected = viaEdge ? std::vector<I>{1, 3, 4} : std::vector<I>{1, 2, 3, 4, 5, 6, 7};
        if (mpi.size > 1 && mpi.rank == host)
            expected = {0};
        const auto found = discovery.ghostIndices.find(G::EntityKind::Node);
        CHECK((found == discovery.ghostIndices.end() ? std::vector<I>{} : found->second) == expected);
    }
    // Complete only the intermediate support required for exact row composition.
    const auto fullCells = cells.WithGhosts(mpi.rank == 0 && mpi.size > 1 ? std::vector<I>{0} : std::vector<I>{});
    const auto fullEdges = edges.WithGhosts(mpi.rank == 0 && mpi.size > 1 ? std::vector<I>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} : std::vector<I>{});
    G::NativeRelation expandedCells(G::Adj::Cell2Node, fullCells, nodes, c2n.Owned());
    G::NativeRelation expandedEdges(G::Adj::Edge2Node, fullEdges, nodes, e2n.Owned());
    auto cellRows = expandedCells.WithGhostRows(G::RelationGhostRoute(fullCells));
    auto edgeRows = expandedEdges.WithGhostRows(G::RelationGhostRoute(fullEdges));
    // Rebind inverse targets to the expanded intermediate canonical orders.
    G::NativeRelation nc(n2c.Read().Kind(), nodes, fullCells, n2c.Owned());
    G::NativeRelation ne(n2e.Read().Kind(), nodes, fullEdges, n2e.Owned());
    auto throughCell = G::NativeConnectivity::Compose(nc.Read(), cellRows.Read(), true);
    auto throughEdge = G::NativeConnectivity::Compose(ne.Read(), edgeRows.Read(), true);
    if (mpi.rank == 0)
    {
        CHECK(throughCell.Read().GlobalRow(0) == std::vector<I>{1, 2, 3, 4, 5, 6, 7});
        CHECK(throughEdge.Read().GlobalRow(0) == std::vector<I>{1, 3, 4});
        CHECK(throughCell.Read().Kind() != throughEdge.Read().Kind());
    }
}
TEST_CASE("Physical periodic face variants traverse direct any all exact ordered definitions")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    const I next = (mpi.rank + 1) % mpi.size;
    auto cells = G::EntitySpace::CreateOwned(G::EntityKind::Cell, mpi, 4);
    auto nodes = cells.CreateSibling(G::EntityKind::Node, 4), faces = cells.CreateSibling(G::EntityKind::Face, 1);
    std::vector<I> ghosts;
    if (mpi.size > 1)
        for (I j = 0; j < 4; ++j)
            ghosts.push_back(next * 4 + j);
    nodes = nodes.WithGhosts(ghosts);
    G::NativeRelation cn(G::Adj::Cell2Node, cells, nodes, Rows({{mpi.rank * 4, mpi.rank * 4 + 1}, {mpi.rank * 4, mpi.rank * 4 + 2}, {mpi.rank * 4, mpi.rank * 4 + 1, mpi.rank * 4 + 3}, {mpi.rank * 4, mpi.rank * 4 + 1}}));
    G::NativeRelation fn(G::Adj::Face2Node, faces, nodes, Rows({{next * 4 + 1, next * 4}}));
    // The physical oriented face at x=[11,10] uses canonical x=[1,0]
    // plus one translation image; every topology variant shares these slots.
    G::Periodicity frames;
    frames.translation[1].map() = G::tPoint{10, 0, 0};
    R::Array<G::NodePeriodicBits> bits(R::RowLayout::Dense(1, 2));
    {
        auto w = bits.Write();
        w.View()(0, 0) = G::NodePeriodicBits{1};
        w.View()(0, 1) = G::NodePeriodicBits{1};
    }
    G::RelationPeriodicBits physical(fn.Read(), bits);
    {
        auto r = bits.Read();
        for (size_t slot = 0; slot < 2; ++slot)
        {
            const auto global = fn.Read().GlobalRow(0)[slot];
            const G::tPoint canonical{double(global % 4), 0, 0};
            CHECK(frames.GetCoordByBits(canonical, r.View()(0, slot))[0] == (slot == 0 ? 11 : 10));
        }
    }
    auto nc = G::NativeConnectivity::Inverse(cn.Read(), mpi);
    auto support = nc.WithGhostRows(G::RelationGhostRoute(nodes));
    auto any = G::NativeConnectivity::Compose(fn.Read(), support.Read());
    auto all = G::NativeConnectivity::ComposeAll(fn.Read(), support.Read());
    auto key = G::AdjKind::Composition(fn.Read().Kind(), support.Read().Kind(), G::RelationSelection::Exact, false, "physical.periodic.incidence", 1, G::RelationPeriodicPolicy::FrameMatched);
    // This explicit globally indexed fixture definition is the input oracle,
    // not a lookup into the result being tested. Cell3 has the same corners as
    // cell0 but a nonuniform image change, so only the real frame matcher can
    // distinguish it from the accepted complete occurrence pattern.
    auto candidatePattern = [](I cell)
    {
        const I base = (cell / 4) * 4;
        switch (cell % 4)
        {
        case 0:
            return G::NativePeriodicNodes{{base, 0}, {base + 1, 0}};
        case 1:
            return G::NativePeriodicNodes{{base, 0}, {base + 2, 0}};
        case 2:
            return G::NativePeriodicNodes{{base, 0}, {base + 1, 0}, {base + 3, 0}};
        default:
            return G::NativePeriodicNodes{{base, 0}, {base + 1, 1}};
        }
    };
    const G::NativePeriodicNodes facePattern{{next * 4 + 1, 1}, {next * 4, 1}};
    CHECK(G::NativePeriodicFrame(candidatePattern(next * 4), facePattern) == std::optional<uint8_t>{1});
    CHECK_FALSE(G::NativePeriodicFrame(candidatePattern(next * 4 + 2), facePattern));
    CHECK_FALSE(G::NativePeriodicFrame(candidatePattern(next * 4 + 3), facePattern));
    auto exact = G::NativeConnectivity::ComposeFiltered(fn.Read(), support.Read(), key, [](I, I, int count)
                                                        { return count == 2; }, [candidatePattern, facePattern](I, I cell, const std::vector<I> &)
                                                        { return G::NativePeriodicFrame(candidatePattern(cell), facePattern).has_value(); });
    auto directKey = key;
    directKey.semanticName = "physical.periodic.direct";
    directKey.path = {G::EntityKind::Face, G::EntityKind::Cell};
    directKey.order = G::RelationOrder::Topological;
    G::NativeRelation direct(directKey, faces, cells, Rows({{next * 4}}));
    auto orderedKey = all.Read().Kind();
    orderedKey.semanticName = "physical.periodic.ordered";
    orderedKey.order = G::RelationOrder::Topological;
    orderedKey.periodic = G::RelationPeriodicPolicy::FrameMatched;
    G::NativeRelation ordered(orderedKey, faces, cells, Rows({{next * 4 + 3, next * 4 + 2, next * 4}}));
    G::NativeConnectivity registry(mpi);
    std::vector<G::NativeRelation *> variants{&direct, &any, &all, &exact, &ordered};
    const std::vector<std::vector<I>> expected{{next * 4}, {next * 4, next * 4 + 1, next * 4 + 2, next * 4 + 3}, {next * 4, next * 4 + 2, next * 4 + 3}, {next * 4}, {next * 4 + 3, next * 4 + 2, next * 4}};
    for (size_t j = 0; j < variants.size(); ++j)
    {
        auto r = variants[j]->Read();
        CHECK(r.GlobalRow(0) == expected[j]);
        registry.Register(std::move(r));
    }
    for (size_t j = 0; j < variants.size(); ++j)
    {
        const auto k = variants[j]->Read().Kind();
        G::GhostSpec spec{{{G::EntityKind::Face, {k}, G::EntityKind::Cell}}};
        auto result = registry.Evaluate(G::CompiledGhostTree::compile(spec));
        auto canonical = expected[j];
        std::sort(canonical.begin(), canonical.end());
        if (mpi.size > 1)
            CHECK(result.ghostIndices.at(G::EntityKind::Cell) == canonical);
        else
            CHECK(result.ghostIndices.empty());
    }
    auto reverse = G::NativeConnectivity::Inverse(exact.Read(), mpi);
    registry.Register(reverse.Read());
    G::GhostSpec reverseSpec{{{G::EntityKind::Cell, {reverse.Read().Kind()}, G::EntityKind::Face}}};
    auto discovered = registry.Evaluate(G::CompiledGhostTree::compile(reverseSpec));
    if (mpi.size > 1)
        CHECK(discovered.ghostIndices.at(G::EntityKind::Face) == std::vector<I>{(mpi.rank + mpi.size - 1) % mpi.size});
}
TEST_CASE("Unrelated remote components do not grow rank0 traversal C++ scratch")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    {
        AllocationRecorder::Phase phase;
        {
            std::vector<unsigned char> control(1024 * 1024);
            control[0] = 17;
            CHECK(phase.Peak() >= 1024 * 1024);
        }
        CHECK(AllocationRecorder::counters[phase.id].live.load() == 0);
    }
    if (mpi.size == 1)
    {
        MESSAGE("remote-component scaling needs at least two ranks; numerical star still runs at np1");
        return;
    }
    std::vector<size_t> peaks;
    for (int extra : {1, 128, 4096})
    {
        auto state = G::ConstructNativeMesh(Fan(mpi, extra), {2, true});
        G::NativeConnectivity registry(mpi);
        auto source = G::UnstructuredMesh::FromNative(mpi, state);
        auto requirements = G::MeshReadRequirements::Primary();
        requirements.inverse = true;
        registry.RegisterMesh(*source, requirements);
        G::GhostSpec spec{{{G::EntityKind::Node, {G::Adj::Node2Cell, G::Adj::Cell2Node, G::Adj::Node2Cell, G::Adj::Cell2Node}, G::EntityKind::Node}}};
        auto tree = G::CompiledGhostTree::compile(spec);
        {
            auto warm = registry.Evaluate(tree);
            (void)warm;
        }
        size_t peak = 0, calls = 0;
        {
            AllocationRecorder::Phase phase;
            {
                auto result = registry.Evaluate(tree);
                if (mpi.rank == 0)
                {
                    const auto it = result.ghostIndices.find(G::EntityKind::Node);
                    CHECK((it == result.ghostIndices.end() || it->second.empty()));
                }
            }
            peak = phase.Peak();
            calls = phase.Calls();
        }
        if (mpi.rank == 0)
        {
            peaks.push_back(peak);
            std::cout << "remote extras=" << extra << " rank0 traversal C++ requested-byte peak=" << peak << " allocations=" << calls << " mesh-state Runtime bytes=" << state->ArrayBytes() << "; excludes registry caches, malloc/MPI internals and setup outside traversal\n";
        }
    }
    if (mpi.rank == 0)
    {
        REQUIRE(peaks.size() == 3);
        CHECK(peaks[1] == peaks[0]);
        CHECK(peaks[2] == peaks[0]);
    }
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
