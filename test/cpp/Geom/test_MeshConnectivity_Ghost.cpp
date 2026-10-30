#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeRelationSynthetic.hpp"
#include <cmath>
#include <fmt/core.h>

using namespace DNDS;
using namespace DNDS::Geom;
namespace T = NativeRelationSynthetic;
static MPIInfo g_mpi;

static MeshAdjacency conesWithManualGhostRows()
{
    auto original = T::FourQuadCones(g_mpi);
    std::vector<DNDS::index> ghosts;
    if (g_mpi.rank == 0 && g_mpi.size > 1)
        ghosts = {2, 3};
    if (g_mpi.rank == 1)
        ghosts = {0, 1};
    auto source = original.Source().WithGhosts(ghosts);
    MeshAdjacency cone(Adj::Cell2Node, source, original.Target(), original.Owned());
    auto full = cone.WithGhostRows(EntityGhostMapping(source));
    CHECK(full.Source().NumGhost() == (g_mpi.size > 1 && g_mpi.rank < 2 ? 2 : 0));
    CHECK(full.Read().RowCount() == full.Source().NumOwned() + full.Source().NumGhost());
    return full;
}

TEST_CASE("Compiled ghost programs expose immutable inspection and own an exact execution signature")
{
    using Tree = CompiledGhostTree;
    static_assert(!std::is_default_constructible_v<Tree>);
    static_assert(std::is_same_v<decltype(std::declval<Tree &>().Roots()), const std::vector<GhostTreeNode> &>);
    static_assert(std::is_same_v<decltype(std::declval<Tree &>().Levels()), const std::vector<std::vector<LevelEntry>> &>);
    GhostSpec spec{{{EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
                    {EntityKind::Cell, {Adj::Cell2Node, Adj::Node2Cell}, EntityKind::Cell}}};
    const auto tree = Tree::compile(spec);
    auto copy = tree;
    CHECK(&copy.Levels() == &tree.Levels());
    const auto signature = tree.Signature();
    spec.chains.erase(spec.chains.begin());
    const auto other = Tree::compile(spec);
    CHECK(tree.Signature() == signature);
    CHECK(other.NodeCount() == tree.NodeCount());
    CHECK(other.Signature() != signature); // Same hops, different collection endpoint.
    CHECK(tree.Levels()[1][0].collect);
    CHECK_FALSE(other.Levels()[1][0].collect);
    copy = other;
    CHECK(tree.Signature() == signature);
    CHECK(copy.Signature() == other.Signature());
}

TEST_CASE("Ghost program agreement rejects different endpoint markers before traversal")
{
    if (g_mpi.size == 1)
        return;
    auto cone = T::FourQuadCones(g_mpi);
    auto c2c = T::AllNeighbors(cone);
    AdjacencyRegistry registry(g_mpi);
    registry.Register(c2c.Read());
    GhostSpec spec{{{EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Cell}, EntityKind::Cell}}};
    if (g_mpi.rank == 0)
        spec.chains.push_back({EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell});
    CHECK_THROWS(registry.Evaluate(CompiledGhostTree::compile(spec)));
    auto common = CompiledGhostTree::compile(GhostSpec{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}});
    CHECK_NOTHROW(registry.Evaluate(common));
}

TEST_CASE("evaluateGhostTree: single hop cell2cell")
{
    if (g_mpi.size < 2)
        return;

    auto cone = T::FourQuadCones(g_mpi);
    auto c2c = T::AllNeighbors(cone);
    AdjacencyRegistry dag(g_mpi);
    dag.Register(c2c.Read());

    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
    }};
    auto tree = CompiledGhostTree::compile(spec);
    CHECK(tree.requiredAdjs().count(Adj::Cell2Cell) == 1);

    auto result = dag.Evaluate(tree);

    // hasGhosts is collective — true on all ranks if any rank has ghosts.
    CHECK(result.hasGhosts(EntityKind::Cell));
    CHECK(!result.hasGhosts(EntityKind::Node)); // no node chain in this spec
    auto &ghostCells = result.ghostIndices[EntityKind::Cell];

    if (g_mpi.rank == 0)
    {
        // Rank 0 owns cells {0,1}. All 4 cells share node 4, so neighbors = {2,3}.
        // Exact: 2 ghost cells, no more, no less.
        CHECK(ghostCells.size() == 2);
        CHECK(result.totalGhosts() == 2);
        CHECK(ghostCells[0] == 2);
        CHECK(ghostCells[1] == 3);
    }
    else if (g_mpi.rank == 1)
    {
        CHECK(ghostCells.size() == 2);
        CHECK(result.totalGhosts() == 2);
        CHECK(ghostCells[0] == 0);
        CHECK(ghostCells[1] == 1);
    }
    else
    {
        // Ranks 2+: no cells, no ghosts on this rank.
        CHECK(ghostCells.empty());
        CHECK(result.totalGhosts() == 0);
    }
}

TEST_CASE("evaluateGhostTree: two-hop cell2cell2node with manual ghost")
{
    if (g_mpi.size < 2)
        return;

    auto cone = conesWithManualGhostRows();
    auto c2c = T::AllNeighbors(cone);
    AdjacencyRegistry dag(g_mpi);
    dag.Register(c2c.Read());
    dag.Register(cone.Read());

    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node},
    }};
    auto tree = CompiledGhostTree::compile(spec);
    auto result = dag.Evaluate(tree);

    // Collective: all ranks agree these kinds have ghosts.
    CHECK(result.hasGhosts(EntityKind::Cell));
    CHECK(result.hasGhosts(EntityKind::Node));

    auto &ghostNodes = result.ghostIndices[EntityKind::Node];

    if (g_mpi.rank == 0)
    {
        // Rank 0 owns nodes 0-4.
        // Ghost cells 2,3 have nodes {3,4,7,6} and {4,5,8,7}.
        // Owned cells 0,1 have nodes {0,1,4,3} and {1,2,5,4}.
        // cell2cell2node = all nodes of owned+ghost cells = {0..8}.
        // Non-owned: {5,6,7,8}. Exact count: 4 ghost nodes.
        CHECK(ghostNodes.size() == 4);
        std::set<DNDS::index> ghostNodeSet(ghostNodes.begin(), ghostNodes.end());
        CHECK(ghostNodeSet == std::set<DNDS::index>{5, 6, 7, 8});
        // Total: 2 ghost cells + 4 ghost nodes = 6.
        CHECK(result.totalGhosts() == 6);
    }
    else if (g_mpi.rank == 1)
    {
        // Rank 1 owns nodes 5-8.
        // cell2cell2node = all nodes of owned+ghost cells = {0..8}.
        // Non-owned: {0,1,2,3,4}. Exact count: 5 ghost nodes.
        CHECK(ghostNodes.size() == 5);
        std::set<DNDS::index> ghostNodeSet(ghostNodes.begin(), ghostNodes.end());
        CHECK(ghostNodeSet == std::set<DNDS::index>{0, 1, 2, 3, 4});
        CHECK(result.totalGhosts() == 7); // 2 ghost cells + 5 ghost nodes
    }
    else
    {
        // Ranks 2+: no cells, no ghosts of any kind.
        CHECK(result.ghostIndices[EntityKind::Node].empty());
        CHECK(result.totalGhosts() == 0);
    }
}

TEST_CASE("evaluateGhostTree: 2-ring cell chain")
{
    if (g_mpi.size < 2)
        return;

    auto cone = conesWithManualGhostRows();
    auto c2c = T::AllNeighbors(cone);
    AdjacencyRegistry dag(g_mpi);
    dag.Register(c2c.Read());

    GhostSpec spec1{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result1 = dag.Evaluate(CompiledGhostTree::compile(spec1));

    GhostSpec spec2{{{EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result2 = dag.Evaluate(CompiledGhostTree::compile(spec2));

    if (g_mpi.rank < 2)
    {
        // On this mesh, 1-ring already covers all cells. 2-ring is identical.
        auto &ghost1 = result1.ghostIndices[EntityKind::Cell];
        auto &ghost2 = result2.ghostIndices[EntityKind::Cell];
        CHECK(ghost1.size() == 2);
        CHECK(ghost2.size() == 2);
        CHECK(ghost1 == ghost2);
    }
}

TEST_CASE("evaluateGhostTree: union of multiple chains")
{
    if (g_mpi.size < 2)
        return;

    auto cone = conesWithManualGhostRows();
    auto c2c = T::AllNeighbors(cone);
    AdjacencyRegistry dag(g_mpi);
    dag.Register(c2c.Read());
    dag.Register(cone.Read());

    GhostSpec spec2{{{EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node}}};
    auto result2 = dag.Evaluate(CompiledGhostTree::compile(spec2));

    // Union of chain 1 (cell2node) + chain 2 (cell2cell2node).
    GhostSpec specUnion{{
        {EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node},
    }};
    auto resultUnion = dag.Evaluate(CompiledGhostTree::compile(specUnion));

    if (g_mpi.rank == 0)
    {
        // Chain 2 alone: cell2cell→cell2node on all 4 cells → nodes {0..8},
        // non-owned [0,5) → ghost = {5,6,7,8}.
        auto &ghost2 = result2.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> set2(ghost2.begin(), ghost2.end());
        CHECK(set2 == std::set<DNDS::index>{5, 6, 7, 8});

        // Union of chain1 (owned cell2node → ghost {5}) + chain2 (→ ghost {5,6,7,8}).
        // Exact: {5,6,7,8}.
        auto &ghostU = resultUnion.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> setU(ghostU.begin(), ghostU.end());
        CHECK(setU == std::set<DNDS::index>{5, 6, 7, 8});
    }
    else if (g_mpi.rank == 1)
    {
        // Chain 2 alone: rank 1 owns nodes [5,9). Ghost = {0,1,2,3,4}.
        auto &ghost2 = result2.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> set2(ghost2.begin(), ghost2.end());
        CHECK(set2 == std::set<DNDS::index>{0, 1, 2, 3, 4});

        // Union: chain1 on rank 1 owned cells {2,3} → nodes {3,4,5,6,7,8},
        // ghost from chain1 = {3,4} (outside [5,9)).
        // Union = {3,4} ∪ {0,1,2,3,4} = {0,1,2,3,4}.
        auto &ghostU = resultUnion.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> setU(ghostU.begin(), ghostU.end());
        CHECK(setU == std::set<DNDS::index>{0, 1, 2, 3, 4});
    }
    else
    {
        // Ranks 2+: no cells, no ghost nodes.
        CHECK(resultUnion.ghostIndices[EntityKind::Node].empty());
    }
}

TEST_CASE("evaluateGhostTree: np=1 produces no ghosts")
{
    if (g_mpi.size != 1)
        return;

    auto cone = T::FourQuadCones(g_mpi);
    auto c2c = T::AllNeighbors(cone);
    AdjacencyRegistry dag(g_mpi);
    dag.Register(c2c.Read());

    GhostSpec spec{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result = dag.Evaluate(CompiledGhostTree::compile(spec));

    CHECK(!result.hasGhosts(EntityKind::Cell));
    CHECK(result.totalGhosts() == 0);
}

struct SyntheticTiledGrid
{
    bool periodic{false};
    DNDS::index N, totalCols, totalRows;
    DNDS::index nCellLocal, nNodeLocal;
    DNDS::index colStart, colEnd, rowStart, rowEnd;
    DNDS::index rankCol, rankRow, ranksPerRow, ranksPerCol;
    std::vector<DNDS::index> ownerNodeOffsets; // pre-computed, no MPI
    std::optional<MeshAdjacency> cell2cell, cell2node;
    std::optional<EntitySpace> cells, nodes;

    DNDS::index cellGlobal(DNDS::index rank, DNDS::index localRow, DNDS::index localCol) const
    {
        return rank * N * N + localRow * N + localCol;
    }
    DNDS::index cellOwnerRank(DNDS::index gr, DNDS::index gc) const
    {
        DNDS::index rc = std::min(gc / N, ranksPerRow - 1);
        DNDS::index rr = std::min(gr / N, ranksPerCol - 1);
        return rr * ranksPerRow + rc;
    }
    /// Node global index via per-rank local matrix.  Node at (r,c) belongs to
    /// cellOwnerRank(r,c).  Its local (row,col) within that rank and its
    /// rank's offset give the 1-D global index.  Periodic nodes wrap.
    DNDS::index nodeGlobal(DNDS::index r, DNDS::index c) const
    {
        DNDS::index wr = r, wc = c;
        if (periodic)
        {
            wr = ((r % totalRows) + totalRows) % totalRows;
            wc = ((c % totalCols) + totalCols) % totalCols;
        }
        DNDS::index owner = cellOwnerRank(wr, wc);
        DNDS::index ownerRC = (owner % ranksPerRow) * N;
        DNDS::index ownerRR = (owner / ranksPerRow) * N;
        DNDS::index ownerNCW = periodic ? N
                                        : N + ((owner % ranksPerRow == ranksPerRow - 1) ? 1 : 0);
        DNDS::index ownerNRH = periodic ? N
                                        : N + ((owner / ranksPerRow == ranksPerCol - 1) ? 1 : 0);
        DNDS::index localCol = wc - ownerRC;
        DNDS::index localRow = wr - ownerRR;
        return ownerNodeOffsets[owner] + localRow * ownerNCW + localCol;
    }

    void build(DNDS::index tileN, const MPIInfo &mpi)
    {
        N = tileN;

        ranksPerRow = DNDS::index(std::sqrt(double(mpi.size)));
        ranksPerCol = mpi.size / ranksPerRow;
        while (ranksPerRow * ranksPerCol != mpi.size && ranksPerRow > 1)
            ranksPerRow--, ranksPerCol = mpi.size / ranksPerRow;
        if (ranksPerRow * ranksPerCol != mpi.size)
            ranksPerRow = mpi.size, ranksPerCol = 1;

        rankCol = mpi.rank % ranksPerRow;
        rankRow = mpi.rank / ranksPerRow;

        totalCols = ranksPerRow * N;
        totalRows = ranksPerCol * N;
        colStart = rankCol * N;
        colEnd = (rankCol + 1) * N;
        rowStart = rankRow * N;
        rowEnd = (rankRow + 1) * N;
        nCellLocal = N * N;

        bool lastInRow = (rankCol == ranksPerRow - 1);
        bool lastInCol = (rankRow == ranksPerCol - 1);

        // Node partition: pre-compute offsets from known shapes.
        ownerNodeOffsets.resize(mpi.size);
        {
            DNDS::index off = 0;
            for (DNDS::index ri = 0; ri < mpi.size; ri++)
            {
                ownerNodeOffsets[ri] = off;
                DNDS::index rc = ri % ranksPerRow;
                DNDS::index rr = ri / ranksPerRow;
                DNDS::index ncw = N + (periodic ? 0 : (rc == ranksPerRow - 1 ? 1 : 0));
                DNDS::index nrh = N + (periodic ? 0 : (rr == ranksPerCol - 1 ? 1 : 0));
                off += nrh * ncw;
            }
        }
        nNodeLocal = (periodic ? N * N
                               : (N + (lastInRow ? 1 : 0)) * (N + (lastInCol ? 1 : 0)));

        nodes.emplace(EntitySpace::CreateOwned(EntityKind::Node, mpi, nNodeLocal));
        cells.emplace(nodes->CreateSibling(EntityKind::Cell, nCellLocal));
        std::vector<std::vector<DNDS::index>> cones(static_cast<size_t>(nCellLocal)), neighborsRows(static_cast<size_t>(nCellLocal));
        for (DNDS::index iLocal = 0; iLocal < nCellLocal; iLocal++)
        {
            DNDS::index r = rowStart + iLocal / N;
            DNDS::index c = colStart + iLocal % N;
            cones[size_t(iLocal)].resize(4);
            cones[size_t(iLocal)][0] = nodeGlobal(r, c);
            cones[size_t(iLocal)][1] = nodeGlobal(r, c + 1);
            cones[size_t(iLocal)][2] = nodeGlobal(r + 1, c + 1);
            cones[size_t(iLocal)][3] = nodeGlobal(r + 1, c);
        }
        for (DNDS::index iLocal = 0; iLocal < nCellLocal; iLocal++)
        {
            DNDS::index r = rowStart + iLocal / N;
            DNDS::index c = colStart + iLocal % N;
            DNDS::index myGlobal = cellGlobal(mpi.rank, iLocal / N, iLocal % N);

            std::set<DNDS::index> neighbors;
            for (DNDS::index dr = -1; dr <= 1; dr++)
            {
                for (DNDS::index dc = -1; dc <= 1; dc++)
                {
                    if (dr == 0 && dc == 0)
                        continue;
                    DNDS::index nr = r + dr;
                    DNDS::index nc = c + dc;
                    if (periodic)
                    {
                        nr = ((nr % totalRows) + totalRows) % totalRows;
                        nc = ((nc % totalCols) + totalCols) % totalCols;
                    }
                    else if (nr < 0 || nr >= totalRows || nc < 0 || nc >= totalCols)
                        continue;
                    DNDS::index owner = cellOwnerRank(nr, nc);
                    DNDS::index nbr = nr - (owner / ranksPerRow) * N;
                    DNDS::index nbc = nc - (owner % ranksPerRow) * N;
                    DNDS::index nbGlobal = cellGlobal(owner, nbr, nbc);
                    if (nbGlobal != myGlobal)
                        neighbors.insert(nbGlobal);
                }
            }
            neighborsRows[size_t(iLocal)] = {neighbors.begin(), neighbors.end()};
        }
        cell2node.emplace(Adj::Cell2Node, *cells, *nodes, NativePeriodicFixtures::Rows(cones));
        cell2cell.emplace(Adj::Cell2Cell, *cells, *cells, NativePeriodicFixtures::Rows(neighborsRows));
        // Fixture storage is local cones + local neighbor incidences + P offsets.
        CHECK(cell2node->Owned().RowCount() == size_t(N * N));
        CHECK(cell2cell->Owned().RowCount() == size_t(N * N));
        CHECK(ownerNodeOffsets.size() == size_t(mpi.size));
    }

    DNDS::index expectedGhostNodesN(DNDS::index nLayers) const
    {
        DNDS::index dx = std::min(N + 2 * nLayers + 1, totalRows);
        DNDS::index dy = std::min(N + 2 * nLayers + 1, totalCols);
        DNDS::index total = (periodic ? N * N : nNodeLocal);
        return dx * dy - total;
    }

    std::set<DNDS::index> expectedGhostCells(const MPIInfo &mpi) const
    {
        std::set<DNDS::index> ghosts;
        // Independent geometric box oracle: do not read generated adjacency.
        for (DNDS::index r = rowStart - 1; r <= rowEnd; ++r)
            for (DNDS::index c = colStart - 1; c <= colEnd; ++c)
            {
                auto wr = r, wc = c;
                if (periodic)
                {
                    wr = (r + totalRows) % totalRows;
                    wc = (c + totalCols) % totalCols;
                }
                else if (r < 0 || r >= totalRows || c < 0 || c >= totalCols)
                    continue;
                const auto owner = cellOwnerRank(wr, wc);
                if (owner != mpi.rank)
                    ghosts.insert(cellGlobal(owner, wr - (owner / ranksPerRow) * N, wc - (owner % ranksPerRow) * N));
            }
        return ghosts;
    }

    std::set<DNDS::index> expectedGhostNodesFromOwnedCells(const MPIInfo &mpi) const
    {
        const DNDS::index nodeOffset = ownerNodeOffsets[size_t(mpi.rank)];
        const DNDS::index nodeEnd = nodeOffset + nNodeLocal;
        std::set<DNDS::index> ghosts;
        // Cartesian vertex box, independent of the generated cell cone rows.
        for (DNDS::index r = rowStart; r <= rowEnd; ++r)
            for (DNDS::index c = colStart; c <= colEnd; ++c)
            {
                const auto global = nodeGlobal(r, c);
                if (global < nodeOffset || global >= nodeEnd)
                    ghosts.insert(global);
            }
        return ghosts;
    }

    /// Analytical n-layer cell ghost count.
    DNDS::index expectedGhostCellsN(DNDS::index nLayers) const
    {
        if (periodic)
        {
            DNDS::index dx = std::min(N + 2 * nLayers, totalCols);
            DNDS::index dy = std::min(N + 2 * nLayers, totalRows);
            return dx * dy - N * N;
        }
        DNDS::index rMin = std::max(DNDS::index(0), rowStart - nLayers * N);
        DNDS::index rMax = std::min(totalRows, rowEnd + nLayers * N);
        DNDS::index cMin = std::max(DNDS::index(0), colStart - nLayers * N);
        DNDS::index cMax = std::min(totalCols, colEnd + nLayers * N);
        return (rMax - rMin) * (cMax - cMin) - N * N;
    }
};

TEST_CASE("evaluateGhostTree: performance benchmark")
{
    if (g_mpi.size < 2)
        return;

    std::vector<DNDS::index> sizes = {32, 100};
    if (g_mpi.size <= 4)
        sizes.push_back(500);

    for (auto N : sizes)
    {
        SyntheticTiledGrid grid;
        grid.build(N, g_mpi);

        AdjacencyRegistry dag(g_mpi);
        dag.Register(grid.cell2cell->Read());
        dag.Register(grid.cell2node->Read());

        // --- 1-hop: cell ghost ---
        GhostSpec spec1{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
        auto tree1 = CompiledGhostTree::compile(spec1);

        auto t0 = MPI_Wtime();
        auto result1 = dag.Evaluate(tree1);
        auto t1 = MPI_Wtime();

        // Correctness: exact ghost cell count.
        auto expectedCells = grid.expectedGhostCells(g_mpi);
        auto &ghostCells = result1.ghostIndices[EntityKind::Cell];
        std::set<DNDS::index> actualCells(ghostCells.begin(), ghostCells.end());
        CHECK(actualCells.size() == expectedCells.size());
        CHECK(actualCells == expectedCells);
        const auto boxRows = std::min(grid.totalRows, grid.rowEnd + 1) - std::max(DNDS::index(0), grid.rowStart - 1);
        const auto boxColumns = std::min(grid.totalCols, grid.colEnd + 1) - std::max(DNDS::index(0), grid.colStart - 1);
        CHECK(ghostCells.size() == size_t(boxRows * boxColumns - N * N));

        // --- 2-hop: cell2cell + cell2node (owned cells only, no ghost c2n) ---
        GhostSpec spec2{{
            {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
            {EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Node},
        }};
        auto tree2 = CompiledGhostTree::compile(spec2);

        auto t2 = MPI_Wtime();
        auto result2 = dag.Evaluate(tree2);
        auto t3 = MPI_Wtime();

        // Correctness: exact ghost node count (from owned cells only).
        auto expectedNodes = grid.expectedGhostNodesFromOwnedCells(g_mpi);
        auto &ghostNodes = result2.ghostIndices[EntityKind::Node];
        std::set<DNDS::index> actualNodes(ghostNodes.begin(), ghostNodes.end());
        CHECK(actualNodes.size() == expectedNodes.size());
        CHECK(actualNodes == expectedNodes);
        CHECK(ghostNodes.size() == size_t((N + 1) * (N + 1) - grid.nNodeLocal));

        if (g_mpi.rank == 0)
        {
            fmt::print("  N={}(per rank): cells/rank={}, ghost_cells={} (expected={}), "
                       "ghost_nodes={} (expected={}), "
                       "1-hop={:.4f}s, 2-hop={:.4f}s\n",
                       N, grid.nCellLocal,
                       ghostCells.size(), expectedCells.size(),
                       ghostNodes.size(), expectedNodes.size(),
                       t1 - t0, t3 - t2);
        }

        CHECK(result1.hasGhosts(EntityKind::Cell));
        CHECK(ghostCells.size() <= size_t(8 * N + 4));
        CHECK(ghostNodes.size() <= size_t(8 * N + 8));
        // Only local owned topology and requested halo rows are retained.
        CHECK(grid.cell2node->Owned().RowCount() == size_t(N * N));
        CHECK(grid.cell2cell->Owned().RowCount() == size_t(N * N));
    }
}

TEST_CASE("evaluateGhostTree: doubly-periodic tiled grid")
{
    if (g_mpi.size < 2)
        return;

    DNDS::index N = 32;
    SyntheticTiledGrid grid;
    grid.periodic = true;
    grid.build(N, g_mpi);

    AdjacencyRegistry dag(g_mpi);
    dag.Register(grid.cell2cell->Read());
    dag.Register(grid.cell2node->Read());

    // --- Cell + node ghost via defaultPrimary-style unified spec ---
    for (int nL = 1; nL <= 4; nL++)
    {
        GhostSpec spec;
        for (int depth = 1; depth <= nL; ++depth)
        {
            std::vector<AdjacencyDescriptor> hops(size_t(depth), Adj::Cell2Cell);
            spec.chains.push_back({EntityKind::Cell, hops, EntityKind::Cell});
            hops.push_back(Adj::Cell2Node);
            spec.chains.push_back({EntityKind::Cell, std::move(hops), EntityKind::Node});
        }
        auto result = dag.Evaluate(CompiledGhostTree::compile(spec));

        auto &gcL = result.ghostIndices[EntityKind::Cell];
        auto &gnL = result.ghostIndices[EntityKind::Node];
        CHECK(gcL.size() == grid.expectedGhostCellsN(nL));
        CHECK(gnL.size() == grid.expectedGhostNodesN(nL));

        const char *pos = (grid.rankCol == 0 || grid.rankCol == grid.ranksPerRow - 1 ||
                           grid.rankRow == 0 || grid.rankRow == grid.ranksPerCol - 1)
                              ? "boundary"
                              : "inner";
        fmt::print("  [{}] {} Periodic N={}: nLayers={}: cell_ghost={} (expected={}), "
                   "node_ghost={} (expected={})\n",
                   g_mpi.rank, pos, N, nL,
                   gcL.size(), grid.expectedGhostCellsN(nL),
                   gnL.size(), grid.expectedGhostNodesN(nL));
    }

    // --- n-layer cell ghost ---
    for (int nL = 2; nL <= 4; nL++)
    {
        GhostSpec specML;
        for (int depth = 1; depth <= nL; ++depth)
            specML.chains.push_back({EntityKind::Cell,
                                     std::vector<AdjacencyDescriptor>(size_t(depth), Adj::Cell2Cell),
                                     EntityKind::Cell});
        auto resultML = dag.Evaluate(CompiledGhostTree::compile(specML));
        auto &gcL = resultML.ghostIndices[EntityKind::Cell];
        CHECK(gcL.size() == grid.expectedGhostCellsN(nL));

        fmt::print("  [{}] Periodic N={}: {}-layer cell ghost: size={}\n",
                   g_mpi.rank, N, nL, gcL.size());
    }
}

TEST_CASE("evaluateGhostTree: small periodic tiled grid")
{
    if (g_mpi.size != 4 && g_mpi.size != 8)
        return;

    DNDS::index N = 2;
    SyntheticTiledGrid grid;
    grid.periodic = true;
    grid.build(N, g_mpi);

    AdjacencyRegistry dag(g_mpi);
    dag.Register(grid.cell2cell->Read());
    dag.Register(grid.cell2node->Read());

    // 1-hop cell ghost.
    GhostSpec spec1{{{EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell}}};
    auto result1 = dag.Evaluate(CompiledGhostTree::compile(spec1));
    auto &ghostCells = result1.ghostIndices[EntityKind::Cell];
    DNDS::index expected1 = grid.expectedGhostCellsN(1);
    CHECK(ghostCells.size() == expected1);

    auto expectedCells = grid.expectedGhostCells(g_mpi);
    CHECK(ghostCells.size() == expectedCells.size());

    // 2-hop cell ghost: check via brute-force expected set.
    {
        GhostSpec spec2{{
            {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
            {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Cell}, EntityKind::Cell},
        }};
        auto result2 = dag.Evaluate(CompiledGhostTree::compile(spec2));
        auto &gc2 = result2.ghostIndices[EntityKind::Cell];
        CHECK(gc2.size() == grid.expectedGhostCellsN(2));
    }

    // Node ghost.
    GhostSpec specNodes{{
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Node}, EntityKind::Node},
    }};
    auto resultNodes = dag.Evaluate(CompiledGhostTree::compile(specNodes));
    auto &ghostNodes = resultNodes.ghostIndices[EntityKind::Node];
    CHECK(ghostNodes.size() == grid.expectedGhostNodesN(1));

    fmt::print("  [{}] np={} Periodic N=2: 1-hop cells={} (expected={}), "
               "nodes={} (expected={})\n",
               g_mpi.rank, g_mpi.size,
               ghostCells.size(), expected1,
               ghostNodes.size(), grid.expectedGhostNodesN(1));
}

TEST_CASE("GhostChain: defaultPrimary compiles")
{
    auto spec = GhostSpec::defaultPrimary();
    auto tree = CompiledGhostTree::compile(spec);

    CHECK(tree.Roots().size() == 2);

    const GhostTreeNode *cellRoot = nullptr;
    const GhostTreeNode *bndRoot = nullptr;
    for (auto &r : tree.Roots())
    {
        if (r.kind == EntityKind::Cell)
            cellRoot = &r;
        if (r.kind == EntityKind::Bnd)
            bndRoot = &r;
    }
    REQUIRE(cellRoot);
    REQUIRE(bndRoot);

    CHECK(!cellRoot->collect);
    CHECK(cellRoot->level == 0);

    REQUIRE(cellRoot->children.size() == 1);
    auto &c2cNode = cellRoot->children[0];
    CHECK(c2cNode.kind == EntityKind::Cell);
    CHECK(c2cNode.hop == Adj::Cell2Cell);
    CHECK(c2cNode.collect);
    CHECK(c2cNode.level == 1);

    REQUIRE(c2cNode.children.size() == 1);
    auto &c2nNode = c2cNode.children[0];
    CHECK(c2nNode.kind == EntityKind::Node);
    CHECK(c2nNode.hop == Adj::Cell2Node);
    CHECK(c2nNode.collect);
    CHECK(c2nNode.level == 2);
    CHECK(c2nNode.children.empty());

    CHECK(!bndRoot->collect);
    CHECK(bndRoot->level == 0);

    REQUIRE(bndRoot->children.size() == 1);
    auto &b2nNode = bndRoot->children[0];
    CHECK(b2nNode.kind == EntityKind::Node);
    CHECK(b2nNode.hop == Adj::Bnd2Node);
    CHECK(b2nNode.collect); // explicit direct Bnd2Node chain ends here
    CHECK(b2nNode.level == 1);

    REQUIRE(b2nNode.children.size() == 1);
    auto &n2bNode = b2nNode.children[0];
    CHECK(n2bNode.kind == EntityKind::Bnd);
    CHECK(n2bNode.hop == Adj::Node2Bnd);
    CHECK(n2bNode.collect);
    CHECK(n2bNode.level == 2);

    REQUIRE(n2bNode.children.size() == 1);
    auto &b2n2Node = n2bNode.children[0];
    CHECK(b2n2Node.kind == EntityKind::Node);
    CHECK(b2n2Node.hop == Adj::Bnd2Node);
    CHECK(b2n2Node.collect);
    CHECK(b2n2Node.level == 3);
    CHECK(b2n2Node.children.empty());

    CHECK(tree.MaxLevel() == 3);

    auto kinds = tree.collectedKinds();
    CHECK(kinds.count(EntityKind::Cell) == 1);
    CHECK(kinds.count(EntityKind::Node) == 1);
    CHECK(kinds.count(EntityKind::Bnd) == 1);
    CHECK(kinds.count(EntityKind::Face) == 0);

    auto adjs = tree.requiredAdjs();
    CHECK(adjs.count(Adj::Cell2Cell) == 1);
    CHECK(adjs.count(Adj::Cell2Node) == 1);
    CHECK(adjs.count(Adj::Bnd2Node) == 1);
    CHECK(adjs.count(Adj::Node2Bnd) == 1);
    CHECK(adjs.size() == 4);
}

TEST_CASE("GhostChain: prefix merging")
{
    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Cell}, EntityKind::Cell},
        {EntityKind::Cell, {Adj::Cell2Cell, Adj::Cell2Cell}, EntityKind::Cell},
    }};
    auto tree = CompiledGhostTree::compile(spec);

    REQUIRE(tree.Roots().size() == 1);
    auto &root = tree.Roots()[0];
    CHECK(root.kind == EntityKind::Cell);
    REQUIRE(root.children.size() == 1);

    auto &ring1 = root.children[0];
    CHECK(ring1.kind == EntityKind::Cell);
    CHECK(ring1.collect);
    CHECK(ring1.level == 1);
    REQUIRE(ring1.children.size() == 1);

    auto &ring2 = ring1.children[0];
    CHECK(ring2.kind == EntityKind::Cell);
    CHECK(ring2.collect);
    CHECK(ring2.level == 2);
    CHECK(ring2.children.empty());

    CHECK(tree.MaxLevel() == 2);
}

TEST_CASE("GhostChain: invalid chain detection")
{
    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Cell, {}, EntityKind::Cell},
        }}),
        std::runtime_error);

    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Node, {Adj::Cell2Node}, EntityKind::Node},
        }}),
        std::runtime_error);

    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Cell, {Adj::Cell2Node}, EntityKind::Cell},
        }}),
        std::runtime_error);

    CHECK_THROWS_AS(
        CompiledGhostTree::compile(GhostSpec{{
            {EntityKind::Bnd, {Adj::Bnd2Node, Adj::Cell2Node}, EntityKind::Node},
        }}),
        std::runtime_error);
}

TEST_CASE("GhostChain: dump")
{
    auto spec = GhostSpec::defaultPrimary();
    auto tree = CompiledGhostTree::compile(spec);
    std::string d = tree.dump();

    CHECK(d.find("Cell") != std::string::npos);
    CHECK(d.find("Bnd") != std::string::npos);
    CHECK(d.find("Node") != std::string::npos);
    CHECK(d.find("COLLECT") != std::string::npos);
}

TEST_CASE("GhostChain: face ghost chain")
{
    GhostSpec spec{{
        {EntityKind::Cell, {Adj::Cell2Face}, EntityKind::Face},
    }};
    auto tree = CompiledGhostTree::compile(spec);

    REQUIRE(tree.Roots().size() == 1);
    auto &root = tree.Roots()[0];
    CHECK(root.kind == EntityKind::Cell);
    REQUIRE(root.children.size() == 1);

    auto &faceNode = root.children[0];
    CHECK(faceNode.kind == EntityKind::Face);
    CHECK(faceNode.hop == Adj::Cell2Face);
    CHECK(faceNode.collect);
    CHECK(faceNode.level == 1);

    auto adjs = tree.requiredAdjs();
    CHECK(adjs.count(Adj::Cell2Face) == 1);
    CHECK(adjs.size() == 1);
}

TEST_CASE("GhostChain: AdjacencyDescriptor equality and hash")
{
    AdjacencyDescriptor a(EntityKind::Cell, EntityKind::Node);
    AdjacencyDescriptor b(EntityKind::Cell, EntityKind::Node, EntityKind::Face);
    CHECK(a == b); // direct: via ignored

    AdjacencyDescriptor c(EntityKind::Cell, EntityKind::Cell, EntityKind::Node);
    AdjacencyDescriptor d(EntityKind::Cell, EntityKind::Cell, EntityKind::Face);
    CHECK(c != d); // intra-level: via matters

    AdjacencyDescriptor e(EntityKind::Cell, EntityKind::Cell, EntityKind::Node);
    CHECK(c == e);

    AdjacencyDescriptorHash h;
    CHECK(h(a) == h(b));
    CHECK(h(c) == h(e));
    (void)h(d);
}

TEST_CASE("GhostChain: entityDepth")
{
    CHECK(entityDepth(EntityKind::Cell, 3) == 3);
    CHECK(entityDepth(EntityKind::Face, 3) == 2);
    CHECK(entityDepth(EntityKind::Edge, 3) == 1);
    CHECK(entityDepth(EntityKind::Node, 3) == 0);
    CHECK(entityDepth(EntityKind::Bnd, 3) == 2);

    CHECK(entityDepth(EntityKind::Cell, 2) == 2);
    CHECK(entityDepth(EntityKind::Face, 2) == 1);
    CHECK(entityDepth(EntityKind::Edge, 2) == 1);
    CHECK(entityDepth(EntityKind::Node, 2) == 0);
    CHECK(entityDepth(EntityKind::Bnd, 2) == 1);
}

TEST_CASE("GhostChain: adjKindName")
{
    CHECK(adjKindName(Adj::Cell2Node) == "Cell2Node");
    CHECK(adjKindName(Adj::Node2Cell) == "Node2Cell");
    CHECK(adjKindName(Adj::Cell2Cell) == "Cell2Cell(Node)");
    CHECK(adjKindName(Adj::Cell2CellFace) == "Cell2Cell(Face)");
    CHECK(adjKindName(Adj::Bnd2Node) == "Bnd2Node");
}

TEST_CASE("GhostChain: checkAvailable native registry rejects omitted required relation")
{
    auto cone = T::FourQuadCones(g_mpi);
    auto adjacent = T::AllNeighbors(cone);
    AdjacencyRegistry registry(g_mpi);
    registry.Register(cone.Read());
    registry.Register(adjacent.Read());
    // The same two required boundary relations are deliberately omitted.
    const auto required = CompiledGhostTree::compile(GhostSpec::defaultPrimary()).requiredAdjs();
    std::unordered_set<AdjacencyDescriptor, AdjacencyDescriptorHash> missing;
    for (const auto &key : required)
        if (!registry.Find(key))
            missing.insert(key);
    CHECK(missing.size() == 2);
    CHECK(missing.count(Adj::Bnd2Node) == 1);
    CHECK(missing.count(Adj::Node2Bnd) == 1);
    CHECK_THROWS(registry.Evaluate(CompiledGhostTree::compile(GhostSpec::defaultPrimary())));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
