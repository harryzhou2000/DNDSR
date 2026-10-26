#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "NativeRelationSynthetic.hpp"
#include "Geom/Elements.hpp"
#include "NativeStructuredInterpolation.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include <filesystem>

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;
namespace T = NativeRelationSynthetic;
static MPIInfo g_mpi;

struct NativeHandCrafted
{
    NativeRelation cone;
    std::vector<Elem::ElemType> types;
};

static NativeHandCrafted makeNativeHandCrafted(
    const MPIInfo &mpi, const std::vector<std::pair<Elem::ElemType, std::vector<DNDS::index>>> &input, DNDS::index numNodes)
{
    auto nodes = EntitySpace::CreateOwned(EntityKind::Node, mpi, mpi.rank == 0 ? numNodes : 0);
    auto cells = nodes.CreateSibling(EntityKind::Cell, mpi.rank == 0 ? DNDS::index(input.size()) : 0);
    std::vector<std::vector<DNDS::index>> rows;
    std::vector<Elem::ElemType> types;
    if (mpi.rank == 0)
        for (const auto &item : input)
        {
            types.push_back(item.first);
            rows.push_back(item.second);
        }
    return {NativeRelation(Adj::Cell2Node, cells, nodes, NativePeriodicFixtures::Rows(rows)), std::move(types)};
}

static SubEntityQueryPbi queryFor(const std::vector<Elem::ElemType> &types, bool edges)
{
    SubEntityQueryPbi query;
    query.numSubEntities = [types, edges](DNDS::index parent)
    { auto e = Elem::Element{types[size_t(parent)]}; return edges ? e.GetNumEdges() : e.GetNumFaces(); };
    query.describe = [types, edges](DNDS::index parent, int slot)
    {
        auto e = Elem::Element{types[size_t(parent)]};
        auto sub = edges ? e.ObtainEdge(t_index(slot)) : e.ObtainFace(t_index(slot));
        return SubEntityDesc{sub.GetNumVertices(), sub.GetNumNodes(), t_index(sub.type)};
    };
    query.extractNodes = [types, edges](DNDS::index parent, int slot, const std::function<DNDS::index(int)> &read, DNDS::index *output)
    {
        auto e = Elem::Element{types[size_t(parent)]};
        auto sub = edges ? e.ObtainEdge(t_index(slot)) : e.ObtainFace(t_index(slot));
        std::vector<DNDS::index> nodes(size_t(e.GetNumNodes())), selected(size_t(sub.GetNumNodes()));
        for (size_t i = 0; i < nodes.size(); ++i)
            nodes[i] = read(int(i));
        if (edges)
            e.ExtractEdgeNodes(t_index(slot), nodes, selected);
        else
            e.ExtractFaceNodes(t_index(slot), nodes, selected);
        std::copy(selected.begin(), selected.end(), output);
    };
    query.extractPbi = [types, edges](DNDS::index parent, int slot, const std::function<NodePeriodicBits(int)> &read, NodePeriodicBits *output)
    {
        auto e = Elem::Element{types[size_t(parent)]};
        auto sub = edges ? e.ObtainEdge(t_index(slot)) : e.ObtainFace(t_index(slot));
        std::vector<NodePeriodicBits> nodes(size_t(e.GetNumNodes())), selected(size_t(sub.GetNumNodes()));
        for (size_t i = 0; i < nodes.size(); ++i)
            nodes[i] = read(int(i));
        if (edges)
            e.ExtractEdgeNodes(t_index(slot), nodes, selected);
        else
            e.ExtractFaceNodes(t_index(slot), nodes, selected);
        std::copy(selected.begin(), selected.end(), output);
    };
    return query;
}

static OwnershipResolverMulti owner()
{
    return [](const std::vector<DNDS::index> &, const std::vector<MPI_int> &ranks, DNDS::index)
    { return OwnershipDecision{*std::min_element(ranks.begin(), ranks.end()) == g_mpi.rank, {}}; };
}

static std::set<DNDS::index> getEntityVertexSet(const NativeRelation &nodes, const R::Array<ElemInfo> &info, DNDS::index row)
{
    auto metadata = info.Read();
    const auto count = Elem::Element{metadata.View()(size_t(row), 0).getElemType()}.GetNumVertices();
    auto read = nodes.Read();
    std::set<DNDS::index> result;
    for (int i = 0; i < count; ++i)
        result.insert(read.ReadEntry(row, size_t(i)).global);
    return result;
}

TEST_CASE("Interpolate: 4-quad faces (2D)")
{
    // Only run on rank 0 (serial test, other ranks have 0 cells)

    //     6---7---8
    //     |   |   |
    //     | 2 | 3 |
    //     |   |   |
    //     3---4---5
    //     |   |   |
    //     | 0 | 1 |
    //     |   |   |
    //     0---1---2
    //
    // Quad4 face table: {0,1}, {1,2}, {2,3}, {3,0}

    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Quad4, {0, 1, 4, 3}},
                                       {Elem::Quad4, {1, 2, 5, 4}},
                                       {Elem::Quad4, {3, 4, 7, 6}},
                                       {Elem::Quad4, {4, 5, 8, 7}},
                                   },
                                   9);

    auto query = queryFor(m.types, false);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Face, query, owner());

    if (g_mpi.rank == 0)
    {

        // 4 quads × 4 edges = 16 half-edges, with 4 internal shared edges.
        // Unique edges: 16 - 4 = 12
        CHECK(res.entities.NumGlobal() == 12);

        // Each cell should reference exactly 4 entities
        for (DNDS::index i = 0; i < 4; i++)
        {
            CAPTURE(i);
            CHECK(res.parent2entity.Read().RowSize(i) == 4);
        }

        // All entities should be Line2
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
        {
            CAPTURE(i);
            CHECK(res.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Line2);
        }

        // Internal edges: count entities with 2+ parents (variable-width entity2parent)
        int nInternal = 0;
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            if (res.entity2parent.Read().RowSize(i) >= 2)
                nInternal++;
        CHECK(nInternal == 4);

        // Boundary edges: 12 - 4 = 8
        CHECK(res.entities.NumGlobal() - nInternal == 8);

        // Verify no duplicate vertex sets
        std::set<std::set<DNDS::index>> allVertSets;
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
        {
            auto vs = getEntityVertexSet(res.entity2node, res.entityElemInfo, i);
            CHECK(allVertSets.count(vs) == 0);
            allVertSets.insert(vs);
        }

        // Verify entity2node has 2 nodes per entity (Line2)
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
        {
            CAPTURE(i);
            CHECK(res.entity2node.Read().RowSize(i) == 2);
        }

        // Verify parent2entity → entity2parent consistency
        for (DNDS::index iCell = 0; iCell < 4; iCell++)
        {
            for (DNDS::rowsize j = 0; j < res.parent2entity.Read().RowSize(iCell); j++)
            {
                DNDS::index iEnt = res.parent2entity.Read().Raw(iCell, size_t(j));
                CAPTURE(iCell);
                CAPTURE(j);
                CAPTURE(iEnt);
                bool found = false;
                for (DNDS::rowsize p = 0; p < res.entity2parent.Read().RowSize(iEnt); p++)
                    if (res.entity2parent.Read().Raw(iEnt, size_t(p)) == iCell)
                        found = true;
                CHECK(found);
            }
        }
    }
}

TEST_CASE("Interpolate: 2-tri faces (2D)")
{

    //     2---3
    //     |\ 1|
    //     | \ |
    //     |0 \|
    //     0---1
    //
    // Tri3 face table: {0,1}, {1,2}, {2,0}
    // Cell 0: nodes {0,1,2} → faces: {0,1}, {1,2}, {2,0}
    // Cell 1: nodes {1,3,2} → faces: {1,3}, {3,2}, {2,1}
    // Shared edge: {1,2}
    // Total unique: 3 + 3 - 1 = 5

    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Tri3, {0, 1, 2}},
                                       {Elem::Tri3, {1, 3, 2}},
                                   },
                                   4);

    auto query = queryFor(m.types, false);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Face, query, owner());

    if (g_mpi.rank == 0)
    {

        CHECK(res.entities.NumGlobal() == 5);

        // One internal edge (shared)
        int nInternal = 0;
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            if (res.entity2parent.Read().RowSize(i) >= 2)
                nInternal++;
        CHECK(nInternal == 1);

        // The shared edge should have vertex set {1, 2}
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
        {
            if (res.entity2parent.Read().RowSize(i) >= 2)
            {
                auto vs = getEntityVertexSet(res.entity2node, res.entityElemInfo, i);
                CHECK(vs == std::set<DNDS::index>{1, 2});
            }
        }
    }
}

TEST_CASE("Interpolate: single tet faces (3D)")
{

    // Tet4: nodes {0,1,2,3}, 4 Tri3 faces
    // face 0: {0,2,1}, face 1: {0,1,3}, face 2: {1,2,3}, face 3: {2,0,3}
    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Tet4, {0, 1, 2, 3}},
                                   },
                                   4);

    auto query = queryFor(m.types, false);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Face, query, owner());

    if (g_mpi.rank == 0)
    {

        CHECK(res.entities.NumGlobal() == 4);

        // All faces Tri3
        for (DNDS::index i = 0; i < 4; i++)
        {
            CHECK(res.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Tri3);
            CHECK(res.entity2node.Read().RowSize(i) == 3);
        }

        // All boundary (single parent)
        for (DNDS::index i = 0; i < 4; i++)
            CHECK(res.entity2parent.Read().RowSize(i) < 2);

        // Verify face vertex sets match Tet4 topology
        std::set<std::set<DNDS::index>> expectedFaces = {
            {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}};
        std::set<std::set<DNDS::index>> actualFaces;
        for (DNDS::index i = 0; i < 4; i++)
            actualFaces.insert(getEntityVertexSet(res.entity2node, res.entityElemInfo, i));
        CHECK(actualFaces == expectedFaces);
    }
}

TEST_CASE("Interpolate: two tets sharing a face (3D)")
{

    // Tet A: {0,1,2,3}  Tet B: {1,2,3,4}
    // Shared face: vertices {1,2,3}
    // Total unique faces: 4 + 4 - 1 = 7
    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Tet4, {0, 1, 2, 3}},
                                       {Elem::Tet4, {1, 2, 3, 4}},
                                   },
                                   5);

    auto query = queryFor(m.types, false);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Face, query, owner());

    if (g_mpi.rank == 0)
    {

        CHECK(res.entities.NumGlobal() == 7);

        int nInternal = 0;
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            if (res.entity2parent.Read().RowSize(i) >= 2)
                nInternal++;
        CHECK(nInternal == 1);

        // Shared face is {1,2,3}
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
        {
            if (res.entity2parent.Read().RowSize(i) >= 2)
            {
                auto vs = getEntityVertexSet(res.entity2node, res.entityElemInfo, i);
                CHECK(vs == std::set<DNDS::index>{1, 2, 3});
            }
        }
    }
}

TEST_CASE("Interpolate: single hex faces (3D)")
{

    // Hex8: 6 Quad4 faces
    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Hex8, {0, 1, 2, 3, 4, 5, 6, 7}},
                                   },
                                   8);

    auto query = queryFor(m.types, false);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Face, query, owner());

    if (g_mpi.rank == 0)
    {

        CHECK(res.entities.NumGlobal() == 6);

        for (DNDS::index i = 0; i < 6; i++)
        {
            CHECK(res.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Quad4);
            CHECK(res.entity2node.Read().RowSize(i) == 4);
            CHECK(res.entity2parent.Read().RowSize(i) < 2);
        }

        // Verify face vertex sets
        std::set<std::set<DNDS::index>> expectedFaces = {
            {0, 1, 2, 3}, {0, 1, 4, 5}, {1, 2, 5, 6}, {2, 3, 6, 7}, {0, 3, 4, 7}, {4, 5, 6, 7}};
        std::set<std::set<DNDS::index>> actualFaces;
        for (DNDS::index i = 0; i < 6; i++)
            actualFaces.insert(getEntityVertexSet(res.entity2node, res.entityElemInfo, i));
        CHECK(actualFaces == expectedFaces);
    }
}

TEST_CASE("Interpolate: single tet edges (3D)")
{

    // Tet4: 6 Line2 edges
    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Tet4, {0, 1, 2, 3}},
                                   },
                                   4);

    auto query = queryFor(m.types, true);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Edge, query, owner());

    if (g_mpi.rank == 0)
    {

        CHECK(res.entities.NumGlobal() == 6);

        for (DNDS::index i = 0; i < 6; i++)
        {
            CHECK(res.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Line2);
            CHECK(res.entity2node.Read().RowSize(i) == 2);
        }

        // Tet4 edges: {0,1},{1,2},{2,0},{0,3},{1,3},{2,3}
        std::set<std::set<DNDS::index>> expectedEdges = {
            {0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
        std::set<std::set<DNDS::index>> actualEdges;
        for (DNDS::index i = 0; i < 6; i++)
        {
            std::set<DNDS::index> vs;
            for (DNDS::rowsize j = 0; j < 2; j++)
                vs.insert(res.entity2node.Read().Raw(i, size_t(j)));
            actualEdges.insert(vs);
        }
        CHECK(actualEdges == expectedEdges);
    }
}

TEST_CASE("Interpolate: two-tet shared edges (3D)")
{

    // Tet A: {0,1,2,3}, Tet B: {1,2,3,4}
    // Tet A edges: {0,1},{1,2},{2,0},{0,3},{1,3},{2,3}
    // Tet B edges: {1,2},{2,3},{3,1},{1,4},{2,4},{3,4}
    // Shared: {1,2},{1,3},{2,3} -> 3 shared
    // Unique: 6 + 6 - 3 = 9
    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Tet4, {0, 1, 2, 3}},
                                       {Elem::Tet4, {1, 2, 3, 4}},
                                   },
                                   5);

    auto query = queryFor(m.types, true);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Edge, query, owner());

    if (g_mpi.rank == 0)
    {

        CHECK(res.entities.NumGlobal() == 9);

        int nShared = 0;
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            if (res.entity2parent.Read().RowSize(i) >= 2)
                nShared++;
        CHECK(nShared == 3);

        // Shared edges: {1,2}, {1,3}, {2,3}
        std::set<std::set<DNDS::index>> expectedShared = {{1, 2}, {1, 3}, {2, 3}};
        std::set<std::set<DNDS::index>> actualShared;
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
        {
            if (res.entity2parent.Read().RowSize(i) >= 2)
            {
                std::set<DNDS::index> vs;
                for (DNDS::rowsize j = 0; j < 2; j++)
                    vs.insert(res.entity2node.Read().Raw(i, size_t(j)));
                actualShared.insert(vs);
            }
        }
        CHECK(actualShared == expectedShared);
    }
}

TEST_CASE("Interpolate: mixed tri+quad faces (2D)")
{

    //     3---4---5
    //     |\ 1| 2 |
    //     | \ |   |
    //     |0 \|   |
    //     0---1---2
    //
    // Cell 0: Tri3  {0,1,3}  → 3 Line2 faces
    // Cell 1: Tri3  {1,4,3}  → 3 Line2 faces
    // Cell 2: Quad4 {1,2,5,4} → 4 Line2 faces
    // Shared edges: {0,1}? No. {1,3} between cell 0 & 1, {1,4} between cell 1 & 2
    // Cell 0 faces: {0,1}, {1,3}, {3,0}
    // Cell 1 faces: {1,4}, {4,3}, {3,1}
    // Cell 2 faces: {1,2}, {2,5}, {5,4}, {4,1}
    // Shared: {1,3} (cells 0&1), {1,4} (cells 1&2)
    // Total: 3 + 3 + 4 - 2 = 8

    auto m = makeNativeHandCrafted(g_mpi,
                                   {
                                       {Elem::Tri3, {0, 1, 3}},
                                       {Elem::Tri3, {1, 4, 3}},
                                       {Elem::Quad4, {1, 2, 5, 4}},
                                   },
                                   6);

    auto query = queryFor(m.types, false);
    auto res = NativeConnectivity::Interpolate(m.cone.Read(), EntityKind::Face, query, owner());

    if (g_mpi.rank == 0)
    {

        CHECK(res.entities.NumGlobal() == 8);

        int nInternal = 0;
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            if (res.entity2parent.Read().RowSize(i) >= 2)
                nInternal++;
        CHECK(nInternal == 2);

        // All should be Line2
        for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            CHECK(res.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Line2);
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex faces (non-periodic)")
{
    auto mesh = NativeStructuredInterpolation::BuildHex(g_mpi, 4, 0);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nLocalCells = mesh.nCellLocal;
    auto query = queryFor(mesh.types, false);

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto cone = mesh.cone.Read();
    RelationPeriodicBits bits(cone, mesh.ownedBits, mesh.ghostBits);
    auto result = NativeConnectivity::Interpolate(cone, EntityKind::Face, query, resolver, nullptr);

    // Check global face count: sum of owned faces across all ranks.
    DNDS::index localOwned = result.entities.NumOwned();
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // Expected: 3 * (np*N) * N * N + 3 * N * N = ... let me compute:
    // X-normal faces: (np*N + 1) * N * N
    // Y-normal faces: np * N * (N + 1) * N
    // Z-normal faces: np * N * N * (N + 1)
    DNDS::index expectedFaces = (np * N + 1) * N * N +
                                np * N * (N + 1) * N +
                                np * N * N * (N + 1);
    CHECK(globalOwned == expectedFaces);

    // Verify entity2node: all face nodes are Quad4 (4 nodes each).
    for (DNDS::index i = 0; i < result.entities.NumOwned(); i++)
    {
        CHECK(result.entity2node.Read().RowSize(i) == 4);
        CHECK(result.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Quad4);
    }

    // Verify entity2parent: boundary faces have 1 parent, internal have 2.
    DNDS::index nBndFaces = 0, nIntFaces = 0;
    for (DNDS::index i = 0; i < result.entities.NumOwned(); i++)
    {
        auto pRow = result.entity2parent.Read().GlobalRow(i);
        if (pRow.size() == 1)
            nBndFaces++;
        else if (pRow.size() == 2)
            nIntFaces++;
        else
            FAIL("face has " << pRow.size() << " parents");
    }

    DNDS::index globalBnd = 0, globalInt = 0;
    MPI_Allreduce(&nBndFaces, &globalBnd, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    MPI_Allreduce(&nIntFaces, &globalInt, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    // Boundary faces: 2 * N * N (X-boundaries) + 2 * np*N * N (Y-boundaries) + 2 * np*N * N (Z-boundaries)
    DNDS::index expectedBnd = 2 * N * N + 2 * np * N * N + 2 * np * N * N;
    CHECK(globalBnd == expectedBnd);
    CHECK(globalInt == expectedFaces - expectedBnd);

    // Verify parent2entity: each local cell should have 6 faces.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.Read().RowSize(iCell) == 6);
        // All entries should be valid global face IDs (not UnInitIndex).
        for (rowsize j = 0; j < 6; j++)
        {
            DNDS::index gFace = result.parent2entity.Read().Raw(iCell, size_t(j));
            CAPTURE(j);
            CHECK(gFace != UnInitIndex);
            CHECK(gFace >= 0);
            CHECK(gFace < globalOwned);
        }
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex faces (X-periodic)")
{
    if (g_mpi.size < 2)
        return; // Periodic X needs at least 2 ranks to be meaningful.

    auto mesh = NativeStructuredInterpolation::BuildHex(g_mpi, 4, 1);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nLocalCells = mesh.nCellLocal;
    auto query = queryFor(mesh.types, false);

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto cone = mesh.cone.Read();
    RelationPeriodicBits bits(cone, mesh.ownedBits, mesh.ghostBits);
    auto result = NativeConnectivity::Interpolate(cone, EntityKind::Face, query, resolver, &bits);

    DNDS::index localOwned = result.entities.NumOwned();
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // X-periodic: no X-boundary faces, all X-faces are internal.
    // X-normal faces: np*N * N * N (no +1 boundary)
    // Y-normal faces: np*N * (N+1) * N
    // Z-normal faces: np*N * N * (N+1)
    // Y and Z boundaries still exist.
    DNDS::index expectedFaces = np * N * N * N +
                                np * N * (N + 1) * N +
                                np * N * N * (N + 1);
    CHECK(globalOwned == expectedFaces);

    // All faces should have exactly 2 parents (no boundary in X, Y/Z still have boundary).
    // Actually Y and Z boundaries have 1 parent.
    DNDS::index nBndFaces = 0;
    for (DNDS::index i = 0; i < result.entities.NumOwned(); i++)
    {
        auto pRow = result.entity2parent.Read().GlobalRow(i);
        if (pRow.size() == 1)
            nBndFaces++;
    }
    DNDS::index globalBnd = 0;
    MPI_Allreduce(&nBndFaces, &globalBnd, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);
    // Only Y and Z boundaries remain: 2 * np*N * N (Y) + 2 * np*N * N (Z)
    DNDS::index expectedBnd = 2 * np * N * N + 2 * np * N * N;
    CHECK(globalBnd == expectedBnd);

    // Each local cell still has 6 faces.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.Read().RowSize(iCell) == 6);
        for (rowsize j = 0; j < 6; j++)
        {
            DNDS::index gFace = result.parent2entity.Read().Raw(iCell, size_t(j));
            CHECK(gFace != UnInitIndex);
        }
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex edges (non-periodic)")
{
    auto mesh = NativeStructuredInterpolation::BuildHex(g_mpi, 4, 0);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nLocalCells = mesh.nCellLocal;
    auto query = queryFor(mesh.types, true);

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto cone = mesh.cone.Read();
    RelationPeriodicBits bits(cone, mesh.ownedBits, mesh.ghostBits);
    auto result = NativeConnectivity::Interpolate(cone, EntityKind::Edge, query, resolver, nullptr);

    // Check global edge count.
    DNDS::index localOwned = result.entities.NumOwned();
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // Expected edges for non-periodic NxNxN stacked np times in X:
    // X-aligned: np*N * (N+1) * (N+1)
    // Y-aligned: (np*N+1) * N * (N+1)
    // Z-aligned: (np*N+1) * (N+1) * N
    DNDS::index expectedEdges = np * N * (N + 1) * (N + 1) +
                                (np * N + 1) * N * (N + 1) +
                                (np * N + 1) * (N + 1) * N;
    CHECK(globalOwned == expectedEdges);

    // All edges should be Line2 (2 nodes each).
    for (DNDS::index i = 0; i < result.entities.NumOwned(); i++)
    {
        CHECK(result.entity2node.Read().RowSize(i) == 2);
        CHECK(result.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Line2);
    }

    // Edge parent counts: internal edges have 4 parents, face-boundary 2, edge-boundary 1.
    // Just verify all edges have >= 1 parent and <= 4 parents.
    for (DNDS::index i = 0; i < result.entities.NumOwned(); i++)
    {
        auto pRow = result.entity2parent.Read().GlobalRow(i);
        CAPTURE(i);
        CHECK(pRow.size() >= 1);
        CHECK(pRow.size() <= 4);
    }

    // Each local cell should have 12 edges.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.Read().RowSize(iCell) == 12);
        for (rowsize j = 0; j < 12; j++)
        {
            DNDS::index gEdge = result.parent2entity.Read().Raw(iCell, size_t(j));
            CAPTURE(j);
            CHECK(gEdge != UnInitIndex);
            CHECK(gEdge >= 0);
            CHECK(gEdge < globalOwned);
        }
    }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex edges (X-periodic)")
{
    if (g_mpi.size < 2)
        return;

    auto mesh = NativeStructuredInterpolation::BuildHex(g_mpi, 4, 1);
    DNDS::index N = 4;
    DNDS::index np = g_mpi.size;
    DNDS::index nLocalCells = mesh.nCellLocal;
    auto query = queryFor(mesh.types, true);

    OwnershipResolverMulti resolver =
        [&](const std::vector<DNDS::index> &parents,
            const std::vector<DNDS::MPI_int> &parentRanks,
            DNDS::index nLocal) -> OwnershipDecision
    {
        DNDS::MPI_int minRank = *std::min_element(parentRanks.begin(), parentRanks.end());
        bool anyLocal = false;
        for (auto p : parents)
            if (p < nLocal)
                anyLocal = true;
        if (!anyLocal)
            return {false, {}};
        if (minRank != g_mpi.rank)
            return {false, {}};
        std::vector<DNDS::MPI_int> peers;
        for (size_t i = 0; i < parents.size(); i++)
            if (parents[i] >= nLocal && parentRanks[i] != g_mpi.rank)
                peers.push_back(parentRanks[i]);
        std::sort(peers.begin(), peers.end());
        peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        return {true, std::move(peers)};
    };

    auto cone = mesh.cone.Read();
    RelationPeriodicBits bits(cone, mesh.ownedBits, mesh.ghostBits);
    auto result = NativeConnectivity::Interpolate(cone, EntityKind::Edge, query, resolver, &bits);

    DNDS::index localOwned = result.entities.NumOwned();
    DNDS::index globalOwned = 0;
    MPI_Allreduce(&localOwned, &globalOwned, 1, DNDS_MPI_INDEX, MPI_SUM, g_mpi.comm);

    // X-periodic: X-aligned edges wrap, no X-boundary.
    // X-aligned: np*N * (N+1) * (N+1)  (same as non-periodic, but no +1 boundary -- wait)
    // Actually for non-periodic, X-aligned edges span from node X=0 to X=np*N,
    // so there are np*N edge segments per Y-Z grid line, times (N+1)^2 grid lines.
    // For X-periodic, same count: np*N edges per Y-Z grid line.
    // Y-aligned: np*N * N * (N+1) (no +1 in X dimension)
    // Z-aligned: np*N * (N+1) * N (no +1 in X dimension)
    DNDS::index expectedEdges = np * N * (N + 1) * (N + 1) +
                                np * N * N * (N + 1) +
                                np * N * (N + 1) * N;
    CHECK(globalOwned == expectedEdges);

    // All edges should be Line2.
    for (DNDS::index i = 0; i < result.entities.NumOwned(); i++)
        CHECK(result.entity2node.Read().RowSize(i) == 2);

    // X-periodic removes X-boundary edges. Internal edges in the bulk have 4 parents.
    // Y/Z boundary edges still have 1 or 2 parents.
    for (DNDS::index i = 0; i < result.entities.NumOwned(); i++)
    {
        auto pRow = result.entity2parent.Read().GlobalRow(i);
        CHECK(pRow.size() >= 1);
        CHECK(pRow.size() <= 4);
    }

    // Each local cell has 12 edges.
    for (DNDS::index iCell = 0; iCell < nLocalCells; iCell++)
    {
        CAPTURE(iCell);
        CHECK(result.parent2entity.Read().RowSize(iCell) == 12);
        for (rowsize j = 0; j < 12; j++)
        {
            DNDS::index gEdge = result.parent2entity.Read().Raw(iCell, size_t(j));
            CHECK(gEdge != UnInitIndex);
        }
    }
}

static void checkFrames(const NativeInterpolation &result, const RelationReadAccess &cone,
                        const RelationPeriodicBits &bits, const SubEntityQueryPbi &query,
                        DNDS::index nx, DNDS::index ny, DNDS::index nz)
{
    auto parent = result.parent2entity.Read();
    const auto entity = result.entity2node.Read(), support = result.entity2parent.Read();
    REQUIRE(result.entity2nodePbi.has_value());
    REQUIRE(result.parent2entityPbi.has_value());
    auto entityBits = result.entity2nodePbi->Read();
    auto parentBits = result.parent2entityPbi->Read();
    // A generated entity's stored orientation/frame is its first parent's
    // occurrence, including when that parent is a source ghost.
    for (DNDS::index row = 0; row < entity.NumOwned(); ++row)
    {
        const auto first = cone.Source().GlobalToLocal(support.ReadEntry(row, 0).global);
        REQUIRE(first.has_value());
        bool found = false;
        for (size_t sub = 0; sub < parent.RowSize(*first); ++sub)
        {
            if (parent.ReadEntry(*first, sub).global != entity.Source().LocalToGlobal(row))
                continue;
            const auto desc = query.describe(*first, int(sub));
            std::vector<DNDS::index> expected(size_t(desc.nNodes));
            std::vector<NodePeriodicBits> expectedBits(size_t(desc.nNodes));
            query.extractNodes(*first, int(sub), [&](int slot)
                               { return cone.ReadEntry(*first, size_t(slot)).global; }, expected.data());
            query.extractPbi(*first, int(sub), [&](int slot)
                             { return bits(*first, size_t(slot)); }, expectedBits.data());
            REQUIRE(entity.RowSize(row) == expected.size());
            for (size_t slot = 0; slot < expected.size(); ++slot)
            {
                CHECK(entity.ReadEntry(row, slot).global == expected[slot]);
                CHECK(entityBits.View()(size_t(row), slot) == expectedBits[slot]);
            }
            found = true;
            break;
        }
        CHECK(found);
    }
    std::set<DNDS::index> remote;
    for (DNDS::index row = 0; row < parent.NumOwned(); ++row)
        for (size_t slot = 0; slot < parent.RowSize(row); ++slot)
        {
            const auto global = parent.ReadEntry(row, slot).global;
            if (!entity.Source().GlobalToLocal(global))
                remote.insert(global);
        }
    auto entities = entity.Source().WithGhosts({remote.begin(), remote.end()});
    NativeRelation owned(entity.Kind(), entities, entity.Target(), result.entity2node.Owned());
    RelationGhostRoute route(entities);
    auto full = owned.WithGhostRows(route);
    auto ghostBits = route.TransferRows(entities, *result.entity2nodePbi,
                                        R::ElementSchema(NodePeriodicBits::CommType(), sizeof(NodePeriodicBits), "Geom.NodePeriodicBits", R::ElementSchema::TypeIdentity<NodePeriodicBits>()));
    const auto fullRead = full.Read();
    auto ghostRead = ghostBits.Read();
    auto point = [=](DNDS::index node, uint8_t frame)
    {
        tPoint p{double(node / (ny * nz)), double((node / nz) % ny), double(node % nz)};
        if (frame & 1)
            p[0] += nx;
        if (frame & 2)
            p[1] += ny;
        if (frame & 4)
            p[2] += nz;
        return p;
    };
    // Check every owned occurrence, including generated entities owned by
    // another rank. This also verifies the incidence-frame XOR physically.
    for (DNDS::index row = 0; row < parent.NumOwned(); ++row)
        for (size_t sub = 0; sub < parent.RowSize(row); ++sub)
        {
            const auto desc = query.describe(row, int(sub));
            std::vector<DNDS::index> expected(size_t(desc.nNodes));
            std::vector<NodePeriodicBits> frames(size_t(desc.nNodes));
            query.extractNodes(row, int(sub), [&](int slot)
                               { return cone.ReadEntry(row, size_t(slot)).global; }, expected.data());
            query.extractPbi(row, int(sub), [&](int slot)
                             { return bits(row, size_t(slot)); }, frames.data());
            const auto local = entities.GlobalToLocal(parent.ReadEntry(row, sub).global);
            REQUIRE(local.has_value());
            const uint8_t incidence = uint8_t(parentBits.View()(size_t(row), sub));
            REQUIRE(fullRead.RowSize(*local) == expected.size());
            for (size_t slot = 0; slot < expected.size(); ++slot)
            {
                bool matched = false;
                for (size_t candidate = 0; candidate < fullRead.RowSize(*local); ++candidate)
                {
                    if (fullRead.ReadEntry(*local, candidate).global != expected[slot])
                        continue;
                    const uint8_t stored = *local < entities.NumOwned() ? uint8_t(entityBits.View()(size_t(*local), candidate))
                                                                        : uint8_t(ghostRead.View()(size_t(*local - entities.NumOwned()), candidate));
                    if ((stored ^ incidence) != uint8_t(frames[slot]))
                        continue;
                    std::vector<DNDS::index> cornerSlots(size_t(desc.nNodes));
                    query.extractNodes(row, int(sub), [](int corner)
                                       { return DNDS::index(corner); }, cornerSlots.data());
                    const int dx[]{0, 1, 1, 0, 0, 1, 1, 0};
                    const int dy[]{0, 0, 1, 1, 0, 0, 1, 1};
                    const int dz[]{0, 0, 0, 0, 1, 1, 1, 1};
                    const auto parentGlobal = cone.Source().LocalToGlobal(row);
                    const auto corner = cornerSlots[slot];
                    const tPoint physicalExpected{double(parentGlobal / (ny * nz) + dx[corner]),
                                                  double((parentGlobal / nz) % ny + dy[corner]),
                                                  double(parentGlobal % nz + dz[corner])};
                    CHECK(physicalExpected.isApprox(point(expected[slot], stored ^ incidence), 0));
                    matched = true;
                    break;
                }
                CHECK(matched);
            }
        }
}

TEST_CASE("InterpolateGlobal: 4x4x4 distributed hex faces (triply-periodic)")
{
    const DNDS::index N = 4;
    auto mesh = NativeStructuredInterpolation::BuildHex(g_mpi, N, 7);
    auto cone = mesh.cone.Read();
    RelationPeriodicBits bits(cone, mesh.ownedBits, mesh.ghostBits);
    for (const bool edges : {false, true})
    {
        auto query = queryFor(mesh.types, edges);
        auto result = NativeConnectivity::Interpolate(cone, edges ? EntityKind::Edge : EntityKind::Face, query, owner(), &bits);
        CHECK(result.entities.NumGlobal() == 3 * g_mpi.size * N * N * N);
        auto support = result.entity2parent.Read();
        for (DNDS::index row = 0; row < support.NumOwned(); ++row)
            CHECK(support.RowSize(row) == size_t(edges ? 4 : 2));
        auto parent = result.parent2entity.Read();
        for (DNDS::index row = 0; row < parent.NumOwned(); ++row)
            CHECK(parent.RowSize(row) == size_t(edges ? 12 : 6));
        checkFrames(result, cone, bits, query, g_mpi.size * N, N, N);
    }
}

TEST_CASE("Regression: Interpolate matches InterpolateFace on UniformSquare_10")
{
    const auto path = (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "data/mesh/UniformSquare_10.cgns");
    auto state = ReadNativeCGNS(g_mpi, path.string(), 2, FBC_Name_2_ID_Default, 0, {1, true});
    state = NativeMeshTopology::Build(*state, g_mpi);
    auto access = state->Snapshot(R::Device::Host()).AcquireRead();
    const auto view = access.HostView();
    for (DNDS::index cell = 0; cell < view.NumCell(); ++cell)
    {
        MeshQueryCoords points;
        view.GetCoordsOnCell(cell, points);
        std::set<std::set<std::pair<DNDS::real, DNDS::real>>> expected, actual;
        // Independent Quad4 polygon edges; no Element face table here.
        for (int side = 0; side < 4; ++side)
            expected.insert(std::set<std::pair<DNDS::real, DNDS::real>>{{points(0, side), points(1, side)}, {points(0, (side + 1) % 4), points(1, (side + 1) % 4)}});
        for (const auto face : view.CellFaces(cell))
        {
            MeshQueryCoords facePoints;
            view.GetCoordsOnFace(face, facePoints);
            REQUIRE(facePoints.cols() == 2);
            actual.insert(std::set<std::pair<DNDS::real, DNDS::real>>{{facePoints(0, 0), facePoints(1, 0)}, {facePoints(0, 1), facePoints(1, 1)}});
        }
        CHECK(actual == expected);
    }
}

TEST_CASE("Interpolate: 2x2 periodic quad mesh — collaborating check required")
{
    auto nodes = EntitySpace::CreateOwned(EntityKind::Node, g_mpi, g_mpi.rank == 0 ? 4 : 0);
    auto cells = nodes.CreateSibling(EntityKind::Cell, g_mpi.rank == 0 ? 4 : 0);
    const int dx[]{0, 1, 1, 0}, dy[]{0, 0, 1, 1};
    std::vector<std::vector<DNDS::index>> rows;
    std::vector<std::vector<NodePeriodicBits>> frames;
    if (g_mpi.rank == 0)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
            {
                rows.emplace_back(4);
                frames.emplace_back(4);
                for (int slot = 0; slot < 4; ++slot)
                {
                    const int px = x + dx[slot], py = y + dy[slot];
                    rows.back()[size_t(slot)] = (py % 2) * 2 + px % 2;
                    frames.back()[size_t(slot)] = NodePeriodicBits{uint8_t((px == 2 ? 1 : 0) | (py == 2 ? 2 : 0))};
                }
            }
    NativeRelation cone(Adj::Cell2Node, cells, nodes, NativePeriodicFixtures::Rows(rows));
    auto read = cone.Read();
    RelationPeriodicBits bits(read, NativePeriodicFixtures::Rows(frames));
    auto query = queryFor(std::vector<Elem::ElemType>(size_t(cells.NumOwned()), Elem::Quad4), false);
    auto falseMerge = NativeConnectivity::Interpolate(read, EntityKind::Face, query, owner());
    bool multiParent = false;
    for (DNDS::index face = 0; face < falseMerge.entities.NumOwned(); ++face)
        multiParent |= falseMerge.entity2parent.Read().RowSize(face) > 2;
    int observed = multiParent;
    MPI_Allreduce(MPI_IN_PLACE, &observed, 1, MPI_INT, MPI_MAX, g_mpi.comm);
    CHECK(observed == 1);
    auto result = NativeConnectivity::Interpolate(read, EntityKind::Face, query, owner(), &bits);
    CHECK(result.entities.NumGlobal() == 8);
    std::map<std::set<DNDS::index>, int> pairCount;
    const auto support = result.entity2parent.Read();
    for (DNDS::index face = 0; face < support.NumOwned(); ++face)
    {
        REQUIRE(support.RowSize(face) == 2);
        const auto row = support.GlobalRow(face);
        CHECK(row[0] != row[1]);
        pairCount[{row[0], row[1]}]++;
    }
    if (g_mpi.rank == 0)
    {
        CHECK(pairCount[{0, 1}] == 2);
        CHECK(pairCount[{0, 2}] == 2);
        CHECK(pairCount[{1, 3}] == 2);
        CHECK(pairCount[{2, 3}] == 2);
        CHECK(pairCount.count({0, 3}) == 0);
        CHECK(pairCount.count({1, 2}) == 0);
    }
    for (DNDS::index cell = 0; cell < cells.NumOwned(); ++cell)
        CHECK(result.parent2entity.Read().RowSize(cell) == 4);
}

TEST_CASE("Periodic 2x2x2: face interpolation (3D, collaborating check)")
{
    auto mesh = NativeStructuredInterpolation::BuildHex(g_mpi, 2, 7, true);
    auto read = mesh.cone.Read();
    RelationPeriodicBits bits(read, mesh.ownedBits, mesh.ghostBits);
    auto query = queryFor(mesh.types, false);
    auto falseMerge = NativeConnectivity::Interpolate(read, EntityKind::Face, query, owner());
    if (g_mpi.rank == 0)
    {
        bool hasMultiParent = false;
        for (DNDS::index row = 0; row < falseMerge.entities.NumOwned(); ++row)
            hasMultiParent |= falseMerge.entity2parent.Read().RowSize(row) > 2;
        CHECK(hasMultiParent);
    }
    // With collaborating check: expect 24 faces, all internal
    {
        auto q = queryFor(mesh.types, false);
        auto res = NativeConnectivity::Interpolate(read, EntityKind::Face, q, owner(), &bits);
        if (g_mpi.rank == 0)
        {
            CHECK(res.entities.NumGlobal() == 24);

            // All faces internal (both parents set)
            int nInternal = 0;
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
                if (res.entity2parent.Read().RowSize(i) >= 2)
                    nInternal++;
            CHECK(nInternal == 24);

            // No entity should have >2 parents (collaborating check prevents false merges)
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
                CHECK(res.entity2parent.Read().RowSize(i) <= 2);

            // Each cell references 6 faces
            for (DNDS::index i = 0; i < 8; i++)
            {
                CAPTURE(i);
                CHECK(res.parent2entity.Read().RowSize(i) == 6);
            }

            // All faces are Quad4
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
                CHECK(res.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Quad4);

            // No face connects a cell to itself
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            {
                REQUIRE(res.entity2parent.Read().RowSize(i) >= 2);
                auto p0 = res.entity2parent.Read().Raw(i, size_t(0));
                auto p1 = res.entity2parent.Read().Raw(i, size_t(1));
                CHECK(p0 != p1);
            }

            // Each cell has exactly 3 face-neighbors (X±, Y±, Z± — but periodic
            // means X- wraps to X+, so each cell is face-adjacent to 6 cells?
            // No: 2×2×2 periodic hex, each cell has 6 faces, each face is shared
            // with exactly 1 other cell. Each cell has at most 6 neighbors, but
            // some neighbors repeat (e.g., cell 0's +X neighbor is cell 1,
            // cell 0's -X neighbor is also cell 1 via periodicity).
            // Actually: cell 0 at (0,0,0), +X neighbor = cell 1 at (1,0,0),
            // -X neighbor = cell 1 via X-periodic wrap. So cell 0 and cell 1
            // share 2 faces (both X-faces). Similarly Y→cell2, Z→cell4.
            // So each cell has 3 distinct face-neighbors, each sharing 2 faces.
            std::map<DNDS::index, std::set<DNDS::index>> cellFN;
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            {
                REQUIRE(res.entity2parent.Read().RowSize(i) >= 2);
                auto p0 = res.entity2parent.Read().Raw(i, size_t(0));
                auto p1 = res.entity2parent.Read().Raw(i, size_t(1));
                cellFN[p0].insert(p1);
                cellFN[p1].insert(p0);
            }
            for (DNDS::index i = 0; i < 8; i++)
            {
                CAPTURE(i);
                CHECK(cellFN[i].size() == 3);
            }
        }
    }
}

TEST_CASE("Periodic 2x2x2: edge interpolation (3D, collaborating check)")
{
    auto mesh = NativeStructuredInterpolation::BuildHex(g_mpi, 2, 7, true);
    auto read = mesh.cone.Read();
    RelationPeriodicBits bits(read, mesh.ownedBits, mesh.ghostBits);
    auto query = queryFor(mesh.types, true);
    auto falseMerge = NativeConnectivity::Interpolate(read, EntityKind::Edge, query, owner());
    if (g_mpi.rank == 0)
    {
        bool hasMultiParent = false;
        for (DNDS::index row = 0; row < falseMerge.entities.NumOwned(); ++row)
            hasMultiParent |= falseMerge.entity2parent.Read().RowSize(row) > 2;
        CHECK(hasMultiParent);
    }
    // With collaborating check: expect 24 edges, all shared by 4 cells
    {
        auto q = queryFor(mesh.types, true);
        auto res = NativeConnectivity::Interpolate(read, EntityKind::Edge, q, owner(), &bits);
        if (g_mpi.rank == 0)
        {
            CHECK(res.entities.NumGlobal() == 24);

            // All edges are Line2
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
                CHECK(res.entityElemInfo.Read().View()(size_t(i), 0).getElemType() == Elem::Line2);

            // Each cell references 12 edges
            for (DNDS::index i = 0; i < 8; i++)
            {
                CAPTURE(i);
                CHECK(res.parent2entity.Read().RowSize(i) == 12);
            }

            // With variable-width entity2parent, edges shared by 4 cells now store
            // all 4 parents. Verify via entity2parent RowSize.
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            {
                CAPTURE(i);
                CHECK(res.entity2parent.Read().RowSize(i) == 4);
            }

            // Also verify via parent2entity reference counting.
            std::vector<int> edgeRefCount(res.entities.NumGlobal(), 0);
            for (DNDS::index iCell = 0; iCell < 8; iCell++)
                for (DNDS::rowsize j = 0; j < res.parent2entity.Read().RowSize(iCell); j++)
                    edgeRefCount[res.parent2entity.Read().Raw(iCell, size_t(j))]++;

            // Each edge should be referenced by exactly 4 cells
            for (DNDS::index i = 0; i < res.entities.NumGlobal(); i++)
            {
                CAPTURE(i);
                CHECK(edgeRefCount[i] == 4);
            }

            // Euler characteristic: V - E + F = 8 - 24 + 24 = 8
            // (For the cell complex: V - E + F - C = 8 - 24 + 24 - 8 = 0 for T³)
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    g_mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
