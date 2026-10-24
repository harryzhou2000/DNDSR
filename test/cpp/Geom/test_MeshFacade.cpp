#ifndef DNDS_MESH_FACADE_TEST_EMBEDDED
#    define DOCTEST_CONFIG_IMPLEMENT
#endif
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/EntitySpace.hpp"
#include "MeshViewKernel.hpp"
#include <memory>
#include <optional>

namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;
using I = DNDS::index;

static DNDS::MPIInfo FacadeMPI()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    return mpi;
}

template <class Pair>
static void PublishRows(Pair &pair, const std::string &name, DNDS::MPIInfo mpi,
                        const std::vector<std::vector<typename Pair::t_arr::value_type>> &rows)
{
    pair.InitPair(name, mpi);
    pair.father->Resize(I(rows.size()), [&](I row)
                        { return DNDS::rowsize(rows[size_t(row)].size()); });
    pair.son->Resize(0, [](I)
                     { return DNDS::rowsize(0); });
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            (*pair.father)(I(i), DNDS::rowsize(j)) = rows[i][j];
}

template <class Pair>
static void PublishFixed(Pair &pair, const std::string &name, DNDS::MPIInfo mpi,
                         const std::vector<std::vector<I>> &rows)
{
    pair.InitPair(name, mpi);
    pair.father->Resize(I(rows.size()));
    pair.son->Resize(0);
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            (*pair.father)(I(i), DNDS::rowsize(j)) = rows[i][j];
}

static void PublishInfo(G::tElemInfoArrayPair &pair, const std::string &name, DNDS::MPIInfo mpi,
                        I count, G::Elem::ElemType type, G::t_index zone)
{
    pair.InitPair(name, G::ElemInfo::CommType(), G::ElemInfo::CommMult(), mpi);
    pair.father->Resize(count);
    pair.son->Resize(0);
    for (I i = 0; i < count; ++i)
    {
        pair(i, 0).setElemType(type);
        pair(i, 0).zone = zone;
    }
}

// One owned cell, no ghost rows. The periodic quad has one self-periodic face
// occurring twice in its ordered face slots, plus two physical boundary faces.
static std::unique_ptr<G::UnstructuredMesh> FacadeFixture(bool periodic = false, bool empty = false)
{
    const auto mpi = FacadeMPI();
    auto mesh = std::make_unique<G::UnstructuredMesh>(mpi, 2);
    const I nodes = empty ? 0 : periodic ? 4
                                         : 3;
    const I cells = empty ? 0 : 1;
    const I faces = empty ? 0 : 3;
    const I boundaries = empty ? 0 : periodic ? 2
                                              : 3;
    mesh->isPeriodic = periodic;
    mesh->coords.InitPair("coords", mpi);
    mesh->coords.father->Resize(nodes);
    mesh->coords.son->Resize(0);
    mesh->nodeWallDist.InitPair("wall", mpi);
    mesh->nodeWallDist.father->Resize(nodes);
    mesh->nodeWallDist.son->Resize(0);
    const G::tPoint points[]{G::tPoint{0, 0, 0}, G::tPoint{1, 0, 0}, G::tPoint{1, 1, 0}, G::tPoint{0, 1, 0}};
    for (I node = 0; node < nodes; ++node)
    {
        mesh->coords[node] = points[node];
        mesh->nodeWallDist[node] = G::tPoint{1 + double(node), 2, 0};
    }
    std::vector<std::vector<I>> cellNodes, faceNodes, boundaryNodes, cellFaces, cellCells, nodeCells, nodeBoundaries;
    if (!empty)
    {
        cellNodes = periodic ? std::vector<std::vector<I>>{{0, 1, 2, 3}} : std::vector<std::vector<I>>{{0, 1, 2}};
        faceNodes = periodic ? std::vector<std::vector<I>>{{0, 3}, {0, 1}, {3, 2}} : std::vector<std::vector<I>>{{0, 1}, {1, 2}, {2, 0}};
        boundaryNodes = periodic ? std::vector<std::vector<I>>{{0, 1}, {3, 2}} : faceNodes;
        cellFaces = periodic ? std::vector<std::vector<I>>{{0, 1, 0, 2}} : std::vector<std::vector<I>>{{0, 1, 2}};
        cellCells = periodic ? std::vector<std::vector<I>>{{0}} : std::vector<std::vector<I>>{{}};
        nodeCells.resize(size_t(nodes), {0});
        nodeBoundaries = periodic ? std::vector<std::vector<I>>{{0}, {0}, {1}, {1}} : std::vector<std::vector<I>>{{0, 2}, {0, 1}, {1, 2}};
    }
    PublishRows(mesh->cell2node, "cellNodes", mpi, cellNodes);
    PublishRows(mesh->bnd2node, "bndNodes", mpi, boundaryNodes);
    PublishRows(mesh->cell2cell, "cellCells", mpi, cellCells);
    PublishRows(mesh->cell2face, "cellFaces", mpi, cellFaces);
    PublishRows(mesh->face2node, "faceNodes", mpi, faceNodes);
    PublishRows(mesh->node2cell, "nodeCells", mpi, nodeCells);
    PublishRows(mesh->node2bnd, "nodeBnd", mpi, nodeBoundaries);
    std::vector<std::vector<I>> bndCells(size_t(boundaries), {0, DNDS::UnInitIndex});
    std::vector<std::vector<I>> faceCells(size_t(faces), {0, DNDS::UnInitIndex});
    if (periodic && !empty)
        faceCells[0] = {0, 0};
    PublishFixed(mesh->bnd2cell, "bndCells", mpi, bndCells);
    PublishFixed(mesh->face2cell, "faceCells", mpi, faceCells);
    std::vector<std::vector<I>> faceBoundaries, boundaryFaces;
    if (!empty)
    {
        faceBoundaries = periodic ? std::vector<std::vector<I>>{{DNDS::UnInitIndex}, {0}, {1}} : std::vector<std::vector<I>>{{0}, {1}, {2}};
        boundaryFaces = periodic ? std::vector<std::vector<I>>{{1}, {2}} : faceBoundaries;
    }
    PublishFixed(mesh->face2bnd, "faceBnd", mpi, faceBoundaries);
    PublishFixed(mesh->bnd2face, "bndFace", mpi, boundaryFaces);
    PublishRows(mesh->cell2edge, "cellEdges", mpi, cellFaces);
    PublishRows(mesh->edge2node, "edgeNodes", mpi, faceNodes);
    PublishRows(mesh->edge2cell, "edgeCells", mpi, std::vector<std::vector<I>>(size_t(faces), {0}));
    PublishRows(mesh->cell2cellFace, "cellFaceCells", mpi, empty ? std::vector<std::vector<I>>{} : periodic ? std::vector<std::vector<I>>{{0, DNDS::UnInitIndex, 0, DNDS::UnInitIndex}}
                                                                                                            : std::vector<std::vector<I>>{{DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex}});
    PublishInfo(mesh->cellElemInfo, "cellInfo", mpi, cells, periodic ? G::Elem::Quad4 : G::Elem::Tri3, 11);
    PublishInfo(mesh->bndElemInfo, "bndInfo", mpi, boundaries, G::Elem::Line2, 7);
    PublishInfo(mesh->faceElemInfo, "faceInfo", mpi, faces, G::Elem::Line2, 7);
    PublishInfo(mesh->edgeElemInfo, "edgeInfo", mpi, faces, G::Elem::Line2, 9);
    auto mappingFor = [&](I count)
    {
        auto global = std::make_shared<DNDS::GlobalOffsetsMapping>();
        global->setMPIAlignBcast(mpi, count);
        return G::AdjIndexInfo::makeFatherOnlyMapping(global, count, mpi);
    };
    auto nodeMap = mappingFor(nodes), cellMap = mappingFor(cells), faceMap = mappingFor(faces),
         boundaryMap = mappingFor(boundaries), edgeMap = mappingFor(faces);
    auto canonical = [&](auto &pair, const auto &map)
    {
        pair.TransAttach();
        pair.trans.createFatherGlobalMapping();
        pair.trans.pLGhostMapping = map;
    };
    canonical(mesh->coords, nodeMap);
    canonical(mesh->cell2node, cellMap);
    canonical(mesh->face2node, faceMap);
    canonical(mesh->bnd2node, boundaryMap);
    canonical(mesh->edge2node, edgeMap);
    auto local = [&](auto &pair, const auto &map)
    { pair.idx.wireTargetMapping(map); pair.idx.markLocal(); };
    local(mesh->cell2node, nodeMap);
    local(mesh->bnd2node, nodeMap);
    local(mesh->bnd2cell, cellMap);
    local(mesh->cell2cell, cellMap);
    local(mesh->cell2face, faceMap);
    local(mesh->face2node, nodeMap);
    local(mesh->face2cell, cellMap);
    local(mesh->face2bnd, boundaryMap);
    local(mesh->bnd2face, faceMap);
    local(mesh->node2cell, cellMap);
    local(mesh->node2bnd, boundaryMap);
    local(mesh->cell2edge, edgeMap);
    local(mesh->edge2node, nodeMap);
    local(mesh->edge2cell, cellMap);
    local(mesh->cell2cellFace, cellMap);
    mesh->adjPrimaryState = mesh->adjFacialState = mesh->adjC2FState = mesh->adjN2CBState = mesh->adjEdgeState = mesh->adjC2CFaceState = G::Adj_PointToLocal;
    if (periodic)
    {
        auto bitsFor = [](const std::vector<std::vector<I>> &rows)
        {
            std::vector<std::vector<G::NodePeriodicBits>> result;
            for (const auto &row : rows)
                result.emplace_back(row.size(), G::NodePeriodicBits{});
            return result;
        };
        auto cellBits = bitsFor(cellNodes), faceBits = bitsFor(faceNodes), slotBits = bitsFor(cellFaces);
        if (!empty)
        {
            cellBits[0][2] = G::nodePB1;
            faceBits[0][1] = G::nodePB1;
            slotBits[0][2] = G::nodePB1;
            mesh->faceElemInfo(0, 0).zone = G::BC_ID_PERIODIC_1;
        }
        PublishRows(mesh->cell2nodePbi, "cellPbi", mpi, cellBits);
        PublishRows(mesh->bnd2nodePbi, "bndPbi", mpi, bitsFor(boundaryNodes));
        PublishRows(mesh->face2nodePbi, "facePbi", mpi, faceBits);
        PublishRows(mesh->cell2facePbi, "slotPbi", mpi, slotBits);
        PublishRows(mesh->cell2edgePbi, "edgeSlotPbi", mpi, slotBits);
        PublishRows(mesh->edge2nodePbi, "edgePbi", mpi, faceBits);
        mesh->periodicInfo.translation[1].map() = G::tPoint{3, 0, 0};
        mesh->periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    }
    // Published companions use the same source-row correspondence as production
    // BuildGhostPrimary/Face/N2CB; target adjacency maps remain independent.
    auto borrowSource = [](auto &pair, const auto &canonicalPair)
    {
        pair.TransAttach();
        pair.trans.BorrowGGIndexing(canonicalPair.trans);
    };
    borrowSource(mesh->cell2cell, mesh->cell2node);
    borrowSource(mesh->cellElemInfo, mesh->cell2node);
    borrowSource(mesh->bnd2cell, mesh->bnd2node);
    borrowSource(mesh->bndElemInfo, mesh->bnd2node);
    borrowSource(mesh->node2cell, mesh->coords);
    borrowSource(mesh->node2bnd, mesh->coords);
    borrowSource(mesh->nodeWallDist, mesh->coords);
    borrowSource(mesh->cell2face, mesh->cell2node);
    borrowSource(mesh->face2cell, mesh->face2node);
    borrowSource(mesh->face2bnd, mesh->face2node);
    borrowSource(mesh->faceElemInfo, mesh->face2node);
    borrowSource(mesh->bnd2face, mesh->bnd2node);
    borrowSource(mesh->cell2edge, mesh->cell2node);
    borrowSource(mesh->edge2cell, mesh->edge2node);
    borrowSource(mesh->edgeElemInfo, mesh->edge2node);
    borrowSource(mesh->cell2cellFace, mesh->cell2node);
    if (periodic)
    {
        borrowSource(mesh->cell2nodePbi, mesh->cell2node);
        borrowSource(mesh->bnd2nodePbi, mesh->bnd2node);
        borrowSource(mesh->cell2facePbi, mesh->cell2node);
        borrowSource(mesh->face2nodePbi, mesh->face2node);
        borrowSource(mesh->cell2edgePbi, mesh->cell2node);
        borrowSource(mesh->edge2nodePbi, mesh->edge2node);
    }
    return mesh;
}

static G::MeshReadRequirements AllFacadeRequirements()
{
    auto result = G::MeshReadRequirements::CellFlux();
    result.inverse = result.edges = result.cellFaceNeighbors = result.wallDistance = true;
    return result;
}

[[maybe_unused]] static void CheckProbeEqual(const G::Test::MeshViewProbeResult &actual, const G::Test::MeshViewProbeResult &expected)
{
    for (int i = 0; i < 15; ++i)
        CHECK(actual.counts[i] == expected.counts[i]);
    CHECK(actual.relationChecksum == expected.relationChecksum);
    CHECK(actual.elementChecksum == expected.elementChecksum);
    CHECK(actual.faceSideChecksum == expected.faceSideChecksum);
    CHECK(actual.coordinateChecksum == doctest::Approx(expected.coordinateChecksum));
    CHECK(actual.wallDistanceChecksum == doctest::Approx(expected.wallDistanceChecksum));
    CHECK(actual.periodicChecksum == doctest::Approx(expected.periodicChecksum));
    CHECK(actual.cellBatchChecksum == doctest::Approx(expected.cellBatchChecksum));
    CHECK(actual.cellExternalChecksum == doctest::Approx(expected.cellExternalChecksum));
    CHECK(actual.cellNodeChecksum == doctest::Approx(expected.cellNodeChecksum));
    CHECK(actual.faceBatchChecksum == doctest::Approx(expected.faceBatchChecksum));
    CHECK(actual.faceExternalChecksum == doctest::Approx(expected.faceExternalChecksum));
    CHECK(actual.faceNodeChecksum == doctest::Approx(expected.faceNodeChecksum));
    for (int d = 0; d < 3; ++d)
    {
        CHECK(actual.transformedCellSample[d] == doctest::Approx(expected.transformedCellSample[d]));
        CHECK(actual.transformedFaceSample[d] == doctest::Approx(expected.transformedFaceSample[d]));
    }
}

TEST_CASE("Mesh facade host acquisition exposes exact local rows and legacy helper semantics")
{
    for (bool periodic : {false, true})
    {
        auto mesh = FacadeFixture(periodic);
        auto access = mesh->AcquireRead(AllFacadeRequirements());
        const auto view = access.HostView();
        CHECK(view.Dimension() == 2);
        CHECK(view.NumCell() == 1);
        CHECK(view.NumNode() == (periodic ? 4 : 3));
        CHECK(view.NumCellGhost() == 0);
        CHECK(view.NumFace() == 3);
        CHECK(view.NumBnd() == (periodic ? 2 : 3));
        CHECK(view.NumEdge() == 3);
        CHECK(view.CellNodes(0)[2] == 2);
        CHECK(view.Coord(1)[0] == 1);
        CHECK(view.GetCellZone(0) == 11);
        CHECK(view.GetCellElement(0).GetNumNodes() == (periodic ? 4 : 3));
        for (I cell = 0; cell < view.NumCellProc(); ++cell)
        {
            G::tSmallCoords expected, actual;
            mesh->GetCoordsOnCell(cell, expected);
            view.GetCoordsOnCell(cell, actual);
            CHECK(actual.isApprox(expected));
            G::MeshQueryCoords bounded;
            view.GetCoordsOnCell(cell, bounded);
            CHECK(bounded.isApprox(expected));
            G::MeshQueryCoords externalCoords;
            view.GetCoordsOnCell(cell, externalCoords, view);
            CHECK(externalCoords.isApprox(expected));
            for (DNDS::rowsize slot = 0; slot < mesh->cell2face.RowSize(cell); ++slot)
            {
                const auto face = mesh->cell2face(cell, slot);
                CHECK(view.CellIsFaceBack(cell, face, slot) == mesh->CellIsFaceBack(cell, face, slot));
                CHECK(view.CellFaceOther(cell, face, slot) == mesh->CellFaceOther(cell, face, slot));
            }
        }
        for (I face = 0; face < view.NumFaceProc(); ++face)
        {
            G::tSmallCoords expected, actual;
            mesh->GetCoordsOnFace(face, expected);
            view.GetCoordsOnFace(face, actual);
            CHECK(actual.isApprox(expected));
            G::MeshQueryCoords bounded;
            view.GetCoordsOnFace(face, bounded);
            CHECK(bounded.isApprox(expected));
            G::MeshQueryCoords externalCoords;
            view.GetCoordsOnFace(face, externalCoords, view);
            CHECK(externalCoords.isApprox(expected));
            CHECK(view.GetCoordNodeOnFace(face, 1).isApprox(mesh->GetCoordNodeOnFace(face, 1)));
            CHECK(view.GetCoordWallDistOnFace(face, 1).isApprox(mesh->GetCoordWallDistOnFace(face, 1)));
        }
        if (periodic)
        {
            CHECK(view.CellFaces(0)[0] == view.CellFaces(0)[2]);
            CHECK(view.CellIsFaceBack(0, 0, 0));
            CHECK_FALSE(view.CellIsFaceBack(0, 0, 2));
            CHECK(view.CellFaceOther(0, 0, 2) == 0);
            CHECK(view.GetCoordNodeOnCell(0, 2).isApprox(G::tPoint{2, 1, 0}));
            CHECK(view.GetCoordWallDistOnCell(0, 2).isApprox(G::tPoint{-2, 3, 0}));
            CHECK(view.CellOtherCellPeriodicHandle(0, 0, []
                                                   { return 1; }, []
                                                   { return 2; }, []
                                                   { return 3; }) == 2);
            CHECK(view.CellOtherCellPeriodicHandle(0, 1, []
                                                   { return 1; }, []
                                                   { return 2; }, []
                                                   { return 3; }) == 1);
        }
        const auto probe = G::Test::MeshViewProbe(view);
        CHECK(probe.coordinateChecksum == (periodic ? 19 : 11));
        CHECK_THROWS(access.DeviceView());
        CHECK_THROWS(mesh->coords.father->HostWriteData());
        CHECK_THROWS(mesh->cell2node.father->HostWriteData());
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
        CHECK_FALSE(mesh->versions.CanMutate());
    }
}

TEST_CASE("Mesh facade scopes retain storage after mesh deletion and reject invalid acquisition")
{
    std::optional<G::MeshReadAccess> retained;
    {
        auto mesh = FacadeFixture();
        retained.emplace(mesh->AcquireRead(AllFacadeRequirements()));
        CHECK_FALSE(mesh->versions.CanMutate());
    }
    CHECK(retained->HostView().CellNodes(0)[2] == 2);
    CHECK(retained->HostView().Coord(2)[1] == 1);
    retained.reset();
    auto empty = FacadeFixture(false, true);
    auto access = empty->AcquireRead(AllFacadeRequirements());
    CHECK(access.HostView().NumNodeProc() == 0);
    CHECK(access.HostView().NumCellProc() == 0);
    CHECK(G::Test::MeshViewProbe(access.HostView()).relationChecksum == 0);
    CHECK_THROWS(empty->AcquireRead(G::MeshReadRequirements::Primary(), {R::Device::Host(), reinterpret_cast<void *>(1)}));

    auto invalid = FacadeFixture();
    invalid->cell2node.idx = G::AdjIndexInfo{};
    invalid->cell2node.idx.markGlobal();
    CHECK_THROWS(invalid->AcquireRead());
    CHECK(invalid->versions.CanMutate());
    auto wrongShape = FacadeFixture();
    wrongShape->cellElemInfo.father->Resize(0);
    CHECK_THROWS(wrongShape->AcquireRead());
    CHECK(wrongShape->versions.CanMutate());
    auto badTarget = FacadeFixture();
    badTarget->cell2node(0, 1) = 99;
    CHECK_THROWS(badTarget->AcquireRead());
    CHECK(badTarget->versions.CanMutate());
    auto missingSourceRows = FacadeFixture();
    missingSourceRows->bnd2cell.father->Resize(0);
    CHECK_THROWS(missingSourceRows->AcquireRead());
    auto badPeriodicShape = FacadeFixture(true);
    badPeriodicShape->cell2nodePbi.father->Resize(1, [](I)
                                                  { return DNDS::rowsize(1); });
    CHECK_THROWS(badPeriodicShape->AcquireRead());
    auto primaryOnly = FacadeFixture();
    auto primary = primaryOnly->AcquireRead();
    CHECK_FALSE(primary.HostView().HasRelation(G::MeshRelation::CellFaces));
    CHECK(primary.HostView().NumFaceProc() == 0);
}

TEST_CASE("Mesh facade rejects distribution mapping and owned closure inconsistencies")
{
    SUBCASE("source owned and ghost split differs despite identical total")
    {
        auto mesh = FacadeFixture();
        mesh->bnd2cell.father->Resize(2);
        mesh->bnd2cell.son->Resize(1);
        CHECK(mesh->bnd2cell.Size() == mesh->bnd2node.Size());
        CHECK_THROWS(mesh->AcquireRead());
        CHECK(mesh->versions.CanMutate());
    }
    SUBCASE("element owned and ghost split differs despite identical total")
    {
        auto mesh = FacadeFixture();
        mesh->cellElemInfo.father->Resize(0);
        mesh->cellElemInfo.son->Resize(1);
        CHECK(mesh->cellElemInfo.Size() == mesh->cell2node.Size());
        CHECK_THROWS(mesh->AcquireRead());
    }
    SUBCASE("target mapping has correct extent but different identity")
    {
        auto mesh = FacadeFixture();
        auto replacement = G::AdjIndexInfo::makeFatherOnlyMapping(mesh->coords.father->pLGlobalMapping, mesh->NumNode(), mesh->mpi);
        mesh->cell2node.idx = G::AdjIndexInfo{};
        mesh->cell2node.idx.wireTargetMapping(replacement);
        mesh->cell2node.idx.markLocal();
        CHECK(replacement != mesh->coords.trans.pLGhostMapping);
        CHECK_THROWS(mesh->AcquireRead());
    }
    SUBCASE("relation source map has equal rows but a stale identity")
    {
        auto mesh = FacadeFixture();
        mesh->cell2cell.trans.pLGhostMapping = std::make_shared<DNDS::OffsetAscendIndexMapping>(*mesh->cell2cell.trans.pLGhostMapping);
        CHECK_THROWS(mesh->AcquireRead());
        CHECK(mesh->versions.CanMutate());
    }
    SUBCASE("element companion source map has equal rows but a stale identity")
    {
        auto mesh = FacadeFixture();
        mesh->cellElemInfo.trans.pLGhostMapping = std::make_shared<DNDS::OffsetAscendIndexMapping>(*mesh->cellElemInfo.trans.pLGhostMapping);
        CHECK_THROWS(mesh->AcquireRead());
        CHECK(mesh->versions.CanMutate());
    }
    SUBCASE("periodic companion source map has equal rows but a stale identity")
    {
        auto mesh = FacadeFixture(true);
        mesh->cell2nodePbi.trans.pLGhostMapping = std::make_shared<DNDS::OffsetAscendIndexMapping>(*mesh->cell2nodePbi.trans.pLGhostMapping);
        CHECK_THROWS(mesh->AcquireRead());
    }
    SUBCASE("element type and incidence widths are validated before dispatch")
    {
        for (auto badType : {G::t_index(G::Elem::UnknownElem), G::t_index(99), G::t_index(G::Elem::Hex27)})
        {
            auto mesh = FacadeFixture();
            mesh->cellElemInfo(0, 0).type = badType;
            CHECK_THROWS(mesh->AcquireRead());
            CHECK(mesh->versions.CanMutate());
        }
        auto mesh = FacadeFixture();
        mesh->cell2face.father->Resize(1, [](I)
                                       { return DNDS::rowsize(2); });
        CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
    }
    SUBCASE("owned cell face is unavailable")
    {
        for (I missing : {I(-4), DNDS::UnInitIndex})
        {
            auto mesh = FacadeFixture();
            mesh->cell2face(0, 0) = missing;
            CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
        }
    }
    SUBCASE("owned face first cell is unavailable")
    {
        for (I missing : {I(-4), DNDS::UnInitIndex})
        {
            auto mesh = FacadeFixture();
            mesh->face2cell(0, 0) = missing;
            CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
        }
    }
    SUBCASE("internal face cannot have external second-cell sentinel")
    {
        auto mesh = FacadeFixture();
        mesh->faceElemInfo(0, 0).zone = G::BC_ID_INTERNAL;
        CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
    }
    SUBCASE("nonperiodic internal face cannot have the same cell on both sides")
    {
        auto mesh = FacadeFixture();
        mesh->faceElemInfo(0, 0).zone = G::BC_ID_INTERNAL;
        mesh->face2cell(0, 1) = mesh->face2cell(0, 0);
        CHECK_FALSE(mesh->isPeriodic);
        CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
        CHECK(mesh->versions.CanMutate());
    }
    SUBCASE("periodic mesh still rejects a self face in an internal or external zone")
    {
        for (auto zone : {G::BC_ID_INTERNAL, G::BC_ID_DEFAULT_WALL})
        {
            auto mesh = FacadeFixture(true);
            CHECK_NOTHROW(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
            mesh->faceElemInfo(0, 0).zone = zone;
            CHECK(mesh->isPeriodic);
            CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
            CHECK(mesh->versions.CanMutate());
        }
    }
    SUBCASE("self-periodic face requires two opposite cell-face sides before publication")
    {
        for (auto sameSide : {G::NodePeriodicBits{}, G::nodePB1})
        {
            auto mesh = FacadeFixture(true);
            mesh->cell2facePbi(0, 0) = sameSide;
            mesh->cell2facePbi(0, 2) = sameSide;
            CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
            CHECK_THROWS(mesh->PublishNative(G::MeshReadRequirements::CellFlux()));
            CHECK_FALSE(mesh->HasNativeState());
            CHECK(mesh->versions.CanMutate());
            mesh->cell2facePbi(0, 0) = G::NodePeriodicBits{};
            mesh->cell2facePbi(0, 2) = G::nodePB2; // Any nonzero side bit retains the existing policy.
            {
                auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
                CHECK(read.HostView().CellIsFaceBack(0, 0, 0));
                CHECK_FALSE(read.HostView().CellIsFaceBack(0, 0, 2));
            }
            CHECK_NOTHROW(mesh->PublishNative(G::MeshReadRequirements::CellFlux()));
        }
        auto missingSide = FacadeFixture(true);
        missingSide->cell2face(0, 2) = 1;
        CHECK_THROWS(missingSide->AcquireRead(G::MeshReadRequirements::CellFlux()));
        CHECK_THROWS(missingSide->PublishNative(G::MeshReadRequirements::CellFlux()));
        CHECK_FALSE(missingSide->HasNativeState());
    }
    SUBCASE("owned cell faces must reciprocally reference the source cell")
    {
        auto mesh = FacadeFixture();
        // Add a second cell with in-range face IDs, but leave the faces pointing
        // only to cell zero. Extents, type widths and mapping identities agree;
        // a range-only acquisition used to publish these inconsistent rows.
        mesh->cell2node.father->Resize(2, [](I)
                                       { return DNDS::rowsize(3); });
        mesh->cell2face.father->Resize(2, [](I)
                                       { return DNDS::rowsize(3); });
        mesh->cell2cell.father->Resize(2, [](I)
                                       { return DNDS::rowsize(0); });
        mesh->cellElemInfo.father->Resize(2);
        for (I cell = 0; cell < 2; ++cell)
        {
            for (DNDS::rowsize slot = 0; slot < 3; ++slot)
            {
                mesh->cell2node(cell, slot) = slot;
                mesh->cell2face(cell, slot) = slot;
            }
            mesh->cellElemInfo(cell, 0).setElemType(G::Elem::Tri3);
            mesh->cellElemInfo(cell, 0).zone = 11;
        }
        mesh->cell2node.TransAttach();
        mesh->cell2node.trans.createFatherGlobalMapping();
        auto cells = G::AdjIndexInfo::makeFatherOnlyMapping(mesh->cell2node.father->pLGlobalMapping, 2, mesh->mpi);
        mesh->cell2node.trans.pLGhostMapping = cells;
        auto borrow = [&](auto &pair)
        {
            pair.TransAttach();
            pair.trans.BorrowGGIndexing(mesh->cell2node.trans);
        };
        borrow(mesh->cell2cell);
        borrow(mesh->cell2face);
        borrow(mesh->cellElemInfo);
        auto wire = [&](auto &pair)
        {
            pair.idx = G::AdjIndexInfo{};
            pair.idx.wireTargetMapping(cells);
            pair.idx.markLocal();
        };
        wire(mesh->cell2cell);
        wire(mesh->bnd2cell);
        wire(mesh->face2cell);
        CHECK_NOTHROW(mesh->AcquireRead());
        CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
        CHECK(mesh->versions.CanMutate());
        for (I face = 0; face < 3; ++face)
        {
            mesh->face2cell(face, 1) = 1;
            mesh->faceElemInfo(face, 0).zone = G::BC_ID_INTERNAL;
        }
        CHECK_NOTHROW(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
    }
    SUBCASE("external sentinel and unavailable inverse support remain valid")
    {
        auto mesh = FacadeFixture();
        mesh->node2cell(0, 0) = -4;
        auto scope = mesh->AcquireRead(AllFacadeRequirements());
        CHECK(scope.HostView().FaceCells(0)[1] == DNDS::UnInitIndex);
        CHECK(scope.HostView().NodeCells(0)[0] == -4);
    }
    SUBCASE("outer ghost face permits unavailable cells until an owned cell needs it")
    {
        auto mesh = FacadeFixture();
        mesh->face2node.son->Resize(1, [](I)
                                    { return DNDS::rowsize(2); });
        (*mesh->face2node.son)(0, 0) = 0;
        (*mesh->face2node.son)(0, 1) = 1;
        mesh->face2cell.son->Resize(1);
        (*mesh->face2cell.son)(0, 0) = -4;
        (*mesh->face2cell.son)(0, 1) = -5;
        mesh->face2bnd.son->Resize(1);
        (*mesh->face2bnd.son)(0, 0) = DNDS::UnInitIndex;
        mesh->faceElemInfo.son->Resize(1);
        (*mesh->faceElemInfo.son)(0, 0).setElemType(G::Elem::Line2);
        (*mesh->faceElemInfo.son)(0, 0).zone = G::BC_ID_INTERNAL;
        // The ghost map includes a self-owned face ID, as allowed by the legacy
        // gather/scatter map. Row 3 is a retained halo copy with incomplete cells.
        std::vector<I> pull{(*mesh->face2node.father->pLGlobalMapping)(mesh->mpi.rank, 0)};
        auto faceMap = std::make_shared<DNDS::OffsetAscendIndexMapping>(
            (*mesh->face2node.father->pLGlobalMapping)(mesh->mpi.rank, 0), 3, pull,
            *mesh->face2node.father->pLGlobalMapping, mesh->mpi);
        mesh->face2node.trans.pLGhostMapping = faceMap;
        mesh->face2cell.trans.pLGhostMapping = faceMap;
        mesh->face2bnd.trans.pLGhostMapping = faceMap;
        mesh->faceElemInfo.trans.pLGhostMapping = faceMap;
        auto rewire = [&](auto &relation)
        {
            relation.idx = G::AdjIndexInfo{};
            relation.idx.wireTargetMapping(faceMap);
            relation.idx.markLocal();
        };
        rewire(mesh->cell2face);
        rewire(mesh->bnd2face);
        {
            auto access = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            CHECK(access.HostView().NumFaceGhost() == 1);
            CHECK(access.HostView().FaceCells(3)[0] == -4);
            CHECK(access.HostView().FaceCells(3)[1] == -5);
        }
        auto allFaces = G::MeshReadRequirements::CellFlux();
        allFaces.allFaceCellsLocal = true;
        CHECK_THROWS(mesh->AcquireRead(allFaces));
        mesh->cell2face(0, 0) = 3;
        CHECK_THROWS(mesh->AcquireRead(G::MeshReadRequirements::CellFlux()));
        CHECK(mesh->versions.CanMutate());
    }
}

TEST_CASE("Mesh host view getters reject moved and sealed scopes")
{
    auto mesh = FacadeFixture();
    auto original = mesh->AcquireRead();
    auto destination = std::move(original);
    CHECK_THROWS(original.HostView());
    destination.Complete(R::Completion::Record(R::ExecutionContext::Host()));
    CHECK_THROWS(destination.HostView());
    destination.Wait();
    CHECK_THROWS(destination.HostView());
    original.Wait();
}

TEST_CASE("Entity space shares legacy indexing with a native route and invalidates tracked layout changes")
{
    const auto mpi = FacadeMPI();
    G::UnstructuredMesh mesh(mpi, 2);
    mesh.cell2node.InitPair("entityCells", mpi);
    mesh.cell2node.father->Resize(2, 0);
    mesh.cell2node.TransAttach();
    mesh.cell2node.trans.createFatherGlobalMapping();
    // Request row 1 from the next rank; a singleton still exercises a self route.
    std::vector<I> pulls{I((mpi.rank + 1) % mpi.size) * 2 + 1};
    mesh.cell2node.trans.createGhostMapping(pulls);
    mesh.cell2node.son->Resize(I(pulls.size()), 0);
    const auto space = mesh.Entities(G::EntityKind::Cell);
    CHECK(space.NumOwned() == 2);
    CHECK(space.NumGhost() == 1);
    CHECK(space.NumGlobal() == 2 * mpi.size);
    CHECK(space.LocalToGlobal(0) == 2 * mpi.rank);
    CHECK(space.LocalToGlobal(2) == pulls[0]);
    CHECK(space.Owner(pulls[0]) == (mpi.rank + 1) % mpi.size);
    CHECK_FALSE(space.GlobalToLocal(-1).has_value());
    CHECK_THROWS(space.LocalToGlobal(-1));
    CHECK_THROWS(space.Owner(space.NumGlobal()));

    DNDS::ArrayPair<DNDS::ParArray<double, 1>> legacyField;
    legacyField.InitPair("field", mpi);
    legacyField.father->Resize(2);
    legacyField.son->Resize(0);
    legacyField.TransAttach();
    G::LegacyFieldBridge::BorrowIndexing(legacyField.trans, space);
    CHECK(legacyField.trans.pLGlobalMapping == mesh.cell2node.trans.pLGlobalMapping);
    CHECK(legacyField.trans.pLGhostMapping == mesh.cell2node.trans.pLGhostMapping);
    legacyField(0, 0) = 100 + 2 * mpi.rank;
    legacyField(1, 0) = 101 + 2 * mpi.rank;
    legacyField.trans.createMPITypes();
    legacyField.trans.pullOnce();
    const auto route = space.BuildRowTransferPlan();
    R::Array<double> source(R::RowLayout::Dense(2, 1)), destination(R::RowLayout::Dense(1, 1));
    {
        auto write = source.Write();
        write.View()(0, 0) = legacyField(0, 0);
        write.View()(1, 0) = legacyField(1, 0);
    }
    R::PayloadPlan payload(route, source.Layout(), destination.Layout(), R::ElementSchema::Arithmetic<double>());
    auto binding = payload.Bind(source, destination);
    binding.Begin().Wait();
    CHECK(destination.Read().View()(0, 0) == legacyField.son->operator()(0, 0));
    CHECK(destination.Read().View()(0, 0) == 100 + pulls[0]);
    CHECK(route.Valid());
    {
        auto mutation = mesh.versions.BeginMutation(R::Change::Layout);
        mutation.Commit();
    }
    CHECK_FALSE(space.Valid());
    CHECK_FALSE(route.Valid());
    CHECK_THROWS(G::LegacyFieldBridge::BorrowIndexing(legacyField.trans, space));
    CHECK_THROWS(binding.Begin());
}

#ifndef DNDS_MESH_FACADE_TEST_EMBEDDED
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
#endif
