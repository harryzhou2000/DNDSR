#pragma once
// TEST ONLY: independent DNDS Array payloads for the established const-view
// kernel comparison. This is not an UnstructuredMesh or a production backend.
#include "Geom/Mesh/Mesh_DeviceView.hpp"
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

struct LegacyConstReferenceArrays
{
    G::tCoordPair coords;
    G::tAdjPair cell2node, face2node, cell2face;
    G::tAdj2Pair face2cell;
    G::tElemInfoArrayPair cellElemInfo, faceElemInfo;
    G::tPbiPair cell2nodePbi, face2nodePbi, cell2facePbi;
    G::Periodicity periodicInfo;
};
static auto LegacyConstReferenceFixture(bool periodic)
{
    const auto mpi = FacadeMPI();
    auto result = std::make_unique<LegacyConstReferenceArrays>();
    auto &data = *result;
    data.coords.InitPair("reference.points", mpi);
    data.coords.father->Resize(periodic ? 4 : 3);
    data.coords.son->Resize(0);
    const G::tPoint points[]{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    for (I node = 0; node < data.coords.Size(); ++node)
        data.coords[node] = points[node];
    const auto cn = periodic ? std::vector<std::vector<I>>{{0, 1, 2, 3}} : std::vector<std::vector<I>>{{0, 1, 2}};
    const auto fn = periodic ? std::vector<std::vector<I>>{{0, 3}, {0, 1}, {3, 2}} : std::vector<std::vector<I>>{{0, 1}, {1, 2}, {2, 0}};
    const auto cf = periodic ? std::vector<std::vector<I>>{{0, 1, 0, 2}} : std::vector<std::vector<I>>{{0, 1, 2}};
    auto fc = std::vector<std::vector<I>>(3, {0, DNDS::UnInitIndex});
    if (periodic)
        fc[0] = {0, 0};
    PublishRows(data.cell2node, "reference.cellNodes", mpi, cn);
    PublishRows(data.face2node, "reference.faceNodes", mpi, fn);
    PublishRows(data.cell2face, "reference.cellFaces", mpi, cf);
    PublishFixed(data.face2cell, "reference.faceCells", mpi, fc);
    PublishInfo(data.cellElemInfo, "reference.cellInfo", mpi, 1, periodic ? G::Elem::Quad4 : G::Elem::Tri3, 11);
    PublishInfo(data.faceElemInfo, "reference.faceInfo", mpi, 3, G::Elem::Line2, 7);
    auto bits = [](const auto &rows)
    {
        std::vector<std::vector<G::NodePeriodicBits>> result;
        for (const auto &row : rows)
            result.emplace_back(row.size(), G::NodePeriodicBits{});
        return result;
    };
    auto cb = bits(cn), fb = bits(fn), side = bits(cf);
    if (periodic)
    {
        cb[0][2] = G::nodePB1;
        fb[0][1] = G::nodePB1;
        side[0][2] = G::nodePB1;
        data.faceElemInfo(0, 0).zone = G::BC_ID_PERIODIC_1;
        data.periodicInfo.translation[1].map() = G::tPoint{3, 0, 0};
        data.periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    }
    PublishRows(data.cell2nodePbi, "reference.cellBits", mpi, cb);
    PublishRows(data.face2nodePbi, "reference.faceBits", mpi, fb);
    PublishRows(data.cell2facePbi, "reference.sideBits", mpi, side);
    return result;
}
