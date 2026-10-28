#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include "Geom/Connectivity/NativeMeshClosure.hpp"
#include "Geom/Connectivity/TopologySubEntityQuery.hpp"
#include "Geom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include <cmath>

namespace DNDS::Geom::detail
{
    NativePartitionGraph BuildOwnedPartitionGraph(const NativeOwnedMeshData &source, const MPIInfo &mpi, int method)
    {
        int expected = method;
        MPI_Bcast(&expected, 1, MPI_INT, 0, mpi.comm);
        FieldTransferDetail::LocalStage(mpi, [&]
                                        { DNDS_check_throw_info(method == expected && (method == 0 || method == 1) &&
                                                                    (source.dimension == 2 || source.dimension == 3),
                                                                "owned partition requires matching weight method 0/1 and dimension two/three"); });
        auto closure = BuildNativeCellClosure(source, mpi);
        auto cone = closure.cellNodes.Read();
        auto metadata = closure.cellInfo.ReadCombined();
        std::optional<RelationPeriodicBits> periodic;
        if (closure.cellNodesPbi)
            periodic.emplace(cone, closure.cellNodesPbi->owned, closure.cellNodesPbi->ghost);
        auto query = Query(metadata.View(), cone, false, periodic ? &*periodic : nullptr);
        std::optional<Runtime::ArrayPair<real>> coordinates;
        std::optional<Runtime::ArrayPair<real>::ReadLease> points;
        if (method)
        {
            FieldTransferDetail::LocalStage(mpi, [&]
                                            { DNDS_check_throw_info(source.coords, "weighted owned partition requires coordinates"); });
            RelationGhostRoute route(cone.Target());
            coordinates.emplace(Runtime::ArrayPair<real>{*source.coords,
                                                         route.TransferRows(cone.Target(), *source.coords, Runtime::ElementSchema::Arithmetic<real>())});
            points.emplace(coordinates->ReadCombined());
        }
        NativePartitionGraph graph;
        std::vector<real> diameters;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            auto rows = ConnectivityRows::EnumerateSubEntities(
                [&](index cell, int slot) { return cone.ConvertedIndex(cell, size_t(slot), Adj_PointToLocal, false); },
                query, cone.Rows(), cone.Target().NumLocal());
            std::vector<std::vector<NativeNeighborWeight>> neighbors(size_t(cone.NumOwned()));
            for (size_t entity = 0; entity < rows.entity2parent.size(); ++entity)
            {
                const auto &parents = rows.entity2parent[entity];
                if (std::none_of(parents.begin(), parents.end(), [&](index cell) { return cell < cone.NumOwned(); })) continue;
                DNDS_check_throw_info(!parents.empty() && parents.size() <= 2 &&
                                          (source.periodic || parents.size() == 1 || parents[0] != parents[1]),
                                      "owned partition face has missing, repeated or nonmanifold incidences");
                if (parents.size() != 2 || parents[0] == parents[1]) continue;
                real diameter = 0;
                if (method)
                {
                    const auto parent = parents[0];
                    const auto &subs = rows.parent2entity[size_t(parent)];
                    const auto sub = std::find(subs.begin(), subs.end(), index(entity)) - subs.begin();
                    DNDS_check_throw_info(sub < subs.size(), "owned partition face has no parent slot");
                    const auto &nodes = rows.entity2node[entity];
                    std::vector<NodePeriodicBits> bits(nodes.size());
                    if (periodic)
                        query.extractPbi(parent, int(sub), [&](int slot) { return (*periodic)(parent, size_t(slot)); }, bits.data());
                    std::vector<tPoint> facePoints;
                    for (size_t slot = 0; slot < nodes.size(); ++slot)
                    {
                        auto row = points->View()[size_t(nodes[slot])];
                        DNDS_check_throw_info(row.size() == 3, "owned partition coordinate width differs");
                        tPoint point{row[0], row[1], row[2]};
                        if (periodic) point = source.periodicInfo.GetCoordByBits(point, bits[slot]);
                        DNDS_check_throw_info(point.allFinite(), "owned partition face coordinate is not finite");
                        for (const auto &other : facePoints) diameter = std::max(diameter, (point - other).norm());
                        facePoints.push_back(point);
                    }
                    DNDS_check_throw_info(std::isfinite(diameter) && diameter > 0, "owned partition face has no finite positive diameter");
                }
                for (size_t side = 0; side < 2; ++side)
                    if (parents[side] < cone.NumOwned())
                    {
                        const auto target = cone.Source().LocalToGlobal(parents[1 - side]);
                        neighbors[size_t(parents[side])].emplace_back(target, diameter);
                    }
            }
            graph.offsets.push_back(0);
            for (auto &row : neighbors)
            {
                CompactAdjacencyRow(row);
                for (const auto &[target, diameter] : row)
                {
                    graph.adjacency.push_back(METIS::indexToIdx(target));
                    if (method) diameters.push_back(diameter);
                }
                graph.offsets.push_back(METIS::indexToIdx(size_t_to_signed<index>(graph.adjacency.size())));
            } });
        if (method)
            ScaleNativePartitionWeights(mpi, diameters, graph);
        return graph;
    }
}
