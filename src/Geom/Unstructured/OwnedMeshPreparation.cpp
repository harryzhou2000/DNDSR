#include "Geom/Unstructured/NativeMeshPreparation.hpp"
#include "Geom/Transition/NativeMeshPartitionGraph.hpp"
#include "Geom/Connectivity/NativeMeshClosure.hpp"
#include "Geom/Connectivity/TopologySubEntityQuery.hpp"
#include "Geom/Connectivity/ConnectivityInterpolateRows.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include <cmath>

namespace DNDS::Geom::detail
{
    NativePartitionGraph BuildOwnedPrimaryCellGraph(const NativeOwnedMeshData &source, const MPIInfo &mpi)
    {
        const auto found = source.relations[size_t(MeshRelation::CellCells)];
        const auto cones = source.relations[size_t(MeshRelation::CellNodes)];
        DNDS_check_throw_info(found && cones && source.cellInfo, "primary owned partition requires cell nodes, types and Cell2Cell rows");
        const auto cells = source.spaces.at(EntityKind::Cell);
        DNDS_check_throw_info(cells.Valid() && cells.NumGhost() == 0, "primary owned partition requires owned cells");
        auto neighborRows = found->Read();
        auto nodeRows = cones->Read();
        auto metadata = source.cellInfo->Read();
        std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> nodeBits;
        if (source.periodic)
        {
            DNDS_check_throw_info(source.cellNodesPbi, "periodic primary partition requires cell-node frame bits");
            nodeBits.emplace(source.cellNodesPbi->Read());
        }
        NativePartitionGraph graph;
        FieldTransferDetail::LocalStage(mpi, [&]
                                        {
            graph.offsets.push_back(0);
            const size_t cellCount = size_t(cells.NumOwned());
            constexpr size_t maxVertices = size_t(Elem::CellNumNodeMax);
            DNDS_check_throw_info(cellCount == 0 || maxVertices <= SIZE_MAX / cellCount,
                                  "primary partition vertex scratch exceeds address space");
            std::vector<index> vertices(cellCount * maxVertices);
            std::vector<uint8_t> frames;
            if (nodeBits) frames.resize(cellCount * maxVertices);
            std::vector<uint8_t> vertexCounts(cellCount);
            for (size_t cell = 0; cell < cellCount; ++cell)
            {
                const auto element = Elem::Element{metadata.View()(cell, 0).getElemType()};
                const auto nodeCount = nodeRows.View().RowSize(cell);
                DNDS_check_throw_info(element.GetNumVertices() > 0 &&
                                          size_t(element.GetNumVertices()) <= maxVertices &&
                                          nodeCount == size_t(element.GetNumNodes()),
                                      "primary partition cell has an invalid element cone");
                vertexCounts[cell] = uint8_t(element.GetNumVertices());
                auto *sorted = vertices.data() + cell * maxVertices;
                std::copy_n(nodeRows.View().Row(cell), element.GetNumVertices(), sorted);
                if (nodeBits)
                {
                    auto *frame = frames.data() + cell * maxVertices;
                    for (int vertex = 0; vertex < element.GetNumVertices(); ++vertex)
                        frame[vertex] = uint8_t(nodeBits->View()(cell, size_t(vertex)));
                    std::array<size_t, maxVertices> order{};
                    for (int vertex = 0; vertex < element.GetNumVertices(); ++vertex)
                        order[size_t(vertex)] = size_t(vertex);
                    std::sort(order.begin(), order.begin() + element.GetNumVertices(), [&](size_t left, size_t right)
                              { return std::pair(sorted[left], frame[left]) < std::pair(sorted[right], frame[right]); });
                    std::array<index, maxVertices> sortedCopy{};
                    std::array<uint8_t, maxVertices> frameCopy{};
                    for (int slot = 0; slot < element.GetNumVertices(); ++slot)
                    {
                        sortedCopy[size_t(slot)] = sorted[order[size_t(slot)]];
                        frameCopy[size_t(slot)] = frame[order[size_t(slot)]];
                    }
                    std::copy_n(sortedCopy.begin(), element.GetNumVertices(), sorted);
                    std::copy_n(frameCopy.begin(), element.GetNumVertices(), frame);
                }
                else
                    std::sort(sorted, sorted + element.GetNumVertices());
            }
            std::vector<index> neighbors;
            for (index cell = 0; cell < cells.NumOwned(); ++cell)
            {
                const auto *current = vertices.data() + size_t(cell) * maxVertices;
                const size_t currentSize = vertexCounts[size_t(cell)];
                neighbors.clear();
                for (auto neighborGlobal : neighborRows.View()[size_t(cell)])
                {
                    if (neighborGlobal == UnInitIndex) continue;
                    const auto neighbor = cells.GlobalToLocal(neighborGlobal);
                    if (!neighbor || *neighbor >= cells.NumOwned() || *neighbor == cell) continue;
                    const auto *other = vertices.data() + size_t(*neighbor) * maxVertices;
                    const size_t otherSize = vertexCounts[size_t(*neighbor)];
                    bool facial = false;
                    if (!nodeBits)
                    {
                        size_t left = 0, right = 0, shared = 0;
                        while (left < currentSize && right < otherSize)
                        {
                            if (current[left] < other[right]) ++left;
                            else if (other[right] < current[left]) ++right;
                            else { ++shared; ++left; ++right; }
                        }
                        facial = shared >= size_t(source.dimension);
                    }
                    else
                    {
                        const auto *currentFrames = frames.data() + size_t(cell) * maxVertices;
                        const auto *otherFrames = frames.data() + size_t(*neighbor) * maxVertices;
                        for (uint8_t mask = 0; mask < 8; ++mask)
                        {
                            size_t left = 0, right = 0, shared = 0;
                            while (left < currentSize && right < otherSize)
                            {
                                const auto currentKey = std::pair(current[left], currentFrames[left]);
                                const auto otherKey = std::pair(other[right], uint8_t(otherFrames[right] ^ mask));
                                if (currentKey < otherKey) ++left;
                                else if (otherKey < currentKey) ++right;
                                else { ++shared; ++left; ++right; }
                            }
                            if (shared >= size_t(source.dimension)) { facial = true; break; }
                        }
                    }
                    if (facial) neighbors.push_back(neighborGlobal);
                }
                CompactAdjacencyRow(neighbors);
                for (auto neighbor : neighbors)
                {
                    DNDS_check_throw_info(neighbor >= 0 && neighbor < cells.NumGlobal(),
                                          "primary partition cell neighbor is outside the mesh domain");
                    graph.adjacency.push_back(METIS::indexToIdx(neighbor));
                }
                graph.offsets.push_back(METIS::indexToIdx(graph.adjacency.size()));
            } });
        return graph;
    }

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
