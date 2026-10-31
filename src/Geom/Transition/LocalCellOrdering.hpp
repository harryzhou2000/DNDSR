#pragma once

#include "Geom/Access/EntityWorkPartition.hpp"
#include "Geom/Transition/SerialAdjReordering.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <map>

namespace DNDS::Geom
{
    enum class CellCoreSplit
    {
        None,
        CellNodeCell
    };
    struct LocalCellOrderingSettings
    {
        int subregions = 1;
        CellCoreSplit coreSplit = CellCoreSplit::None;
        bool aggregateSendRegions = false;
    };
    struct LocalCellOrderingResult
    {
        std::vector<index> oldToNew, subregionStarts, coreEnds;
    };

    /// Subregion -> optional node-star core split -> optional identical send
    /// destination grouping -> RCM. Application remains a normal local entity
    /// permutation. Unlike the historical cell reorder, partition boundaries
    /// cannot be destroyed by an interior-first pass across inner subregions.
    inline LocalCellOrderingResult BuildLocalCellOrdering(UnstructuredMesh &mesh, LocalCellOrderingSettings settings = {})
    {
        const auto mpi = mesh.getMPI();
        std::optional<Runtime::VersionTracker::ReadLease> pin;
        MPI::LocalStage(mpi, [&]
                        {
        pin.emplace(mesh.versions.Read());
        (void)mesh.CurrentNativeState();
        DNDS_check_throw_info(settings.subregions >= 1 &&
                                  (settings.coreSplit == CellCoreSplit::None || settings.coreSplit == CellCoreSplit::CellNodeCell),
                              "invalid local cell ordering settings"); });
        int aggregate = settings.aggregateSendRegions, minimum = 0, maximum = 0;
        MPI_Allreduce(&aggregate, &minimum, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(&aggregate, &maximum, 1, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(minimum == maximum, "send aggregation options differ across ranks");
        const auto cells = mesh.Entities(EntityKind::Cell);
        const index nCell = cells.NumOwned();
        tLocalMatStruct graph;
        std::vector<uint8_t> dependent;
        std::vector<std::vector<int>> destinations;
        MPI::LocalStage(mpi, [&]
                        {
        auto adjacency = ReadPublishedAdjacency(mesh, Adj::Cell2CellFace);
        graph.resize(size_t(nCell));
        dependent.assign(size_t(nCell), 0);
        destinations.resize(size_t(nCell));
        for (index iCell = 0; iCell < nCell; ++iCell)
            for (index slot = 0; slot < adjacency.RowSize(iCell); ++slot)
            {
                const auto entry = adjacency.ReadEntry(iCell, size_t(slot));
                if (entry.absent) continue;
                DNDS_check_throw_info(!entry.unavailable, "local cell ordering requires complete face-neighbor mapping");
                const auto other = cells.GlobalToLocal(entry.global);
                if (other && *other < nCell && *other != iCell) graph[iCell].push_back(*other);
            }
        if (settings.coreSplit == CellCoreSplit::CellNodeCell)
        {
            const auto work = EntityWorkPartition::CellNodeCell(mesh);
            for (auto iCell : work.HaloDependent()) dependent[iCell] = 1;
        } });
        if (settings.aggregateSendRegions)
        {
            // Actual canonical mapping, not neighbor-rank inference. All ranks
            // must agree on this option before entering the collective.
            const auto route = cells.BuildRowTransferPlan();
            for (int peer = 0; peer < int(route.SendRows().size()); ++peer)
                for (auto iCell : route.SendRows()[peer])
                    destinations[iCell].push_back(peer);
        }
        LocalCellOrderingResult result;
        MPI::LocalStage(mpi, [&]
                        {
        result.oldToNew.assign(size_t(nCell), -1);
        result.subregionStarts.push_back(0);
        std::vector<int> regions(size_t(nCell), 0);
        if (nCell && settings.subregions > 1)
        {
            const auto partition = PartitionSerialAdj_Metis(graph.begin(), graph.end(), int(std::min<index>(settings.subregions, nCell)), 0);
            std::copy(partition.begin(), partition.end(), regions.begin());
        }
        index next = 0;
        std::vector<index> inGroup(size_t(nCell), -1);
        for (int region = 0; region < settings.subregions; ++region)
        {
            for (int dependency = 0; dependency < 2; ++dependency)
            {
                std::map<std::vector<int>, std::vector<index>> groups;
                for (index iCell = 0; iCell < nCell; ++iCell)
                    if (regions[iCell] == region && int(dependent[iCell]) == dependency)
                        groups[settings.aggregateSendRegions ? destinations[iCell] : std::vector<int>{}].push_back(iCell);
                for (const auto &[unused, rows] : groups)
                {
                    for (index iRow = 0; iRow < index(rows.size()); ++iRow) inGroup[rows[iRow]] = iRow;
                    tLocalMatStruct subgraph(rows.size());
                    for (index iRow = 0; iRow < index(rows.size()); ++iRow)
                        for (auto other : graph[rows[iRow]])
                            if (inGroup[other] >= 0) subgraph[iRow].push_back(inGroup[other]);
                    index oldBandwidth = 0, newBandwidth = 0;
                    auto order = ReorderSerialAdj_CorrectRCM(subgraph.begin(), subgraph.end(), oldBandwidth, newBandwidth);
                    for (auto iRow : order.first) result.oldToNew[rows[iRow]] = next++;
                    for (auto iCell : rows) inGroup[iCell] = -1;
                }
                if (dependency == 0) result.coreEnds.push_back(next);
            }
            result.subregionStarts.push_back(next);
        } });
        return result;
    }
}
