#pragma once

#include "Geom/Metis.hpp"
#include "Geom/Connectivity/MeshRelationTypes.hpp"
#include "DNDS/MPI.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace DNDS::Geom
{
    class UnstructuredMeshState;
    class EntitySpace;
    struct NativeOwnedMeshData;
    struct NativePartitionOptions;

    /// Private setup descriptor for exact face-neighbor cell rows. Partition
    /// construction computes these rows for METIS; retaining them across the
    /// owned transition avoids rebuilding the same closure for local reorder.
    inline AdjKind NativeSetupFacialCellNeighborsKey()
    {
        auto key = Adj::Cell2Cell;
        key.path = {EntityKind::Cell, EntityKind::Cell};
        key.selection = RelationSelection::Exact;
        key.order = RelationOrder::SortedUnique;
        key.periodic = RelationPeriodicPolicy::EntityIdentity;
        key.semanticName = "geom.setup.exact-facial-neighbors";
        key.semanticVersion = 1;
        return key;
    }
    namespace detail
    {
        using NativeNeighborWeight = std::pair<index, real>;

        /// Small adjacency rows are sorted in place. Callers build with
        /// push_back, then compact once instead of paying tree allocation per
        /// neighbor.
        inline void CompactAdjacencyRow(std::vector<index> &row)
        {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }

        /// Merge duplicate face incidences to one row entry and retain the
        /// largest diameter, matching the historical map update policy.
        inline void CompactAdjacencyRow(std::vector<NativeNeighborWeight> &row)
        {
            std::sort(row.begin(), row.end(), [](const auto &lhs, const auto &rhs)
                      { return lhs.first < rhs.first; });
            size_t output = 0;
            for (size_t input = 0; input < row.size(); ++input)
            {
                if (output && row[output - 1].first == row[input].first)
                    row[output - 1].second = std::max(row[output - 1].second, row[input].second);
                else
                    row[output++] = row[input];
            }
            row.resize(output);
        }

        /// Setup scratch passed directly to METIS/ParMETIS: owned CSR rows,
        /// current global cell targets, and optional parallel integer weights.
        /// Duplicate faces for one neighbor use the maximum physical diameter.
        struct NativePartitionGraph
        {
            std::vector<idx_t> offsets, adjacency, weights;
        };
        NativePartitionGraph BuildNativePartitionGraph(const UnstructuredMeshState &source, const MPIInfo &mpi, int edgeWeightMethod = 0);
        /// Temporary exact facial incidence from owned cell cones. Remote
        /// inputs are sparse closure scratch, with no Face family/publication.
        NativePartitionGraph BuildOwnedPartitionGraph(const NativeOwnedMeshData &source, const MPIInfo &mpi, int edgeWeightMethod = 0);
        void ScaleNativePartitionWeights(const MPIInfo &mpi, const std::vector<real> &diameters, NativePartitionGraph &graph);
        std::vector<MPI_int> PartitionNativeGraph(const EntitySpace &cells, const MPIInfo &mpi,
                                                  NativePartitionGraph graph, const NativePartitionOptions &options);
    }
}
