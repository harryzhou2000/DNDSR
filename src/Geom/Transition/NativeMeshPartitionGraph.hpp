#pragma once

#include "Geom/Metis.hpp"
#include "Geom/Connectivity/MeshRelationTypes.hpp"
#include "DNDS/MPI.hpp"

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
