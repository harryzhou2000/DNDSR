#pragma once

#include "Geom/Metis.hpp"
#include "DNDS/MPI.hpp"

namespace DNDS::Geom
{
    class UnstructuredMeshState;
    class EntitySpace;
    struct NativeOwnedMeshData;
    struct NativePartitionOptions;
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
