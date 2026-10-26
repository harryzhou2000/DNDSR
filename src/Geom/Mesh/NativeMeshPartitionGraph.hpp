#pragma once

#include "Geom/Metis.hpp"
#include "DNDS/MPI.hpp"

namespace DNDS::Geom
{
    class NativeMeshState;
    namespace detail
    {
        /// Setup scratch passed directly to METIS/ParMETIS: owned CSR rows,
        /// current global cell targets, and optional parallel integer weights.
        /// Duplicate faces for one neighbor use the maximum physical diameter.
        struct NativePartitionGraph
        {
            std::vector<idx_t> offsets, adjacency, weights;
        };
        NativePartitionGraph BuildNativePartitionGraph(const NativeMeshState &source, const MPIInfo &mpi, int edgeWeightMethod = 0);
    }
}
