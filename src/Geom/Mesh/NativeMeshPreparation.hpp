#pragma once

#include "NativeMeshTransition.hpp"

namespace DNDS::Geom
{
    /// Distributed facial-graph partitioning. Unsupported legacy methods and
    /// weighting policies are rejected rather than silently substituted.
    struct NativePartitionOptions
    {
        std::string metisType = "KWAY";
        int metisUfactor = 20;
        int metisSeed = 0;
        int edgeWeightMethod = 0;
        int metisNcuts = 3;
        int ghostLayers = 1;
    };

    class NativeMeshPreparation
    {
    public:
        /// Detached topology completion; retains all existing families/IDs.
        static std::shared_ptr<const NativeMeshState> PrepareTopology(
            const NativeMeshState &source, const MPIInfo &mpi,
            bool faces = true, bool edges = false, int ghostLayers = 1);
        /// Partition cells by the exact facial graph, then native redistribution.
        /// Node/boundary owners follow cells. Existing face/edge owners remain
        /// retained; all families and external field transfer maps are returned.
        /// sourceSpaces retains exact original descriptors for every original
        /// family. Newly generated topology families name their intermediate
        /// source order; no external field existed on them before this call.
        /// Empty ranks participate through an active-rank communicator. A single
        /// active rank runs METIS on its already-local graph without a gather.
        static NativeMeshTransitionResult Partition(
            const NativeMeshState &source, const MPIInfo &mpi,
            const NativePartitionOptions &options = {});
    };
}
