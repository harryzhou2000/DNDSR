#pragma once

#include "NativeMeshTransition.hpp"

namespace DNDS::Geom
{
    /// Facial-graph partitioning without gathering a distributed graph.
    struct NativePartitionOptions
    {
        std::string metisType = "KWAY";
        int metisUfactor = 20;
        int metisSeed = 0;
        /// 0: unweighted. 1: maximum physical face-node distance, normalized
        /// globally. Multiple faces of a cell pair contribute their maximum.
        /// Integer weights are scaled so backend aggregate sums cannot overflow.
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
        /// Node/boundary owners and newly generated facial owners follow cells.
        /// Existing face/edge owners remain retained; all families and external
        /// field transfer maps are returned.
        /// sourceSpaces retains exact original descriptors for every original
        /// family. Newly generated topology families name their intermediate
        /// source order; no external field existed on them before this call.
        /// Empty ranks participate through an active-rank communicator. A single
        /// active rank runs METIS on its already-local graph without a gather.
        /// RB requires this single active rank; distributed ownership uses
        /// KWAY through ParMETIS. ReaderOwned serial input supports RB before
        /// distribution. Both methods support weighted and unweighted graphs.
        static NativeMeshTransitionResult Partition(
            const NativeMeshState &source, const MPIInfo &mpi,
            const NativePartitionOptions &options = {});
    };
}
