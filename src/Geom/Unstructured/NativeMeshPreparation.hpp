#pragma once

#include "Geom/Transition/NativeMeshTransition.hpp"

namespace DNDS::Geom
{
    /// Explicit frontend policy extension for requested generated families.
    /// Retains a saved recipe (or selects the default when absent) and appends
    /// only the requested facial/edge traversals. No mesh/ghost storage changes.
    GhostSpec ExtendNativeMeshGhostSpecForTopology(const NativeOwnedMeshData &owned, bool faces, bool edges, int cellLayers = 1);
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
        /// Setup-only opt-in: unweighted reader-owned primary input may remain
        /// primary until final preparation, avoiding an intermediate Face family.
        /// Existing facial input and all other partition paths are unchanged.
        bool deferNewFacesUntilPreparation = false;
    };

    class NativeMeshPreparation
    {
    public:
        /// Owned-only partition and redistribution, before final ghost
        /// assembly. Exact facial graph scratch is derived from primary cones;
        /// no intermediate Face family or read state is produced. Existing
        /// generated families are retained. All partition methods/weights have
        /// the same meaning as Partition; deferNewFacesUntilPreparation is not
        /// needed because this operation never creates persistent faces.
        static NativeOwnedMeshTransitionResult PartitionOwned(
            const NativeOwnedMeshData &source, const MPIInfo &mpi,
            const NativePartitionOptions &options = {});
        /// Detached topology completion; retains all existing families/IDs.
        static std::shared_ptr<const UnstructuredMeshState> PrepareTopology(
            const UnstructuredMeshState &source, const MPIInfo &mpi,
            bool faces = true, bool edges = false, int ghostLayers = 1);
        /// Partition cells by the exact facial graph, then native redistribution.
        /// Node/boundary owners and newly generated facial owners follow cells.
        /// Existing face/edge owners remain retained; all families and external
        /// field transfer maps are returned.
        /// sourceSpaces retains exact original descriptors for every original
        /// family. Unweighted reader-owned primary input derives only graph
        /// scratch before redistribution and normally creates faces afterward.
        /// deferNewFacesUntilPreparation retains primary output only for that
        /// reader path; final preparation must create faces before flux queries.
        /// These new
        /// faces receive fresh IDs and have no sourceSpaces/field transfer entry;
        /// no external face field existed before this call. Other topology
        /// completion paths retain their intermediate source associations.
        /// Empty ranks participate through an active-rank communicator. A single
        /// active rank runs METIS on its already-local graph without a gather.
        /// RB requires this single active rank; distributed ownership uses
        /// KWAY through ParMETIS. ReaderOwned serial input supports RB before
        /// distribution. Both methods support weighted and unweighted graphs.
        static NativeMeshTransitionResult Partition(
            const UnstructuredMeshState &source, const MPIInfo &mpi,
            const NativePartitionOptions &options = {});
    };
}
