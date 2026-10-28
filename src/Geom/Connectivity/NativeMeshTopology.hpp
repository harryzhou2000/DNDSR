#pragma once

#include "Geom/Unstructured/OwnedMeshData.hpp"

namespace DNDS::Geom
{
    /// Complete native primary topology without a legacy workspace, including
    /// canonical periodic nodes with exact ordered image companions.
    /// Sparse shared-node parent closure is rebuilt from authoritative owned
    /// cones, independently of the source's existing cell ghosts.
    /// Unassigned exterior faces retain BC_ID_NULL and no boundary association.
    class NativeMeshTopology
    {
    public:
        /// Existing requested families are refreshed by exact full-node/image
        /// keys. Surviving IDs and ancestry are retained across owner/order
        /// changes; unmatched new entities allocate above the old high-water.
        /// Named rows are remapped or rejected when no payload can be inferred.
        /// Explicit generated-ID floors are used only for reconstructed targets;
        /// every rank supplies identical floors and removed IDs are never reused.
        /// Owned-to-owned topology construction. Only temporary sparse algorithm
        /// closure is fetched; no final ghost storage or mesh state is produced.
        static NativeOwnedMeshData Build(NativeOwnedMeshData source, const MPIInfo &mpi,
                                         bool buildFaces = true, bool buildEdges = false,
                                         const std::map<EntityKind, index> &generatedIdStarts = {});
    };

    /// Explicit complete topology-change operation on an existing publication.
    /// Captures owned data, constructs topology, then finalizes once. The face
    /// builder itself has no dependency on the source publication or its ghosts.
    std::shared_ptr<const UnstructuredMeshState> PrepareNativeMeshTopology(
        const UnstructuredMeshState &source, const MPIInfo &mpi,
        bool buildFaces = true, bool buildEdges = false, int ghostLayers = 1,
        const std::map<EntityKind, index> &generatedIdStarts = {});
}
