#pragma once

#include "NativeMeshState.hpp"

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
        /// Explicit generated-ID floors are used only for reconstructed targets;
        /// every rank supplies identical floors and removed IDs are never reused.
        static std::shared_ptr<const NativeMeshState> Build(const NativeMeshState &source, const MPIInfo &mpi,
                                                            bool buildFaces = true, bool buildEdges = false, int ghostLayers = 1,
                                                            const std::map<EntityKind, index> &generatedIdStarts = {});
    };
}
