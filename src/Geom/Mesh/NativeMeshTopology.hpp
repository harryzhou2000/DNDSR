#pragma once

#include "NativeMeshState.hpp"

namespace DNDS::Geom
{
    /// Complete nonperiodic native primary topology without a legacy workspace.
    /// Source cell ghosts must contain the complete incidences of generated
    /// faces/edges (the primary node-neighbor ring provides this closure).
    /// Unassigned exterior faces retain BC_ID_NULL and no boundary association.
    class NativeMeshTopology
    {
    public:
        static std::shared_ptr<const NativeMeshState> Build(const NativeMeshState &source, const MPIInfo &mpi,
                                                            bool buildFaces = true, bool buildEdges = false, int ghostLayers = 1);
    };
}
