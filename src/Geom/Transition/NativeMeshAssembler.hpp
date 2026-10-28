#pragma once

#include "Geom/Unstructured/OwnedMeshData.hpp"

namespace DNDS::Geom
{
    /// Explicit default selected by a construction frontend. Includes the
    /// boundary-node-boundary ring; this is policy, not an inferred traversal.
    GhostSpec DefaultNativeMeshGhostSpec(const NativeOwnedMeshData &owned, int cellLayers = 1,
                                         bool completeOwnedNodeStars = true);

    /// Completed-operation convenience for callers requesting a mesh state.
    /// Zero preserves a retained recipe, or selects the one-ring default when
    /// absent. A positive depth overrides standard Cell2Cell prefixes only;
    /// custom paths remain intact. Owned builders do not invoke this operation.
    std::shared_ptr<const UnstructuredMeshState> AssembleNativeMesh(
        const MPIInfo &mpi, NativeOwnedMeshData owned, int cellLayers = 0);

    /// Final ghost materialization from authoritative owned global rows.
    /// The caller supplies the exact requested recipe on every rank. This class
    /// does not invent boundary rings, node stars or other coverage policy.
    class NativeMeshAssembler
    {
    public:
        static NativeOwnedMeshData CaptureOwned(const UnstructuredMeshState &source, const MPIInfo &mpi);
        static std::shared_ptr<const UnstructuredMeshState> Build(const MPIInfo &mpi, NativeOwnedMeshData owned,
                                                                  const GhostSpec &specification);
    };
}
