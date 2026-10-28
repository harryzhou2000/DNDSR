#pragma once

#include "Geom/Unstructured/NativeMeshBuilder.hpp"
#include "Geom/Unstructured/NativeMeshProvenance.hpp"

namespace DNDS::Geom
{
    struct NativeMeshDerivedResult
    {
        std::shared_ptr<const UnstructuredMeshState> state;
        NativeMeshProvenance provenance;
    };
    struct NativeOwnedMeshDerivedResult
    {
        NativeOwnedMeshData owned;
        NativeMeshProvenance provenance;
    };
    class NativeMeshRefinement
    {
    public:
        /// Owned construction variants: same mathematics/identity policy as
        /// their completed-state counterparts, with temporary sparse closure
        /// only. Return a new namespace without assembling compute ghosts.
        static NativeOwnedMeshDerivedResult ElevateOwned(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                         NativeMeshBuildOptions options = {}, bool preserveCellIds = false);
        static NativeOwnedMeshDerivedResult BisectOwned(const NativeOwnedMeshData &source, const MPIInfo &mpi,
                                                        NativeMeshBuildOptions options = {});
        /// Add quadratic nodes by exact shared vertex-span identity. Coordinates
        /// use the established span mean. Node ancestry records all span nodes.
        /// Returns a new namespace; source numbering and buffers stay unchanged.
        /// preserveCellIds selects the legacy setup identity policy: retain
        /// Cell/Boundary and original Node IDs; new node IDs use a separate
        /// rank-prefix span range above the source high-water. The resulting
        /// namespace is still new. False gives every family fresh IDs.
        static NativeMeshDerivedResult Elevate(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                               NativeMeshBuildOptions options = {}, bool preserveCellIds = false);
        /// Split O2 elements into their established O1 children. Returns an
        /// independent logical mesh and fresh IDs, retaining original node rows.
        /// Periodic node-image labels follow the exact child slots; physical
        /// coordinates determine element-dependent subdivision variants.
        static NativeMeshDerivedResult Bisect(const UnstructuredMeshState &source, const MPIInfo &mpi,
                                              NativeMeshBuildOptions options = {});
    };
}
