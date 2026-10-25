#pragma once

#include "NativeMeshBuilder.hpp"
#include "NativeMeshProvenance.hpp"

namespace DNDS::Geom
{
    struct NativeMeshDerivedResult
    {
        std::shared_ptr<const NativeMeshState> state;
        NativeMeshProvenance provenance;
    };
    class NativeMeshRefinement
    {
    public:
        /// Add quadratic nodes by exact shared vertex-span identity. Coordinates
        /// use the established span mean. Node ancestry records all span nodes.
        /// Returns a new namespace; source numbering and buffers stay unchanged.
        static NativeMeshDerivedResult Elevate(const NativeMeshState &source, const MPIInfo &mpi,
                                               NativeMeshBuildOptions options = {});
        /// Split O2 elements into their established O1 children. Returns an
        /// independent logical mesh and fresh IDs, retaining original node rows.
        /// Periodic node-image labels follow the exact child slots; physical
        /// coordinates determine element-dependent subdivision variants.
        static NativeMeshDerivedResult Bisect(const NativeMeshState &source, const MPIInfo &mpi,
                                              NativeMeshBuildOptions options = {});
    };
}
