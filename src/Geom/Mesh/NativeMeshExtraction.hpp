#pragma once

#include "NativeMeshRefinement.hpp"

namespace DNDS::Geom
{
    class NativeMeshExtraction
    {
    public:
        /// Extract external boundary elements into an independent mesh. Periodic
        /// seam main/donor rows are omitted; tangential node images are retained.
        /// Used nodes remain on their source owner and are compacted there.
        /// Mapped physical face order takes precedence over boundary row order.
        /// The new namespace retains Node->Node and Cell->Bnd persistent-ID
        /// ancestry, including the compact old-to-boundary node correspondence.
        /// The result has no boundary elements of its own; dimension-one primary
        /// curves are supported; zero-dimensional output is unsupported. Nodes
        /// used only by omitted periodic rows are compacted away.
        static NativeMeshDerivedResult ExtractBoundary(const NativeMeshState &source, const MPIInfo &mpi,
                                                       NativeMeshBuildOptions options = {});
    };
}
