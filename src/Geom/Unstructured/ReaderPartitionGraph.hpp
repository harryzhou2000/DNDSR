#pragma once

#include "Geom/Transition/NativeMeshPartitionGraph.hpp"

namespace DNDS::Geom::detail
{
    /// Exact unweighted graph scratch for dimension-two/three primary input
    /// with at most one owned cell rank. Node ownership may be independent of
    /// the reader. This creates
    /// no face family, IDs, placement, or persistent topology publication.
    NativePartitionGraph BuildReaderPartitionGraph(const UnstructuredMeshState &source, const MPIInfo &mpi);
    /// Rank-local exact primary facial scratch for local permutation. Uses only
    /// owned/appended cells and nodes, returns sorted unique owned-cell targets.
    /// Caller coordinates failures before its next collective. Creates no Face
    /// family or cache and does not replace CellCells private-cell classification.
    std::vector<std::vector<index>> BuildPrimaryLocalCellNeighbors(const UnstructuredMeshState &source);
}
