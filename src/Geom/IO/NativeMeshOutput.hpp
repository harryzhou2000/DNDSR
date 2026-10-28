#pragma once

#include "Geom/Unstructured/UnstructuredMeshState.hpp"

namespace DNDS::Geom
{
    /// Owned output-only buffers. Periodic node images stay with their node's
    /// owner and pointSourceRows maps them to original local point-field rows.
    /// This is not a mesh workspace: no topology or legacy arrays are cloned.
    struct NativeMeshOutput
    {
        Runtime::Array<real> coordinates;
        std::vector<index> pointSourceRows, connectivity, offsets;
        std::vector<uint8_t> types;
        index nodeOffset = 0, cellOffset = 0, globalNodes = 0, globalCells = 0, globalConnectivity = 0;
        /// Collective setup on mpi.comm. Uses direct row exchange for periodic
        /// masks and labels, with local node/ghost storage plus O(rank) offsets.
        static NativeMeshOutput Build(const UnstructuredMeshState &source, const MPIInfo &mpi);
    };
}
