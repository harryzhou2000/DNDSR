#pragma once
#include "Geom/Connectivity/MeshRelationTypes.hpp"
#include "Geom/Connectivity/MeshTopologyTypes.hpp"
#include <functional>
namespace DNDS::Geom
{
    class MeshAdjacencyRead;
    class MeshAdjacency;
    /// Compatibility spelling for the native setup mathematics. Storage ownership,
    /// relation registration and sparse ghost evaluation belong to AdjacencyRegistry.
    struct MeshConnectivity
    {
        static MeshAdjacency Inverse(const MeshAdjacencyRead &, const MPIInfo &);
        static MeshAdjacency Compose(const MeshAdjacencyRead &, const MeshAdjacencyRead &, bool removeSelf = false);
        static MeshAdjacency ComposeAll(const MeshAdjacencyRead &, const MeshAdjacencyRead &, bool removeSelf = false);
        static MeshAdjacency ComposeFiltered(const MeshAdjacencyRead &, const MeshAdjacencyRead &, const AdjacencyDescriptor &,
                                             const std::function<bool(index, index, int)> &,
                                             const std::function<bool(index, index, const std::vector<index> &)> &extra = {});
    };
}
