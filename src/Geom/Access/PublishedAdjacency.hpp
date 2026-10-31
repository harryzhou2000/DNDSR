#pragma once

#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::Geom
{
    /// MPI-local exact-descriptor acquisition over fixed and named publications.
    /// Built-in facade slots and custom descriptor entries keep their existing
    /// storage/lease authorities. This performs no registration or ghost build.
    inline MeshAdjacencyRead ReadPublishedAdjacency(UnstructuredMesh &mesh, const AdjacencyDescriptor &descriptor)
    {
        const auto keys = mesh.AdjacencyKeys();
        if (std::find(keys.begin(), keys.end(), descriptor) != keys.end())
            return mesh.ReadAdjacency(descriptor);
        return MeshAdjacencyRead::Current(mesh, descriptor, mesh.CurrentNativeRequirements());
    }
}
