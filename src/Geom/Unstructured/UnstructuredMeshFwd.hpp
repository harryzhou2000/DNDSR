#pragma once

namespace DNDS::Geom
{
    struct UnstructuredMesh;
    class UnstructuredMeshState;
    class UnstructuredMeshSnapshot;
    // Source compatibility for the former implementation-oriented names.
    using NativeMeshState = UnstructuredMeshState;
    using NativeMeshSnapshot = UnstructuredMeshSnapshot;
}
