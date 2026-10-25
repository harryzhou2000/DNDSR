#pragma once

#include "NativeMeshState.hpp"

namespace DNDS::Geom
{
    /// Detached owned-only host input. Every topology entry is a current GLOBAL
    /// row index or UnInitIndex; companions and IDs are payload. Family spaces
    /// are siblings without ghosts. Named descriptors retain their full keys.
    struct NativeOwnedMeshData
    {
        MeshIdentity identity;
        MeshReadRequirements requirements;
        int dimension = -1;
        bool periodic = false;
        Periodicity periodicInfo;
        std::map<EntityKind, EntitySpace> spaces;
        std::array<std::optional<Runtime::Array<index>>, size_t(MeshRelation::Count)> relations;
        std::map<AdjKind, Runtime::Array<index>> namedRelations;
        std::optional<Runtime::Array<real>> coords, nodeWallDist;
        std::optional<Runtime::Array<ElemInfo>> cellInfo, bndInfo, faceInfo, edgeInfo;
        std::optional<Runtime::Array<NodePeriodicBits>> cellNodesPbi, bndNodesPbi, cellFacesPbi, faceNodesPbi, cellEdgesPbi, edgeNodesPbi;
        std::map<EntityKind, Runtime::Array<index>> ids;
        std::optional<NativeMeshProvenance> provenance;
        std::vector<std::shared_ptr<Runtime::Array<index>::ReadLease>> retainedProvenance;
        /// CaptureOwned shares source companion allocations; this retained scope
        /// protects them even after the source state is released.
        std::shared_ptr<const MeshReadAccess> retainedRead;
    };

    class NativeMeshAssembler
    {
    public:
        static NativeOwnedMeshData CaptureOwned(const NativeMeshState &source, const MPIInfo &mpi);
        static std::shared_ptr<const NativeMeshState> Build(const MPIInfo &mpi, NativeOwnedMeshData owned, int ghostLayers = 1);
    };
}
