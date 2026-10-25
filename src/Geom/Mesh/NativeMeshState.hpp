#pragma once

#include "NativeMeshSnapshot.hpp"
#include "EntitySpace.hpp"
#include <map>

namespace DNDS::Geom
{
    class RelationReadAccess;

    /// Explicitly published native query authority. Legacy public arrays remain
    /// a separate edit workspace; no raw edit silently updates this state.
    /// All map/array ownership is private and immutable after publication.
    class NativeMeshState
    {
        friend struct UnstructuredMesh;
        friend class MeshEdit;
        friend class NativeMeshTransition;
        struct Space
        {
            EntitySpace identity;
            index owned, ghost;
            std::shared_ptr<GlobalOffsetsMapping> global;
            std::shared_ptr<OffsetAscendIndexMapping> ghosts;
        };
        struct NamedRelation
        {
            NativeNamedRelation data;
            std::shared_ptr<Runtime::Array<index>::ReadLease> owned, ghost;
        };
        std::map<AdjKind, NamedRelation> _namedRelations;
        MeshReadRequirements _requirements;
        MeshIdentity _meshIdentity;
        std::map<EntityKind, Space> _spaces;
        std::map<EntityKind, Runtime::Array<index>> _originalIds;
        std::map<EntityKind, std::shared_ptr<Runtime::Array<index>::ReadLease>> _originalReads;
        std::map<int, NativeMeshSnapshot> _placements;
        std::shared_ptr<MeshReadAccess> _hostRead;
        UnstructuredMeshView _hostView;
        Runtime::VersionTracker _versions;
        Runtime::VersionStamp _epoch;
        std::vector<index> _localPartStarts;
        tLocalMatStruct _localCellNeighbors;

        NativeMeshState() = default;
        void BindPreparedEpoch(Runtime::VersionTracker versions);
        static int PlacementKey(Runtime::Device device)
        {
            return device.space == Runtime::MemorySpace::Host ? -1 : device.ordinal;
        }

    public:
        /// Collective native-only publication assembly; all inputs use the same entity
        /// domain/order. ids contains one owned/appended persistent-ID column
        /// for each published family. Validates owned-ID uniqueness across ranks
        /// and ghost IDs against owners using sparse routes. Each rank must pass
        /// the same communicator, requirements and entity families. No legacy
        /// allocation or global-entity-sized local storage is constructed.
        static std::shared_ptr<const NativeMeshState> FromNative(
            const MPIInfo &mpi, NativeMeshData data, std::map<EntityKind, EntitySpace> spaces,
            std::map<EntityKind, Runtime::Array<index>> ids, MeshReadRequirements requirements);
        /// Always imports/validates the legacy workspace, even if a prior native
        /// state is currently published. Caller serializes publication/setup.
        static std::shared_ptr<const NativeMeshState> CaptureLegacy(UnstructuredMesh &source,
                                                                    MeshReadRequirements requirements);
        /// Bind only after a successful publication transaction has committed.
        std::shared_ptr<const NativeMeshState> BindEpoch(Runtime::VersionTracker versions) const;
        std::shared_ptr<const NativeMeshState> WithPlacement(Runtime::Device device, Runtime::ExecutionContext context) const;
        void Validate() const { _epoch.Validate(); }
        /// Exact descriptor lookup over host setup storage. Device placement
        /// does not move this registry or change its canonical entity spaces.
        RelationReadAccess Relation(const AdjKind &key) const;
        std::vector<AdjKind> RelationKeys() const;
        void CheckRequirements(MeshReadRequirements requested) const;
        const NativeMeshSnapshot &Snapshot(Runtime::Device device) const;
        const UnstructuredMeshView &BorrowHostView() const { return _hostView; }
        bool HasEntities(EntityKind kind) const { return _spaces.find(kind) != _spaces.end(); }
        index NumOwned(EntityKind kind) const;
        index NumGhost(EntityKind kind) const;
        index NumGlobal(EntityKind kind) const;
        EntitySpace Entities(EntityKind kind) const;
        MeshReadRequirements Requirements() const { return _requirements; }
        index OriginalEntityId(EntityKind kind, index row) const;
        const std::string &MeshNamespace() const { return _meshIdentity.meshNamespace; }
        const std::map<EntityKind, index> &IdHighWater() const { return _meshIdentity.nextIds; }
        index NextEntityId(EntityKind kind) const
        {
            auto found = _meshIdentity.nextIds.find(kind);
            DNDS_check_throw_info(found != _meshIdentity.nextIds.end(), "persistent-ID high-water mark is unavailable");
            return found->second;
        }
        int NLocalParts() const { return _localPartStarts.empty() ? 1 : int(_localPartStarts.size() - 1); }
        index LocalPartStart(int part) const { return _localPartStarts.empty() ? 0 : _localPartStarts.at(size_t(part)); }
        index LocalPartEnd(int part) const { return _localPartStarts.empty() ? NumOwned(EntityKind::Cell) : _localPartStarts.at(size_t(part + 1)); }
        const tLocalMatStruct &LocalCellNeighbors() const { return _localCellNeighbors; }
    };
}
