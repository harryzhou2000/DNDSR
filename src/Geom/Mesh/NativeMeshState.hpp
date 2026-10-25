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
        friend class NativeMeshAssembler;
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
        std::optional<NativeMeshProvenance> _provenance;
        std::vector<std::shared_ptr<Runtime::Array<index>::ReadLease>> _provenanceReads;
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
        /// Setup metadata only: retain current part boundaries and owned local
        /// neighbors without changing entity order, buffers or their epochs.
        std::shared_ptr<const NativeMeshState> WithLocalCellSetup(std::vector<index> partStarts,
                                                                  tLocalMatStruct neighbors) const;
        /// Owned face-neighbor graph in current face-slot order. Requires faces;
        /// optionally excludes neighbors outside each current local partition.
        tLocalMatStruct BuildLocalCellNeighbors(bool onLocalPartition = false) const;
        /// Detached setup metadata for a newly prepared cell order. Entity maps
        /// and payloads stay unchanged; neighbors derive from exact face slots.
        std::shared_ptr<const NativeMeshState> WithLocalPartitions(std::vector<index> partStarts) const;
        /// Replace wall vectors while retaining the exact coordinate allocation.
        std::shared_ptr<const NativeMeshState> WithWallDistance(Runtime::ArrayPair<real> walls) const;
        /// Detached geometry-only replacement with identical entity order and
        /// shared topology/identity. Both owned and ghost coordinates must be
        /// ready and coherent; callers arrange any owner/ghost refresh. Clearing
        /// wallVectors explicitly invalidates the derived wall-distance data.
        /// Device placements are discarded and must be rebuilt for the new state.
        std::shared_ptr<const NativeMeshState> WithGeometry(
            Runtime::ArrayPair<real> coordinates, std::optional<Runtime::ArrayPair<real>> wallVectors,
            const Periodicity &periodicInfo) const;
        void Validate() const { _epoch.Validate(); }
        /// Rank-local capacity of retained Runtime array allocations, including
        /// every placement and host/device row metadata. Shared allocation IDs
        /// count once, including named/fixed aliases, IDs and ancestry. Excludes
        /// STL containers, entity mappings, access controllers and route caches.
        size_t ArrayBytes() const;
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
        /// Shared owned-row metadata. Array reads retain their own leases;
        /// publication prevents writes for this state's lifetime.
        const std::optional<NativeMeshProvenance> &Provenance() const
        {
            Validate();
            return _provenance;
        }
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
