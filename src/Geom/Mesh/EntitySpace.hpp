#pragma once

#include "MeshRelationTypes.hpp"
#include "DNDS/IndexMapping.hpp"
#include "DNDS/Runtime/RowTransfer.hpp"
#include <optional>

namespace DNDS::Geom
{
    struct UnstructuredMesh;
    class LegacyFieldBridge;
    class RelationReadAccess;
    template <class T, class L>
    class EntityField;

    /// Immutable snapshot of one entity family's numbering and distribution.
    /// No topology, Array or transformer is exposed. Global IDs here are the
    /// current rank-offset numbering, not persistent entity IDs across reorder.
    /// The identity token is process-local. Native sibling families share the
    /// logical mesh domain; numbering changes publish new descriptors and epochs.
    /// Exported field maps cannot mutate the retained canonical numbering.
    /// The caller retains the MPI communicator through descriptor/plan use.
    class EntitySpace
    {
        friend struct UnstructuredMesh;
        friend class LegacyFieldBridge;
        friend class RelationReadAccess;
        template <class T, class L>
        friend class EntityField;
        friend class NativeMeshState;
        friend class NativeMeshTransition;
        friend class NativeMeshAssembler;
        friend struct NativeMeshOutput;
        friend struct NativeMeshSerialOutput;
        friend class NativeConnectivity;
        friend class RelationPeriodicBits;

        EntityKind _kind{};
        MPIInfo _mpi;
        index _owned = 0, _ghost = 0, _globalCount = 0;
        std::shared_ptr<const uint8_t> _identity;
        Runtime::VersionStamp _stamp;
        Runtime::VersionTracker _versions;
        std::shared_ptr<GlobalOffsetsMapping> _global;
        std::shared_ptr<OffsetAscendIndexMapping> _ghostMap;
        bool _publishedMaps = false;

        EntitySpace(EntityKind kind, MPIInfo mpi, index owned, index ghost,
                    std::shared_ptr<const uint8_t> identity, Runtime::VersionStamp stamp,
                    std::shared_ptr<GlobalOffsetsMapping> global,
                    std::shared_ptr<OffsetAscendIndexMapping> ghostMap,
                    Runtime::VersionTracker versions);
        static void CheckCommunicator(MPI_Comm actual, MPI_Comm expected);
        // RelationReadAccess owns read leases for both numbering epochs, so
        // these private map operations need no generation mutex per incidence.
        // Public descriptor queries still validate on every call.
        index LocalToGlobalPinned(index local) const;
        std::optional<index> GlobalToLocalPinned(index global) const;

    public:
        EntitySpace(const EntitySpace &) = default;
        EntitySpace(EntitySpace &&) noexcept = default;
        /// Replacing a descriptor handle never mutates the retained mesh epoch.
        /// VersionTracker assignment is an owner operation; descriptors only
        /// share its controller and therefore explicitly rebind it here.
        EntitySpace &operator=(EntitySpace other)
        {
            _kind = other._kind;
            _mpi = other._mpi;
            _owned = other._owned;
            _ghost = other._ghost;
            _globalCount = other._globalCount;
            _identity = std::move(other._identity);
            _stamp = std::move(other._stamp);
            _versions.RebindPrepared(other._versions);
            _global = std::move(other._global);
            _ghostMap = std::move(other._ghostMap);
            _publishedMaps = other._publishedMaps;
            return *this;
        }
        /// Collective native construction, with no legacy mesh/Array storage.
        /// Each rank supplies its owned count; only O(nRanks) offsets and local
        /// ghost/request metadata are stored. The caller retains mpi.comm.
        static EntitySpace CreateOwned(EntityKind kind, MPIInfo mpi, index owned);
        /// Construct another entity family in this logical mesh domain. All
        /// ranks name the same kind; the new order has its own canonical maps.
        EntitySpace CreateSibling(EntityKind kind, index owned) const;
        /// Collective detached ghost-map replacement. globals names only the
        /// required remote rows, in current rank-offset numbering. Existing
        /// descriptors and plans continue referring to the previous order.
        EntitySpace WithGhosts(std::vector<index> globals) const;
        EntityKind Kind() const { return _kind; }
        /// Opaque process-local identity; never serialize this value.
        uint64_t MeshIdentity() const { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(_identity.get())); }
        index NumOwned() const { return _owned; }
        index NumGhost() const { return _ghost; }
        index NumLocal() const { return _owned + _ghost; }
        index NumGlobal() const { return _globalCount; }
        Runtime::VersionStamp Stamp() const { return _stamp; }
        bool Valid() const { return _identity && _global && _ghostMap && _stamp.Valid(); }
        void Validate() const { DNDS_check_throw_info(Valid(), "stale or invalid entity-space descriptor"); }
        bool SameDomain(const EntitySpace &other) const { return _identity == other._identity && _kind == other._kind; }
        /// Map identity plus current stamps describe the current row ordering. Values
        /// are deliberately not compared as a substitute for semantic identity.
        bool SameOrder(const EntitySpace &other) const
        {
            return Valid() && other.Valid() && SameDomain(other) &&
                   _global == other._global && _ghostMap == other._ghostMap;
        }

        index LocalToGlobal(index local) const;
        /// Missing entries stay explicit; unavailable globals are not confused
        /// with boundary sentinels or converted to a dereferenceable local row.
        std::optional<index> GlobalToLocal(index global) const;
        MPI_int Owner(index global) const;
        index OwnerLocalRow(index global) const;

        /// Collective on the mesh communicator, including empty ranks. Builds
        /// an owned-to-ghost route every call. Retain/share the returned plan to
        /// reuse its immutable route; descriptor lookup never initiates MPI.
        Runtime::RowTransferPlan BuildRowTransferPlan() const;
    };

    /// Transitional field-setup adapter. It borrows only maps, leaving field
    /// allocation, TransAttach, createMPITypes and request creation with callers.
    /// The returned legacy transformer retains its documented lifetime contract;
    /// this bridge cannot stamp arbitrary subsequent direct transformer calls.
    /// cloneMaps additionally isolates compatibility callers that can edit map
    /// objects; published native maps are always cloned regardless of this flag.
    class LegacyFieldBridge
    {
    public:
        template <class TTransformer>
        static void BorrowIndexing(TTransformer &transformer, const EntitySpace &space, bool cloneMaps = false)
        {
            space.Validate();
            DNDS_check_throw_info(transformer.father && transformer.son, "attach field arrays before borrowing entity indexing");
            DNDS_check_throw_info(transformer.father->Size() == space.NumOwned(), "field owned rows differ from entity space");
            // Legacy createMPITypes sizes an initially empty ghost allocation.
            DNDS_check_throw_info(transformer.son->Size() == 0 || transformer.son->Size() == space.NumGhost(),
                                  "field ghost rows differ from entity space");
            EntitySpace::CheckCommunicator(transformer.mpi.comm, space._mpi.comm);
            EntitySpace::CheckCommunicator(transformer.father->getMPI().comm, space._mpi.comm);
            EntitySpace::CheckCommunicator(transformer.son->getMPI().comm, space._mpi.comm);
            // Publish only after all recoverable local checks have succeeded.
            auto global = cloneMaps || space._publishedMaps ? std::make_shared<GlobalOffsetsMapping>(*space._global) : space._global;
            auto ghost = cloneMaps || space._publishedMaps ? std::make_shared<OffsetAscendIndexMapping>(*space._ghostMap) : space._ghostMap;
            transformer.pLGlobalMapping = global;
            transformer.pLGhostMapping = ghost;
            transformer.father->pLGlobalMapping = std::move(global);
        }
    };
}
