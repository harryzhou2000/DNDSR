#pragma once

#include "MeshAccess.hpp"
#include "MeshRelationTypes.hpp"
#include "MeshIdentity.hpp"
#include <optional>
#include <map>

namespace DNDS::Geom
{
    /// Host setup relation, keyed separately by its complete AdjKind definition.
    /// Fixed facade slots may share these allocations without duplicating rows.
    struct NativeNamedRelation
    {
        Runtime::ArrayPair<index> rows;
        MeshAdjState state = Adj_PointToGlobal;
    };

    /// Native host construction input. Values and metadata are adopted without
    /// copying. Mutable aliases must obey the Array access contract; publication
    /// retains read leases. No legacy mesh or transformer is required.
    struct NativeMeshData
    {
        /// Consumed by NativeMeshState; setup relations remain host-only even
        /// when facade placements are prepared on a device.
        std::map<AdjKind, NativeNamedRelation> namedRelations;
        int dimension = -1;
        bool periodic = false;
        Periodicity periodicInfo;
        std::array<std::optional<Runtime::ArrayPair<index>>, static_cast<size_t>(MeshRelation::Count)> relations;
        std::optional<Runtime::ArrayPair<real>> coords, nodeWallDist;
        std::optional<Runtime::ArrayPair<ElemInfo>> cellInfo, bndInfo, faceInfo, edgeInfo;
        std::optional<Runtime::ArrayPair<NodePeriodicBits>> cellNodesPbi, bndNodesPbi, cellFacesPbi, faceNodesPbi, cellEdgesPbi, edgeNodesPbi;
        /// Durable UUID and non-reuse marks, consumed by NativeMeshState.
        MeshIdentity identity;
    };

    /// Detached immutable native backing for the query facade. This is an
    /// intermediate snapshot, not the authoritative storage of UnstructuredMesh.
    /// Capture validates/copies once; repeat AcquireRead only retains native
    /// allocations and acquires their controllers. Source mutations do not alter
    /// or invalidate this snapshot. Provenance is informative, not a dependency.
    class NativeMeshSnapshot
    {
        friend class MeshReadAccess;
        friend class NativeMeshTransition;
        template <class T>
        using Pair = Runtime::ArrayPair<T>;
        struct State
        {
            Runtime::VersionTracker versions;
            Runtime::VersionStamp source;
            Runtime::Device device;
            MeshViewStorage semantics;
            std::array<std::optional<Pair<index>>, static_cast<size_t>(MeshRelation::Count)> relations;
            std::optional<Pair<real>> coords, nodeWallDist;
            std::optional<Pair<ElemInfo>> cellInfo, bndInfo, faceInfo, edgeInfo;
            std::optional<Pair<NodePeriodicBits>> cellNodesPbi, bndNodesPbi, cellFacesPbi, faceNodesPbi, cellEdgesPbi, edgeNodesPbi;
            /// FromNative may adopt externally aliased allocations. Keep them
            /// read-protected for this immutable publication's entire lifetime.
            std::shared_ptr<MeshReadAccess> immutableRead;
        };
        std::shared_ptr<const State> _state;
        explicit NativeMeshSnapshot(std::shared_ptr<const State> state) : _state(std::move(state)) {}
        const State &Live() const
        {
            DNDS_check_throw_info(_state, "native mesh snapshot was moved or released");
            return *_state;
        }
        template <class T>
        static Pair<T> ClonePair(Runtime::PairReadView<T> source);
        static NativeMeshSnapshot FromRead(MeshReadAccess read);
        template <class T>
        static std::optional<Pair<T>> Place(const std::optional<Pair<T>> &source, Runtime::Device device,
                                            Runtime::ExecutionContext context);

    public:
        /// Consume validated local-index host buffers. Unsupported or missing
        /// required topology fails before returning a usable snapshot.
        static NativeMeshSnapshot FromNative(NativeMeshData data, MeshReadRequirements requirements);
        static NativeMeshSnapshot Capture(UnstructuredMesh &source, MeshReadRequirements requirements = MeshReadRequirements::Primary());
        static NativeMeshSnapshot ImportLegacyWorkspace(UnstructuredMesh &source, MeshReadRequirements requirements = MeshReadRequirements::Primary());
        NativeMeshSnapshot To(Runtime::Device device, Runtime::ExecutionContext context) const;
        NativeMeshSnapshot To(Runtime::Device device) const { return To(device, {device == Runtime::Device::Host() ? GetDevice() : device, nullptr}); }
        Runtime::Device GetDevice() const { return Live().device; }
        Runtime::VersionStamp SourceStamp() const { return Live().source; }
        MeshReadAccess AcquireRead(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const
        {
            return MeshReadAccess(*this, context);
        }
    };
}
