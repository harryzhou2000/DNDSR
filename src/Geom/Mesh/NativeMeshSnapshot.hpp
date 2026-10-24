#pragma once

#include "MeshAccess.hpp"
#include <optional>

namespace DNDS::Geom
{
    /// Detached immutable native backing for the query facade. This is an
    /// intermediate snapshot, not the authoritative storage of UnstructuredMesh.
    /// Capture validates/copies once; repeat AcquireRead only retains native
    /// allocations and acquires their controllers. Source mutations do not alter
    /// or invalidate this snapshot. Provenance is informative, not a dependency.
    class NativeMeshSnapshot
    {
        friend class MeshReadAccess;
        template <class T>
        struct Pair
        {
            Runtime::Array<T> owned, ghost;
        };
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
