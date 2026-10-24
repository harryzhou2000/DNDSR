#include "NativeMeshSnapshot.hpp"
#include "Mesh.hpp"
#include <cstring>

namespace DNDS::Geom
{
    template <class T>
    NativeMeshSnapshot::Pair<T> NativeMeshSnapshot::ClonePair(Runtime::PairReadView<T> source)
    {
        auto clone = [](Runtime::ArrayView<const T> view)
        {
            Runtime::RowLayout layout;
            if (view.offsets)
            {
                std::vector<size_t> lengths(view.rows);
                for (size_t i = 0; i < view.rows; ++i)
                    lengths[i] = view.offsets[i + 1] - view.offsets[i];
                layout = Runtime::RowLayout::Ragged(lengths);
            }
            else if (view.sizes)
                layout = Runtime::RowLayout::Padded(std::vector<size_t>(view.sizes, view.sizes + view.rows), view.stride);
            else
                layout = Runtime::RowLayout::Dense(view.rows, view.width, view.stride);
            Runtime::Array<T> result(layout);
            if (layout.Elements())
            {
                auto write = result.Write();
                std::memcpy(write.View().data, view.data, CheckedSize::Multiply(layout.Elements(), sizeof(T)));
            }
            return result;
        };
        return {clone(source.owned), clone(source.ghost)};
    }

    NativeMeshSnapshot NativeMeshSnapshot::Capture(UnstructuredMesh &source, MeshReadRequirements requirements)
    {
        return FromRead(source.AcquireRead(requirements));
    }
    NativeMeshSnapshot NativeMeshSnapshot::ImportLegacyWorkspace(UnstructuredMesh &source, MeshReadRequirements requirements)
    {
        return FromRead(MeshReadAccess(source, requirements, Runtime::ExecutionContext::Host()));
    }
    NativeMeshSnapshot NativeMeshSnapshot::FromRead(MeshReadAccess read)
    {
        auto state = std::make_shared<State>();
        state->source = read.Stamp();
        state->device = Runtime::Device::Host();
        const auto &s = read._storage;
        // Copy semantic values only. Pointer bundles are assembled from private
        // native arrays on read; no source pointer may escape into this state.
        state->semantics.dim = s.dim;
        state->semantics.isPeriodic = s.isPeriodic;
        state->semantics.periodicInfo = s.periodicInfo;
        state->semantics.hasNodeWallDist = s.hasNodeWallDist;
        for (size_t i = 0; i < state->relations.size(); ++i)
        {
            state->semantics.relationStates[i] = s.relationStates[i];
            state->semantics.relationPresent[i] = s.relationPresent[i];
            if (s.relationPresent[i])
                state->relations[i] = ClonePair(s.relations[i]);
        }
        state->coords = ClonePair(s.coords);
        state->cellInfo = ClonePair(s.cellElemInfo);
        state->bndInfo = ClonePair(s.bndElemInfo);
        if (s.relationPresent[size_t(MeshRelation::FaceNodes)])
            state->faceInfo = ClonePair(s.faceElemInfo);
        if (s.relationPresent[size_t(MeshRelation::EdgeNodes)])
            state->edgeInfo = ClonePair(s.edgeElemInfo);
        if (s.hasNodeWallDist)
            state->nodeWallDist = ClonePair(s.nodeWallDist);
        if (s.isPeriodic)
        {
            state->cellNodesPbi = ClonePair(s.cell2nodePbi);
            state->bndNodesPbi = ClonePair(s.bnd2nodePbi);
            if (s.relationPresent[size_t(MeshRelation::FaceNodes)])
            {
                state->cellFacesPbi = ClonePair(s.cell2facePbi);
                state->faceNodesPbi = ClonePair(s.face2nodePbi);
            }
            if (s.relationPresent[size_t(MeshRelation::EdgeNodes)])
            {
                state->cellEdgesPbi = ClonePair(s.cell2edgePbi);
                state->edgeNodesPbi = ClonePair(s.edge2nodePbi);
            }
        }
        return NativeMeshSnapshot(std::move(state));
    }

    template <class T>
    std::optional<NativeMeshSnapshot::Pair<T>> NativeMeshSnapshot::Place(const std::optional<Pair<T>> &source,
                                                                         Runtime::Device device, Runtime::ExecutionContext context)
    {
        if (!source)
            return {};
        return Pair<T>{source->owned.To(device, context), source->ghost.To(device, context)};
    }

    NativeMeshSnapshot NativeMeshSnapshot::To(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        Live();
        DNDS_check_throw_info(context.device == (device == Runtime::Device::Host() ? GetDevice() : device),
                              "snapshot placement context mismatch");
        auto next = std::make_shared<State>();
        next->source = _state->source;
        next->device = device;
        next->semantics = _state->semantics;
        for (size_t i = 0; i < next->relations.size(); ++i)
            next->relations[i] = Place(_state->relations[i], device, context);
        next->coords = Place(_state->coords, device, context);
        next->nodeWallDist = Place(_state->nodeWallDist, device, context);
        next->cellInfo = Place(_state->cellInfo, device, context);
        next->bndInfo = Place(_state->bndInfo, device, context);
        next->faceInfo = Place(_state->faceInfo, device, context);
        next->edgeInfo = Place(_state->edgeInfo, device, context);
        next->cellNodesPbi = Place(_state->cellNodesPbi, device, context);
        next->bndNodesPbi = Place(_state->bndNodesPbi, device, context);
        next->cellFacesPbi = Place(_state->cellFacesPbi, device, context);
        next->faceNodesPbi = Place(_state->faceNodesPbi, device, context);
        next->cellEdgesPbi = Place(_state->cellEdgesPbi, device, context);
        next->edgeNodesPbi = Place(_state->edgeNodesPbi, device, context);
        return NativeMeshSnapshot(std::move(next));
    }

    template <class T>
    Runtime::PairReadView<T> MeshReadAccess::CaptureNative(const Runtime::Array<T> &owned, const Runtime::Array<T> &ghost)
    {
        auto capture = [&](const auto &array)
        {
            auto hold = std::make_unique<ArrayHold<T>>(array, _context);
            auto view = hold->read.View();
            _holds.push_back(std::move(hold));
            return view;
        };
        return {capture(owned), capture(ghost)};
    }

    MeshReadAccess::MeshReadAccess(const NativeMeshSnapshot &snapshot, Runtime::ExecutionContext context)
        : _meshRead(snapshot.Live().versions.Read()), _stamp(snapshot.Live().versions.Stamp()), _context(context)
    {
        DNDS_check_throw_info(context.device == snapshot.GetDevice() &&
                                  (context.device != Runtime::Device::Host() || !context.stream),
                              "native snapshot acquisition placement mismatch");
        const auto &s = *snapshot._state;
        _storage = s.semantics;
        auto capture = [&](const auto &field, auto &destination)
        {
            if (field)
                destination = CaptureNative(field->owned, field->ghost);
        };
        for (size_t i = 0; i < s.relations.size(); ++i)
            capture(s.relations[i], _storage.relations[i]);
        capture(s.coords, _storage.coords);
        capture(s.nodeWallDist, _storage.nodeWallDist);
        capture(s.cellInfo, _storage.cellElemInfo);
        capture(s.bndInfo, _storage.bndElemInfo);
        capture(s.faceInfo, _storage.faceElemInfo);
        capture(s.edgeInfo, _storage.edgeElemInfo);
        capture(s.cellNodesPbi, _storage.cell2nodePbi);
        capture(s.bndNodesPbi, _storage.bnd2nodePbi);
        capture(s.cellFacesPbi, _storage.cell2facePbi);
        capture(s.faceNodesPbi, _storage.face2nodePbi);
        capture(s.cellEdgesPbi, _storage.cell2edgePbi);
        capture(s.edgeNodesPbi, _storage.edge2nodePbi);
    }
    MeshReadAccess::MeshReadAccess(const NativeMeshSnapshot &snapshot, Runtime::ExecutionContext context,
                                   Runtime::VersionTracker::ReadLease source, Runtime::VersionStamp stamp)
        : MeshReadAccess(snapshot, context)
    {
        // Adopt the source publication lease after all native buffers were
        // captured. Detached snapshot reads retain their independent identity.
        _sourceRead.emplace(std::move(source));
        _stamp = std::move(stamp);
    }
}
