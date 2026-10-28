#pragma once

#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/IO/NativeMeshOutput.hpp"
#include <mutex>

namespace DNDS::Geom
{
    enum class TransitionCachePolicy
    {
        Invalidate,
        RebuildOutput
    };
    enum class CurrentMeshCheckpointPolicy
    {
        RequireNoAuxiliaryValues,
        DropAuxiliary
    };
    /// One serialized publication point for a mesh and its registered fields.
    /// Access to external field handles must use the same caller serialization.
    /// Old shared mesh owners and retained read scopes remain valid after swap.
    class MeshTransitionHandle
    {
        friend class PreparedMeshTransition;
        mutable std::mutex _mutex;
        std::shared_ptr<UnstructuredMesh> _mesh;

    public:
        explicit MeshTransitionHandle(std::shared_ptr<UnstructuredMesh> mesh);
        MeshTransitionHandle(const MeshTransitionHandle &) = delete;
        MeshTransitionHandle &operator=(const MeshTransitionHandle &) = delete;
        std::shared_ptr<UnstructuredMesh> Get() const;
    };

    /// Prepared host transition, collectively constructed and published.
    /// Native primary, inverse, facial and optional face-neighbor relations are
    /// assembled on detached storage. Faces may be preserved or explicitly destroyed
    /// and reconstructed. Native owned fields use the exact mesh reorder plan.
    /// Wall vectors move as Node companions. Edges and their registered fields
    /// preserve exact row slots; destroyKinds{Edge} explicitly reconstructs them
    /// and prohibits attached Edge fields/maps. Derived caches are explicitly invalidated
    /// or native output-only buffers reconstructed. Persistent identity and
    /// durable ancestry follow native row transfers. Setup-only parent/extraction
    /// row caches are invalidated on the new facade; the source is unchanged.
    ///
    /// The handle and registered field objects must outlive this object. Caller
    /// orchestration and external field writes remain serialized through Publish.
    class PreparedMeshTransition
    {
        MeshTransitionHandle *_handle;
        std::shared_ptr<UnstructuredMesh> _source, _next;
        std::unique_ptr<MeshReadAccess> _sourceRead;
        Runtime::VersionStamp _sourceStamp;
        std::vector<EntitySpace> _sourceSpaces;
        PreparedCompanionBundle _fields;
        std::vector<CompanionEntry> _attachments;
        std::vector<CompanionAttachment> _attachmentStamps;
        std::vector<std::pair<EntityKind, PreparedCompanion *>> _bindings;
        std::vector<bool> _bindingPrepared;
        bool _published = false;
        std::shared_ptr<const NativeMeshOutput> _output;

        explicit PreparedMeshTransition(MeshTransitionHandle &handle) : _handle(&handle) {}

    public:
        PreparedMeshTransition(const PreparedMeshTransition &) = delete;
        PreparedMeshTransition &operator=(const PreparedMeshTransition &) = delete;
        PreparedMeshTransition(PreparedMeshTransition &&) noexcept = default;
        PreparedMeshTransition &operator=(PreparedMeshTransition &&) = delete;

        /// Local failures at named checkpoints are agreed before subsequent MPI.
        /// The optional hook is for bounded preparation checks/fault injection;
        /// it must not initiate communication or mutate source mesh/field handles.
        static PreparedMeshTransition Prepare(
            MeshTransitionHandle &handle, const ReorderInput &input,
            const std::vector<CompanionEntry> &fields = {},
            const std::function<void(const char *)> &checkpoint = {}, int ghostLayers = 1,
            TransitionCachePolicy caches = TransitionCachePolicy::Invalidate);

        /// All validation and MPI agreement precede nonthrowing publication.
        /// A failed validation leaves the handle and every field unpublished.
        void Publish();
        /// Prepared output-only native payload, or null for Invalidate. It
        /// retains its own buffers after transition/source destruction.
        const std::shared_ptr<const NativeMeshOutput> &PreparedOutput() const { return _output; }
    };

    /// Collective primary-only checkpoint, including per-rank JSON. Preserves
    /// primary topology/geometry/original IDs/PBI; optional topology rebuilds.
    /// Wall values require explicit DropAuxiliary authorization. This does not
    /// checkpoint registered fields or preserve face/edge numbering.
    /// The legacy file schema is encoded one payload at a time from a retained
    /// read scope; no complete legacy mesh workspace is created. ghostLayers is
    /// retained for source compatibility; existing source closure is serialized.
    void WriteCurrentPrimaryMeshSnapshot(UnstructuredMesh &source, Serializer::SerializerBaseSSP serializer,
                                         const std::string &name, int ghostLayers = 1,
                                         CurrentMeshCheckpointPolicy policy = CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues);

    /// Compatibility spelling; same strict primary-only policy. Prefer the
    /// explicit primary name. All ranks agree on authority and serializer mode.
    void WriteCurrentMeshSnapshot(UnstructuredMesh &source, Serializer::SerializerBaseSSP serializer,
                                  const std::string &name, int ghostLayers = 1,
                                  CurrentMeshCheckpointPolicy policy = CurrentMeshCheckpointPolicy::RequireNoAuxiliaryValues);
}
