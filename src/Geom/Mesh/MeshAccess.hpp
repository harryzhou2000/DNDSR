#pragma once

#include "MeshView.hpp"
#include "DNDS/Runtime/Version.hpp"
#include <memory>
#include <vector>
#include <optional>

namespace DNDS::Geom
{
    struct UnstructuredMesh;
    class NativeMeshSnapshot;
    class NativeMeshState;

    /// A view exposes only explicitly requested relation groups. Local index
    /// validation occurs during acquisition, never in a release kernel's loop.
    struct MeshReadRequirements
    {
        bool faces = false;
        bool inverse = false;
        bool edges = false;
        bool cellFaceNeighbors = false;
        bool wallDistance = false;
        /// Require every captured face's cells to be locally dereferenceable,
        /// except the external second-cell sentinel. Generic CellFlux only
        /// requires owned faces and faces incident to owned cells.
        bool allFaceCellsLocal = false;
        static MeshReadRequirements Primary() { return {}; }
        static MeshReadRequirements CellFlux()
        {
            MeshReadRequirements result;
            result.faces = true;
            return result;
        }
    };

    /// Host owner of a plain host/CUDA query view. This scope retains both value
    /// and shape allocations plus the mesh read lease. Copying HostView or
    /// DeviceView does not retain this scope. Keep it through all uses, or attach
    /// a completion that joins every using stream. Acquisition is serialized
    /// with native setup/publication. Device reads use an explicit placement.
    class MeshReadAccess
    {
        friend struct UnstructuredMesh;
        friend class RelationReadAccess;
        friend class RelationPeriodicBits;
        friend class NativeMeshSnapshot;
        struct Hold
        {
            virtual ~Hold() = default;
            virtual void Complete(const Runtime::Completion &completion) = 0;
        };
        template <class T>
        struct ArrayHold final : Hold
        {
            Runtime::Array<T> array;
            typename Runtime::Array<T>::ReadLease read;
            ArrayHold(Runtime::Array<T> source, Runtime::ExecutionContext context)
                : array(std::move(source)), read(array.Read(context)) {}
            void Complete(const Runtime::Completion &completion) override { read.Complete(completion); }
        };

        Runtime::VersionTracker::ReadLease _meshRead;
        std::optional<Runtime::VersionTracker::ReadLease> _sourceRead;
        std::shared_ptr<const NativeMeshState> _nativeOwner;
        Runtime::VersionStamp _stamp;
        Runtime::ExecutionContext _context;
        MeshViewStorage _storage;
        std::vector<std::unique_ptr<Hold>> _holds;
        Runtime::Completion _completion;
        bool _completed = false;
        bool _active = true;

        template <class T>
        Runtime::PairReadView<T> CaptureNative(const Runtime::Array<T> &owned, const Runtime::Array<T> &ghost);
        void Validate(const MeshReadRequirements &requirements) const;
        void Restrict(const MeshReadRequirements &requirements);

    public:
        MeshReadAccess(const NativeMeshSnapshot &snapshot, Runtime::ExecutionContext context);
        MeshReadAccess(const NativeMeshSnapshot &snapshot, Runtime::ExecutionContext context,
                       Runtime::VersionTracker::ReadLease source, Runtime::VersionStamp stamp);
        MeshReadAccess(const MeshReadAccess &) = delete;
        MeshReadAccess &operator=(const MeshReadAccess &) = delete;
        MeshReadAccess(MeshReadAccess &&other) noexcept;
        MeshReadAccess &operator=(MeshReadAccess &&) = delete;
        ~MeshReadAccess();
        UnstructuredMeshView HostView() const;
        UnstructuredMeshView DeviceView() const;
        const Runtime::VersionStamp &Stamp() const { return _stamp; }
        /// Seal this scope after submitting its last use. Completion must cover
        /// the acquisition stream and joined other uses. No new view may be
        /// obtained afterward. A host worker retains this owner until it returns.
        void Complete(Runtime::Completion completion);
        /// Seal if needed, then wait for the declared prefix. Retains allocations
        /// and read leases until destruction; the caller must not reuse old raw
        /// views after sealing. No unrelated later stream work is included.
        void Wait();
    };
}
