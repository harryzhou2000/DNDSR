#pragma once

#include "Geom/Access/MeshView.hpp"
#include "Geom/Unstructured/UnstructuredMeshFwd.hpp"
#include "DNDS/Runtime/Version.hpp"
#include <memory>
#include <vector>
#include <optional>

namespace DNDS::Geom
{
    class UnstructuredMeshLegacyView;
    class UnstructuredMeshLegacyRead;

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

    /// Per-publication diagnostic audits, separate from required topology.
    /// Full checks are the default. Disabling either flag is an explicit caller
    /// assertion that immutable input identity/consistency was established by
    /// its producer. Shape/type/index/sentinel validity, required local closure,
    /// leases/generations and collective schedules are always checked. Closure
    /// controls diagnostic incidence reciprocity, not ghost construction or
    /// higher-order/periodic matching. Re-enable audits at checkpoints explicitly.
    struct MeshAuditOptions
    {
        bool persistentIdentity = true;
        bool closure = true;
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
        friend class UnstructuredMeshSnapshot;
        friend class UnstructuredMeshState;
        friend class UnstructuredMeshLegacyRead;
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
        std::shared_ptr<const UnstructuredMeshState> _nativeOwner;
        Runtime::VersionStamp _stamp;
        Runtime::ExecutionContext _context;
        MeshViewStorage _storage;
        std::vector<std::unique_ptr<Hold>> _holds;
        Runtime::Completion _completion;
        bool _completed = false;
        bool _active = true;

        template <class T>
        Runtime::PairReadView<T> CaptureNative(const Runtime::Array<T> &owned, const Runtime::Array<T> &ghost);
        void Validate(const MeshReadRequirements &requirements, MeshAuditOptions audit = {}) const;
        void Restrict(const MeshReadRequirements &requirements);

    public:
        MeshReadAccess(const UnstructuredMeshSnapshot &snapshot, Runtime::ExecutionContext context);
        MeshReadAccess(const UnstructuredMeshSnapshot &snapshot, Runtime::ExecutionContext context,
                       Runtime::VersionTracker::ReadLease source, Runtime::VersionStamp stamp);
        MeshReadAccess(const MeshReadAccess &) = delete;
        MeshReadAccess &operator=(const MeshReadAccess &) = delete;
        MeshReadAccess(MeshReadAccess &&other) noexcept;
        MeshReadAccess &operator=(MeshReadAccess &&) = delete;
        ~MeshReadAccess();
        UnstructuredMeshView HostView() const;
        UnstructuredMeshView DeviceView() const;
        UnstructuredMeshLegacyView LegacyHostView() const;
        UnstructuredMeshLegacyView LegacyDeviceView() const;
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
