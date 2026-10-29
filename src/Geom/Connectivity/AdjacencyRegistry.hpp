#pragma once

#include "DNDS/Runtime/RowTransfer.hpp"
#include "Geom/Access/RelationAccess.hpp"
#include "Geom/Connectivity/MeshTopologyTypes.hpp"

namespace DNDS::Geom
{
    /// Node-parallel periodic metadata, bound to one ordered cone. Bits are
    /// values in the parent's frame, never target indices. Native values and
    /// layouts remain protected by the same retained lease boundary.
    class RelationPeriodicBits
    {
        EntitySpace _source, _target;
        AdjacencyDescriptor _kind;
        Runtime::Array<NodePeriodicBits> _owned;
        std::optional<Runtime::Array<NodePeriodicBits>> _ghost;
        std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> _ownedRead;
        std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> _ghostRead;
        Runtime::PairReadView<NodePeriodicBits> _view;
        std::shared_ptr<MeshReadAccess> _meshRead;

    public:
        RelationPeriodicBits(const MeshAdjacencyRead &relation, Runtime::Array<NodePeriodicBits> owned,
                             std::optional<Runtime::Array<NodePeriodicBits>> ghost = {});
        explicit RelationPeriodicBits(const MeshAdjacencyRead &relation);
        void Validate(const MeshAdjacencyRead &relation) const;
        NodePeriodicBits operator()(index row, size_t slot) const { return _view(size_t(row), slot); }
    };

    struct NativeInterpolation
    {
        EntitySpace entities;
        MeshAdjacency parent2entity, entity2node, entity2parent;
        /// Both parent arrays retain the input parent's owned/appended order.
        /// Empty optionals mean nonperiodic, rather than a zero-width PBI row.
        std::optional<Runtime::Array<NodePeriodicBits>> parent2entityPbi, parent2entityGhostPbi, entity2nodePbi;
        Runtime::Array<ElemInfo> entityElemInfo;
    };

    struct AdjacencyGhostPublication
    {
        std::unordered_map<EntityKind, EntitySpace> spaces;
        std::unordered_map<AdjacencyDescriptor, MeshAdjacency, AdjacencyDescriptorHash> relations;
        /// Copies share each entity family's single owned-to-ghost mapping.
        std::unordered_map<EntityKind, EntityGhostMapping> routes;
    };

    using NativeGhostPublication = AdjacencyGhostPublication; // compatibility name

    /// CPU setup DSL. Registration retains source/target identities and read
    /// scopes; no compute-kernel dispatch or silent legacy-storage fallback.
    /// Collective operations must be called in the same order on every rank.
    /// Serialize registry operations; immutable global payloads and one-time
    /// local-to-global conversions are cached for repeated traversal hops.
    class AdjacencyRegistry
    {
        MPIInfo _mpi;
        std::unordered_map<AdjacencyDescriptor, std::shared_ptr<MeshAdjacencyRead>, AdjacencyDescriptorHash> _relations;
        std::unordered_map<EntityKind, EntitySpace> _spaces;
        mutable std::unordered_map<AdjacencyDescriptor, Runtime::Array<index>, AdjacencyDescriptorHash> _fetchValues;
        void ValidateCollectively(const std::function<void()> &validate) const;
        Runtime::Array<index> Fetch(const MeshAdjacencyRead &relation, const std::vector<index> &globals,
                                    const Runtime::RowTransferPlan *reusedRoute = nullptr,
                                    std::shared_ptr<Runtime::CommunicationContext> context = {}) const;
        static EntitySpace MakeSpace(EntityKind kind, index owned, const EntitySpace &anchor);

    public:
        explicit AdjacencyRegistry(MPIInfo mpi) : _mpi(mpi) {}
        /// Native setup mathematics; composition requires the intermediate
        /// support rows to be present. Legacy DSL entry points forward here.
        static MeshAdjacency Inverse(const MeshAdjacencyRead &cone, const MPIInfo &mpi);
        static MeshAdjacency Compose(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf = false);
        static MeshAdjacency ComposeAll(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf = false);
        static MeshAdjacency ComposeFiltered(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC,
                                             const AdjacencyDescriptor &resultKind,
                                             const std::function<bool(index, index, int)> &predicate,
                                             const std::function<bool(index, index, const std::vector<index> &)> &matchExtra = {});
        void Register(MeshAdjacencyRead relation);
        /// Rank-local lookup by full semantic definition; null means absent.
        /// The returned immutable read owns its leases and remains valid after
        /// registry destruction. Serialize lookup with registry modifications.
        std::shared_ptr<const MeshAdjacencyRead> Find(const AdjacencyDescriptor &kind) const;
        size_t RegisteredCount() const { return _relations.size(); }
        /// The caller selects required relation groups at acquisition.
        void RegisterMesh(UnstructuredMesh &mesh, MeshReadRequirements requirements);
        /// Traversal discovers arbitrary remote rows through native direct
        /// transient routes. Existing target ghosts are not required.
        GhostResult Evaluate(const CompiledGhostTree &tree) const;
        /// Collective detached publication: resolve all requested entity maps
        /// before compiling one shared native payload route per source family.
        AdjacencyGhostPublication BuildGhosted(const GhostResult &discovery) const;
        static NativeInterpolation Interpolate(const MeshAdjacencyRead &parent2node, EntityKind entityKind,
                                               const SubEntityQueryPbi &query, const OwnershipResolverMulti &resolver,
                                               const RelationPeriodicBits *periodic = nullptr);
    };

    using NativeConnectivity = AdjacencyRegistry; // compatibility name
}
