#pragma once

#include "RelationAccess.hpp"
#include "MeshTopologyTypes.hpp"

namespace DNDS::Geom
{
    /// Node-parallel periodic metadata, bound to one ordered cone. Bits are
    /// values in the parent's frame, never target indices. Native values and
    /// layouts remain protected by the same retained lease boundary.
    class RelationPeriodicBits
    {
        EntitySpace _source, _target;
        AdjKind _kind;
        Runtime::Array<NodePeriodicBits> _owned;
        std::optional<Runtime::Array<NodePeriodicBits>> _ghost;
        std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> _ownedRead;
        std::optional<Runtime::Array<NodePeriodicBits>::ReadLease> _ghostRead;
        Runtime::PairReadView<NodePeriodicBits> _view;
        std::shared_ptr<MeshReadAccess> _meshRead;

    public:
        RelationPeriodicBits(const RelationReadAccess &relation, Runtime::Array<NodePeriodicBits> owned,
                             std::optional<Runtime::Array<NodePeriodicBits>> ghost = {});
        explicit RelationPeriodicBits(const RelationReadAccess &relation);
        void Validate(const RelationReadAccess &relation) const;
        NodePeriodicBits operator()(index row, size_t slot) const { return _view(size_t(row), slot); }
    };

    struct NativeInterpolation
    {
        EntitySpace entities;
        NativeRelation parent2entity, entity2node, entity2parent;
        /// Both parent arrays retain the input parent's owned/appended order.
        /// Empty optionals mean nonperiodic, rather than a zero-width PBI row.
        std::optional<Runtime::Array<NodePeriodicBits>> parent2entityPbi, parent2entityGhostPbi, entity2nodePbi;
        Runtime::Array<ElemInfo> entityElemInfo;
    };

    struct NativeGhostPublication
    {
        std::unordered_map<EntityKind, EntitySpace> spaces;
        std::unordered_map<AdjKind, NativeRelation, AdjKindHash> relations;
        /// Copies share each entity family's single native route.
        std::unordered_map<EntityKind, RelationGhostRoute> routes;
    };

    /// CPU setup DSL. Registration retains source/target identities and read
    /// scopes; no compute-kernel dispatch or silent legacy-storage fallback.
    /// Collective operations must be called in the same order on every rank.
    /// Serialize registry operations; immutable global payloads and one-time
    /// local-to-global conversions are cached for repeated traversal hops.
    class NativeConnectivity
    {
        MPIInfo _mpi;
        std::unordered_map<AdjKind, std::shared_ptr<RelationReadAccess>, AdjKindHash> _relations;
        std::unordered_map<EntityKind, EntitySpace> _spaces;
        mutable std::unordered_map<AdjKind, Runtime::Array<index>, AdjKindHash> _fetchValues;
        void ValidateCollectively(const std::function<void()> &validate) const;
        Runtime::Array<index> Fetch(const RelationReadAccess &relation, const std::vector<index> &globals) const;
        static EntitySpace MakeSpace(EntityKind kind, index owned, const EntitySpace &anchor);

    public:
        explicit NativeConnectivity(MPIInfo mpi) : _mpi(mpi) {}
        /// Native setup mathematics; composition requires the intermediate
        /// support rows to be present. Legacy DSL entry points forward here.
        static NativeRelation Inverse(const RelationReadAccess &cone, const MPIInfo &mpi);
        static NativeRelation Compose(const RelationReadAccess &AB, const RelationReadAccess &BC, bool removeSelf = false);
        static NativeRelation ComposeAll(const RelationReadAccess &AB, const RelationReadAccess &BC, bool removeSelf = false);
        static NativeRelation ComposeFiltered(const RelationReadAccess &AB, const RelationReadAccess &BC,
                                              const AdjKind &resultKind,
                                              const std::function<bool(index, index, int)> &predicate,
                                              const std::function<bool(index, index, const std::vector<index> &)> &matchExtra = {});
        void Register(RelationReadAccess relation);
        /// The caller selects required relation groups at acquisition.
        void RegisterMesh(UnstructuredMesh &mesh, MeshReadRequirements requirements);
        /// Traversal discovers arbitrary remote rows through native direct
        /// transient routes. Existing target ghosts are not required.
        GhostResult Evaluate(const CompiledGhostTree &tree) const;
        /// Collective detached publication: resolve all requested entity maps
        /// before compiling one shared native payload route per source family.
        NativeGhostPublication BuildGhosted(const GhostResult &discovery) const;
        static NativeInterpolation Interpolate(const RelationReadAccess &parent2node, EntityKind entityKind,
                                               const SubEntityQueryPbi &query, const OwnershipResolverMulti &resolver,
                                               const RelationPeriodicBits *periodic = nullptr);
    };
}
