#pragma once

#include "RelationAccess.hpp"

namespace DNDS::Geom
{
    /// Node-parallel periodic metadata, bound to one ordered cone. Bits are
    /// values in the parent's frame, never target indices. Native and legacy
    /// storage share the same retained value/layout lease boundary.
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
        RelationPeriodicBits(const RelationReadAccess &relation, const tPbiPair &legacy);
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
    class NativeConnectivity
    {
        MPIInfo _mpi;
        std::unordered_map<AdjKind, std::shared_ptr<RelationReadAccess>, AdjKindHash> _relations;
        std::unordered_map<EntityKind, EntitySpace> _spaces;
        void ValidateCollectively(const std::function<void()> &validate) const;
        Runtime::Array<index> Fetch(const RelationReadAccess &relation, const std::vector<index> &globals) const;
        static EntitySpace MakeSpace(EntityKind kind, index owned, const EntitySpace &anchor);

    public:
        explicit NativeConnectivity(MPIInfo mpi) : _mpi(mpi) {}
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
