#pragma once
/// @file MeshConnectivity.hpp
/// @brief Layered DAG of mesh adjacency relations with composable DSL operations.
///
/// MeshConnectivity manages adjacency (connectivity) tables between entity strata
/// of different topological depths (cells, faces, edges, nodes). It provides three
/// core operations:
///   - Inverse:         cone (A→B) → support (B→A)
///   - Compose:         A→B + B→C → A→C
///   - ComposeFiltered: A→B + B→C → A→C with on-the-fly predicate filtering
///
/// Cone adjacencies (downward: higher→lower depth) are ordered by element topology.
/// Support adjacencies (upward: lower→higher depth) are ordered by creation method
/// (typically the order entities were discovered during inversion).
///
/// Periodic bits (pbi) are only stored on cones whose target depth is 0 (nodes),
/// since pbi tracks how each node's coordinates transform under periodicity.

#include "DNDS/ArrayPair.hpp"
#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "Mesh_DeviceView.hpp"
#include "MeshRelationTypes.hpp"
#include "MeshTopologyTypes.hpp"
#include "ConnectivityComposeRows.hpp"

#include <variant>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace DNDS::Geom
{
    class RelationReadAccess;
    class NativeRelation;
    namespace ConnectivityRows
    {
        template <class Read, class Write>
        void Inverse(index nFrom, index nTo, Read &&read, Write &&write, const MPIInfo &mpi,
                     const std::function<index(index)> &fromGlobal,
                     const std::function<index(index)> &toGlobal,
                     const ssp<GlobalOffsetsMapping> &toMapping);
    }
    // Forward declaration for registerAdj overload
    template <class TPair>
    struct AdjPairTracked;

    // =================================================================
    // Type-erased adjacency storage
    // =================================================================

    /// Variant of all supported fixed-width adjacency pair types.
    /// tAdjPair (NonUniformSize) is the general variable-width CSR;
    /// tAdj1Pair..tAdj8Pair are compile-time fixed-width for performance
    /// on hot-path adjacencies (face2cell = Adj2, bnd2face = Adj1, etc.).
    using AdjVariant = std::variant<
        tAdjPair,   // variable-width (cell2node, cell2face, node2cell, cell2cell, ...)
        tAdj1Pair,  // 1 per row (bnd2face, face2bnd, cell2cellOrig, ...)
        tAdj2Pair,  // 2 per row (face2cell, bnd2cell)
        tAdj3Pair,  // 3 per row
        tAdj4Pair,  // 4 per row
        tAdj8Pair>; // 8 per row

    /// Convenience: create a shared AdjVariant from a specific pair type.
    /// Usage: `auto adj = makeAdjVariant(myTAdj2Pair);`
    /// or:    `auto adj = makeAdjVariant<tAdj2Pair>();` (empty)
    template <class TPair>
    static ssp<AdjVariant> makeAdjVariant(TPair &&pair)
    {
        return make_ssp<AdjVariant>(std::forward<TPair>(pair));
    }

    /// Create a shared AdjVariant holding a default-constructed TPair.
    template <class TPair>
    static ssp<AdjVariant> makeAdjVariant()
    {
        return make_ssp<AdjVariant>(TPair{});
    }

    // =================================================================
    // narrowAdjToFixed: variable-width → fixed-width conversion
    // =================================================================

    /// Copy a variable-width adjacency (father-only) into a fixed-width
    /// target. Each row is truncated or padded with `fill` to exactly
    /// `target_rs` entries.
    ///
    /// @tparam target_rs  Compile-time row width of the target.
    /// @tparam source_rs  Row width of the source (typically NonUniformSize).
    /// @param source      Source adjacency (father-only or father+son).
    /// @param target      Target fixed-width adjacency. Father must be pre-Resize'd
    ///                    to at least `nRows`.
    /// @param nRows       Number of rows to copy.
    /// @param fill        Value for missing entries (default: UnInitIndex).
    template <rowsize target_rs, rowsize source_rs = NonUniformSize>
    static void narrowAdjToFixed(
        const ArrayAdjacencyPair<source_rs> &source,
        ArrayAdjacencyPair<target_rs> &target,
        index nRows,
        index fill = UnInitIndex)
    {
        static_assert(target_rs > 0, "narrowAdjToFixed: target must be fixed-width");
        for (index i = 0; i < nRows; i++)
        {
            auto row = source.father->operator[](i);
            for (rowsize j = 0; j < target_rs; j++)
                target.father->operator()(i, j) =
                    (j < static_cast<rowsize>(row.size())) ? row[j] : fill;
        }
    }

    // =================================================================
    // ConeAdj: downward adjacency (higher depth → lower depth)
    // =================================================================

    /// A cone (downward) adjacency from entities at `fromDepth` to entities at
    /// `toDepth`. Row order is defined by element topology and must not be
    /// permuted (e.g., cell2node ordering determines element shape functions).
    ///
    /// Periodic bits (`pbi`) are available on all cones as an optional
    /// payload.  For non-periodic meshes or cones where the caller does
    /// not populate pbi, the member remains uninitialised and `hasPbi()`
    /// returns false.
    ///
    /// Cones to nodes (`toDepth == 0`) are the most common carrier because
    /// pbi naturally lives on node-periodic data, but cell2face or cell2edge
    /// cones may also carry pbi when needed.
    ///
    /// The adjacency data is stored via `ssp<AdjVariant>` for shared ownership.
    /// Both the DAG and legacy mesh members can share the same allocation.
    struct ConeAdj
    {
        int fromDepth{-1};   ///< Source stratum (e.g., dim for cells, dim-1 for faces)
        int toDepth{-1};     ///< Target stratum (e.g., 0 for nodes, dim-1 for faces)
        ssp<AdjVariant> adj; ///< Shared adjacency pair (typed by row width)
        tPbiPair pbi;        ///< Periodic bits per entry (optional, any toDepth)

        /// Access as variable-width tAdjPair. Throws if adj holds a fixed-width type.
        tAdjPair &asAdj() { return std::get<tAdjPair>(*adj); }
        [[nodiscard]] const tAdjPair &asAdj() const { return std::get<tAdjPair>(*adj); }

        /// Access as fixed-width tAdj2Pair (e.g., for cell2face with 2 entries).
        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(*adj); }
        [[nodiscard]] const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(*adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(*adj); }
        [[nodiscard]] const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(*adj); }

        /// Typed access: `as<tAdj2Pair>()` etc.
        template <class TPair>
        TPair &as() { return std::get<TPair>(*adj); }
        template <class TPair>
        [[nodiscard]] const TPair &as() const { return std::get<TPair>(*adj); }

        /// Number of local (father) rows.
        [[nodiscard]] index fatherSize() const
        {
            if (!adj)
                return 0;
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, *adj);
        }

        /// Check if the adjacency pair is initialized (father is non-null).
        [[nodiscard]] bool initialized() const
        {
            if (!adj)
                return false;
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, *adj);
        }

        /// Check if pbi is attached.
        [[nodiscard]] bool hasPbi() const { return bool(pbi.father); }
    };

    // =================================================================
    // SupportAdj: upward adjacency (lower depth → higher depth)
    // =================================================================

    /// A support (upward) adjacency from entities at `fromDepth` to entities at
    /// `toDepth`. Row order is determined by the creation method (typically
    /// Inverse), and is stable but not semantically ordered.
    ///
    /// Supports never carry periodic bits — pbi is a property of cones.
    ///
    /// The adjacency data is stored via `ssp<AdjVariant>` for shared ownership.
    struct SupportAdj
    {
        int fromDepth{-1};   ///< Source stratum (e.g., 0 for nodes)
        int toDepth{-1};     ///< Target stratum (e.g., dim for cells)
        ssp<AdjVariant> adj; ///< Shared adjacency pair (typed by row width)

        /// Access as variable-width tAdjPair.
        tAdjPair &asAdj() { return std::get<tAdjPair>(*adj); }
        [[nodiscard]] const tAdjPair &asAdj() const { return std::get<tAdjPair>(*adj); }

        tAdj2Pair &asAdj2() { return std::get<tAdj2Pair>(*adj); }
        [[nodiscard]] const tAdj2Pair &asAdj2() const { return std::get<tAdj2Pair>(*adj); }

        tAdj1Pair &asAdj1() { return std::get<tAdj1Pair>(*adj); }
        [[nodiscard]] const tAdj1Pair &asAdj1() const { return std::get<tAdj1Pair>(*adj); }

        /// Typed access: `as<tAdj2Pair>()` etc.
        template <class TPair>
        TPair &as() { return std::get<TPair>(*adj); }
        template <class TPair>
        [[nodiscard]] const TPair &as() const { return std::get<TPair>(*adj); }

        [[nodiscard]] index fatherSize() const
        {
            if (!adj)
                return 0;
            return std::visit([](const auto &p) -> index
                              { return p.father ? p.father->Size() : 0; }, *adj);
        }

        [[nodiscard]] bool initialized() const
        {
            if (!adj)
                return false;
            return std::visit([](const auto &p) -> bool
                              { return bool(p.father); }, *adj);
        }
    };

    /// Result of distributed interpolation with globally-unique B entities.
    ///
    /// Given A→C cone (global C indices, A is ghosted via A→C→A), creates
    /// globally-unique B entities (faces or edges) with ownership resolution.
    ///
    /// ## Outputs
    ///
    ///   - **parent2entity**: A→B mapping in global B indices. Father = local A,
    ///     son = ghost A. Slot ordering matches element topology: slot j is
    ///     face/edge j as defined by Element::ObtainFace(j)/ObtainEdge(j).
    ///     Some son entries may be UnInitIndex (fully-ghost B not resolved by
    ///     the push protocol — resolved later by the caller's ghost B pull).
    ///
    ///   - **parent2entityPbi**: Parallel to parent2entity. One NodePeriodicBits
    ///     per (A, B) slot — the uniform XOR between A's sub-entity node-pbi
    ///     and B's stored entity2nodePbi. To get B's node coords in A's frame:
    ///     apply parent2entityPbi to entity2nodePbi, then GetCoordByBits.
    ///     For faces, this is at most 1-bit (a face crosses one periodic
    ///     boundary). For edges, can be multi-bit (e.g., P1|P2 for a corner
    ///     edge). Zero for the first parent. Empty if not periodic.
    ///     Computed locally — no MPI push needed (depends only on cell2nodePbi
    ///     and entity2nodePbi, both available after Step 2b).
    ///
    ///   - **entity2node**: B→C mapping in global C indices. Father-only (owned B).
    ///     Node ordering is from the first-discovered parent's ExtractFaceNodes/
    ///     ExtractEdgeNodes call — this is the entity's own canonical ordering.
    ///
    ///   - **entity2nodePbi**: B→C pbi. Father-only. Parallel to entity2node.
    ///     Pbi from the first-discovered parent's perspective (the entity's own
    ///     frame). Other parents' perspectives differ by parent2entityPbi.
    ///     Empty if not periodic.
    ///
    ///   - **entity2parent**: B→A mapping in global A indices. Father-only.
    ///     Variable-width: 1 (boundary) or 2 (internal) for faces, 1..N for edges.
    ///     Complete under A→C→A ghosting (all parents sharing any C-vertex with
    ///     a local A are present as ghosts, so all parents of any locally-visible
    ///     B entity are enumerated).
    ///
    ///   - **entityElemInfo**: Per-entity element info. Father-only.
    ///
    /// ## Ghost B entities
    ///
    /// NOT included. Use evaluateGhostTree(A → A2B → B) to pull them, then
    /// BorrowAndPull on entity2node / entity2nodePbi / entityElemInfo.
    /// parent2entityPbi for ghost A parents is already populated (the push
    /// only carries the global B index; pbi is local). After ghost B pull,
    /// entity2parent for ghost B can be obtained via Inverse(parent2entity)
    /// or by ghost-pulling entity2parent itself.
    /// @tparam e2p_rs  Row-size of entity2parent (NonUniformSize = variable,
    ///                 2 = fixed for faces, etc.).
    template <rowsize e2p_rs = NonUniformSize>
    struct InterpolateGlobalResultT
    {
        tAdjPair parent2entity;                   ///< A → B (global B indices). Father = local A,
                                                  ///< son = ghost A. Slot j = face/edge j per topology.
        tPbiPair parent2entityPbi;                ///< A → B pbi (parallel to parent2entity).
                                                  ///< Uniform XOR: A's sub-entity node-pbi vs B's stored
                                                  ///< entity2nodePbi. Faces: at most 1 bit. Edges: multi-bit.
                                                  ///< 0 for B's first parent. Empty if not periodic.
                                                  ///< No push needed — computed locally in Step 2b.
        tAdjPair entity2node;                     ///< B → C (global C indices). Father-only (owned B).
                                                  ///< Node order: first-discovered parent's extraction order.
        ArrayAdjacencyPair<e2p_rs> entity2parent; ///< B → A (global A indices). Father-only (owned B).
                                                  ///< Width 1-2 for faces, 1-N for edges.
                                                  ///< Complete under A→C→A ghosting.
        tPbiPair entity2nodePbi;                  ///< B → C pbi. Father-only (owned B).
                                                  ///< First-discovered parent's perspective (entity's own frame).
                                                  ///< Parallel to entity2node. Empty if not periodic.
        tElemInfoArrayPair entityElemInfo;        ///< Per-B elem info. Father-only (owned B).
        index nOwnedEntities{0};                  ///< Number of owned B entities (father size).
    };

    /// Default: all fields variable-width.
    using InterpolateGlobalResult = InterpolateGlobalResultT<>;

    // =================================================================
    // InterpolateResult: output of LOCAL sub-entity interpolation
    // =================================================================

    /// Result of interpolating (extracting) sub-entities from parent→node connectivity.
    ///
    /// Given parent→node (e.g., cell→node), Interpolate creates intermediate entities
    /// (e.g., faces or edges) by extracting sub-entities from element topology,
    /// deduplicating by sorted vertex comparison, and building both parent→entity
    /// and entity→node adjacencies.
    ///
    /// All indices are local (0-based within the input arrays). No MPI communication
    /// is performed — the caller is responsible for providing a complete view
    /// (local + ghost cells) and for subsequent ownership resolution / ghost exchange.
    /// Result of local (rank-only) sub-entity interpolation.
    ///
    /// Given parent→node (e.g., cell→node), Interpolate creates intermediate entities
    /// (e.g., faces or edges) by extracting sub-entities from element topology,
    /// deduplicating by sorted vertex comparison (+ optional matchExtra for periodic),
    /// and building both parent→entity and entity→node adjacencies.
    ///
    /// All indices are local (0-based within the input arrays). No MPI communication
    /// is performed. The caller provides a complete view (local + ghost cells) and
    /// handles subsequent ownership resolution / ghost exchange.
    ///
    /// ## Ordering guarantees
    ///
    ///   - parent2entity[iParent][j] corresponds to sub-entity j of parent iParent
    ///     as defined by query.numSubEntities / query.describe / query.extractNodes.
    ///   - entity2node[iEnt] stores nodes in the order extracted by the **first parent**
    ///     that created entity iEnt. Later parents that find the same entity do not
    ///     alter the node order.
    ///   - entity2parent[iEnt] lists parents in discovery order (first parent first).
    ///
    /// @tparam p2e_rs  Row-size of parent2entity (NonUniformSize = variable).
    /// @tparam e2n_rs  Row-size of entity2node (NonUniformSize = variable).
    /// @tparam e2p_rs  Row-size of entity2parent (NonUniformSize = variable).
    template <rowsize p2e_rs = NonUniformSize,
              rowsize e2n_rs = NonUniformSize,
              rowsize e2p_rs = NonUniformSize>
    struct InterpolateResultT
    {
        ArrayAdjacencyPair<p2e_rs> parent2entity; ///< parent → entities. Father-only. Slot j = sub-entity j.
        ArrayAdjacencyPair<e2n_rs> entity2node;   ///< entity → nodes. Father-only. First-parent extraction order.
        ArrayAdjacencyPair<e2p_rs> entity2parent; ///< entity → parents. Father-only. Variable-width:
                                                  ///< 1 for boundary faces, 2 for internal faces, N for edges.
                                                  ///< Discovery order (first parent first).
        std::vector<ElemInfo> entityElemInfo;     ///< Per-entity element info (zone=0, type from SubEntityDesc::typeTag).
        std::vector<std::vector<NodePeriodicBits>> parent2entityPbi;
        ///< Per-parent, per-sub pbi. Parallel to parent2entity.
        ///< Populated by InterpolateGlobal Step 2b, not by
        ///< InterpolateLocal itself. Empty if not periodic or
        ///< if InterpolateLocal was called directly.
        index nEntities{0}; ///< Total number of unique entities created.
    };

    /// Default InterpolateResult: all fields are variable-width (NonUniformSize).
    using InterpolateResult = InterpolateResultT<>;

    // (InterpolateDistributedResult and legacy types moved above)

    // =================================================================
    // InterpolateDistributedResult: output of distributed interpolation
    // =================================================================

    /// Result of legacy distributed interpolation (2-parent only).
    ///
    /// Extends InterpolateResult with ghost communication: entity arrays have
    /// father (owned) + son (ghost) populated. parent2entity entries use
    /// local-appended entity indices that span both father and son.
    ///
    /// **Legacy**: superseded by InterpolateGlobal for production use.
    /// Retained for backward compatibility with InterpolateDistributed.
    /// Limitations: entity2parent is fixed-2 (tAdj2Pair), does not support
    /// N-parent edges, does not produce parent2entityPbi.
    ///
    /// @tparam p2e_rs  Row-size of parent2entity (NonUniformSize = variable).
    /// @tparam e2n_rs  Row-size of entity2node (NonUniformSize = variable).
    /// @tparam e2p_rs  Row-size of entity2parent (2 = fixed face-to-parent).
    template <rowsize p2e_rs = NonUniformSize,
              rowsize e2n_rs = NonUniformSize,
              rowsize e2p_rs = 2>
    struct InterpolateDistributedResultT
    {
        ArrayAdjacencyPair<p2e_rs> parent2entity; ///< parent → entities. Father = local parents,
                                                  ///< son = ghost parents. Local-appended entity indices.
        ArrayAdjacencyPair<e2n_rs> entity2node;   ///< entity → nodes. Father = owned, son = ghost.
        ArrayAdjacencyPair<e2p_rs> entity2parent; ///< entity → (parentL, parentR).
                                                  ///< Father = owned, son = ghost. Local-appended parent indices.
                                                  ///< parentR = UnInitIndex for boundary entities.
        tElemInfoArrayPair entityElemInfo;        ///< Per-entity element info. Father = owned, son = ghost.
        index nOwnedEntities{0};                  ///< Number of owned entities (father size).
    };

    /// Default InterpolateDistributedResult: entity2parent is fixed-2 (as before).
    using InterpolateDistributedResult = InterpolateDistributedResultT<>;

    // =================================================================
    // MeshConnectivity: the layered DAG
    // =================================================================

    /// Manages the layered DAG of mesh adjacency relations.
    ///
    /// Cones (downward adjacencies) and supports (upward adjacencies) are stored
    /// in separate vectors. Each is identified by a `(fromDepth, toDepth)` pair
    /// using dynamic depth tags (e.g., `(dim, 0)` for cell→node).
    ///
    /// Adjacency data is stored via `ssp<AdjVariant>` for shared ownership.
    /// Both the DAG's ConeAdj/SupportAdj and the legacy mesh members can share
    /// the same underlying arrays (shallow copy of ssp<Array> pointers).
    ///
    /// The adjacency registry (`adjRegistry`) maps AdjKind tags to
    /// `ssp<AdjVariant>` for use by the ghost traversal system. No raw pointers.
    /// Legacy adapters and complete native semantic descriptors may coexist.
    /// Endpoint-only lookup never falls back to a different named relation.
    struct MeshConnectivity
    {
        /// Storage-neutral setup paths. These retain their readers' explicit
        /// source/target spaces; no legacy width variant or topology copy is
        /// required. Inverse remains collective; composition needs all
        /// referenced intermediate rows already present in the reader.
        static NativeRelation Inverse(const RelationReadAccess &cone, const MPIInfo &mpi);
        static NativeRelation Compose(const RelationReadAccess &AB, const RelationReadAccess &BC, bool removeSelf = false);
        static NativeRelation ComposeAll(const RelationReadAccess &AB, const RelationReadAccess &BC, bool removeSelf = false);
        /// Custom and exact predicates must carry a named/versioned descriptor.
        static NativeRelation ComposeFiltered(const RelationReadAccess &AB, const RelationReadAccess &BC,
                                              const AdjKind &resultKind,
                                              const std::function<bool(index, index, int)> &predicate,
                                              const std::function<bool(index, index, const std::vector<index> &)> &matchExtra = {});
        int meshDim{0};
        std::vector<ConeAdj> cones;
        std::vector<SupportAdj> supports;

        // -----------------------------------------------------------------
        // Adjacency registry (restricted set for ghost traversal)
        // -----------------------------------------------------------------

        /// Maps AdjKind to the shared AdjVariant used by ghost chain evaluation.
        /// Only direct cones/supports and intra-level (via Node/Face)
        /// adjacencies are allowed. Registered via registerAdj().
        /// Owns shared references (ssp) — no raw pointers.
        std::unordered_map<AdjKind, ssp<AdjVariant>, AdjKindHash> adjRegistry;

        /// Per-EntityKind global offsets mapping (for ownership determination).
        /// Must be registered for every EntityKind that appears as a root anchor
        /// or COLLECT target in any ghost chain.
        std::unordered_map<EntityKind, ssp<GlobalOffsetsMapping>> globalMappings;

        /// Register an adjacency for ghost traversal.
        /// The ssp shares ownership with the ConeAdj/SupportAdj that holds the data.
        /// Overwrites any existing registration for the same AdjKind.
        void registerAdj(AdjKind kind, ssp<AdjVariant> adjPtr);

        /// Convenience: register a specific pair type (tAdjPair, tAdj2Pair, etc.).
        /// Stores a father-only shallow reference. The evaluator creates its
        /// own scratch transformers internally for any ghost data it needs.
        template <class TPair>
        void registerAdj(AdjKind kind, TPair &pair)
        {
            auto adjVar = makeAdjVariant<TPair>();
            auto &stored = std::get<TPair>(*adjVar);
            stored.father = pair.father;
            // son and ghost mapping are NOT copied. The evaluator's scratch
            // pull creates its own temporary son/transformer when needed.
            registerAdj(kind, std::move(adjVar));
        }

        /// Const overload: same as above but accepts a const reference.
        /// Safe because we only copy the father shared_ptr (no mutation).
        template <class TPair>
        void registerAdj(AdjKind kind, const TPair &pair)
        {
            auto adjVar = makeAdjVariant<TPair>();
            auto &stored = std::get<TPair>(*adjVar);
            stored.father = pair.father;
            registerAdj(kind, std::move(adjVar));
        }

        /// Overload for AdjPairTracked<TPair>: unwrap to base TPair.
        template <class TPair>
        void registerAdj(AdjKind kind, AdjPairTracked<TPair> &pair)
        {
            registerAdj(kind, static_cast<TPair &>(pair));
        }

        /// Const overload for AdjPairTracked<TPair>.
        template <class TPair>
        void registerAdj(AdjKind kind, const AdjPairTracked<TPair> &pair)
        {
            registerAdj(kind, static_cast<const TPair &>(pair));
        }

        /// Register a GlobalOffsetsMapping for an EntityKind.
        void registerGlobalMapping(EntityKind kind, const ssp<GlobalOffsetsMapping> &gm);

        /// Resolve an AdjKind to the registered AdjVariant.
        /// Returns nullptr if not registered.
        ssp<AdjVariant> resolveAdj(AdjKind kind) const;

        /// Resolve a GlobalOffsetsMapping for an EntityKind.
        /// Returns nullptr if not registered.
        const ssp<GlobalOffsetsMapping> &getGlobalMapping(EntityKind kind) const;

        /// Check whether an AdjKind is registered.
        [[nodiscard]] bool hasAdj(AdjKind kind) const;

        // -----------------------------------------------------------------
        // Cone management
        // -----------------------------------------------------------------

        /// Add a new cone for (fromDepth, toDepth). Returns reference.
        /// Asserts no duplicate exists.
        ConeAdj &addCone(int fromDepth, int toDepth);

        /// Find a cone by (fromDepth, toDepth). Returns nullptr if not found.
        ConeAdj *findCone(int fromDepth, int toDepth);
        const ConeAdj *findCone(int fromDepth, int toDepth) const;
        bool hasCone(int fromDepth, int toDepth) const;

        // -----------------------------------------------------------------
        // Support management
        // -----------------------------------------------------------------

        /// Add a new support for (fromDepth, toDepth). Returns reference.
        /// Asserts no duplicate exists.
        SupportAdj &addSupport(int fromDepth, int toDepth);

        /// Find a support by (fromDepth, toDepth). Returns nullptr if not found.
        SupportAdj *findSupport(int fromDepth, int toDepth);
        const SupportAdj *findSupport(int fromDepth, int toDepth) const;
        bool hasSupport(int fromDepth, int toDepth) const;

        // -----------------------------------------------------------------
        // Core DSL operations (static, templated on adjacency row-size)
        // -----------------------------------------------------------------
        // Input adjacencies can be any ArrayAdjacencyPair<rs> (fixed or variable).
        // Output adjacencies are always tAdjPair (NonUniformSize) because the
        // result width is generally unpredictable at compile time.
        // Default template args = NonUniformSize for backward compatibility.

        /// Invert a cone to get its support (distributed, MPI push-back).
        ///
        /// Given a cone adjacency A→B (for each A-entity, list of B-entities),
        /// compute the support adjacency B→A (for each B-entity, list of A-entities
        /// that reference it). Result is globally complete via MPI push-back.
        ///
        /// The input cone can be any row-size (fixed or variable). The output
        /// support is always variable-width (tAdjPair) because fan-in is not
        /// known at compile time.
        ///
        /// @tparam cone_rs      Row-size of the input cone (NonUniformSize or fixed).
        /// @param cone         Adjacency: from → to (global indices).
        /// @param nToLocal     Local number of "to" entities on this rank.
        /// @param mpi          MPI communicator.
        /// @param fromLocal2Global  Maps local from-entity index to global index.
        /// @param toLocal2Global    Maps local to-entity index to global index.
        /// @param toGlobalMapping   Global mapping for to-entities (ownership lookup).
        /// @return             CSR adjacency: to → from (global, complete). Father-only.
        template <rowsize cone_rs = NonUniformSize>
        static tAdjPair Inverse(
            const ArrayAdjacencyPair<cone_rs> &cone,
            index nToLocal,
            const MPIInfo &mpi,
            const std::function<index(index)> &fromLocal2Global,
            const std::function<index(index)> &toLocal2Global,
            const ssp<GlobalOffsetsMapping> &toGlobalMapping);

        /// Compose two adjacencies: A→B + B→C → A→C (delegates to ComposeFiltered
        /// with SharedCountPredicate{1}).
        ///
        /// @tparam rs_AB  Row-size of AB adjacency.
        /// @tparam rs_BC  Row-size of BC adjacency.
        /// @tparam out_rs Row-size of output (NonUniformSize or fixed).
        template <rowsize rs_AB = NonUniformSize, rowsize rs_BC = NonUniformSize,
                  rowsize out_rs = NonUniformSize>
        static ArrayAdjacencyPair<out_rs> Compose(
            const ArrayAdjacencyPair<rs_AB> &AB,
            const ArrayAdjacencyPair<rs_BC> &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            bool removeSelf = false);

        /// Compose two adjacencies with on-the-fly predicate filtering.
        ///
        /// For each row a in AB, iterates b in AB[a], collects c in BC[b].
        /// Counts shared B-entities per candidate, applies predicate.
        ///
        /// ## Ghost prerequisite
        ///
        /// **BC must contain (father+son) all B-entities referenced by AB.**
        /// The caller must ghost-pull BC for every global B-index that appears
        /// in any row of AB. The `bGlobal2Local` map must cover all such
        /// B-globals. If a B-global from AB is not in `bGlobal2Local`, the
        /// method asserts (fatal error indicating incomplete ghost pull).
        ///
        /// Typical pattern: before calling ComposeFiltered, the caller builds
        /// BC (e.g., node2cell via Inverse), then ghost-pulls it for all
        /// off-rank B-entities referenced by local AB rows.
        ///
        /// @tparam rs_AB        Row-size of AB adjacency.
        /// @tparam rs_BC        Row-size of BC adjacency.
        /// @tparam Predicate    Predicate type.
        /// @param AB           A → B (father only, global B indices).
        /// @param BC           B → C (father+son, global C indices).
        /// @param nALocal      Number of local A-entities.
        /// @param bGlobal2Local Maps global B-index to local-appended index in BC.
        ///                      Must contain every B-global that appears in AB.
        /// @param aLocal2Global Maps local A-index to global A-index.
        /// @param pred         Predicate(a_global, c_global, nShared) → keep?
        /// @param matchExtra   Optional second predicate called after pred passes.
        ///                     Receives (aLocal, cGlobal, sharedBGlobals) where
        ///                     sharedBGlobals is the list of B-entities through
        ///                     which A and C are connected. If set and returns
        ///                     false, the (A, C) pair is rejected.
        ///                     Used for periodic pbi filtering.
        /// @return             A → C adjacency, father-only. Variable or fixed-width
        ///                     depending on out_rs.
        /// @tparam out_rs       Row-size of output (NonUniformSize = variable,
        ///                      or fixed: rows shorter than out_rs are padded with
        ///                      UnInitIndex, rows longer trigger assert).
        template <rowsize rs_AB = NonUniformSize, rowsize rs_BC = NonUniformSize,
                  rowsize out_rs = NonUniformSize, class Predicate = SharedCountPredicate>
        static ArrayAdjacencyPair<out_rs> ComposeFiltered(
            const ArrayAdjacencyPair<rs_AB> &AB,
            const ArrayAdjacencyPair<rs_BC> &BC,
            index nALocal,
            const std::unordered_map<index, index> &bGlobal2Local,
            const std::function<index(index)> &aLocal2Global,
            Predicate &&pred,
            const std::function<bool(index aLocal, index cGlobal,
                                     const std::vector<index> &sharedBGlobals)>
                &matchExtra = nullptr);

        // -----------------------------------------------------------------
        // InterpolateLocal: rank-local sub-entity extraction (no MPI)
        // -----------------------------------------------------------------

        /// Extract sub-entities (faces or edges) from parent→node connectivity.
        /// This is a **local-only** operation — no MPI communication.
        ///
        /// Enumerates and deduplicates B entities within a contiguous block of
        /// parent A entities (typically local + ghost cells). Produces local-indexed
        /// parent→entity, entity→node, and entity→parent adjacencies.
        ///
        /// Used internally by InterpolateGlobal (Step 1). Can also be used
        /// directly for rank-local analysis or testing.
        ///
        /// ## Deduplication
        ///
        /// Two sub-entities are considered the same B entity if:
        ///   1. They have the same typeTag (from SubEntityDesc).
        ///   2. Their sorted vertex sets (first nVertices nodes) are equal.
        ///   3. query.matchExtra (if set) returns true — used for periodic meshes
        ///      to implement the collaborating pbi check.
        ///
        /// ## Ordering guarantees
        ///
        ///   - parent2entity[iParent][j]: slot j = sub-entity j per query callbacks.
        ///   - entity2node[iEnt]: nodes in first-discovered parent's extraction order.
        ///   - entity2parent[iEnt]: parents in discovery order (first parent first).
        ///
        /// ## Ghost note
        ///
        /// The caller typically passes all A parents (father+son) so that
        /// shared sub-entities are deduplicated within the local view. If ghost
        /// A parents are omitted, shared B entities at rank boundaries will
        /// appear as separate single-sided entities.
        ///
        /// ## Pbi note
        ///
        /// InterpolateLocal does not handle pbi. parent2entityPbi in the result
        /// is left empty. InterpolateGlobal computes it in Step 2b after calling
        /// InterpolateLocal, using the parent2nodePbi input and extractPbi callback.
        ///
        /// @param parent2node    Parent → nodes (father-only or father+son).
        ///                       Accessed via operator[] for indices [0, nParent).
        /// @tparam p2n_rs       Row-size of parent2node (NonUniformSize or fixed).
        /// @param query          User-provided callbacks describing sub-entity topology.
        /// @param nParent        Number of parent entities to process.
        /// @param nNode          Total number of nodes (for reverse-index sizing).
        /// @param mpi            MPI info (only for array allocation, no communication).
        /// @return               InterpolateResult with all adjacencies (local indices).
        template <rowsize p2n_rs = NonUniformSize>
        static InterpolateResult Interpolate(
            const ArrayAdjacencyPair<p2n_rs> &parent2node,
            const SubEntityQuery &query,
            index nParent,
            index nNode,
            const MPIInfo &mpi);

        /// Legacy distributed interpolation (2-parent only, no pbi output).
        ///
        /// Superseded by InterpolateGlobal for production use. Retained for
        /// backward compatibility. Limitations vs InterpolateGlobal:
        ///   - entity2parent is fixed-2 (tAdj2Pair), cannot represent N-parent edges.
        ///   - No parent2entityPbi output.
        ///   - No entity2nodePbi output.
        ///   - Uses push-based ghost exchange (not pull via evaluateGhostTree).
        ///
        /// Extends Interpolate by:
        ///   1. Calling the local Interpolate on all parents (local + ghost).
        ///   2. Using the OwnershipResolver callback to decide which rank owns
        ///      each entity.
        ///   3. Compacting owned entities and assigning global IDs.
        ///   4. Push-based ghost exchange so non-owning ranks receive the
        ///      entities they need.
        ///   5. Resolving parent2entity so all entries (for both local and ghost
        ///      parents) use local-appended entity indices.
        ///
        /// @param parent2node       CSR: parent → nodes (father + son).
        /// @param parentGhostMapping Ghost mapping for parent entities (for global
        ///                           index lookups). Must map local-appended index
        ///                           to global parent index.
        /// @param query             Sub-entity topology callbacks.
        /// @param nLocalParents     Number of local (owned/father) parents.
        /// @param nTotalParents     Total parents (father + son).
        /// @param nNode             Total number of nodes.
        /// @param resolver          Ownership callback.
        /// @param nodeGhostMapping  Ghost mapping for nodes (local-appended → global).
        ///                          Used to convert entity2node to global before push
        ///                          and back to local on the receiver.
        /// @param mpi               MPI communicator.
        /// @return InterpolateDistributedResult with owned + ghost entities.
        template <rowsize p2n_rs = NonUniformSize>
        static InterpolateDistributedResult InterpolateDistributed(
            const ArrayAdjacencyPair<p2n_rs> &parent2node,
            const OffsetAscendIndexMapping &parentGhostMapping,
            const SubEntityQuery &query,
            index nLocalParents,
            index nTotalParents,
            index nNode,
            const OwnershipResolver2 &resolver,
            const OffsetAscendIndexMapping &nodeGhostMapping,
            const MPIInfo &mpi);

        /// Distributed interpolation producing globally-unique B entities.
        ///
        /// Given an A→C cone with pbi (where A is ghosted via A→C→A), creates
        /// globally-unique B entities (faces or edges). B entities are created
        /// only from **local (non-ghost) A parents**. Ghost A parents are used
        /// solely for deduplication — when a local A's sub-entity shares C-vertices
        /// with a ghost A's sub-entity, they are recognized as the same B entity.
        ///
        /// ## Ghost prerequisites (caller's responsibility)
        ///
        /// **A must be ghosted by at least the A→C→A 1-hop node-neighbor set.**
        /// Every A entity sharing at least one C-vertex with a local A must be
        /// present as a ghost. This guarantees:
        ///   - All B entities at rank boundaries are locally deduplicated.
        ///   - entity2parent is complete (all parents of any locally-visible B
        ///     are enumerated).
        ///
        /// If the ghost set is insufficient, shared B entities across ranks will
        /// appear as single-sided. This is a silent error the method cannot detect
        /// (boundary faces/edges are legitimately single-sided).
        ///
        /// **C (nodes) must be ghosted** to cover all C-vertices referenced by
        /// ghost A parents. Typically guaranteed by the A→C→A ghost chain.
        ///
        /// ## Algorithm
        ///
        ///   1. Local enumeration + dedup (InterpolateLocal) on all A (father+son).
        ///   2. Extract B→C pbi via extractPbi. Compute parent2entityPbi (uniform
        ///      XOR per matched node between each parent's view and entity's stored
        ///      first-parent view).
        ///   3. Classify: fully local → owned; fully ghost → discard; straddling →
        ///      ownership resolver callback.
        ///   4. Compact owned entities, convert indices to global.
        ///   5. Assign global B indices via createFatherGlobalMapping.
        ///   6. Push (globalA, globalB, subIdx) triplets to peer ranks so they can
        ///      fill parent2entity. parent2entityPbi is NOT pushed — it is a local
        ///      property of (cell, slot) computed entirely from cell2nodePbi and
        ///      entity2nodePbi.
        ///
        /// Ghost B entities are NOT produced. Use evaluateGhostTree(A→A2B→B)
        /// after this call to pull them.
        ///
        /// ## Ordering guarantees
        ///
        /// parent2entity[iParent][j] corresponds to sub-entity j per element topology
        /// (ObtainFace(j) / ObtainEdge(j)). entity2node stores nodes in the first-
        /// discovered parent's extraction order. All pbi arrays are parallel to their
        /// adjacency counterparts.
        ///
        /// @param parent2node       CSR: A → C (father+son, local C indices).
        /// @param parent2nodePbi    A → C pbi (father+son). Pass empty pair if not periodic.
        /// @param parentGhostMapping Ghost mapping for A (local-appended → global).
        /// @param parentGlobalMapping Global offsets mapping for A (for rank lookup).
        /// @param nodeGhostMapping   Ghost mapping for C (local-appended → global).
        /// @param query             Sub-entity topology + pbi extraction callbacks.
        /// @param nLocalParents     Number of owned A.
        /// @param nTotalParents     Total A (father + son).
        /// @param nNode             Total C (father + son).
        /// @param resolver          N-parent ownership callback.
        /// @param mpi               MPI communicator.
        /// @return InterpolateGlobalResultT<e2p_rs> with owned B entities in global indices.
        /// @tparam p2n_rs  Row-size of parent2node input.
        /// @tparam e2p_rs  Row-size of entity2parent output (NonUniformSize = variable,
        ///                 2 = fixed for faces). Fixed-width rows shorter than e2p_rs
        ///                 are padded with UnInitIndex.
        template <rowsize p2n_rs = NonUniformSize, rowsize e2p_rs = NonUniformSize>
        static InterpolateGlobalResultT<e2p_rs> InterpolateGlobal(
            const ArrayAdjacencyPair<p2n_rs> &parent2node,
            const tPbiPair &parent2nodePbi,
            const OffsetAscendIndexMapping &parentGhostMapping,
            const GlobalOffsetsMapping &parentGlobalMapping,
            const OffsetAscendIndexMapping &nodeGhostMapping,
            const SubEntityQueryPbi &query,
            index nLocalParents,
            index nTotalParents,
            index nNode,
            const OwnershipResolverMulti &resolver,
            const MPIInfo &mpi);

        // -----------------------------------------------------------------
        // Ghost evaluation
        // -----------------------------------------------------------------

        /// Evaluate a compiled ghost tree to determine which entities to ghost.
        ///
        /// BFS level-by-level evaluation with scratch pulls between levels.
        /// Produces a GhostResult containing per-EntityKind ghost index sets
        /// (union of all COLLECT nodes).
        ///
        /// @param tree          Compiled ghost tree (from CompiledGhostTree::compile).
        /// @param mpi           MPI communicator.
        /// @return              Per-EntityKind sorted, deduplicated ghost indices.
        ///
        /// Preconditions:
        ///   - All AdjKind values in the tree must be registered via registerAdj().
        ///   - The registered tAdjPair arrays must be in Adj_PointToGlobal state.
        ///   - GlobalOffsetsMapping must be available for entity kinds at root
        ///     level (to determine ownership).
        GhostResult evaluateGhostTree(
            const CompiledGhostTree &tree,
            const MPIInfo &mpi) const;
    };

    // =====================================================================
    // Template implementation: ComposeFiltered
    // =====================================================================

    template <rowsize rs_AB, rowsize rs_BC, rowsize out_rs, class Predicate>
    ArrayAdjacencyPair<out_rs> MeshConnectivity::ComposeFiltered(
        const ArrayAdjacencyPair<rs_AB> &AB,
        const ArrayAdjacencyPair<rs_BC> &BC,
        index nALocal,
        const std::unordered_map<index, index> &bGlobal2Local,
        const std::function<index(index)> &aLocal2Global,
        Predicate &&pred,
        const std::function<bool(index aLocal, index cGlobal,
                                 const std::vector<index> &sharedBGlobals)>
            &matchExtra)
    {
        const auto &mpi = AB.father->getMPI();
        const bool hasMatchExtra = bool(matchExtra);
        ArrayAdjacencyPair<out_rs> result;
        result.InitPair("ComposeFiltered_result", mpi);
        result.father->Resize(nALocal);

        ConnectivityRows::Compose(nALocal, [&](index row)
                                  { return AB.father->operator[](row); }, [&](index row)
                                  { return BC[row]; }, [&](index global)
                                  {
                auto found = bGlobal2Local.find(global);
                DNDS_check_throw_info(found != bGlobal2Local.end(),
                                      "ComposeFiltered: intermediate global row missing; ghost pull incomplete");
                return found->second; }, aLocal2Global, std::forward<Predicate>(pred), [&](index row, const std::vector<index> &accepted)
                                  {
                if constexpr (out_rs == NonUniformSize)
                {
                    result.father->ResizeRow(row, accepted.size());
                    for (rowsize j = 0; j < static_cast<rowsize>(accepted.size()); ++j)
                        result.father->operator()(row, j) = accepted[j];
                }
                else
                {
                    DNDS_check_throw_info(accepted.size() <= size_t(out_rs), "ComposeFiltered: fixed result row too narrow");
                    for (rowsize j = 0; j < out_rs; ++j)
                        result.father->operator()(row, j) = size_t(j) < accepted.size() ? accepted[j] : UnInitIndex;
                } }, matchExtra);

        return result;
    }

    // =====================================================================
    // Template implementation: Compose
    // =====================================================================

    template <rowsize rs_AB, rowsize rs_BC, rowsize out_rs>
    ArrayAdjacencyPair<out_rs> MeshConnectivity::Compose(
        const ArrayAdjacencyPair<rs_AB> &AB,
        const ArrayAdjacencyPair<rs_BC> &BC,
        index nALocal,
        const std::unordered_map<index, index> &bGlobal2Local,
        const std::function<index(index)> &aLocal2Global,
        bool removeSelf)
    {
        return ComposeFiltered<rs_AB, rs_BC, out_rs>(
            AB, BC, nALocal, bGlobal2Local, aLocal2Global,
            SharedCountPredicate{.minShared = 1, .removeSelf = removeSelf});
    }

    // =====================================================================
    // Template implementation: Inverse
    // =====================================================================

    template <rowsize cone_rs>
    tAdjPair MeshConnectivity::Inverse(
        const ArrayAdjacencyPair<cone_rs> &cone,
        index nToLocal,
        const MPIInfo &mpi,
        const std::function<index(index)> &fromLocal2Global,
        const std::function<index(index)> &toLocal2Global,
        const ssp<GlobalOffsetsMapping> &toGlobalMapping)
    {
        tAdjPair result;
        result.InitPair("Inverse_result", mpi);
        result.father->Resize(nToLocal);
        ConnectivityRows::Inverse(cone.father->Size(), nToLocal, [&](index row)
                                  { return cone.father->operator[](row); }, [&](index row, const std::vector<index> &entries)
                                  {
                result.father->ResizeRow(row, entries.size());
                for (rowsize j = 0; j < static_cast<rowsize>(entries.size()); ++j)
                    result.father->operator()(row, j) = entries[j]; }, mpi, fromLocal2Global, toLocal2Global, toGlobalMapping);
        return result;
    }

    template <class Read, class Write>
    void ConnectivityRows::Inverse(index nFromLocal, index nToLocal, Read &&read, Write &&write, const MPIInfo &mpi,
                                   const std::function<index(index)> &fromLocal2Global,
                                   const std::function<index(index)> &toLocal2Global,
                                   const ssp<GlobalOffsetsMapping> &toGlobalMapping)
    {
        // ----- Step 1: Local inversion -----
        std::unordered_map<index, std::unordered_set<index>> to2fromRecord;
        std::vector<index> ghostToIndices;
        std::unordered_set<index> ghostToSet;

        for (index iFrom = 0; iFrom < nFromLocal; iFrom++)
        {
            index fromGlobal = fromLocal2Global(iFrom);
            for (auto iTo : read(iFrom))
            {
                // Fixed-width adjacencies may have UnInitIndex padding — skip.
                if (iTo == UnInitIndex)
                    continue;

                to2fromRecord[iTo].insert(fromGlobal);

                auto [ret, rank, val] = toGlobalMapping->search(iTo);
                DNDS_assert_info(ret, fmt::format("Inverse: to-entity {} not found in global mapping", iTo));
                if (rank != mpi.rank && ghostToSet.find(iTo) == ghostToSet.end())
                {
                    ghostToIndices.push_back(iTo);
                    ghostToSet.insert(iTo);
                }
            }
        }

        // ----- Step 2: Build initial support pair -----
        tAdjPair support;
        support.InitPair("Inverse_support", mpi);
        support.father->Resize(nToLocal);

        for (index iTo = 0; iTo < nToLocal; iTo++)
        {
            index toGlobal = toLocal2Global(iTo);
            auto it = to2fromRecord.find(toGlobal);
            if (it != to2fromRecord.end())
            {
                support.father->ResizeRow(iTo, it->second.size());
                rowsize j = 0;
                for (auto fromG : it->second)
                    support.father->operator()(iTo, j++) = fromG;
            }
        }

        // Set up ghost communication
        support.TransAttach();
        support.trans.createFatherGlobalMapping();
        support.trans.createGhostMapping(ghostToIndices);

        support.son->Resize(support.trans.pLGhostMapping->ghostIndex.size());
        for (auto &[toGlobal, fromSet] : to2fromRecord)
        {
            MPI_int rank{-1};
            index val{-1};
            if (!support.trans.pLGhostMapping->search(toGlobal, rank, val))
                DNDS_check_throw_info(false, "Inverse: ghost search failed");
            if (rank >= 0)
            {
                support.son->ResizeRow(val, fromSet.size());
                rowsize j = 0;
                for (auto fromG : fromSet)
                    support.son->operator()(val, j++) = fromG;
            }
        }

        // ----- Step 3: Reverse-push -----
        using tSupportArr = typename decltype(support.son)::element_type;
        ssp<tSupportArr> supportPast = make_ssp<tSupportArr>(ObjName{"Inverse_supportPast"}, mpi);

        typename DNDS::ArrayTransformerType<tSupportArr>::Type supportPastTrans;
        supportPastTrans.setFatherSon(support.son, supportPast);
        supportPastTrans.createFatherGlobalMapping();

        std::vector<index> pushSonSeries(support.son->Size());
        for (index i = 0; i < support.son->Size(); i++)
            pushSonSeries[i] = i;
        supportPastTrans.createGhostMapping(pushSonSeries, support.trans.pLGhostMapping->ghostStart);
        supportPastTrans.createMPITypes();
        supportPastTrans.pullOnce();

        // ----- Step 4: Merge remote contributions -----
        DNDS_assert(DNDS::size_to_index(support.trans.pLGhostMapping->ghostIndex.size()) == support.son->Size());
        DNDS_assert(DNDS::size_to_index(support.trans.pLGhostMapping->pushingIndexGlobal.size()) == supportPast->Size());

        for (index i = 0; i < supportPast->Size(); i++)
        {
            index toGlobal = support.trans.pLGhostMapping->pushingIndexGlobal[i];
            for (auto fromG : (*supportPast)[i])
                to2fromRecord[toGlobal].insert(fromG);
        }

        // ----- Step 5: Rebuild father -----
        for (index iTo = 0; iTo < nToLocal; iTo++)
        {
            index toGlobal = toLocal2Global(iTo);
            auto it = to2fromRecord.find(toGlobal);
            if (it != to2fromRecord.end())
            {
                std::vector<index> fromVec(it->second.begin(), it->second.end());
                std::sort(fromVec.begin(), fromVec.end());
                write(iTo, fromVec);
            }
            else
                write(iTo, std::vector<index>{});
        }
    }

} // namespace DNDS::Geom

// Template implementations for Interpolate family (too large for inline)
#include "MeshConnectivity_Interpolate.hxx"
