/// @file Mesh_Reorder.cpp
/// @brief Implementation of ReorderPlan::build, ReorderPlan::apply, and
///        UnstructuredMesh::buildReorderRegistry / ReorderEntities.

#include "ReorderPlan.hpp"
#include "Mesh.hpp"
#include "Mesh_CellPermutation.hpp"

#include <algorithm>
#include <limits>
#include <set>

namespace DNDS::Geom
{
    // =================================================================
    // ComputeFollowMapFromAdj: derive follower placement from leader
    // =================================================================

    /// Compute a follow map for `follower` based on `leader`'s explicit map.
    ///
    /// Uses the support adjacency follower->leader (e.g., node2cell) to
    /// determine: for each follower entity, go to min(leader.targetRank)
    /// over all leaders referencing it.
    ///
    /// @param follower2leader  Raw follower->leader adj (father-only, global entries).
    /// @param nFollower        Number of follower entities (father size).
    /// @param leaderGM         Global mapping for the leader entity kind.
    /// @param leaderTargetRanks  Per-leader-slot target rank (the explicit map).
    /// @param mpi              MPI communicator.
    /// @return Per-follower-slot target rank.
    /// @warning Collective.
    template <rowsize f2l_rs>
    static std::vector<MPI_int> ComputeFollowMapFromAdj(
        const ArrayAdjacencyPair<f2l_rs> &follower2leader,
        index nFollower,
        const ssp<GlobalOffsetsMapping> &leaderGM,
        const std::vector<MPI_int> &leaderTargetRanks,
        const MPIInfo &mpi)
    {
        DNDS_assert(leaderGM);
        DNDS_assert(follower2leader.father);

        // Step 1: Build a ghost-pullable lookup of leader's targetRanks.
        ArrayAdjacencyPair<1> leaderLookup;
        leaderLookup.InitPair("followMap_leaderLookup", mpi);
        auto nLeader = static_cast<index>(leaderTargetRanks.size());
        leaderLookup.father->Resize(nLeader);
        for (index i = 0; i < nLeader; i++)
            leaderLookup(i, 0) = static_cast<index>(leaderTargetRanks[i]);
        leaderLookup.TransAttach();
        leaderLookup.trans.createFatherGlobalMapping();

        // Step 2: Collect off-rank leader globals from follower2leader entries.
        std::set<index> offRankLeaderGlobals;
        for (index i = 0; i < nFollower; i++)
            for (rowsize j = 0; j < follower2leader.RowSize(i); j++)
            {
                index leaderGlobal = follower2leader(i, j);
                if (leaderGlobal == UnInitIndex)
                    continue;
                auto [found, rank, val] = leaderGM->search(leaderGlobal);
                if (found && rank != mpi.rank)
                    offRankLeaderGlobals.insert(leaderGlobal);
            }

        // Step 3: Ghost-pull leaderLookup for off-rank leaders.
        std::vector<index> pullSet(offRankLeaderGlobals.begin(), offRankLeaderGlobals.end());
        leaderLookup.trans.createGhostMapping(pullSet);
        leaderLookup.trans.createMPITypes();
        leaderLookup.trans.pullOnce();

        // Step 4: For each follower, find min leader targetRank.
        std::vector<MPI_int> followMap(nFollower, mpi.size); // init to max
        for (index i = 0; i < nFollower; i++)
        {
            MPI_int minRank = mpi.size;
            for (rowsize j = 0; j < follower2leader.RowSize(i); j++)
            {
                index leaderGlobal = follower2leader(i, j);
                if (leaderGlobal == UnInitIndex)
                    continue;
                // Resolve to local-appended index in leaderLookup
                MPI_int rank;
                index val;
                bool found = leaderLookup.trans.pLGhostMapping->search_indexAppend(
                    leaderGlobal, rank, val);
                DNDS_assert(found);
                auto leaderTarget = static_cast<MPI_int>(leaderLookup(val, 0));
                minRank = std::min(minRank, leaderTarget);
            }
            // If no leaders found (should not happen for valid mesh), stay put
            followMap[i] = (minRank < mpi.size) ? minRank : mpi.rank;
        }

        return followMap;
    }

    /// Shorthand for computing a follow map from a follower→leader adjacency.
    ///
    /// Calls ComputeFollowMapFromAdj(), automatically extracting the row-size
    /// template parameter and forwarding the follower→leader base class.
    /// `UnInitIndex` entries in the support adjacency are skipped.
    ///
    /// @tparam AdjPair  Concrete AdjPairTracked type (e.g., AdjPairTracked<tAdjPair>).
    /// @param f2l      Follower→leader adjacency, entries must be global.
    /// @param nFollower  Number of father follower entities.
    /// @param leaderGM   Global mapping for the leader entity kind.
    /// @param leaderTargetRanks  Explicit target rank vector for leaders.
    /// @param mpi       MPI communicator.
    /// @return  Per-follower target rank vector, size == nFollower.
    template <class AdjPair>
    static std::vector<MPI_int> followVia(
        const AdjPair &f2l,
        index nFollower,
        const ssp<GlobalOffsetsMapping> &leaderGM,
        const std::vector<MPI_int> &leaderTargetRanks,
        const MPIInfo &mpi)
    {
        constexpr rowsize Rs = std::remove_reference_t<AdjPair>::t_arr::rs;
        return ComputeFollowMapFromAdj(
            static_cast<const ArrayAdjacencyPair<Rs> &>(f2l),
            nFollower, leaderGM, leaderTargetRanks, mpi);
    }

    // =================================================================
    // ReorderPlan::build
    // =================================================================

    ReorderPlan ReorderPlan::build(
        const ReorderInput &input,
        const ReorderRegistry &registry,
        const MPIInfo &mpi)
    {
        ReorderPlan plan;

        // --- Step 1: Merge explicit maps into allMaps ---
        std::unordered_map<EntityKind, std::vector<MPI_int>> allMaps;
        for (const auto &em : input.explicitMaps)
            allMaps[em.kind] = em.targetRanks;

        // --- Step 2: Compute follow maps ---
        // Follow computation is handled by the mesh wrapper (resolveFollows)
        // before calling build. By the time we reach here, all entity kinds
        // should be in explicitMaps. The input.follows field is not processed
        // here — it is a mesh-wrapper-level concept.
        DNDS_assert_info(input.follows.empty(),
                         "ReorderPlan::build: input.follows must be empty; "
                         "use UnstructuredMesh::resolveFollows to compute follow maps "
                         "before calling build");

        // --- Step 3: Build PermutationTransfer per entity kind ---
        plan.reorderedKinds.clear();
        for (auto &[kind, ranks] : allMaps)
        {
            plan.reorderedKinds.insert(kind);
            auto gm = registry.getGlobalMapping(kind);
            DNDS_assert_info(gm, fmt::format("ReorderPlan::build: no global mapping for kind {}",
                                             entityKindName(kind)));
            plan.transfers[kind] = PermutationTransfer::fromPartition(ranks, gm, mpi);
        }

        // --- Step 4: Detect global local-only ---
        plan.isLocalOnly = true;
        for (auto &[kind, transfer] : plan.transfers)
            if (!transfer.isLocalOnly)
            {
                plan.isLocalOnly = false;
                break;
            }
        int globalFlag;
        int localFlag = plan.isLocalOnly ? 1 : 0;
        MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_LAND, mpi.comm);
        plan.isLocalOnly = (globalFlag != 0);

        // --- Step 5: Collect pull sets and build lookups ---
        for (auto kind : plan.reorderedKinds)
        {
            // Use pre-collected pull sets from the registry
            std::vector<index> pullSet;
            auto psIt = registry.pullSets.find(kind);
            if (psIt != registry.pullSets.end())
                pullSet = psIt->second;

            plan.lookups[kind] = plan.transfers.at(kind).buildLookup(pullSet, mpi);
        }

        return plan;
    }

    // =================================================================
    // ReorderPlan::apply
    // =================================================================

    PreparedCompanionBundle ReorderPlan::prepareCompanions(const ReorderRegistry &registry, const MPIInfo &mpi) const
    {
        std::string signature;
        int valid = 1;
        try
        {
            std::unordered_map<EntityKind, std::pair<uint64_t, uint64_t>> distributions;
            std::unordered_set<std::string> names;
            std::unordered_set<const void *> destinations;
            auto append = [&](const std::string &value)
            { signature += std::to_string(value.size()) + ":" + value + ";"; };
            for (const auto &field : registry.companions)
            {
                DNDS_check_throw_info(field.name.empty() || names.insert(field.name).second,
                                      "duplicate companion registration name");
                append(std::to_string(static_cast<int>(field.kind)));
                append(field.name);
                append(std::to_string(static_cast<int>(field.disposition)));
                append(field.prepare ? "staged" : "legacy");
                DNDS_check_throw_info(!field.prepare || (field.preflight && field.attachment), "incomplete staged field registration");
                DNDS_check_throw_info(field.prepare || (!field.preflight && !field.attachment), "incomplete staged field registration");
                const bool active = reorderedKinds.count(field.kind);
                DNDS_check_throw_info(!active || field.disposition == CompanionDisposition::TransferIfReordered,
                                      "field disposition prohibits reordering its entity space");
                if (!field.prepare)
                    continue;
                auto attached = field.attachment();
                DNDS_check_throw_info(attached.destination && destinations.insert(attached.destination).second,
                                      "staged destination handle registered more than once");
                append(attached.schema);
                append(attached.association);
                append(attached.hasDistribution ? "distributed" : "array");
                if (attached.hasDistribution)
                {
                    append(std::to_string(attached.domain));
                    append(std::to_string(attached.epoch));
                    auto metadata = std::make_pair(attached.domain, attached.epoch);
                    auto [entry, inserted] = distributions.emplace(field.kind, metadata);
                    DNDS_check_throw_info(inserted || entry->second == metadata,
                                          "fields attached to one entity space have different domains/epochs");
                }
                if (active)
                    field.preflight(transfers.at(field.kind));
            }
            DNDS_check_throw_info(signature.size() <= size_t(std::numeric_limits<int>::max()), "field schedule too large");
        }
        catch (...)
        {
            valid = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "staged field attachment, disposition or source validation failed on a rank");
        int length = static_cast<int>(signature.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string expected;
        try
        {
            expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0');
        }
        catch (...)
        {
            valid = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "cannot retain staged field schedule on a rank");
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
        valid = signature == expected;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "staged field schemas/attachments/dispositions differ between ranks");

        PreparedCompanionBundle result;
        try
        {
            result.fields.reserve(registry.companions.size());
        }
        catch (...)
        {
            valid = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "cannot retain staged field bundle on a rank");
        for (const auto &field : registry.companions)
            if (field.prepare && reorderedKinds.count(field.kind))
            {
                std::unique_ptr<PreparedCompanion> prepared;
                try
                {
                    prepared = field.prepare(transfers.at(field.kind), mpi);
                    valid = bool(prepared);
                }
                catch (...)
                {
                    valid = 0;
                }
                MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
                DNDS_check_throw_info(valid, "staged field preparation failed on a rank; sources remain unpublished");
                result.fields.push_back(std::move(prepared));
            }
        return result;
    }

    void ReorderPlan::apply(ReorderRegistry &registry, const MPIInfo &mpi) const
    {
        PreparedCompanionBundle prepared;
        if (std::any_of(registry.companions.begin(), registry.companions.end(), [](const auto &field)
                        { return bool(field.prepare) || field.disposition != CompanionDisposition::TransferIfReordered; }))
            prepared = prepareCompanions(registry, mpi);
        // Phase 1: REMAP all adj entries
        for (auto &adj : registry.adjs)
        {
            auto action = classifyAdj(adj.kind, reorderedKinds);
            if (action == AdjAction::REMAP ||
                action == AdjAction::RELOCATE_REMAP ||
                action == AdjAction::SELF)
            {
                EntityKind targetKind = adj.kind.isIntraLevel()
                                            ? adj.kind.from
                                            : adj.kind.to;
                auto it = lookups.find(targetKind);
                DNDS_assert_info(it != lookups.end(),
                                 fmt::format("ReorderPlan::apply REMAP: no lookup for target kind {}",
                                             entityKindName(targetKind)));
                if (adj.remapFn)
                    adj.remapFn(it->second);
            }
        }

        // Phase 2: RELOCATE all adj rows
        for (auto &adj : registry.adjs)
        {
            auto action = classifyAdj(adj.kind, reorderedKinds);
            if (action == AdjAction::RELOCATE ||
                action == AdjAction::RELOCATE_REMAP ||
                action == AdjAction::SELF)
            {
                EntityKind sourceKind = adj.kind.from;
                auto it = transfers.find(sourceKind);
                DNDS_assert_info(it != transfers.end(),
                                 fmt::format("ReorderPlan::apply RELOCATE: no transfer for source kind {}",
                                             entityKindName(sourceKind)));
                if (adj.relocateFn)
                    adj.relocateFn(it->second, mpi);
            }
        }

        // Phase 3: RELOCATE all companions of reordered kinds
        for (auto &comp : registry.companions)
        {
            if (!comp.prepare && reorderedKinds.count(comp.kind))
            {
                auto it = transfers.find(comp.kind);
                DNDS_assert_info(it != transfers.end(),
                                 fmt::format("ReorderPlan::apply COMPANION: no transfer for kind {}",
                                             entityKindName(comp.kind)));
                comp.fn(it->second, mpi);
            }
        }
        prepared.Publish();
    }

    // =================================================================
    // UnstructuredMesh::buildReorderRegistry
    // =================================================================

    ReorderRegistry UnstructuredMesh::buildReorderRegistry(
        const std::unordered_set<EntityKind> &destroyKinds)
    {
        ReorderRegistry reg;
        std::vector<std::function<void()>> collectPullSets;

        auto shouldSkip = [&](AdjKind kind)
        {
            return destroyKinds.count(kind.from) || destroyKinds.count(kind.to);
        };

        // --- Helper: register a tracked adj member ---
        auto regAdj = [&](AdjKind kind, auto &trackedPair)
        {
            if (!trackedPair.father || shouldSkip(kind))
                return;

            // Require every built registered adjacency to NOT be in local state.
            // Local entries get treated as globals during follow/remap lookup
            // resolution, corrupting topology.  Adj_Unknown is acceptable
            // (legacy adjs built before per-adj idx tracking was added).
            DNDS_check_throw_info(
                !trackedPair.idx.isLocal(),
                fmt::format("buildReorderRegistry: {} entries must NOT be "
                            "Adj_PointToLocal before reorder (currently {})",
                            adjKindName(kind), int(trackedPair.idx.state())));

            AdjRemapFn remap = [&trackedPair](const PermutationTransfer::LookupResult &lookup)
            {
                // Remap owned source rows only. Ghost/son rows describe the
                // pre-reorder ghost layout and are rebuilt by the normal
                // BuildGhost* pipeline after ReorderEntities(); resolving them
                // here would require extra pull sets and can map stale ghost
                // topology through the wrong lookup.
                index nRows = trackedPair.father->Size();
                for (index i = 0; i < nRows; i++)
                    for (rowsize j = 0; j < trackedPair.RowSize(i); j++)
                    {
                        index &v = trackedPair(i, j);
                        if (v != UnInitIndex)
                            v = lookup.resolve(v);
                    }
            };

            AdjRelocateFn relocate = [&trackedPair](
                                         const PermutationTransfer &t, const MPIInfo &m)
            {
                t.transferRows(trackedPair, m);
            };

            reg.registerAdj(kind, std::move(remap), std::move(relocate),
                            adjKindName(kind));

            // The remap runs on owned source rows before relocation. Build
            // coverage from precisely those rows, so every arriving row has
            // already had its target IDs resolved on its original owner.
            // Keep coverage with registration rather than a second inventory
            // that can omit a relation (notably cell2cellFace).
            collectPullSets.emplace_back([&, kind, pair = &trackedPair]()
                                         {
                auto targetKind = kind.isIntraLevel() ? kind.from : kind.to;
                auto gm = reg.getGlobalMapping(targetKind);
                if (!gm)
                    return;
                auto &ps = reg.pullSets[targetKind];
                for (index i = 0; i < pair->father->Size(); ++i)
                    for (rowsize j = 0; j < pair->RowSize(i); ++j)
                    {
                        auto v = (*pair)(i, j);
                        if (v == UnInitIndex)
                            continue;
                        auto [found, rank, slot] = gm->search(v);
                        if (found && rank != mpi.rank)
                            ps.push_back(v);
                    } });
        };

        // Register tracked adj members
        regAdj(Adj::Cell2Node, cell2node);
        regAdj(Adj::Cell2Cell, cell2cell);
        regAdj(Adj::Bnd2Node, bnd2node);
        regAdj(Adj::Bnd2Cell, bnd2cell);
        regAdj(Adj::Node2Cell, node2cell);
        regAdj(Adj::Node2Bnd, node2bnd);
        regAdj(Adj::Cell2Face, cell2face);
        regAdj(Adj::Face2Node, face2node);
        regAdj(Adj::Face2Cell, face2cell);
        regAdj(Adj::Face2Bnd, face2bnd);
        regAdj(Adj::Bnd2Face, bnd2face);
        regAdj(Adj::Cell2CellFace, cell2cellFace);
        regAdj(Adj::Cell2Edge, cell2edge);
        regAdj(Adj::Edge2Node, edge2node);
        regAdj(Adj::Edge2Cell, edge2cell);

        // --- Helper: register a companion ---
        auto regComp = [&](EntityKind kind, auto &pair, const char *name)
        {
            if (!pair.father || destroyKinds.count(kind))
                return;
            reg.registerCompanion(kind, [&pair](const PermutationTransfer &t, const MPIInfo &m)
                                  { t.transferRows(pair, m); }, name);
        };

        // Register companions
        regComp(EntityKind::Cell, cellElemInfo, "cellElemInfo");
        regComp(EntityKind::Cell, cell2cellOrig, "cell2cellOrig");
        regComp(EntityKind::Node, coords, "coords");
        regComp(EntityKind::Node, node2nodeOrig, "node2nodeOrig");
        regComp(EntityKind::Bnd, bndElemInfo, "bndElemInfo");
        regComp(EntityKind::Bnd, bnd2bndOrig, "bnd2bndOrig");

        if (isPeriodic)
        {
            regComp(EntityKind::Cell, cell2nodePbi, "cell2nodePbi");
            regComp(EntityKind::Bnd, bnd2nodePbi, "bnd2nodePbi");
            if (!destroyKinds.count(EntityKind::Face))
            {
                regComp(EntityKind::Face, face2nodePbi, "face2nodePbi");
                if (cell2facePbi.father)
                    regComp(EntityKind::Cell, cell2facePbi, "cell2facePbi");
            }
            if (!destroyKinds.count(EntityKind::Edge))
            {
                regComp(EntityKind::Edge, edge2nodePbi, "edge2nodePbi");
                regComp(EntityKind::Cell, cell2edgePbi, "cell2edgePbi");
            }
        }
        if (!destroyKinds.count(EntityKind::Face))
            regComp(EntityKind::Face, faceElemInfo, "faceElemInfo");
        if (!destroyKinds.count(EntityKind::Edge))
            regComp(EntityKind::Edge, edgeElemInfo, "edgeElemInfo");
        if (coordsElevDisp.father)
            regComp(EntityKind::Node, coordsElevDisp, "coordsElevDisp");
        if (nodeWallDist.father)
            regComp(EntityKind::Node, nodeWallDist, "nodeWallDist");

        // --- Register global mappings ---
        auto getGM = [](const auto &pair) -> ssp<GlobalOffsetsMapping>
        {
            if (pair.father && pair.father->pLGlobalMapping)
                return pair.father->pLGlobalMapping;
            return nullptr;
        };

        auto firstValid = [](std::initializer_list<ssp<GlobalOffsetsMapping>> candidates)
            -> ssp<GlobalOffsetsMapping>
        {
            for (const auto &gm : candidates)
                if (gm)
                    return gm;
            return nullptr;
        };

        if (auto gm = firstValid({getGM(cell2node), getGM(cell2cell), getGM(cell2face)}))
            reg.registerGlobalMapping(EntityKind::Cell, gm);
        if (auto gm = coords.father ? coords.father->pLGlobalMapping : nullptr)
            reg.registerGlobalMapping(EntityKind::Node, gm);
        if (auto gm = firstValid({getGM(bnd2node), getGM(bnd2cell), getGM(bnd2face)}))
            reg.registerGlobalMapping(EntityKind::Bnd, gm);
        if (auto gm = firstValid({getGM(face2node), getGM(face2cell), getGM(face2bnd)}))
            reg.registerGlobalMapping(EntityKind::Face, gm);
        if (auto gm = firstValid({getGM(edge2node), getGM(edge2cell)}))
            reg.registerGlobalMapping(EntityKind::Edge, gm);

        // --- Pre-collect pull sets from the registered relation inventory ---
        for (const auto &collect : collectPullSets)
            collect();

        // Deduplicate and sort pull sets
        for (auto &[kind, ps] : reg.pullSets)
        {
            std::sort(ps.begin(), ps.end());
            ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
        }

        return reg;
    }

    // =================================================================
    // UnstructuredMesh::resolveFollows
    // =================================================================

    ReorderInput UnstructuredMesh::resolveFollows(
        const ReorderInput &input,
        const ReorderRegistry &reg)
    {
        ReorderInput augmented = input;
        std::unordered_set<EntityKind> explicitKinds;
        for (auto &em : augmented.explicitMaps)
            explicitKinds.insert(em.kind);

        // When follows is empty, apply default policy: all non-explicit
        // entity kinds with a support adjacency to an explicit kind follow it.
        // Currently: Node and Bnd follow Cell if Cell is explicit.
        if (augmented.follows.empty())
        {
            if (explicitKinds.count(EntityKind::Cell))
            {
                if (!explicitKinds.count(EntityKind::Node) && node2cell.father)
                    augmented.follows.push_back(
                        FollowSpec{EntityKind::Node, EntityKind::Cell, Adj::Node2Cell});
                if (!explicitKinds.count(EntityKind::Bnd) && bnd2cell.father)
                    augmented.follows.push_back(
                        FollowSpec{EntityKind::Bnd, EntityKind::Cell, Adj::Bnd2Cell});
            }
        }

        // Compute follow maps and merge into explicit maps
        std::unordered_map<EntityKind, std::vector<MPI_int>> allMaps;
        for (auto &em : augmented.explicitMaps)
            allMaps[em.kind] = em.targetRanks;

        for (auto &spec : augmented.follows)
        {
            if (allMaps.count(spec.follower))
                continue; // explicit takes precedence

            auto leaderIt = allMaps.find(spec.leader);
            DNDS_assert_info(leaderIt != allMaps.end(),
                             fmt::format("resolveFollows: follow spec references leader {} "
                                         "which has no map",
                                         entityKindName(spec.leader)));

            auto followerGM = reg.getGlobalMapping(spec.follower);
            DNDS_assert_info(followerGM,
                             fmt::format("resolveFollows: no global mapping for follower {}",
                                         entityKindName(spec.follower)));
            index nFollower = followerGM->RLengths()[mpi.rank];

            std::vector<MPI_int> followMap;
            // ── Follower → Cell ──
            if (spec.follower2leader == Adj::Node2Cell && node2cell.father)
            {
                followMap = followVia(
                    node2cell, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Bnd2Cell && bnd2cell.father)
            {
                followMap = followVia(
                    bnd2cell, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Face2Cell && face2cell.father)
            {
                followMap = followVia(
                    face2cell, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Edge2Cell && edge2cell.father)
            {
                followMap = followVia(
                    edge2cell, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            // ── Follower → Node ──
            else if (spec.follower2leader == Adj::Cell2Node && cell2node.father)
            {
                followMap = followVia(
                    cell2node, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Bnd2Node && bnd2node.father)
            {
                followMap = followVia(
                    bnd2node, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Face2Node && face2node.father)
            {
                followMap = followVia(
                    face2node, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            else if (spec.follower2leader == Adj::Edge2Node && edge2node.father)
            {
                followMap = followVia(
                    edge2node, nFollower,
                    reg.getGlobalMapping(spec.leader), leaderIt->second, mpi);
            }
            else
            {
                DNDS_assert_info(false,
                                 fmt::format("resolveFollows: unsupported or missing "
                                             "follow adjacency {} (follower {} → leader {}).  "
                                             "The follower→leader adj must be built.",
                                             adjKindName(spec.follower2leader),
                                             entityKindName(spec.follower),
                                             entityKindName(spec.leader)));
            }

            allMaps[spec.follower] = std::move(followMap);
        }

        // Build finalised input with all maps explicit
        ReorderInput finalInput;
        for (auto &[kind, ranks] : allMaps)
            finalInput.explicitMaps.push_back(EntityReorderMap{kind, ranks});
        finalInput.destroyKinds = input.destroyKinds;
        return finalInput;
    }

    // =================================================================
    // UnstructuredMesh::buildReorderPlan
    // =================================================================

    ReorderPlan UnstructuredMesh::buildReorderPlan(const ReorderInput &input)
    {
        auto reg = buildReorderRegistry(input.destroyKinds);
        auto finalInput = resolveFollows(input, reg);
        return ReorderPlan::build(finalInput, reg, mpi);
    }

    // =================================================================
    // UnstructuredMesh::ReorderEntities
    // =================================================================

    void UnstructuredMesh::ReorderEntities(const ReorderInput &input)
    {
        ReorderEntitiesWithFields(input, {});
    }

    void UnstructuredMesh::ReorderEntitiesWithFields(const ReorderInput &input,
                                                     const std::vector<CompanionEntry> &fields)
    {
        CheckCollectiveMutationAvailable();
        // Step 0: Validate precondition
        DNDS_assert_info(adjPrimaryState == Adj_PointToGlobal,
                         "ReorderEntities: adjPrimaryState must be Adj_PointToGlobal");

        // Step 1: Build registry (with destroy skip)
        auto reg = buildReorderRegistry(input.destroyKinds);

        // Validate the collective callback schedule before changing mesh data.
        std::unordered_set<std::string> names;
        std::string signature;
        int valid = 1;
        for (const auto &field : fields)
        {
            valid = valid && field.fn && !field.name.empty() && names.insert(field.name).second &&
                    !input.destroyKinds.count(field.kind) && bool(reg.getGlobalMapping(field.kind));
            signature += std::to_string(static_cast<int>(field.kind)) + ":" +
                         std::to_string(field.name.size()) + ":" + field.name + ";";
        }
        valid = valid && signature.size() <= size_t(std::numeric_limits<int>::max());
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "invalid field registration on a rank");
        int length = static_cast<int>(signature.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string expected = mpi.rank == 0 ? signature : std::string(size_t(length), '\0');
        MPI_Bcast(expected.data(), length, MPI_CHAR, 0, mpi.comm);
        valid = signature == expected;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "field registration order/kinds/names differ between ranks");
        reg.companions.insert(reg.companions.end(), fields.begin(), fields.end());

        // Step 2: Resolve follows and build plan
        auto finalInput = resolveFollows(input, reg);
        auto plan = ReorderPlan::build(finalInput, reg, mpi);

        // Stage all native fields while mesh arrays and source bindings still
        // describe the old distribution. No field handle publishes here.
        auto preparedFields = plan.prepareCompanions(reg, mpi);
        reg.companions.erase(std::remove_if(reg.companions.begin(), reg.companions.end(),
                                            [](const auto &field)
                                            { return bool(field.prepare); }),
                             reg.companions.end());

        auto changes = Runtime::Change::Distribution | Runtime::Change::Layout | Runtime::Change::Allocation;
        if (!input.destroyKinds.empty())
            changes = changes | Runtime::Change::Topology;
        MeshMutationToken mutationToken(this, versions, changes);

        ExecuteReorderPlanImpl(mutationToken, input, reg, plan);
        preparedFields.Publish();
        mutationToken.Commit();
    }

    void UnstructuredMesh::ExecuteReorderPlanImpl(MeshMutationToken &mutationToken, const ReorderInput &input,
                                                  ReorderRegistry &reg, const ReorderPlan &plan)
    {
        auto changes = Runtime::Change::Distribution | Runtime::Change::Layout | Runtime::Change::Allocation;
        if (!input.destroyKinds.empty())
            changes = changes | Runtime::Change::Topology;
        auto mutationStep = mutationToken.Enter(this, changes);
        // Step 3: Destroy adjacencies for destroyKinds
        auto destroyAdj = [&](auto &trackedPair)
        {
            trackedPair.father.reset();
            trackedPair.son.reset();
            trackedPair.idx = AdjIndexInfo{};
        };
        for (auto kind : input.destroyKinds)
        {
            if (kind == EntityKind::Face)
            {
                destroyAdj(cell2face);
                destroyAdj(face2node);
                destroyAdj(face2cell);
                destroyAdj(face2bnd);
                destroyAdj(bnd2face);
                destroyAdj(cell2cellFace);
                faceElemInfo.father.reset();
                faceElemInfo.son.reset();
                if (isPeriodic)
                {
                    face2nodePbi.father.reset();
                    face2nodePbi.son.reset();
                    cell2facePbi.father.reset();
                    cell2facePbi.son.reset();
                }
                adjFacialState = Adj_Unknown;
                adjC2FState = Adj_Unknown;
                adjC2CFaceState = Adj_Unknown;
                // Invalidate side caches
                bnd2faceV.clear();
                face2bndM.clear();
            }
            if (kind == EntityKind::Edge)
            {
                destroyAdj(cell2edge);
                destroyAdj(edge2node);
                destroyAdj(edge2cell);
                edgeElemInfo.father.reset();
                edgeElemInfo.son.reset();
                if (isPeriodic)
                {
                    cell2edgePbi.father.reset();
                    cell2edgePbi.son.reset();
                    edge2nodePbi.father.reset();
                    edge2nodePbi.son.reset();
                }
                adjEdgeState = Adj_Unknown;
            }
        }

        // Step 4: Apply plan (REMAP entries, RELOCATE rows + companions)
        plan.apply(reg, mpi);

        // Step 5: Rebuild global mappings for reordered entities
        for (auto kind : plan.reorderedKinds)
        {
            if (kind == EntityKind::Cell && cell2node.father)
            {
                cell2node.father->createGlobalMapping();
                // Borrow to other cell-parallel arrays
                if (cell2cell.father)
                    cell2cell.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
                if (cellElemInfo.father)
                    cellElemInfo.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
                if (cell2cellOrig.father)
                    cell2cellOrig.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
                if (cell2face.father)
                    cell2face.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
                if (cell2cellFace.father)
                    cell2cellFace.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
                if (cell2edge.father)
                    cell2edge.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Node && coords.father)
            {
                coords.father->createGlobalMapping();
                if (node2nodeOrig.father)
                    node2nodeOrig.father->pLGlobalMapping = coords.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Bnd && bnd2node.father)
            {
                bnd2node.father->createGlobalMapping();
                if (bndElemInfo.father)
                    bndElemInfo.father->pLGlobalMapping = bnd2node.father->pLGlobalMapping;
                if (bnd2bndOrig.father)
                    bnd2bndOrig.father->pLGlobalMapping = bnd2node.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Face && face2node.father)
            {
                face2node.father->createGlobalMapping();
                if (faceElemInfo.father)
                    faceElemInfo.father->pLGlobalMapping = face2node.father->pLGlobalMapping;
                if (face2cell.father)
                    face2cell.father->pLGlobalMapping = face2node.father->pLGlobalMapping;
                if (face2bnd.father)
                    face2bnd.father->pLGlobalMapping = face2node.father->pLGlobalMapping;
            }
            else if (kind == EntityKind::Edge && edge2node.father)
            {
                edge2node.father->createGlobalMapping();
                if (edgeElemInfo.father)
                    edgeElemInfo.father->pLGlobalMapping = edge2node.father->pLGlobalMapping;
                if (edge2cell.father)
                    edge2cell.father->pLGlobalMapping = edge2node.father->pLGlobalMapping;
            }
        }

        // Step 5b: Re-attach transformers with empty sons (prepare for ghost rebuild)
        // After transferRows, sons are null. The rebuild pipeline
        // (RecoverNode2CellAndNode2Bnd, etc.) needs TransAttach-ready pairs.
        auto reattach = [&](auto &pair)
        {
            if (!pair.father)
                return;
            using TArr = typename std::remove_reference_t<decltype(pair)>::t_arr;
            if (!pair.son)
                pair.son = make_ssp<TArr>(ObjName{"reorder.son"}, mpi);
            pair.TransAttach();
        };

        for (auto kind : plan.reorderedKinds)
        {
            if (kind == EntityKind::Cell)
            {
                reattach(cell2node);
                reattach(cell2cell);
                reattach(cellElemInfo);
                reattach(cell2cellOrig);
                if (isPeriodic)
                    reattach(cell2nodePbi);
                if (cell2face.father)
                    reattach(cell2face);
                if (cell2cellFace.father)
                    reattach(cell2cellFace);
                if (cell2edge.father)
                    reattach(cell2edge);
                if (isPeriodic)
                {
                    if (cell2facePbi.father)
                        reattach(cell2facePbi);
                    if (cell2edgePbi.father)
                        reattach(cell2edgePbi);
                }
            }
            else if (kind == EntityKind::Node)
            {
                reattach(coords);
                reattach(node2nodeOrig);
                // Optional node companions
                if (coordsElevDisp.father)
                {
                    reattach(coordsElevDisp);
                    coordsElevDisp.father->pLGlobalMapping = coords.father->pLGlobalMapping;
                }
                if (nodeWallDist.father)
                {
                    reattach(nodeWallDist);
                    nodeWallDist.father->pLGlobalMapping = coords.father->pLGlobalMapping;
                }
            }
            else if (kind == EntityKind::Bnd)
            {
                reattach(bnd2node);
                reattach(bnd2cell);
                reattach(bndElemInfo);
                reattach(bnd2bndOrig);
                if (isPeriodic)
                    reattach(bnd2nodePbi);
                if (bnd2face.father)
                    reattach(bnd2face);
            }
            else if (kind == EntityKind::Face)
            {
                reattach(face2node);
                reattach(face2cell);
                reattach(face2bnd);
                reattach(cell2face);
                reattach(bnd2face);
                reattach(cell2cellFace);
                reattach(faceElemInfo);
                if (isPeriodic)
                {
                    reattach(face2nodePbi);
                    if (cell2facePbi.father)
                        reattach(cell2facePbi);
                }
            }
            else if (kind == EntityKind::Edge)
            {
                reattach(cell2edge);
                reattach(edge2node);
                reattach(edge2cell);
                reattach(edgeElemInfo);
                if (isPeriodic)
                {
                    reattach(cell2edgePbi);
                    reattach(edge2nodePbi);
                }
            }
        }
        // Also reattach inverse adjacencies if they exist
        if (node2cell.father)
            reattach(node2cell);
        if (node2bnd.father)
            reattach(node2bnd);

        // Re-wire target mappings after reattach/recreate. Reorder can replace
        // the source/target transformers while preserving adjacencies; stale
        // AdjIndexInfo target mappings would make later toLocal/toGlobal use old
        // ghost mappings.
        if (cell2node.father && coords.trans.pLGhostMapping)
            cell2node.idx.wireTargetMapping(coords.trans.pLGhostMapping);
        if (cell2cell.father && cell2node.trans.pLGhostMapping)
            cell2cell.idx.wireTargetMapping(cell2node.trans.pLGhostMapping);
        if (bnd2cell.father && cell2node.trans.pLGhostMapping)
            bnd2cell.idx.wireTargetMapping(cell2node.trans.pLGhostMapping);
        if (bnd2node.father && coords.trans.pLGhostMapping)
            bnd2node.idx.wireTargetMapping(coords.trans.pLGhostMapping);
        if (node2cell.father && cell2node.trans.pLGhostMapping)
            node2cell.idx.wireTargetMapping(cell2node.trans.pLGhostMapping);
        if (node2bnd.father && bnd2node.trans.pLGhostMapping)
            node2bnd.idx.wireTargetMapping(bnd2node.trans.pLGhostMapping);
        if (cell2face.father && face2node.trans.pLGhostMapping)
            cell2face.idx.wireTargetMapping(face2node.trans.pLGhostMapping);
        if (bnd2face.father && face2node.trans.pLGhostMapping)
            bnd2face.idx.wireTargetMapping(face2node.trans.pLGhostMapping);
        if (face2node.father && coords.trans.pLGhostMapping)
            face2node.idx.wireTargetMapping(coords.trans.pLGhostMapping);
        if (face2cell.father && cell2node.trans.pLGhostMapping)
            face2cell.idx.wireTargetMapping(cell2node.trans.pLGhostMapping);
        if (face2bnd.father && bnd2node.trans.pLGhostMapping)
            face2bnd.idx.wireTargetMapping(bnd2node.trans.pLGhostMapping);
        if (cell2edge.father && edge2node.trans.pLGhostMapping)
            cell2edge.idx.wireTargetMapping(edge2node.trans.pLGhostMapping);
        if (edge2node.father && coords.trans.pLGhostMapping)
            edge2node.idx.wireTargetMapping(coords.trans.pLGhostMapping);
        if (edge2cell.father && cell2node.trans.pLGhostMapping)
            edge2cell.idx.wireTargetMapping(cell2node.trans.pLGhostMapping);

        // Step 6: Update idx states
        auto markGlobalIfBuilt = [](auto &trackedPair)
        {
            if (trackedPair.father && trackedPair.idx.state() != Adj_Unknown)
                trackedPair.idx.markGlobal();
        };
        // For all tracked adj that were affected (non-SKIP), mark global.
        // Simplification: mark all existing adj as global since we're in
        // Adj_PointToGlobal state overall.
        if (cell2node.father)
            cell2node.idx.markGlobal();
        if (cell2cell.father)
            cell2cell.idx.markGlobal();
        if (bnd2node.father)
            bnd2node.idx.markGlobal();
        if (bnd2cell.father)
            bnd2cell.idx.markGlobal();
        if (node2cell.father)
            node2cell.idx.markGlobal();
        if (node2bnd.father)
            node2bnd.idx.markGlobal();

        // Step 7: Update mesh-level state
        adjPrimaryState = Adj_PointToGlobal;
        if (node2cell.father)
            adjN2CBState = Adj_PointToGlobal;

        // Invalidate local vectors
        cell2parentCell.clear();
        node2parentNode.clear();
        node2bndNode.clear();
        vtkCell2nodeOffsets.clear();
        vtkCellType.clear();
        vtkCell2node.clear();
        nodeRecreated2nodeLocal.clear();
        localPartitionStarts.clear();
        cell2cellFaceVLocalParts.clear();

        // Invalidate side caches when Face or Bnd was reordered (or destroyed above)
        if (plan.reorderedKinds.count(EntityKind::Face) ||
            plan.reorderedKinds.count(EntityKind::Bnd))
        {
            bnd2faceV.clear();
            face2bndM.clear();
        }
    }

    // =================================================================
    // UnstructuredMesh::ReorderLocalCells (new, using ReorderEntities)
    // =================================================================

    void UnstructuredMesh::ReorderLocalCells(int nParts, int nPartsInner)
    {
        CheckCollectiveMutationAvailable();
        // Guard against edge-interpolated state.
        // ReorderLocalCells runs before InterpolateFace and InterpolateEdge in
        // the standard pipeline.  Calling it after edge interpolation would
        // silently corrupt cell2edge, edge2cell, PBI arrays, and edge mappings
        // because none of the edge local/global conversions, row transfers, or
        // re-wire operations are implemented here.
        // Face/cell2face/face2cell/C2CFace have partial handling (L2G/G2L,
        // remap, relocate, re-wire) and are intentionally NOT rejected by this
        // guard — see the TODO comment below for remaining face gaps.
        DNDS_check_throw_info(
            adjEdgeState == Adj_Unknown && !cell2edge.father,
            "ReorderLocalCells: edges are built (adjEdgeState=" + std::to_string(int(adjEdgeState)) + ").  ReorderLocalCells does not handle edge adjacency "
                                                                                                      "remapping; call it before InterpolateEdge.  "
                                                                                                      "If you need to reorder after edge interpolation, use "
                                                                                                      "ReorderEntities instead.");
        // TODO(reorder-edges): this function does not handle cell2face/face2cell
        // or edge adjacencies (cell2edge, cell2edgePbi, edge2cell).  In the
        // standard pipeline ReorderLocalCells runs before InterpolateFace and
        // InterpolateEdge, so these arrays are never built at this point.
        // If a user calls ReorderLocalCells after face or edge interpolation,
        // these adjacencies will be silently corrupted.
        // Missing operations include:
        //   - L2G/G2L conversions (AdjLocal2GlobalEdge, AdjGlobal2LocalEdge)
        //   - Cell ref recording from edge2cell (addCellRefs)
        //   - Cell index remap in edge2cell (remapCellEntries)
        //   - Row permutation of cell2edge, cell2edgePbi (cellTransfer.transferRows)
        //   - Global mapping borrow for cell2edge, cell2edgePbi.father->pLGlobalMapping
        //   - Ghost borrowAndPull for cell2edge, cell2edgePbi
        //   - edge2cell re-wire (idx.wireTargetMapping)
        //   - edge2cell ghost pull (trans.pullOnce)
        DNDS_assert(this->adjPrimaryState == Adj_PointToLocal);
        DNDS_assert(cell2node.isLocal() && bnd2node.isLocal() &&
                    cell2cell.isLocal() && bnd2cell.isLocal());
        nParts = std::max(nParts, 1);
        nPartsInner = std::max(nPartsInner, 1);
        MeshMutationToken mutationToken(this, versions, Runtime::Change::Layout | Runtime::Change::Allocation | Runtime::Change::Distribution);

        // --- Section A: Compute cell permutation (same as legacy) ---
        // We need the local face-adjacency graph and cell2cell.
        // Convert to global first to get the permutation computation right,
        // then convert back and redo with the framework.
        // Actually: ComputeCellPermutation works on LOCAL indices (it uses
        // cell2cell in local state). So compute the permutation first.
        auto cell2cellFaceV = this->GetCell2CellFaceVLocal();

        auto perm = detail::ComputeCellPermutation(
            cell2cellFaceV, cell2cell, NumCell(), nParts, nPartsInner);
        this->localPartitionStarts = std::move(perm.localPartitionStarts);

        MPI::AllreduceOneIndex(perm.bwOld, MPI_MAX, mpi);
        MPI::AllreduceOneIndex(perm.bwNew, MPI_MAX, mpi);
        if (mpi.rank == mRank)
            log() << fmt::format("UnstructuredMesh === ReorderLocalCells, nPart0 [{}], "
                                 "got reordering, bw [{}] to [{}]",
                                 nParts, perm.bwOld, perm.bwNew)
                  << std::endl;

        // --- Convert to global ---
        if (this->adjFacialState == Adj_PointToLocal && face2cell.isBuilt())
            this->AdjLocal2GlobalFacialImpl(mutationToken);
        if (this->adjC2FState == Adj_PointToLocal && cell2face.isBuilt())
            this->AdjLocal2GlobalC2FImpl(mutationToken);
        if (this->adjC2CFaceState == Adj_PointToLocal && cell2cellFace.isBuilt())
            this->AdjLocal2GlobalC2CFaceImpl(mutationToken);
        if (this->adjN2CBState == Adj_PointToLocal && node2cell.isBuilt())
            this->AdjLocal2GlobalN2CBImpl(mutationToken);
        this->AdjLocal2GlobalPrimaryImpl(mutationToken);

        // --- Build cell partition map (all local, use permutation) ---
        // fromLocalPermutation expects old2new. perm.cellOld2New is that.
        std::vector<MPI_int> cellPartition(NumCell(), mpi.rank);

        // We need to communicate the permutation to ReorderEntities.
        // The framework's fromPartition computes new globals automatically,
        // but for a local permutation we want a specific ordering.
        // Use fromLocalPermutation by overriding the transfer after build.
        //
        // Actually, the simplest approach: call ReorderEntities with the
        // all-same-rank partition (which is an identity from the framework's
        // perspective), then separately apply the local permutation.
        //
        // Better: don't use ReorderEntities here. Instead, use the framework
        // components directly with fromLocalPermutation.

        // Build a PermutationTransfer from the computed permutation
        auto cellGM = cell2node.father->pLGlobalMapping;
        DNDS_assert(cellGM);
        auto cellTransfer = PermutationTransfer::fromLocalPermutation(
            perm.cellOld2New, cellGM, mpi);
        DNDS_assert(cellTransfer.isLocalOnly);

        // Build lookup for cell entry remapping
        // Collect off-rank cell globals from adj entries targeting cells
        std::set<index> cellPullSet;
        auto addCellRefs = [&](const auto &adj, index nRows)
        {
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index v = adj(i, j);
                    if (v == UnInitIndex)
                        continue;
                    auto [found, rank, val] = cellGM->search(v);
                    if (found && rank != mpi.rank)
                        cellPullSet.insert(v);
                }
        };
        addCellRefs(cell2cell, NumCell());
        addCellRefs(bnd2cell, NumBnd());
        if (node2cell.father)
            addCellRefs(node2cell, NumNode());
        if (face2cell.father)
            addCellRefs(face2cell, NumFace());
        if (cell2cellFace.father)
            addCellRefs(cell2cellFace, NumCell());

        // The retained halo can extend beyond cells referenced by owned rows
        // (BuildGhostPrimary with two or more layers). Its complete index list
        // is remapped below, so every halo cell needs its owner's permutation.
        if (cell2node.trans.pLGhostMapping)
            for (index global : cell2node.trans.pLGhostMapping->ghostIndex)
                cellPullSet.insert(global);

        std::vector<index> pullVec(cellPullSet.begin(), cellPullSet.end());
        auto cellLookup = cellTransfer.buildLookup(pullVec, mpi);

        // --- REMAP: update cell indices in xxx2cell adjacencies ---
        auto remapCellEntries = [&](auto &adj, index nRows)
        {
            for (index i = 0; i < nRows; i++)
                for (rowsize j = 0; j < adj.RowSize(i); j++)
                {
                    index &v = adj(i, j);
                    if (v != UnInitIndex)
                        v = cellLookup.resolve(v);
                }
        };

        remapCellEntries(cell2cell, NumCell());
        remapCellEntries(bnd2cell, NumBnd());
        if (node2cell.father)
            remapCellEntries(node2cell, NumNode());
        if (face2cell.father)
            remapCellEntries(face2cell, NumFace());
        if (cell2cellFace.father)
            remapCellEntries(cell2cellFace, NumCell());

        // --- RELOCATE: permute cell2xxx rows ---
        cellTransfer.transferRows(cell2node, mpi);
        cellTransfer.transferRows(cell2cell, mpi);
        cellTransfer.transferRows(cellElemInfo, mpi);
        cellTransfer.transferRows(cell2cellOrig, mpi);
        if (cell2face.father)
            cellTransfer.transferRows(cell2face, mpi);
        if (cell2cellFace.father)
            cellTransfer.transferRows(cell2cellFace, mpi);
        if (isPeriodic && cell2nodePbi.father)
            cellTransfer.transferRows(cell2nodePbi, mpi);
        if (isPeriodic && cell2facePbi.father)
            cellTransfer.transferRows(cell2facePbi, mpi);

        // --- Rebuild global mapping ---
        cell2node.father->createGlobalMapping();
        if (cell2cell.father)
            cell2cell.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
        if (cellElemInfo.father)
            cellElemInfo.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;
        if (cell2cellOrig.father)
            cell2cellOrig.father->pLGlobalMapping = cell2node.father->pLGlobalMapping;

        // --- Rebuild ghost mappings (local-only optimization) ---
        // Permute the ghost index list to match new cell globals
        {
            std::vector<index> ghostCellGlobalsNew;
            if (cell2node.trans.pLGhostMapping)
            {
                ghostCellGlobalsNew = cell2node.trans.pLGhostMapping->ghostIndex;
                for (index &g : ghostCellGlobalsNew)
                    g = cellLookup.resolve(g);
            }
            // Reattach son and create ghost mapping
            if (!cell2node.son)
                cell2node.son = make_ssp<tAdjPair::t_arr>(ObjName{"reorder.son"}, mpi);
            cell2node.TransAttach();
            cell2node.trans.createFatherGlobalMapping();
            cell2node.trans.createGhostMapping(ghostCellGlobalsNew);
            cell2node.trans.createMPITypes();
            cell2node.trans.pullOnce();
        }
        // Borrow ghost indexing for other cell arrays
        {
            auto borrowAndPull = [&](auto &pair)
            {
                if (!pair.father)
                    return;
                if (!pair.son)
                {
                    using TArr = typename std::remove_reference_t<decltype(pair)>::t_arr;
                    pair.son = make_ssp<TArr>(ObjName{"reorder.son"}, mpi);
                }
                pair.TransAttach();
                pair.trans.BorrowGGIndexing(cell2node.trans);
                pair.trans.createMPITypes();
                pair.trans.pullOnce();
            };
            borrowAndPull(cell2cell);
            borrowAndPull(cell2cellOrig);
            borrowAndPull(cellElemInfo);
            if (cell2face.father)
                borrowAndPull(cell2face);
            if (cell2cellFace.father)
                borrowAndPull(cell2cellFace);
            if (isPeriodic && cell2nodePbi.father)
                borrowAndPull(cell2nodePbi);
            if (isPeriodic && cell2facePbi.father)
                borrowAndPull(cell2facePbi);
        }

        // --- Re-wire target mappings ---
        {
            auto cellGhostMap = cell2node.trans.pLGhostMapping;
            cell2cell.idx.wireTargetMapping(cellGhostMap);
            bnd2cell.idx.wireTargetMapping(cellGhostMap);
            if (cell2cellFace.father && cell2cellFace.idx.isWired())
                cell2cellFace.idx.wireTargetMapping(cellGhostMap);
            if (node2cell.father && node2cell.idx.isWired())
                node2cell.idx.wireTargetMapping(cellGhostMap);
            if (face2cell.father && face2cell.idx.isWired())
                face2cell.idx.wireTargetMapping(cellGhostMap);
        }

        // --- Pull ghost data for non-cell adjacencies (face2cell, node2cell) ---
        if (face2cell.father && face2cell.trans.pLGhostMapping)
            face2cell.trans.pullOnce();
        if (node2cell.father && node2cell.trans.pLGhostMapping)
            node2cell.trans.pullOnce();
        DNDS_check_throw_info(bnd2cell.father && bnd2cell.trans.pLGhostMapping,
                              "ReorderLocalCells: bnd2cell must have ghost mapping for pull");
        bnd2cell.trans.pullOnce();

        // --- Convert back to local ---
        if (this->adjFacialState == Adj_PointToGlobal && face2cell.isBuilt())
            this->AdjGlobal2LocalFacialImpl(mutationToken);
        if (this->adjC2FState == Adj_PointToGlobal && cell2face.isBuilt())
            this->AdjGlobal2LocalC2FImpl(mutationToken);
        if (this->adjC2CFaceState == Adj_PointToGlobal && cell2cellFace.isBuilt())
            this->AdjGlobal2LocalC2CFaceImpl(mutationToken);
        if (this->adjN2CBState == Adj_PointToGlobal && node2cell.isBuilt())
            this->AdjGlobal2LocalN2CBImpl(mutationToken);
        this->AdjGlobal2LocalPrimaryImpl(mutationToken);

        if (mpi.rank == mRank)
            log() << fmt::format("UnstructuredMesh === ReorderLocalCells finished") << std::endl;
        mutationToken.Commit();
    }

} // namespace DNDS::Geom
