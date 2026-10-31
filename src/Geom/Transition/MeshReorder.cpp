#include "Geom/Transition/ReorderPlan.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include <algorithm>
#include <limits>
#include <set>
namespace DNDS::Geom
{
    ReorderPlan ReorderPlan::build(
        const ReorderInput &input,
        const ReorderRegistry &registry,
        const MPIInfo &mpi)
    {
        ReorderPlan plan;
        int nativeOnly = !input.localPermutations.empty();
        MPI_Allreduce(MPI_IN_PLACE, &nativeOnly, 1, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(!nativeOnly, "local entity permutations require NativeMeshTransition");

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
                DNDS_check_throw_info(!field.requiresEntityBinding,
                                      "entity-bound fields require detached PreparedMeshTransition destination binding");
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

}
