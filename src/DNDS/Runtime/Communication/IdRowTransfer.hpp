#pragma once

#include "DNDS/Runtime/Communication/SparseBatchExchange.hpp"

namespace DNDS::Runtime
{
    /// Build an ordinary row transfer by exact persistent IDs, with a distributed
    /// hash directory. No allocation depends on the numeric ID range or global
    /// row count. Hashes select directory ranks; full IDs decide matches.
    /// Source IDs must be globally unique. Repeated target IDs are allowed for
    /// gathering unless requireBijection requests exact one-to-one coverage.
    ///
    /// All ranks enter with the same policy/context. CPU ID producers must be
    /// complete; input leases are retained until lookup finishes. The returned
    /// plan is a snapshot of row correspondence, not an owner of the ID arrays.
    /// Rebuild it after either row permutation or identity changes, or supply
    /// version dependencies from the owning mesh/publication. Payloads use the
    /// normal PayloadPlan/binding contract, independently of this cold lookup.
    inline RowTransferPlan BuildRowTransferById(std::shared_ptr<CommunicationContext> context,
                                                const Array<index> &sourceIds, const Array<index> &targetIds,
                                                bool requireBijection = false, std::vector<VersionStamp> dependencies = {})
    {
        DNDS_check_throw_info(context, "ID row transfer requires a communication context");
        const MPIInfo mpi(context->Communicator());
        int policy[]{int(requireBijection), -int(requireBijection)};
        MPI_Allreduce(MPI_IN_PLACE, policy, 2, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(policy[0] == -policy[1], "ID transfer policies differ between ranks");
        std::optional<Array<index>::ReadLease> sourceRead, targetRead;
        Array<index> records, queries;
        std::vector<std::vector<index>> sourcePeers, queryPeers;
        const auto bucket = [&](index id)
        {
            uint64_t hash = uint64_t(id) + 0x9e3779b97f4a7c15ULL;
            hash = (hash ^ (hash >> 30)) * 0xbf58476d1ce4e5b9ULL;
            hash = (hash ^ (hash >> 27)) * 0x94d049bb133111ebULL;
            return int((hash ^ (hash >> 31)) % uint64_t(mpi.size));
        };
        MPI::LocalStage(mpi, [&]
                        {
            for (const auto *ids : {&sourceIds, &targetIds})
                DNDS_check_throw_info(ids->GetDevice() == Device::Host() && !ids->Layout().IsVariable() && ids->Layout().Width() == 1,
                                      "ID transfer needs CPU scalar ID tables");
            sourceRead.emplace(sourceIds.Read()); targetRead.emplace(targetIds.Read());
            records = Array<index>(RowLayout::Dense(sourceIds.RowCount(), 3));
            queries = Array<index>(RowLayout::Dense(targetIds.RowCount(), 1));
            sourcePeers.resize(mpi.size); queryPeers.resize(mpi.size);
            auto directory = records.Write(), requested = queries.Write();
            for (index iRow = 0; iRow < sourceIds.RowCount(); ++iRow)
            {
                const index id = sourceRead->View<UniformTableRows>()(iRow, 0);
                DNDS_check_throw_info(id >= 0, "negative source persistent ID");
                directory.View()(iRow, 0) = id; directory.View()(iRow, 1) = mpi.rank; directory.View()(iRow, 2) = iRow;
                sourcePeers[bucket(id)].push_back(iRow);
            }
            for (index iRow = 0; iRow < targetIds.RowCount(); ++iRow)
            {
                const index id = targetRead->View<UniformTableRows>()(iRow, 0);
                DNDS_check_throw_info(id >= 0, "negative target persistent ID");
                requested.View()(iRow, 0) = id; queryPeers[bucket(id)].push_back(iRow);
            } });
        auto published = SparseBatchExchange::Build(context, sourceIds.RowCount(), std::move(sourcePeers)).Exchange(records);
        const auto queryRoute = SparseBatchExchange::Build(context, targetIds.RowCount(), std::move(queryPeers));
        auto requested = queryRoute.Exchange(queries);
        struct Entry
        {
            index id, rank, row;
        };
        std::vector<Entry> directory;
        Array<index> answers;
        MPI::LocalStage(mpi, [&]
                        {
            const auto read = published.Read(), wanted = requested.Read();
            directory.reserve(published.RowCount());
            for (index iRow = 0; iRow < published.RowCount(); ++iRow)
                directory.push_back({read.View()(iRow, 0), read.View()(iRow, 1), read.View()(iRow, 2)});
            std::sort(directory.begin(), directory.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
            for (index iRow = 1; iRow < index(directory.size()); ++iRow)
                DNDS_check_throw_info(directory[iRow - 1].id != directory[iRow].id, "duplicate source persistent ID");
            if (requireBijection)
            {
                std::vector<index> ids; ids.reserve(requested.RowCount());
                for (index iRow = 0; iRow < requested.RowCount(); ++iRow) ids.push_back(wanted.View()(iRow, 0));
                std::sort(ids.begin(), ids.end());
                DNDS_check_throw_info(ids.size() == directory.size(), "ID transfer target coverage differs");
                for (index iRow = 0; iRow < index(ids.size()); ++iRow)
                    DNDS_check_throw_info(ids[iRow] == directory[iRow].id, "ID transfer targets are not a bijection");
            }
            answers = Array<index>(RowLayout::Dense(requested.RowCount(), 2));
            auto write = answers.Write();
            for (index iRow = 0; iRow < requested.RowCount(); ++iRow)
            {
                const index id = wanted.View()(iRow, 0);
                const auto found = std::lower_bound(directory.begin(), directory.end(), id, [](const Entry &entry, index key) { return entry.id < key; });
                DNDS_check_throw_info(found != directory.end() && found->id == id, "target persistent ID has no source");
                write.View()(iRow, 0) = found->rank; write.View()(iRow, 1) = found->row;
            } });
        Array<index> locations;
        MPI::LocalStage(mpi, [&]
                        { locations = Array<index>(RowLayout::Dense(targetIds.RowCount(), 2)); });
        PayloadPlan(queryRoute.Route().Reverse(), answers.Layout(), locations.Layout(), ElementSchema::Arithmetic<index>())
            .Bind(answers, locations, TransferTransport::Direct, {TransferMode::Transient, context, 0})
            .Begin()
            .Wait();
        std::vector<RowRequest> requests;
        MPI::LocalStage(mpi, [&]
                        {
            const auto read = locations.Read(); requests.reserve(targetIds.RowCount());
            for (index iRow = 0; iRow < targetIds.RowCount(); ++iRow)
                requests.push_back({int(read.View()(iRow, 0)), read.View()(iRow, 1), iRow}); });
        return RowTransferPlan(std::move(context), sourceIds.RowCount(), targetIds.RowCount(), requests, std::move(dependencies));
    }
}
