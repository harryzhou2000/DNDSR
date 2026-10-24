#include "RelationAccess.hpp"
#include <algorithm>
#include <limits>

namespace DNDS::Geom
{
    void RelationReadAccess::Validate() const
    {
        CheckActive();
        _source.Validate();
        _target.Validate();
        EntitySpace::CheckCommunicator(_source._mpi.comm, _target._mpi.comm);
        DNDS_check_throw_info(_source.MeshIdentity() == _target.MeshIdentity() &&
                                  _kind.from == _source.Kind() && _kind.to == _target.Kind(),
                              "relation association differs from its source/target entity spaces");
        DNDS_check_throw_info(_state == Adj_PointToGlobal || _state == Adj_PointToLocal,
                              "relation setup requires an explicit global/local index state");
        DNDS_check_throw_info(_view.owned.Rows() == size_t(_source.NumOwned()) &&
                                  (_view.ghost.Rows() == 0 || _view.ghost.Rows() == size_t(_source.NumGhost())),
                              "relation owned/ghost rows differ from its source space");
        for (index row = 0; row < Rows(); ++row)
            for (size_t slot = 0; slot < RowSize(row); ++slot)
                (void)ReadEntry(row, slot);
    }

    RelationReadAccess::Entry RelationReadAccess::ReadEntry(index row, size_t slot) const
    {
        const auto value = Raw(row, slot);
        if (value == UnInitIndex)
            return {true, false, UnInitIndex};
        index global;
        bool unavailable;
        if (_state == Adj_PointToLocal && value >= 0)
        {
            DNDS_check_throw_info(value < _target.NumLocal(), "relation local target is outside its entity space");
            global = _target.LocalToGlobal(value);
            unavailable = false;
        }
        else
        {
            DNDS_check_throw_info(_state == Adj_PointToLocal || value >= 0,
                                  "global relation contains a local unavailable-index encoding");
            global = _state == Adj_PointToLocal ? -1 - value : value;
            DNDS_check_throw_info(global >= 0 && global < _target.NumGlobal(), "relation global target is outside its entity space");
            unavailable = !_target.GlobalToLocal(global).has_value();
            DNDS_check_throw_info(_state != Adj_PointToLocal || unavailable,
                                  "encoded unavailable target is present in the supplied local target mapping");
        }
        return {false, unavailable, global};
    }

    static void WriteNativeRow(Runtime::RowBuilder &builder, index row, const std::vector<index> &entries)
    {
        builder.ResizeRow(size_t(row), entries.size(), false);
        auto target = builder.Row<index>(size_t(row));
        for (size_t slot = 0; slot < entries.size(); ++slot)
            target[slot] = entries[slot];
    }

    NativeRelation MeshConnectivity::ComposeFiltered(const RelationReadAccess &AB, const RelationReadAccess &BC,
                                                     const std::function<bool(index, index, int)> &predicate,
                                                     const std::function<bool(index, index, const std::vector<index> &)> &matchExtra)
    {
        AB.Validate();
        BC.Validate();
        DNDS_check_throw_info(AB.Target().SameOrder(BC.Source()), "composition intermediate entity/map order mismatch");
        DNDS_check_throw_info(bool(predicate), "composition predicate is missing");
        Runtime::RowBuilder builder(size_t(AB.NumOwned()), sizeof(index));
        ConnectivityRows::Compose(AB.NumOwned(), [&](index row)
                                  { return AB.GlobalRow(row); }, [&](index row)
                                  { return BC.GlobalRow(row); }, [&](index global)
                                  {
                auto row = BC.Source().GlobalToLocal(global);
                DNDS_check_throw_info(row && *row < BC.Rows(), "composition intermediate row is unavailable; transfer it explicitly");
                return *row; }, [&](index row)
                                  { return AB.Source().LocalToGlobal(row); }, predicate, [&](index row, const std::vector<index> &entries)
                                  { WriteNativeRow(builder, row, entries); }, matchExtra);
        return NativeRelation({AB.Source().Kind(), BC.Target().Kind()}, AB.Source(), BC.Target(),
                              Runtime::Array<index>::FromSharedStorage(builder.Commit()));
    }

    NativeRelation MeshConnectivity::Compose(const RelationReadAccess &AB, const RelationReadAccess &BC, bool removeSelf)
    {
        DNDS_check_throw_info(!removeSelf || AB.Source().SameOrder(BC.Target()),
                              "removeSelf requires identical source and final target spaces");
        return ComposeFiltered(AB, BC, SharedCountPredicate{1, removeSelf});
    }

    NativeRelation MeshConnectivity::Inverse(const RelationReadAccess &cone, const MPIInfo &mpi)
    {
        int valid = 1;
        try
        {
            cone.Validate();
            cone.CheckCommunicator(mpi.comm);
        }
        catch (...)
        {
            valid = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "inverse relation/communicator validation failed on a rank");
        // Each peer receives one ragged batch of (target-owned row, source
        // global ID) contributions. Only the owning rank sees that incidence;
        // the all-to-all discovery communicates presence, never global topology.
        std::vector<std::vector<index>> batches(size_t(mpi.size));
        for (index source = 0; source < cone.NumOwned(); ++source)
        {
            const auto sourceGlobal = cone.Source().LocalToGlobal(source);
            for (size_t slot = 0; slot < cone.RowSize(source); ++slot)
            {
                const auto entry = cone.ReadEntry(source, slot);
                if (entry.absent)
                    continue;
                const auto owner = cone.Target().Owner(entry.global);
                batches[size_t(owner)].push_back(cone.Target().OwnerLocalRow(entry.global));
                batches[size_t(owner)].push_back(sourceGlobal);
            }
        }
        std::vector<int> sendPresence(size_t(mpi.size)), receivePresence(size_t(mpi.size));
        Runtime::RowBuilder outgoing(size_t(mpi.size), sizeof(index));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            sendPresence[size_t(peer)] = !batches[size_t(peer)].empty();
            WriteNativeRow(outgoing, peer, batches[size_t(peer)]);
        }
        auto source = Runtime::Array<index>::FromSharedStorage(outgoing.Commit());
        MPI_Alltoall(sendPresence.data(), 1, MPI_INT, receivePresence.data(), 1, MPI_INT, mpi.comm);
        std::vector<std::vector<size_t>> send(size_t(mpi.size)), receive(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            if (sendPresence[size_t(peer)])
                send[size_t(peer)].push_back(size_t(peer));
            if (receivePresence[size_t(peer)])
                receive[size_t(peer)].push_back(size_t(peer));
        }
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        auto route = Runtime::RowTransferPlan::Known(context, size_t(mpi.size), size_t(mpi.size),
                                                     std::move(send), std::move(receive),
                                                     {cone.Source().Stamp(), cone.Target().Stamp()});
        Runtime::Array<index> incoming(route.DestinationLayout(source.Layout()));
        Runtime::TransferOptions options;
        options.mode = Runtime::TransferMode::Transient;
        options.context = context;
        Runtime::PayloadPlan payload(route, source.Layout(), incoming.Layout(), Runtime::ElementSchema::Arithmetic<index>());
        payload.Bind(source, incoming, Runtime::TransferTransport::Direct, options).Begin().Wait();

        std::vector<std::vector<index>> support(size_t(cone.Target().NumOwned()));
        auto values = incoming.Read();
        auto view = values.View();
        valid = 1;
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            const auto length = view.RowSize(size_t(peer));
            valid = valid && length % 2 == 0;
            for (size_t slot = 0; slot + 1 < length; slot += 2)
            {
                const auto targetRow = view(size_t(peer), slot), sourceGlobal = view(size_t(peer), slot + 1);
                const bool entryValid = targetRow >= 0 && size_t(targetRow) < support.size() &&
                                        sourceGlobal >= 0 && sourceGlobal < cone.Source().NumGlobal() &&
                                        cone.Source().Owner(sourceGlobal) == peer;
                valid = valid && entryValid;
                if (entryValid)
                    support[size_t(targetRow)].push_back(sourceGlobal);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "inverse contribution payload contains an invalid row or source entity");
        Runtime::RowBuilder builder(support.size(), sizeof(index));
        for (size_t row = 0; row < support.size(); ++row)
        {
            auto &entries = support[row];
            std::sort(entries.begin(), entries.end());
            entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
            WriteNativeRow(builder, index(row), entries);
        }
        return NativeRelation({cone.Target().Kind(), cone.Source().Kind()}, cone.Target(), cone.Source(),
                              Runtime::Array<index>::FromSharedStorage(builder.Commit()));
    }

    NativeRelation NativeRelation::WithGhostRows(const RelationGhostRoute &halo) const
    {
        const auto mpi = RelationReadAccess::SourceMPI(_source);
        std::optional<RelationReadAccess> read;
        int valid = 1;
        try
        {
            read.emplace(Read()); // Pins both entity orders through completion.
            read->CheckCommunicator(halo._route.Communicator());
            valid = halo.Valid() && _source.SameOrder(halo._source) && _state == Adj_PointToGlobal &&
                    halo._route.SourceRows() == size_t(_source.NumOwned()) &&
                    halo._route.DestinationRows() == size_t(_source.NumGhost());
        }
        catch (...)
        {
            valid = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, mpi.comm);
        DNDS_check_throw_info(valid, "native halo route has stale/wrong source order or nonglobal relation payload");
        Runtime::Array<index> ghosts(halo._route.DestinationLayout(_owned.Layout()));
        Runtime::TransferOptions options;
        options.mode = Runtime::TransferMode::Transient;
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        options.context = context;
        Runtime::PayloadPlan payload(halo._route, _owned.Layout(), ghosts.Layout(), Runtime::ElementSchema::Arithmetic<index>());
        payload.Bind(_owned, ghosts, Runtime::TransferTransport::Direct, options).Begin().Wait();
        NativeRelation result(_kind, _source, _target, _owned, _state);
        result._ghost.emplace(std::move(ghosts));
        auto validated = result.Read();
        return result;
    }
}
