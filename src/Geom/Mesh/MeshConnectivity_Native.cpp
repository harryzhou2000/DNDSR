#include "NativeConnectivity.hpp"
#include "ConnectivityComposeRows.hpp"
#include <algorithm>
#include <limits>

namespace DNDS::Geom
{
    void RelationReadAccess::Validate() const
    {
        CheckActive();
        _kind.Validate();
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
        DNDS_check_throw_info(!_kind.removeSelf || _source.SameOrder(_target),
                              "removeSelf relation requires identical source/target entity order");
        for (index row = 0; row < Rows(); ++row)
        {
            index previous = -1;
            for (size_t slot = 0; slot < RowSize(row); ++slot)
            {
                const auto entry = ReadEntry(row, slot);
                if (!_kind.IsLegacy() && _kind.order == RelationOrder::SortedUnique)
                {
                    // Sorting describes entity globals, never local encodings:
                    // owned rows precede ghosts even when a ghost global is lower.
                    DNDS_check_throw_info(!entry.absent && entry.global > previous,
                                          "SortedUnique relation has absent, duplicate or unordered targets");
                    previous = entry.global;
                }
                DNDS_check_throw_info(!_kind.removeSelf || entry.absent || entry.global != _source.LocalToGlobal(row),
                                      "removeSelf relation contains its source entity");
            }
        }
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

    index RelationReadAccess::ConvertedIndex(index row, size_t slot, MeshAdjState targetState,
                                             bool allowUnavailable) const
    {
        DNDS_check_throw_info(targetState == Adj_PointToGlobal || targetState == Adj_PointToLocal,
                              "relation conversion requires an explicit global/local destination state");
        const auto entry = ReadEntry(row, slot);
        if (entry.absent)
            return UnInitIndex;
        if (targetState == Adj_PointToGlobal)
            return entry.global;
        if (!entry.unavailable)
            return *_target.GlobalToLocal(entry.global);
        DNDS_check_throw_info(allowUnavailable, "relation local conversion requires missing target closure");
        const index encoded = -1 - entry.global;
        DNDS_check_throw_info(encoded != UnInitIndex, "relation unavailable target collides with absent sentinel");
        return encoded;
    }

    NativeRelation NativeRelation::Convert(const RelationReadAccess &source, MeshAdjState targetState,
                                           bool allowUnavailable)
    {
        source.Validate();
        DNDS_check_throw_info(targetState == Adj_PointToGlobal || targetState == Adj_PointToLocal,
                              "relation conversion requires an explicit global/local destination state");
        auto convert = [&](const Runtime::ArrayView<const index> &view, index first)
        {
            std::vector<size_t> lengths;
            if (view.offsets || view.sizes)
            {
                lengths.resize(view.Rows());
                for (size_t row = 0; row < view.Rows(); ++row)
                    lengths[row] = view.RowSize(row);
            }
            const auto shape = view.offsets ? Runtime::RowLayout::Ragged(lengths) : view.sizes ? Runtime::RowLayout::Padded(lengths, view.stride)
                                                                                               : Runtime::RowLayout::Dense(view.Rows(), view.width, view.stride);
            Runtime::Array<index> output(shape);
            {
                auto write = output.Write();
                for (size_t row = 0; row < view.Rows(); ++row)
                    for (size_t slot = 0; slot < view.RowSize(row); ++slot)
                        write.View()(row, slot) = source.ConvertedIndex(first + index(row), slot, targetState, allowUnavailable);
            }
            return output;
        };
        NativeRelation result(source.Kind(), source.Source(), source.Target(), convert(source._view.owned, 0), targetState);
        // A mesh scope retains both portions in one MeshHold; it need not have
        // a separate ghost holder even though ghost rows are present.
        if (source._ghost || source._view.ghost.Rows())
            result._ghost.emplace(convert(source._view.ghost, source.NumOwned()));
        auto check = result.Read();
        return result;
    }

    static void WriteNativeRow(Runtime::RowBuilder &builder, index row, const std::vector<index> &entries)
    {
        builder.ResizeRow(size_t(row), entries.size(), false);
        auto target = builder.Row<index>(size_t(row));
        for (size_t slot = 0; slot < entries.size(); ++slot)
            target[slot] = entries[slot];
    }

    NativeRelation NativeConnectivity::ComposeFiltered(const RelationReadAccess &AB, const RelationReadAccess &BC,
                                                       const AdjKind &resultKind,
                                                       const std::function<bool(index, index, int)> &predicate,
                                                       const std::function<bool(index, index, const std::vector<index> &)> &matchExtra)
    {
        AB.Validate();
        BC.Validate();
        DNDS_check_throw_info(AB.Target().SameOrder(BC.Source()), "composition intermediate entity/map order mismatch");
        DNDS_check_throw_info(bool(predicate), "composition predicate is missing");
        resultKind.Validate();
        DNDS_check_throw_info(!resultKind.IsLegacy() && resultKind == AdjKind::Composition(AB.Kind(), BC.Kind(),
                                                                                           resultKind.selection, resultKind.removeSelf, resultKind.semanticName, resultKind.semanticVersion, resultKind.periodic),
                              "composition output descriptor differs from its input definitions");
        DNDS_check_throw_info(!resultKind.removeSelf || AB.Source().SameOrder(BC.Target()),
                              "removeSelf requires identical source and final target spaces");
        DNDS_check_throw_info(resultKind.selection == RelationSelection::Any || resultKind.selection == RelationSelection::All ||
                                  resultKind.selection == RelationSelection::Custom || resultKind.selection == RelationSelection::Exact,
                              "composition requires an any/all/custom/exact selection");
        DNDS_check_throw_info(!matchExtra || resultKind.selection == RelationSelection::Custom || resultKind.selection == RelationSelection::Exact,
                              "additional matching requires a named custom/exact selection");
        DNDS_check_throw_info(resultKind.periodic != RelationPeriodicPolicy::FrameMatched || bool(matchExtra),
                              "frame-matched selection requires an explicit periodic matcher");
        DNDS_check_throw_info(resultKind.selection != RelationSelection::Exact || bool(matchExtra),
                              "exact selection requires an explicit incidence/frame matcher");
        auto unique = [](std::vector<index> entries)
        {
            entries.erase(std::remove(entries.begin(), entries.end(), UnInitIndex), entries.end());
            std::sort(entries.begin(), entries.end());
            entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
            return entries;
        };
        auto selected = [&](index a, index c, int count)
        {
            if (resultKind.removeSelf && a == c)
                return false;
            if (resultKind.selection == RelationSelection::Any)
                return count >= 1;
            if (resultKind.selection == RelationSelection::All)
            {
                const auto local = AB.Source().GlobalToLocal(a);
                DNDS_check_throw_info(local, "composition source row is unavailable");
                return count == int(unique(AB.GlobalRow(*local)).size());
            }
            return predicate(a, c, count);
        };
        Runtime::RowBuilder builder(size_t(AB.NumOwned()), sizeof(index));
        ConnectivityRows::Compose(AB.NumOwned(), [&](index row)
                                  { return unique(AB.GlobalRow(row)); }, [&](index row)
                                  { return unique(BC.GlobalRow(row)); }, [&](index global)
                                  {
                auto row = BC.Source().GlobalToLocal(global);
                DNDS_check_throw_info(row && *row < BC.Rows(), "composition intermediate row is unavailable; transfer it explicitly");
                return *row; }, [&](index row)
                                  { return AB.Source().LocalToGlobal(row); }, selected, [&](index row, const std::vector<index> &entries)
                                  { WriteNativeRow(builder, row, entries); }, matchExtra);
        return NativeRelation(resultKind, AB.Source(), BC.Target(),
                              Runtime::Array<index>::FromSharedStorage(builder.Commit()));
    }

    NativeRelation NativeConnectivity::Compose(const RelationReadAccess &AB, const RelationReadAccess &BC, bool removeSelf)
    {
        DNDS_check_throw_info(!removeSelf || AB.Source().SameOrder(BC.Target()),
                              "removeSelf requires identical source and final target spaces");
        return ComposeFiltered(AB, BC, AdjKind::Composition(AB.Kind(), BC.Kind(), RelationSelection::Any, removeSelf),
                               SharedCountPredicate{1, removeSelf});
    }

    NativeRelation NativeConnectivity::ComposeAll(const RelationReadAccess &AB, const RelationReadAccess &BC, bool removeSelf)
    {
        return ComposeFiltered(AB, BC, AdjKind::Composition(AB.Kind(), BC.Kind(), RelationSelection::All, removeSelf),
                               SharedCountPredicate{1, removeSelf});
    }

    NativeRelation NativeConnectivity::Inverse(const RelationReadAccess &cone, const MPIInfo &mpi)
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
        return NativeRelation(cone.Kind().Inverted(), cone.Target(), cone.Source(),
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
