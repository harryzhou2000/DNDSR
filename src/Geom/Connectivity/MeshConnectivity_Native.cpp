#include "Geom/Connectivity/NativeConnectivity.hpp"
#include "Geom/Connectivity/ConnectivityComposeRows.hpp"
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
        if (_contentsValidated)
            return;
        const bool sorted = !_kind.IsLegacy() && _kind.order == RelationOrder::SortedUnique;
        const bool needsGlobal = sorted || _kind.removeSelf;
        const auto scan = [&](const Runtime::ArrayView<const index> &view, index first)
        {
            // The scope and both mappings are pinned above. Internal validation
            // can traverse raw rows without re-entering public access checks
            // for every incidence. Public random access remains checked.
            for (size_t row = 0; row < view.Rows(); ++row)
            {
                index previous = -1;
                const auto sourceGlobal = _kind.removeSelf ? _source.LocalToGlobalPinned(first + index(row)) : -1;
                const auto count = view.RowSize(row);
                const auto *values = count ? view.Row(row) : nullptr;
                for (size_t slot = 0; slot < count; ++slot)
                {
                    const auto value = values[slot];
                    const bool absent = value == UnInitIndex;
                    index global = value;
                    if (!absent)
                    {
                        if (_state == Adj_PointToGlobal)
                            DNDS_check_throw_info(value >= 0 && value < _target.NumGlobal(),
                                                  "relation global target is outside its entity space");
                        else if (value >= 0)
                        {
                            DNDS_check_throw_info(value < _target.NumLocal(), "relation local target is outside its entity space");
                            if (needsGlobal)
                                global = _target.LocalToGlobalPinned(value);
                        }
                        else
                        {
                            global = -1 - value;
                            DNDS_check_throw_info(global >= 0 && global < _target.NumGlobal(),
                                                  "relation global target is outside its entity space");
                            DNDS_check_throw_info(!_target.GlobalToLocalPinned(global),
                                                  "encoded unavailable target is present in the supplied local target mapping");
                        }
                    }
                    if (sorted)
                    {
                        // Ordering is in global IDs, including local encodings
                        // whose appended ghost globals precede owned globals.
                        DNDS_check_throw_info(!absent && global > previous,
                                              "SortedUnique relation has absent, duplicate or unordered targets");
                        previous = global;
                    }
                    DNDS_check_throw_info(!_kind.removeSelf || absent || global != sourceGlobal,
                                          "removeSelf relation contains its source entity");
                }
            }
        };
        scan(_view.owned, 0);
        scan(_view.ghost, _source.NumOwned());
        _contentsValidated = true;
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
            global = _target.LocalToGlobalPinned(value);
            unavailable = false;
        }
        else
        {
            DNDS_check_throw_info(_state == Adj_PointToLocal || value >= 0,
                                  "global relation contains a local unavailable-index encoding");
            global = _state == Adj_PointToLocal ? -1 - value : value;
            DNDS_check_throw_info(global >= 0 && global < _target.NumGlobal(), "relation global target is outside its entity space");
            unavailable = !_target.GlobalToLocalPinned(global).has_value();
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
        // The construction scan validated these encodings while the retained
        // leases pinned values and both mappings. Preserve slot bounds, but
        // translate only when the requested representation actually differs.
        const auto value = Raw(row, slot);
        if (value == UnInitIndex)
            return UnInitIndex;
        if (_state == Adj_PointToLocal)
        {
            if (targetState == Adj_PointToGlobal)
                return value >= 0 ? _target.LocalToGlobalPinned(value) : -1 - value;
            DNDS_check_throw_info(value >= 0 || allowUnavailable, "relation local conversion requires missing target closure");
            return value;
        }
        if (targetState == Adj_PointToGlobal)
            return value;
        const auto local = _target.GlobalToLocalPinned(value);
        if (local)
            return *local;
        DNDS_check_throw_info(allowUnavailable, "relation local conversion requires missing target closure");
        const index encoded = -1 - value;
        DNDS_check_throw_info(encoded != UnInitIndex, "relation unavailable target collides with absent sentinel");
        return encoded;
    }

    void RelationReadAccess::ConvertRowTo(index row, MeshAdjState targetState, index *output,
                                          bool allowUnavailable) const
    {
        DNDS_check_throw_info(targetState == Adj_PointToGlobal || targetState == Adj_PointToLocal,
                              "relation conversion requires an explicit global/local destination state");
        DNDS_check_throw_info(row >= 0 && row < Rows(), "relation row is not present locally");
        const auto count = _view.RowSize(size_t(row));
        DNDS_check_throw_info(!count || output, "relation row conversion needs destination storage");
        const auto *source = count ? _view.Row(size_t(row)) : nullptr;
        for (size_t slot = 0; slot < count; ++slot)
        {
            const auto value = source[slot];
            if (value == UnInitIndex)
            {
                output[slot] = UnInitIndex;
                continue;
            }
            if (_state == Adj_PointToLocal)
            {
                if (targetState == Adj_PointToGlobal)
                    output[slot] = value >= 0 ? _target.LocalToGlobalPinned(value) : -1 - value;
                else
                {
                    DNDS_check_throw_info(value >= 0 || allowUnavailable,
                                          "relation local conversion requires missing target closure");
                    output[slot] = value;
                }
                continue;
            }
            if (targetState == Adj_PointToGlobal)
            {
                output[slot] = value;
                continue;
            }
            if (const auto local = _target.GlobalToLocalPinned(value))
                output[slot] = *local;
            else
            {
                DNDS_check_throw_info(allowUnavailable,
                                      "relation local conversion requires missing target closure");
                output[slot] = -1 - value;
            }
        }
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
                    source.ConvertRowTo(first + index(row), targetState, write.View().Row(row), allowUnavailable);
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

    static Runtime::Array<index> BuildNativeRows(const std::vector<std::vector<index>> &rows)
    {
        return Runtime::BuildArrayRows<index>(rows.size(), [&](size_t row)
                                              { return rows[row].size(); }, [&](size_t row, Runtime::RowSpan<index> target)
                                              {
            if (!rows[row].empty())
                std::copy(rows[row].begin(), rows[row].end(), target.data()); });
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
        // Node support is revisited by every incident cell. Normalize each
        // retained row once instead of allocating/sorting it at every hop.
        // These caches scale with local owned/appended rows, never global size.
        std::vector<std::vector<index>> sourceRows(size_t(AB.NumOwned())), intermediateRows(size_t(BC.Rows()));
        for (index row = 0; row < AB.NumOwned(); ++row)
            sourceRows[size_t(row)] = unique(AB.GlobalRow(row));
        for (index row = 0; row < BC.Rows(); ++row)
            intermediateRows[size_t(row)] = unique(BC.GlobalRow(row));
        auto selected = [&](index a, index c, int count)
        {
            if (resultKind.removeSelf && a == c)
                return false;
            if (resultKind.selection == RelationSelection::Any)
                return count >= 1;
            if (resultKind.selection == RelationSelection::All)
            {
                const auto local = AB.Source().GlobalToLocalPinned(a);
                DNDS_check_throw_info(local, "composition source row is unavailable");
                return count == int(sourceRows[size_t(*local)].size());
            }
            return predicate(a, c, count);
        };
        std::vector<std::vector<index>> rows(size_t(AB.NumOwned()));
        ConnectivityRows::Compose(AB.NumOwned(), [&](index row) -> const std::vector<index> &
                                  { return sourceRows[size_t(row)]; }, [&](index row) -> const std::vector<index> &
                                  { return intermediateRows[size_t(row)]; }, [&](index global)
                                  {
                auto row = BC.Source().GlobalToLocalPinned(global);
                DNDS_check_throw_info(row && *row < BC.Rows(), "composition intermediate row is unavailable; transfer it explicitly");
                return *row; }, [&](index row)
                                  { return AB.Source().LocalToGlobalPinned(row); }, selected, [&](index row, const std::vector<index> &entries)
                                  { rows[size_t(row)] = entries; }, matchExtra);
        return NativeRelation(resultKind, AB.Source(), BC.Target(),
                              BuildNativeRows(rows));
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
        int sourceOwnerLocal = cone.Source().NumOwned() ? mpi.rank : -1;
        int sourceOwner = -1;
        MPI_Allreduce(&sourceOwnerLocal, &sourceOwner, 1, MPI_INT, MPI_MAX, mpi.comm);
        int localOnly = sourceOwner < 0 ||
                        (mpi.rank == sourceOwner && cone.Source().NumOwned() == cone.Source().NumGlobal() &&
                         cone.Target().NumOwned() == cone.Target().NumGlobal()) ||
                        (mpi.rank != sourceOwner && cone.Source().NumOwned() == 0 && cone.Target().NumOwned() == 0);
        MPI_Allreduce(MPI_IN_PLACE, &localOnly, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (localOnly)
        {
            int validLocal = 1;
            std::vector<std::vector<index>> support(size_t(cone.Target().NumOwned()));
            try
            {
                cone.Validate();
                cone.CheckCommunicator(mpi.comm);
                for (index sourceRow = 0; sourceRow < cone.NumOwned(); ++sourceRow)
                {
                    const auto sourceGlobal = cone.Source().LocalToGlobalPinned(sourceRow);
                    for (size_t slot = 0; slot < cone.RowSize(sourceRow); ++slot)
                    {
                        const auto entry = cone.ReadEntry(sourceRow, slot);
                        if (entry.absent)
                            continue;
                        const auto targetRow = cone.Target().GlobalToLocalPinned(entry.global);
                        DNDS_check_throw_info(targetRow && size_t(*targetRow) < support.size(),
                                              "local inverse target row is unavailable");
                        support[size_t(*targetRow)].push_back(sourceGlobal);
                    }
                }
                for (auto &entries : support)
                {
                    std::sort(entries.begin(), entries.end());
                    entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
                }
            }
            catch (...)
            {
                validLocal = 0;
            }
            MPI_Allreduce(MPI_IN_PLACE, &validLocal, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(validLocal, "local inverse relation/communicator validation failed on a rank");
            return NativeRelation(cone.Kind().Inverted(), cone.Target(), cone.Source(), BuildNativeRows(support));
        }
        int valid = 1;
        std::vector<std::vector<index>> batches;
        try
        {
            cone.Validate();
            cone.CheckCommunicator(mpi.comm);
            // Packing can detect an unmanaged corrupted entry even when the
            // retained scope's construction scan is cached. Agree on those
            // errors before any rank enters the exchange collectives.
            batches.resize(size_t(mpi.size));
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
        std::vector<int> sendPresence(size_t(mpi.size)), receivePresence(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            sendPresence[size_t(peer)] = !batches[size_t(peer)].empty();
        }
        auto source = BuildNativeRows(batches);
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
        for (size_t row = 0; row < support.size(); ++row)
        {
            auto &entries = support[row];
            std::sort(entries.begin(), entries.end());
            entries.erase(std::unique(entries.begin(), entries.end()), entries.end());
        }
        return NativeRelation(cone.Kind().Inverted(), cone.Target(), cone.Source(),
                              BuildNativeRows(support));
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
