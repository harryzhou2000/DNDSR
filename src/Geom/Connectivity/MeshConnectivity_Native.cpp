#include "Geom/Connectivity/AdjacencyRegistry.hpp"
#include "Geom/Connectivity/ConnectivityComposeRows.hpp"
#include <algorithm>
#include <limits>
#include <type_traits>

namespace DNDS::Geom
{
    void MeshAdjacencyRead::Validate() const
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
        DNDS_check_throw_info(_view.owned.RowCount() == size_t(_source.NumOwned()) &&
                                  (_view.ghost.RowCount() == 0 || _view.ghost.RowCount() == size_t(_source.NumGhost())),
                              "relation owned/ghost rows differ from its source space");
        DNDS_check_throw_info(!_kind.removeSelf || _source.SameOrder(_target),
                              "removeSelf relation requires identical source/target entity order");
        if (_contentsValidated)
            return;
        const bool sorted = ContentChecks(_kind) & 1U;
        const bool removeSelf = _kind.removeSelf;
        const bool needsGlobal = sorted || removeSelf;
        const bool globalState = _state == Adj_PointToGlobal;
        const auto targetGlobalCount = _target.NumGlobal(), targetLocalCount = _target.NumLocal();
        const auto sourceOwned = _source.NumOwned();
        const auto sourceFirst = removeSelf && sourceOwned ? _source.LocalToGlobalPinned(0) : 0;
        unsigned errors = 0;
        const auto scan = [&](const Runtime::ArrayView<const index> &view, auto ownedRows)
        {
            // The scope and both mappings are pinned above. Internal validation
            // can traverse raw rows without re-entering public access checks
            // for every incidence. Public random access remains checked.
            for (index iRow = 0; iRow < view.RowCount(); ++iRow)
            {
                index previous = -1;
                // Owned rows occupy the validated contiguous global interval.
                // Ghost rows still use their exact retained numbering map.
                const auto sourceGlobal = !removeSelf                  ? index(-1)
                                          : decltype(ownedRows)::value ? sourceFirst + iRow
                                                                       : _source.LocalToGlobalPinned(sourceOwned + iRow);
                const auto count = view.RowSize(iRow);
                const auto *values = count ? view.Row(iRow) : nullptr;
                for (index iSlot = 0; iSlot < count; ++iSlot)
                {
                    const auto value = values[iSlot];
                    const bool absent = value == UnInitIndex;
                    index global = value;
                    if (!absent)
                    {
                        if (globalState)
                            errors |= unsigned(value < 0 || value >= targetGlobalCount);
                        else if (value >= 0)
                        {
                            DNDS_check_throw_info(value < targetLocalCount, "relation local target is outside its entity space");
                            if (needsGlobal)
                                global = _target.LocalToGlobalPinned(value);
                        }
                        else
                        {
                            global = -1 - value;
                            DNDS_check_throw_info(global >= 0 && global < targetGlobalCount,
                                                  "relation global target is outside its entity space");
                            DNDS_check_throw_info(!_target.GlobalToLocalPinned(global),
                                                  "encoded unavailable target is present in the supplied local target mapping");
                        }
                    }
                    if (sorted)
                    {
                        // Ordering is in global IDs, including local encodings
                        // whose appended ghost globals precede owned globals.
                        errors |= unsigned(absent || global <= previous) << 1;
                        previous = global;
                    }
                    errors |= unsigned(removeSelf && !absent && global == sourceGlobal) << 2;
                }
            }
        };
        scan(_view.owned, std::true_type{});
        scan(_view.ghost, std::false_type{});
        // No indexed target lookup uses unchecked global input. Deferring these
        // read-only content errors lets the global scan remain a plain loop;
        // local lookup bounds/encoding checks above still precede each lookup.
        DNDS_check_throw_info(!(errors & 1U), "relation global target is outside its entity space");
        DNDS_check_throw_info(!(errors & 2U), "SortedUnique relation has absent, duplicate or unordered targets");
        DNDS_check_throw_info(!(errors & 4U), "removeSelf relation contains its source entity");
        _contentsValidated = true;
    }

    std::vector<MeshAdjacencyRead> MeshAdjacencyRead::AcquireAliases(const std::vector<AdjacencyDescriptor> &kinds,
                                                                     EntitySpace source, EntitySpace target, MeshAdjState state,
                                                                     const Runtime::Array<index> &owned,
                                                                     const Runtime::Array<index> *ghost)
    {
        std::vector<MeshAdjacencyRead> result;
        if (kinds.empty())
            return result;
        // Prefer the strictest available content contract before admitting any
        // weaker alias. Sorted-only and self-only are incomparable: a second
        // full scan still proves whichever constraint the basis did not check.
        index basis = 0;
        for (index iKind = 1; iKind < index(kinds.size()); ++iKind)
            if (ContentChecks(kinds[iKind]) > ContentChecks(kinds[basis]))
                basis = iKind;
        result.reserve(kinds.size());
        auto validated = MeshAdjacencyRead(kinds[basis], source, target, state, owned, ghost);
        unsigned proved = ContentChecks(kinds[basis]);
        for (index iKind = 0; iKind < index(kinds.size()); ++iKind)
        {
            if (iKind == basis)
                result.push_back(std::move(validated));
            else
            {
                const auto required = ContentChecks(kinds[iKind]);
                result.push_back(MeshAdjacencyRead(kinds[iKind], source, target, state, owned, ghost,
                                                   (required & proved) == required));
                proved |= required;
            }
        }
        // The basis scope remains alive throughout construction, either in
        // validated or result; each result has its own acquired value leases.
        return result;
    }

    MeshAdjacencyRead::Entry MeshAdjacencyRead::ReadEntry(index row, size_t slot) const
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

    index MeshAdjacencyRead::ConvertedIndex(index row, size_t slot, MeshAdjState targetState,
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

    void MeshAdjacencyRead::ConvertRowTo(index row, MeshAdjState targetState, index *output,
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

    MeshAdjacency MeshAdjacency::Convert(const MeshAdjacencyRead &source, MeshAdjState targetState,
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
                lengths.resize(view.RowCount());
                for (size_t row = 0; row < view.RowCount(); ++row)
                    lengths[row] = view.RowSize(row);
            }
            const auto shape = view.offsets ? Runtime::RowLayout::Ragged(lengths) : view.sizes ? Runtime::RowLayout::Padded(lengths, view.stride)
                                                                                               : Runtime::RowLayout::Dense(view.RowCount(), view.width, view.stride);
            Runtime::Array<index> output(shape);
            {
                auto write = output.Write();
                for (size_t row = 0; row < view.RowCount(); ++row)
                    source.ConvertRowTo(first + index(row), targetState, write.View().Row(row), allowUnavailable);
            }
            return output;
        };
        MeshAdjacency result(source.Kind(), source.Source(), source.Target(), convert(source._view.owned, 0), targetState);
        // A mesh scope retains both portions in one MeshHold; it need not have
        // a separate ghost holder even though ghost rows are present.
        if (source._ghost || source._view.ghost.RowCount())
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

    MeshAdjacency AdjacencyRegistry::ComposeFiltered(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC,
                                                     const AdjacencyDescriptor &resultKind,
                                                     const std::function<bool(index, index, int)> &predicate,
                                                     const std::function<bool(index, index, const std::vector<index> &)> &matchExtra)
    {
        AB.Validate();
        BC.Validate();
        DNDS_check_throw_info(AB.Target().SameOrder(BC.Source()), "composition intermediate entity/map order mismatch");
        DNDS_check_throw_info(bool(predicate), "composition predicate is missing");
        resultKind.Validate();
        DNDS_check_throw_info(!resultKind.IsLegacy() && resultKind == AdjacencyDescriptor::Composition(AB.Kind(), BC.Kind(),
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
        std::vector<std::vector<index>> sourceRows(size_t(AB.NumOwned())), intermediateRows(size_t(BC.RowCount()));
        for (index row = 0; row < AB.NumOwned(); ++row)
            sourceRows[size_t(row)] = unique(AB.GlobalRow(row));
        for (index row = 0; row < BC.RowCount(); ++row)
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
                DNDS_check_throw_info(row && *row < BC.RowCount(), "composition intermediate row is unavailable; transfer it explicitly");
                return *row; }, [&](index row)
                                  { return AB.Source().LocalToGlobalPinned(row); }, selected, [&](index row, const std::vector<index> &entries)
                                  { rows[size_t(row)] = entries; }, matchExtra);
        return MeshAdjacency(resultKind, AB.Source(), BC.Target(),
                             BuildNativeRows(rows));
    }

    MeshAdjacency AdjacencyRegistry::Compose(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf)
    {
        DNDS_check_throw_info(!removeSelf || AB.Source().SameOrder(BC.Target()),
                              "removeSelf requires identical source and final target spaces");
        return ComposeFiltered(AB, BC, AdjacencyDescriptor::Composition(AB.Kind(), BC.Kind(), RelationSelection::Any, removeSelf),
                               SharedCountPredicate{1, removeSelf});
    }

    MeshAdjacency AdjacencyRegistry::ComposeAll(const MeshAdjacencyRead &AB, const MeshAdjacencyRead &BC, bool removeSelf)
    {
        return ComposeFiltered(AB, BC, AdjacencyDescriptor::Composition(AB.Kind(), BC.Kind(), RelationSelection::All, removeSelf),
                               SharedCountPredicate{1, removeSelf});
    }

    MeshAdjacency AdjacencyRegistry::Inverse(const MeshAdjacencyRead &cone, const MPIInfo &mpi)
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
            return MeshAdjacency(cone.Kind().Inverted(), cone.Target(), cone.Source(), BuildNativeRows(support));
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
                const auto sourceGlobal = cone.Source().LocalToGlobalPinned(source);
                for (size_t slot = 0; slot < cone.RowSize(source); ++slot)
                {
                    const auto entry = cone.ReadEntry(source, slot);
                    if (entry.absent)
                        continue;
                    // The retained adjacency scope pins both immutable maps.
                    // Find owner and row together; public descriptor queries
                    // would each repeat the epoch lock and binary search.
                    MPI_int owner = -1;
                    index ownerRow = -1;
                    DNDS_check_throw_info(cone.Target()._global->search(entry.global, owner, ownerRow),
                                          "inverse target is outside its entity space");
                    batches[size_t(owner)].push_back(ownerRow);
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
        std::vector<std::vector<index>> send(size_t(mpi.size)), receive(size_t(mpi.size));
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            if (sendPresence[size_t(peer)])
                send[size_t(peer)].push_back(index(peer));
            if (receivePresence[size_t(peer)])
                receive[size_t(peer)].push_back(index(peer));
        }
        auto context = std::make_shared<Runtime::CommunicationContext>(mpi.comm);
        auto route = Runtime::RowTransferPlan::Known(context, size_t(mpi.size), size_t(mpi.size),
                                                     std::move(send), std::move(receive),
                                                     {cone.Source().Stamp(), cone.Target().Stamp()});
        Runtime::PayloadPlan payload(route, source.Layout(), Runtime::ElementSchema::Arithmetic<index>());
        Runtime::Array<index> incoming(payload.DestinationLayout());
        Runtime::TransferOptions options;
        options.mode = Runtime::TransferMode::Transient;
        options.context = context;
        payload.Bind(source, incoming, Runtime::TransferTransport::Direct, options).Begin().Wait();

        std::vector<std::vector<index>> support(size_t(cone.Target().NumOwned()));
        auto values = incoming.Read();
        auto view = values.View();
        valid = 1;
        for (int peer = 0; peer < mpi.size; ++peer)
        {
            const auto length = view.RowSize(size_t(peer));
            valid = valid && length % 2 == 0;
            const auto &sourceOffsets = cone.Source()._global->ROffsets();
            for (size_t slot = 0; slot + 1 < length; slot += 2)
            {
                const auto targetRow = view(size_t(peer), slot), sourceGlobal = view(size_t(peer), slot + 1);
                const bool entryValid = targetRow >= 0 && size_t(targetRow) < support.size() &&
                                        sourceGlobal >= sourceOffsets[size_t(peer)] &&
                                        sourceGlobal < sourceOffsets[size_t(peer + 1)];
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
        return MeshAdjacency(cone.Kind().Inverted(), cone.Target(), cone.Source(),
                             BuildNativeRows(support));
    }

    MeshAdjacency MeshAdjacency::WithGhostRows(const EntityGhostMapping &halo) const
    {
        const auto mpi = MeshAdjacencyRead::SourceMPI(_source);
        std::optional<MeshAdjacencyRead> read;
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
        Runtime::PayloadPlan payload(halo._route, _owned.Layout(), Runtime::ElementSchema::Arithmetic<index>());
        Runtime::Array<index> ghosts(payload.DestinationLayout());
        Runtime::TransferOptions options;
        options.mode = Runtime::TransferMode::Transient;
        options.context = halo._route.Context();
        payload.Bind(_owned, ghosts, Runtime::TransferTransport::Direct, options).Begin().Wait();
        MeshAdjacency result(_kind, _source, _target, _owned, _state);
        result._ghost.emplace(std::move(ghosts));
        auto validated = result.Read();
        return result;
    }

}
