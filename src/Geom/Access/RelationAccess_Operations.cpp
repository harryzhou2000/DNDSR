#include "Geom/Access/RelationAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
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
        const auto *mesh = dynamic_cast<const MeshAdjacencyRead::MeshHold *>(source._owned.get());
        auto convert = [&](index first, const MeshAdjacencyRead::Hold *hold, bool ghost)
        {
            // Empty padded arrays legitimately have a null row-size pointer.
            // Their retained storage, rather than pointer presence, determines
            // the physical family. This also shares immutable row metadata.
            const auto *array = dynamic_cast<const MeshAdjacencyRead::ArrayHold *>(hold);
            const auto *layout = array  ? &array->read.Layout()
                                 : mesh ? (ghost ? &mesh->ghostLayout : &mesh->ownedLayout)
                                        : nullptr;
            DNDS_check_throw_info(layout, "adjacency conversion requires retained storage metadata");
            return Runtime::BuildArrayRows<index>(*layout, [&](index iRow, auto output)
                                                  { source.ConvertRowTo(first + iRow, targetState, output.data(), allowUnavailable); }, true);
        };
        MeshAdjacency result(source.Kind(), source.Source(), source.Target(), convert(0, source._owned.get(), false), targetState);
        // A mesh scope retains both portions in one MeshHold; it need not have
        // a separate ghost holder even though ghost rows are present.
        if (source._ghost || mesh)
            result._ghost.emplace(convert(source.NumOwned(), source._ghost.get(), true));
        auto check = result.Read();
        return result;
    }

    MeshAdjacency MeshAdjacency::WithGhostRows(const EntityGhostMapping &halo, Runtime::TransferLayout layout) const
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
            MPI::AbortIfSolverFailure("native connectivity preparation failed");
            valid = 0;
        }
        _source.Preflight(valid, "native halo route has stale/wrong source order or nonglobal relation payload");
        DNDS_check_throw_info(valid, "native halo route has stale/wrong source order or nonglobal relation payload");
        Runtime::PayloadPlan payload(halo._route, _owned.Layout(), Runtime::ElementSchema::Arithmetic<index>(), layout);
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
