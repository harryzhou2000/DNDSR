#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Access/EntitySpace.hpp"

#include <algorithm>
#include <limits>

namespace DNDS::Geom
{
    namespace
    {
        // Native setup must agree on recoverable input errors before entering
        // the next mapping collective, including ranks with zero owned rows.
        void CheckNativeSpaceInput(const MPIInfo &mpi, EntityKind kind, index owned, bool valid)
        {
            DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL, "native entity space requires a communicator");
            int rank = -1, size = 0;
            MPI_Comm_rank(mpi.comm, &rank);
            MPI_Comm_size(mpi.comm, &size);
            int values[]{int(valid && mpi.rank == rank && mpi.size == size && owned >= 0 &&
                             int(kind) >= 0 && int(kind) < int(EntityKind::NUM_KINDS)),
                         int(kind), -int(kind)};
            MPI_Allreduce(MPI_IN_PLACE, values, 3, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(values[0] && values[1] == -values[2],
                                  "native entity-space inputs or entity kinds differ between ranks");
        }
    }

    EntitySpace EntitySpace::CreateOwned(EntityKind kind, MPIInfo mpi, index owned)
    {
        CheckNativeSpaceInput(mpi, kind, owned, true);
        auto global = std::make_shared<GlobalOffsetsMapping>();
        global->setMPIAlignBcast(mpi, owned);
        std::vector<index> empty;
        auto ghost = std::make_shared<OffsetAscendIndexMapping>((*global)(mpi.rank, 0), owned, empty, *global, mpi);
        Runtime::VersionTracker versions;
        EntitySpace result(kind, mpi, owned, 0, std::make_shared<const uint8_t>(0),
                           versions.Stamp(), std::move(global), std::move(ghost), versions);
        result._publishedMaps = true;
        return result;
    }

    EntitySpace EntitySpace::CreateSibling(EntityKind kind, index owned) const
    {
        CheckNativeSpaceInput(_mpi, kind, owned, Valid());
        auto result = CreateOwned(kind, _mpi, owned);
        result._identity = _identity;
        result._versions.RebindPrepared(_versions);
        result._stamp = _stamp;
        return result;
    }

    EntitySpace EntitySpace::WithGhosts(std::vector<index> globals) const
    {
        bool valid = Valid();
        if (valid)
            for (auto global : globals)
            {
                MPI_int owner = -1;
                index row = -1;
                valid = _global->search(global, owner, row) && owner != _mpi.rank && valid;
            }
        CheckNativeSpaceInput(_mpi, _kind, _owned, valid);
        auto ghost = std::make_shared<OffsetAscendIndexMapping>((*_global)(_mpi.rank, 0), _owned,
                                                                globals, *_global, _mpi);
        const auto count = index(ghost->ghostIndex.size());
        EntitySpace result(_kind, _mpi, _owned, count, _identity, _stamp,
                           _global, std::move(ghost), _versions);
        result._publishedMaps = true;
        return result;
    }

    void EntitySpace::CheckCommunicator(MPI_Comm actual, MPI_Comm expected)
    {
        DNDS_check_throw_info(actual != MPI_COMM_NULL && expected != MPI_COMM_NULL, "entity indexing needs a valid communicator");
        int relation = MPI_UNEQUAL;
        DNDS_check_throw_info(MPI_Comm_compare(actual, expected, &relation) == MPI_SUCCESS &&
                                  (relation == MPI_IDENT || relation == MPI_CONGRUENT),
                              "field communicator differs from entity-space rank ordering");
    }

    EntitySpace::EntitySpace(EntityKind kind, MPIInfo mpi, index owned, index ghost,
                             std::shared_ptr<const uint8_t> identity, Runtime::VersionStamp stamp,
                             std::shared_ptr<GlobalOffsetsMapping> global,
                             std::shared_ptr<OffsetAscendIndexMapping> ghostMap, Runtime::VersionTracker versions)
        : _kind(kind), _mpi(mpi), _owned(owned), _ghost(ghost), _identity(std::move(identity)),
          _stamp(std::move(stamp)), _versions(std::move(versions)), _global(std::move(global)), _ghostMap(std::move(ghostMap))
    {
        Validate();
        DNDS_check_throw_info(owned >= 0 && ghost >= 0 && ghost <= std::numeric_limits<index>::max() - owned,
                              "entity-space row count overflow");
        DNDS_check_throw_info(mpi.comm != MPI_COMM_NULL && mpi.size > 0 && mpi.rank >= 0 && mpi.rank < mpi.size,
                              "entity-space MPI information is invalid");
        int actualRank = -1, actualSize = -1;
        MPI_Comm_rank(mpi.comm, &actualRank);
        MPI_Comm_size(mpi.comm, &actualSize);
        DNDS_check_throw_info(actualRank == mpi.rank && actualSize == mpi.size, "entity-space MPI information differs from communicator");
        const auto &lengths = _global->RLengths();
        const auto &offsets = _global->ROffsets();
        DNDS_check_throw_info(lengths.size() == size_t(mpi.size) && offsets.size() == size_t(mpi.size) + 1 && offsets.front() == 0,
                              "entity-space global map has the wrong rank extent");
        for (int r = 0; r < mpi.size; ++r)
            DNDS_check_throw_info(lengths[r] >= 0 && offsets[r] >= 0 && offsets[r + 1] >= offsets[r] &&
                                      offsets[r + 1] - offsets[r] == lengths[r],
                                  "entity-space global map has inconsistent offsets");
        DNDS_check_throw_info(lengths[mpi.rank] == owned, "entity-space owned rows differ from global map");
        _globalCount = _global->globalSize();
        DNDS_check_throw_info(_ghostMap->ghostIndex.size() == size_t(ghost) &&
                                  _ghostMap->ghostStart.size() == size_t(mpi.size) + 1 &&
                                  _ghostMap->ghostSizes.size() == size_t(mpi.size) &&
                                  _ghostMap->ghostStart.front() == 0 && _ghostMap->ghostStart.back() == ghost,
                              "entity-space ghost map has inconsistent extents");
        DNDS_check_throw_info(std::is_sorted(_ghostMap->ghostIndex.begin(), _ghostMap->ghostIndex.end()) &&
                                  std::adjacent_find(_ghostMap->ghostIndex.begin(), _ghostMap->ghostIndex.end()) == _ghostMap->ghostIndex.end(),
                              "entity-space ghost numbering is not ordered");
        if (owned)
        {
            index first = -1, last = -1;
            DNDS_check_throw_info(_ghostMap->searchInMain(offsets[mpi.rank], first) && first == 0 &&
                                      _ghostMap->searchInMain(offsets[mpi.rank + 1] - 1, last) && last == owned - 1,
                                  "entity-space ghost map has a different owned block");
        }
        DNDS_check_throw_info(_ghostMap->pushIndexSizes.size() == size_t(mpi.size) &&
                                  _ghostMap->pushIndexStarts.size() == size_t(mpi.size) + 1 &&
                                  _ghostMap->pushIndexStarts.front() == 0 &&
                                  _ghostMap->pushIndexStarts.back() >= 0 &&
                                  size_t(_ghostMap->pushIndexStarts.back()) == _ghostMap->pushingIndexGlobal.size(),
                              "entity-space ghost map has inconsistent push extents");
        for (int r = 0; r < mpi.size; ++r)
        {
            auto first = _ghostMap->ghostStart[r], end = _ghostMap->ghostStart[r + 1];
            DNDS_check_throw_info(first >= 0 && end >= first && end <= ghost && _ghostMap->ghostSizes[r] == end - first,
                                  "entity-space ghost map has inconsistent peer ranges");
            for (index j = first; j < end; ++j)
            {
                MPI_int owner = -1;
                index local = -1;
                DNDS_check_throw_info(_global->search(_ghostMap->ghostIndex[j], owner, local) && owner == r,
                                      "entity-space ghost row has the wrong owner");
            }
            auto pushFirst = _ghostMap->pushIndexStarts[r], pushEnd = _ghostMap->pushIndexStarts[r + 1];
            DNDS_check_throw_info(pushFirst >= 0 && pushEnd >= pushFirst &&
                                      size_t(pushEnd) <= _ghostMap->pushingIndexGlobal.size() &&
                                      _ghostMap->pushIndexSizes[r] == pushEnd - pushFirst,
                                  "entity-space ghost map has inconsistent push ranges");
            for (index j = pushFirst; j < pushEnd; ++j)
                DNDS_check_throw_info(_ghostMap->pushingIndexGlobal[j] >= offsets[mpi.rank] &&
                                          _ghostMap->pushingIndexGlobal[j] < offsets[mpi.rank + 1],
                                      "entity-space pushed row is not locally owned");
        }
    }

    index EntitySpace::LocalToGlobal(index local) const
    {
        Validate();
        return LocalToGlobalPinned(local);
    }

    index EntitySpace::LocalToGlobalPinned(index local) const
    {
        DNDS_check_throw_info(local >= 0 && local < NumLocal(), "local entity index outside entity space");
        // Constructor validated this immutable map and rank; the retained
        // numbering lease pins it. Only the queried local bound is dynamic.
        return local < _owned ? _global->ROffsets()[size_t(_mpi.rank)] + local : _ghostMap->ghostIndex[local - _owned];
    }

    std::optional<index> EntitySpace::GlobalToLocal(index global) const
    {
        Validate();
        return GlobalToLocalPinned(global);
    }

    std::optional<index> EntitySpace::GlobalToLocalPinned(index global) const
    {
        if (global < 0 || global >= _globalCount)
            return std::nullopt;
        const auto start = _global->ROffsets()[size_t(_mpi.rank)];
        if (global >= start && global - start < _owned)
            return global - start;
        auto found = std::lower_bound(_ghostMap->ghostIndex.begin(), _ghostMap->ghostIndex.end(), global);
        if (found == _ghostMap->ghostIndex.end() || *found != global)
            return std::nullopt;
        return _owned + index(found - _ghostMap->ghostIndex.begin());
    }

    MPI_int EntitySpace::Owner(index global) const
    {
        Validate();
        MPI_int rank = -1;
        index local = -1;
        DNDS_check_throw_info(_global->search(global, rank, local), "global entity index outside entity space");
        return rank;
    }

    index EntitySpace::OwnerLocalRow(index global) const
    {
        Validate();
        MPI_int rank = -1;
        index local = -1;
        DNDS_check_throw_info(_global->search(global, rank, local), "global entity index outside entity space");
        return local;
    }

    Runtime::RowTransferPlan EntitySpace::BuildRowTransferPlan() const
    {
        // All ranks agree before route construction can enter its collectives.
        int valid = Valid();
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _mpi.comm);
        DNDS_check_throw_info(valid, "cannot build a row route from a stale entity space on a rank");
        int kindRange[]{int(_kind), -int(_kind)};
        MPI_Allreduce(MPI_IN_PLACE, kindRange, 2, MPI_INT, MPI_MIN, _mpi.comm);
        DNDS_check_throw_info(kindRange[0] == -kindRange[1], "row route entity kind differs between ranks");
        std::vector<Runtime::RowRequest> requests;
        requests.reserve(size_t(_ghost));
        for (index row = 0; row < _ghost; ++row)
        {
            MPI_int rank = -1;
            index local = -1;
            // Validated global IDs and ownership come from the retained map.
            const bool found = _global->search(_ghostMap->ghostIndex[row], rank, local);
            DNDS_check_throw_info(found, "entity-space ghost global index is invalid");
            requests.push_back({rank, local, size_t(row)});
        }
        return Runtime::RowTransferPlan(_mpi.comm, size_t(_owned), size_t(_ghost), requests, {_stamp});
    }

    EntitySpace UnstructuredMesh::Entities(EntityKind kind) const
    {
        DNDS_check_throw_info(_nativeState, "entity numbering requires native mesh authority");
        return _nativeState->Entities(kind);
    }
}
