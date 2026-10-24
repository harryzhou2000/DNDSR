#pragma once

#include "EntitySpace.hpp"
#include "DNDS/Runtime/LegacyArrayBridge.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <optional>

namespace DNDS::Geom
{
    /// Retained host setup scope. Values and row metadata share their original
    /// controllers; source/target version leases pin the numbering. Global
    /// sparse relations need not have local target closure.
    class RelationReadAccess
    {
        friend struct MeshConnectivity;
        friend class NativeRelation;
        EntitySpace _source, _target;
        Runtime::VersionTracker::ReadLease _sourceLease, _targetLease;
        struct Hold
        {
            virtual ~Hold() = default;
        };
        struct ArrayHold final : Hold
        {
            Runtime::Array<index> array;
            Runtime::Array<index>::ReadLease read;
            explicit ArrayHold(Runtime::Array<index> values)
                : array(std::move(values)), read(array.Read()) {}
        };
        std::unique_ptr<Hold> _owned, _ghost;
        Runtime::PairReadView<index> _view;
        AdjKind _kind;
        MeshAdjState _state;
        bool _active = true;

        void CheckActive() const
        {
            DNDS_check_throw_info(_active, "relation read access was moved");
        }

        Runtime::ArrayView<const index> Capture(Runtime::Array<index> array, std::unique_ptr<Hold> &destination)
        {
            DNDS_check_throw_info(array.GetDevice() == Runtime::Device::Host(), "relation setup requires explicit host placement");
            auto hold = std::make_unique<ArrayHold>(std::move(array));
            auto view = hold->read.View();
            destination = std::move(hold);
            return view;
        }
        void Validate() const;
        const ssp<GlobalOffsetsMapping> &TargetMapping() const { return _target._global; }
        const MPIInfo &MPI() const { return _source._mpi; }
        static MPIInfo SourceMPI(const EntitySpace &source) { return source._mpi; }
        void CheckCommunicator(MPI_Comm communicator) const { EntitySpace::CheckCommunicator(_source._mpi.comm, communicator); }

    public:
        struct Entry
        {
            bool absent = false;
            bool unavailable = false;
            index global = UnInitIndex;
        };
        RelationReadAccess(AdjKind kind, EntitySpace source, EntitySpace target,
                           MeshAdjState state, const Runtime::Array<index> &owned,
                           const Runtime::Array<index> *ghost = nullptr)
            : _source(std::move(source)), _target(std::move(target)),
              _sourceLease(_source._versions.Read()), _targetLease(_target._versions.Read()),
              _kind(kind), _state(state)
        {
            _view.owned = Capture(owned, _owned);
            if (ghost)
                _view.ghost = Capture(*ghost, _ghost);
            Validate();
        }

        template <class Pair>
        RelationReadAccess(AdjKind kind, const AdjPairTracked<Pair> &pair, EntitySpace source, EntitySpace target)
            : _source(std::move(source)), _target(std::move(target)),
              _sourceLease(_source._versions.Read()), _targetLease(_target._versions.Read()),
              _kind(kind), _state(pair.idx.state())
        {
            DNDS_check_throw_info(pair.father && pair.son, "legacy relation arrays are missing");
            DNDS_check_throw_info(pair.father->pLGlobalMapping == _source._global &&
                                      (pair.son->Size() == 0 || pair.trans.pLGhostMapping == _source._ghostMap),
                                  "legacy relation row mapping differs from its source space");
            DNDS_check_throw_info(_state != Adj_PointToLocal || pair.idx.mapping() == _target._ghostMap,
                                  "legacy relation target mapping differs from its target space");
            _view.owned = Capture(Runtime::ShareLegacyHost(*pair.father), _owned);
            _view.ghost = Capture(Runtime::ShareLegacyHost(*pair.son), _ghost);
            Validate();
        }
        RelationReadAccess(const RelationReadAccess &) = delete;
        RelationReadAccess &operator=(const RelationReadAccess &) = delete;
        RelationReadAccess(RelationReadAccess &&other) noexcept
            : _source(std::move(other._source)), _target(std::move(other._target)),
              _sourceLease(std::move(other._sourceLease)), _targetLease(std::move(other._targetLease)),
              _owned(std::move(other._owned)), _ghost(std::move(other._ghost)),
              _view(other._view), _kind(other._kind), _state(other._state), _active(other._active)
        {
            other._active = false;
            other._view = {};
        }
        RelationReadAccess &operator=(RelationReadAccess &&) = delete;
        const EntitySpace &Source() const
        {
            CheckActive();
            return _source;
        }
        const EntitySpace &Target() const
        {
            CheckActive();
            return _target;
        }
        AdjKind Kind() const
        {
            CheckActive();
            return _kind;
        }
        MeshAdjState State() const
        {
            CheckActive();
            return _state;
        }
        index NumOwned() const
        {
            CheckActive();
            return index(_view.owned.Rows());
        }
        index Rows() const
        {
            CheckActive();
            return index(_view.owned.Rows() + _view.ghost.Rows());
        }
        size_t RowSize(index row) const
        {
            DNDS_check_throw_info(row >= 0 && row < Rows(), "relation row is not present locally");
            return _view.RowSize(size_t(row));
        }
        index Raw(index row, size_t slot) const
        {
            DNDS_check_throw_info(slot < RowSize(row), "relation slot is out of range");
            return _view(size_t(row), slot);
        }
        Entry ReadEntry(index row, size_t slot) const;
        std::vector<index> GlobalRow(index row) const
        {
            std::vector<index> result;
            result.reserve(RowSize(row));
            for (size_t slot = 0; slot < RowSize(row); ++slot)
            {
                auto entry = ReadEntry(row, slot);
                result.push_back(entry.absent ? UnInitIndex : entry.global);
            }
            return result;
        }
    };

    /// Typed collectively created halo route. Copies share its native route;
    /// the retained entity identity/order prevents same-sized route confusion.
    class RelationGhostRoute
    {
        friend class NativeRelation;
        EntitySpace _source;
        Runtime::RowTransferPlan _route;

    public:
        explicit RelationGhostRoute(EntitySpace source)
            : _source(std::move(source)), _route(_source.BuildRowTransferPlan()) {}
        const EntitySpace &Source() const { return _source; }
        bool Valid() const { return _source.Valid() && _route.Valid(); }
    };

    /// Native CSR relation produced by setup mathematics. Owned output is
    /// published first; explicit WithGhostRows transfers missing source rows.
    class NativeRelation
    {
        AdjKind _kind;
        EntitySpace _source, _target;
        Runtime::Array<index> _owned;
        std::optional<Runtime::Array<index>> _ghost;
        MeshAdjState _state;

    public:
        NativeRelation(AdjKind kind, EntitySpace source, EntitySpace target,
                       Runtime::Array<index> owned, MeshAdjState state = Adj_PointToGlobal)
            : _kind(kind), _source(std::move(source)), _target(std::move(target)),
              _owned(std::move(owned)), _state(state)
        {
            auto check = Read();
        }
        const Runtime::Array<index> &Owned() const { return _owned; }
        const EntitySpace &Source() const { return _source; }
        const EntitySpace &Target() const { return _target; }
        RelationReadAccess Read() const { return {_kind, _source, _target, _state, _owned, _ghost ? &*_ghost : nullptr}; }
        /// Collective transient direct transfer using a route explicitly bound
        /// to this source identity/order. The original owned relation is retained.
        /// Local-index payloads require target-map translation across ranks;
        /// publish global entries before invoking this setup transfer.
        NativeRelation WithGhostRows(const RelationGhostRoute &route) const;
        RelationReadAccess Read(const EntitySpace &source, const EntitySpace &target) const
        {
            DNDS_check_throw_info(_source.SameOrder(source) && _target.SameOrder(target),
                                  "native relation does not belong to the requested entity orders");
            return Read();
        }
    };
}
