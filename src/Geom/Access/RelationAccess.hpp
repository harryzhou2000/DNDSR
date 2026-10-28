#pragma once

#include "Geom/Access/EntitySpace.hpp"
#include "Geom/Access/MeshAccess.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <optional>

namespace DNDS::Geom
{
    template <class Pair>
    struct AdjPairTracked;
    /// Retained host setup scope. Values and row metadata share their original
    /// controllers; source/target version leases pin the numbering. Global
    /// sparse relations need not have local target closure.
    class RelationReadAccess
    {
        friend struct MeshConnectivity;
        friend class NativeRelation;
        friend class NativeConnectivity;
        friend class RelationPeriodicBits;
        EntitySpace _source, _target;
        Runtime::VersionTracker::ReadLease _sourceLease, _targetLease;
        struct Hold
        {
            virtual ~Hold() = default;
            virtual void ValidateScope() const {}
        };
        struct MeshHold final : Hold
        {
            std::shared_ptr<MeshReadAccess> read;
            explicit MeshHold(std::shared_ptr<MeshReadAccess> owner) : read(std::move(owner)) {}
            void ValidateScope() const override
            {
                // Construction acquired the host view and pinned its epoch.
                // Entry bounds still validate every query; checking a retained
                // scope must not copy the entire mesh view or relock its epoch.
                DNDS_check_throw_info(read && read->_active && !read->_completed, "mesh read access was moved or sealed");
            }
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
        // Set only by construction, before this scope escapes. Retained Array
        // and entity-order leases keep every validated entry immutable until
        // release; association/stamp checks still run on later validations.
        mutable bool _contentsValidated = false;

        void CheckActive() const
        {
            DNDS_check_throw_info(_active, "relation read access was moved");
            if (_owned)
                _owned->ValidateScope();
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

        /// Legacy adapter definition is provided by RelationAccessLegacy.hpp.
        template <class Pair>
        RelationReadAccess(AdjKind kind, const AdjPairTracked<Pair> &pair, EntitySpace source, EntitySpace target);

    private:
        /// Used only with a scope acquired from the same mesh by Current/RegisterMesh.
        RelationReadAccess(AdjKind kind, std::shared_ptr<MeshReadAccess> meshRead, EntitySpace source, EntitySpace target);

    public:
        static RelationReadAccess Current(UnstructuredMesh &mesh, AdjKind kind, MeshReadRequirements requirements);
        RelationReadAccess(const RelationReadAccess &) = delete;
        RelationReadAccess &operator=(const RelationReadAccess &) = delete;
        RelationReadAccess(RelationReadAccess &&other) noexcept
            : _source(std::move(other._source)), _target(std::move(other._target)),
              _sourceLease(std::move(other._sourceLease)), _targetLease(std::move(other._targetLease)),
              _owned(std::move(other._owned)), _ghost(std::move(other._ghost)),
              _view(other._view), _kind(other._kind), _state(other._state), _active(other._active),
              _contentsValidated(other._contentsValidated)
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
        /// Setup-only conversion against this exact target order. Global means
        /// current distributed rank-offset numbering, never a persistent ID.
        /// Missing targets encode as -1-global in local state; UnInitIndex
        /// remains an absent incidence. Strict local conversion rejects missing
        /// closure. No mapping/storage mutation or communication is performed.
        index ConvertedIndex(index row, size_t slot, MeshAdjState targetState,
                             bool allowUnavailable = true) const;
        std::vector<index> GlobalRow(index row) const
        {
            std::vector<index> result;
            result.reserve(RowSize(row));
            for (size_t slot = 0; slot < RowSize(row); ++slot)
            {
                result.push_back(ConvertedIndex(row, slot, Adj_PointToGlobal));
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
        /// Transfer an ordered companion with the same route. The caller names
        /// its source entity order and explicit wire schema; no map escapes.
        template <class T>
        Runtime::Array<T> TransferRows(const EntitySpace &source, const Runtime::Array<T> &owned,
                                       const Runtime::ElementSchema &schema) const
        {
            int valid = Valid() && _source.SameOrder(source) && owned.Layout().Rows() == size_t(_source.NumOwned());
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _route.Communicator());
            DNDS_check_throw_info(valid, "companion halo route has stale/wrong source order or row extent");
            Runtime::Array<T> ghost(_route.DestinationLayout(owned.Layout()));
            Runtime::TransferOptions options;
            options.mode = Runtime::TransferMode::Transient;
            options.context = std::make_shared<Runtime::CommunicationContext>(_route.Communicator());
            Runtime::PayloadPlan payload(_route, owned.Layout(), ghost.Layout(), schema);
            payload.Bind(owned, ghost, Runtime::TransferTransport::Direct, options).Begin().Wait();
            return ghost;
        }
    };

    /// Native CSR relation produced by setup mathematics. Owned output is
    /// published first; explicit WithGhostRows transfers missing source rows.
    class NativeRelation
    {
        friend class NativeConnectivity;
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
        const std::optional<Runtime::Array<index>> &Ghost() const { return _ghost; }
        AdjKind Kind() const
        {
            _source.Validate();
            _target.Validate();
            return _kind;
        }
        const EntitySpace &Source() const { return _source; }
        const EntitySpace &Target() const { return _target; }
        RelationReadAccess Read() const { return {_kind, _source, _target, _state, _owned, _ghost ? &*_ghost : nullptr}; }
        /// Detached host conversion of every owned and present ghost row.
        /// Retains source/target identity, slot order and logical row shape.
        /// Both directions validate explicit state; even same-state conversion
        /// returns independent values. Live compute mesh publications stay local.
        /// No MPI is issued; wrap in the caller's collective failure boundary
        /// before a subsequent collective setup operation.
        static NativeRelation Convert(const RelationReadAccess &source, MeshAdjState targetState,
                                      bool allowUnavailable = true);
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
