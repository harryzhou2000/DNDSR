#pragma once

#include "DNDS/Algebra/DofBundle.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"

namespace DNDS::Algebra
{
    /// Reserved CPU DoF payloads with explicit, deterministic slot selection.
    /// Each slot owns full owned/ghost fields. Prepare all space-specific halo
    /// bindings collectively before borrowing; borrow/return performs no MPI.
    /// Reconstruction scratch is a separate resource family.
    /// Controller-serialized CPU facility: no concurrent Reserve/Prepare/Acquire
    /// or return, and no asynchronous return before numerical/transfer joins.
    class DofScratchPool
    {
    public:
        enum class Use
        {
            OwnedOnly,
            Communicating
        };
        enum class Initialize
        {
            Dirty,
            ZeroOwned
        };
        using HaloBinder = std::function<Runtime::TransferBinding(DofField<real> &)>;

    private:
        struct Slot
        {
            DofBundle fields;
            bool busy = false;
            explicit Slot(DofBundle values) : fields(std::move(values)) {}
        };
        struct Prepared
        {
            index slot;
            DofBundle fields;
            std::vector<Runtime::TransferBinding> halos;
        };
        struct State
        {
            std::vector<Slot> slots;
            std::vector<Prepared> prepared;
            index active = 0, peak = 0;
            size_t bytes = 0;
        };
        std::shared_ptr<State> _state = std::make_shared<State>();
        static void CheckIdle(const DofBundle &fields)
        {
            for (index i = 0; i < fields.BlockNumber(); ++i)
            {
                const auto &field = fields.Block(fields.Name(i));
                DNDS_check_throw_info(field.Owned().Storage().IsIdle() && field.Ghost().Storage().IsIdle(),
                                      "DoF scratch still has active access or communication");
            }
        }

    public:
        class Borrow
        {
            std::shared_ptr<State> _state;
            index _prepared = 0;
            Use _use = Use::OwnedOnly;
            friend class DofScratchPool;
            Borrow(std::shared_ptr<State> state, index prepared, Use use)
                : _state(std::move(state)), _prepared(prepared), _use(use) {}

        public:
            Borrow(const Borrow &) = delete;
            Borrow &operator=(const Borrow &) = delete;
            Borrow(Borrow &&) noexcept = default;
            Borrow &operator=(Borrow &&) = delete;
            ~Borrow()
            {
                if (!_state)
                    return;
                try
                {
                    Release();
                }
                catch (...)
                {
                    MPI::Abort(MPI_COMM_WORLD, "DoF scratch returned with outstanding use");
                }
            }
            /// Sharing aliases are valid only within this borrow. Do not retain
            /// them past Release, rebind their storage, or infer halo freshness.
            const DofBundle &Fields() const
            {
                DNDS_check_throw_info(_state, "DoF scratch borrow was released");
                return _state->prepared[_prepared].fields;
            }
            Runtime::TransferOperation BeginExchange(index iField)
            {
                DNDS_check_throw_info(_state && _use == Use::Communicating, "owned-only scratch cannot exchange");
                auto &halos = _state->prepared[_prepared].halos;
                DNDS_check_throw_info(iField >= 0 && iField < index(halos.size()), "scratch field index is invalid");
                return halos[iField].Begin();
            }
            /// Explicit return can reject a live use without losing the borrow.
            /// Normal RAII return requires all CPU operators/transfers joined.
            void Release()
            {
                if (!_state)
                    return;
                auto &prepared = _state->prepared[_prepared];
                CheckIdle(prepared.fields);
                _state->slots[prepared.slot].busy = false;
                --_state->active;
                _state.reset();
            }
        };

        /// Setup only. No free-list decision during distributed execution;
        /// every rank supplies the same logical slot/descriptor sequence.
        index Reserve(const DofBundle &prototype)
        {
            DNDS_check_throw_info(!_state->active, "cannot reserve scratch during execution");
            prototype.Validate();
            DofBundle fields(prototype.Space());
            size_t bytes = 0;
            for (index i = 0; i < prototype.BlockNumber(); ++i)
            {
                const auto &name = prototype.Name(i);
                const auto &source = prototype.Block(name);
                DNDS_check_throw_info(source.Device() == Runtime::Device::Host(), "reserved DoF scratch currently requires CPU storage");
                DofField<real> field(source.Space(), Runtime::Array<real>(source.Owned().Layout()), Runtime::Array<real>(source.Ghost().Layout()));
                bytes += field.Owned().Storage().Bytes() + field.Ghost().Storage().Bytes();
                fields.Add(name, std::move(field));
            }
            _state->slots.emplace_back(std::move(fields));
            _state->bytes += bytes;
            return index(_state->slots.size()) - 1;
        }
        /// Build a level-specific row-space alias and its own saved bindings.
        /// Equal shape does not prove equal routing: binder supplies the actual
        /// level mapping. Caller proves cell order and schema compatibility.
        index Prepare(index slot, const DofSpace &space, const HaloBinder &binder)
        {
            DNDS_check_throw_info(!_state->active && slot >= 0 && slot < index(_state->slots.size()) && binder,
                                  "invalid scratch preparation or active borrow");
            const auto &storage = _state->slots[slot].fields;
            DNDS_check_throw_info(space.OwnedRows() == storage.Space().OwnedRows() && space.GhostRows() == storage.Space().GhostRows(),
                                  "scratch row counts differ from level space");
            Prepared prepared{slot, DofBundle(space), {}};
            for (index i = 0; i < storage.BlockNumber(); ++i)
            {
                const auto &name = storage.Name(i);
                const auto &field = storage.Block(name);
                prepared.fields.Add(name, DofField<real>(space, field.Owned(), field.Ghost()));
                prepared.halos.push_back(binder(prepared.fields.Block(name)));
            }
            _state->prepared.push_back(std::move(prepared));
            return index(_state->prepared.size()) - 1;
        }
        Borrow Acquire(index prepared, Use use = Use::OwnedOnly, Initialize initialization = Initialize::Dirty)
        {
            DNDS_check_throw_info(prepared >= 0 && prepared < index(_state->prepared.size()), "unknown prepared scratch descriptor");
            auto &entry = _state->prepared[prepared];
            auto &slot = _state->slots[entry.slot];
            DNDS_check_throw_info(!slot.busy, "reserved DoF scratch slot is already borrowed");
            entry.fields.Validate();
            CheckIdle(entry.fields);
            if (initialization == Initialize::ZeroOwned)
                entry.fields.Fill(0);
            else
                for (index i = 0; i < entry.fields.BlockNumber(); ++i)
                {
                    // New borrowing epoch invalidates previous freshness/cache
                    // assumptions without resetting monotone storage versions.
                    auto write = entry.fields.Block(entry.fields.Name(i)).Owned().Write();
                }
            slot.busy = true;
            _state->peak = std::max(_state->peak, ++_state->active);
            return Borrow(_state, prepared, use);
        }
        size_t RetainedBytes() const { return _state->bytes; }
        index SlotNumber() const { return index(_state->slots.size()); }
        index PeakBorrows() const { return _state->peak; }
    };
}
