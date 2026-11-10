#pragma once

#include "Geom/Access/EntitySpace.hpp"
#include "Geom/Transition/FieldTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Geom
{
    /// Geom-side owned field association. Runtime Arrays remain generic;
    /// callers opt into checked mesh/domain/order boundaries through this owner.
    /// Storage may be table, padded or CSR. Select a checked layout projection
    /// at ReadScope/WriteScope::View<Policy>(), independently of migration layout.
    template <class T>
    class EntityField
    {
        struct State
        {
            EntitySpace space;
            Runtime::Array<T> values;
            Runtime::ElementSchema schema;
            Runtime::Device device;
            uint64_t allocation, shape;
            State(EntitySpace entity, Runtime::Array<T> array, Runtime::ElementSchema elementSchema, Runtime::ExecutionContext execution)
                : space(std::move(entity)), values(std::move(array)), schema(std::move(elementSchema)), device(execution.device),
                  allocation(values.Identity()), shape(values.ShapeVersion())
            {
                space.Validate();
                DNDS_check_throw_info(values.IsValid() && values.RowCount() == size_t(space.NumOwned()) && schema.Extent() == sizeof(T) &&
                                          schema.Compatible(Runtime::ElementSchema::TypeIdentity<T>()) && device == values.GetDevice(),
                                      "entity field requires exact owned rows and matching schema/context");
            }
            Runtime::ExecutionContext HostContext() const
            {
                DNDS_check_throw_info(device == Runtime::Device::Host(), "device entity field needs an explicit execution stream/context");
                return Runtime::ExecutionContext::Host();
            }
            void Validate(const EntitySpace &expected) const
            {
                DNDS_check_throw_info(space.SameOrder(expected) && values.IsValid() && values.Identity() == allocation &&
                                          values.ShapeVersion() == shape && values.RowCount() == size_t(space.NumOwned()),
                                      "entity field mesh/domain/order or allocation/layout changed");
            }
        };
        struct Slot
        {
            std::shared_ptr<State> current;
        };
        std::shared_ptr<Slot> _slot;
        static Runtime::VersionTracker::ReadLease Retain(const EntitySpace &space) { return space._versions.Read(); }

        struct Prepared final : PreparedCompanion
        {
            std::shared_ptr<Runtime::ExecutionBorrow> execution;
            std::shared_ptr<Slot> destination;
            Runtime::Array<T> values;
            Runtime::ElementSchema schema;
            Runtime::Device device;
            std::shared_ptr<typename Runtime::Array<T>::ReadLease> sourceRead;
            std::shared_ptr<State> next;
            Prepared(std::shared_ptr<Slot> target, Runtime::Array<T> array, Runtime::ElementSchema elementSchema, Runtime::ExecutionContext execution,
                     std::shared_ptr<typename Runtime::Array<T>::ReadLease> retained, std::shared_ptr<Runtime::ExecutionBorrow> borrow)
                : execution(std::move(borrow)), destination(std::move(target)), values(std::move(array)), schema(std::move(elementSchema)), device(execution.device), sourceRead(std::move(retained)) {}
            void BindEntitySpace(const EntitySpace &space) override
            {
                next = std::make_shared<State>(space, std::move(values), schema, Runtime::ExecutionContext{device, nullptr});
            }
            void Publish() noexcept override { destination->current.swap(next); }
        };

    public:
        class ReadScope
        {
            std::optional<Runtime::ExecutionBorrow> _borrow;
            std::shared_ptr<State> _state;
            Runtime::VersionTracker::ReadLease _mesh;
            typename Runtime::Array<T>::ReadLease _read;
            bool _sealed = false;
            friend class EntityField;
            ReadScope(std::shared_ptr<State> state, const EntitySpace &expected, Runtime::ExecutionContext context)
                : _state(std::move(state)), _mesh(Retain(_state->space)), _read(_state->values.Read(context))
            {
                _state->Validate(expected);
            }

        public:
            template <class Policy = Runtime::RowLayout, bool Checked = false>
            auto View() const
            {
                DNDS_check_throw_info(_state && !_sealed, "entity field read is moved or sealed");
                return _read.template View<Policy, Checked>();
            }
            void Validate(const EntitySpace &space) const
            {
                DNDS_check_throw_info(_state, "entity field read was moved");
                _state->Validate(space);
            }
            EntitySpace Space() const
            {
                DNDS_check_throw_info(_state, "entity field read was moved");
                return _state->space;
            }
            void Complete(const Runtime::Completion &completion)
            {
                DNDS_check_throw_info(_state && !_sealed, "entity field read moved or already sealed");
                _read.Complete(completion);
                _sealed = true;
            }
        };

        /// Reads retain the state whose identity was checked. Mutating values
        /// uses an explicit scoped writer; structural edits stay outside this API.
        class WriteScope
        {
            std::optional<Runtime::ExecutionBorrow> _borrow;
            std::shared_ptr<State> _state;
            Runtime::VersionTracker::ReadLease _mesh;
            typename Runtime::Array<T>::WriteLease _write;
            bool _sealed = false;
            friend class EntityField;
            WriteScope(std::shared_ptr<State> state, const EntitySpace &space, Runtime::ExecutionContext context)
                : _state(std::move(state)), _mesh(Retain(_state->space)), _write(_state->values.Write(context)) { _state->Validate(space); }

        public:
            template <class Policy = Runtime::RowLayout, bool Checked = false>
            auto View() const
            {
                DNDS_check_throw_info(_state && !_sealed, "entity field write is moved or sealed");
                return _write.template View<Policy, Checked>();
            }
            void Complete(const Runtime::Completion &completion)
            {
                DNDS_check_throw_info(_state && !_sealed, "entity field write moved or already sealed");
                _write.Complete(completion);
                _sealed = true;
            }
        };
        WriteScope Write(const EntitySpace &space, const Runtime::ExecutionStream &stream)
        {
            auto borrow = stream.BorrowContext();
            auto result = Write(space, borrow.Context());
            result._borrow.emplace(std::move(borrow));
            return result;
        }
        WriteScope Write(const EntitySpace &space)
        {
            Validate(space);
            return Write(space, _slot->current->HostContext());
        }
        WriteScope Write(const EntitySpace &space, Runtime::ExecutionContext context)
        {
            Validate(space);
            return WriteScope(_slot->current, space, context);
        }

        class HaloRoute
        {
            EntitySpace _space;
            Runtime::RowTransferPlan _route;
            friend class EntityField;
            explicit HaloRoute(EntitySpace space) : _space(std::move(space)), _route(_space.BuildRowTransferPlan()) {}

        public:
            /// Retained fields and mesh maps are checked before execution, even
            /// if this binding's old allocations remain alive after publication.
            class Binding
            {
                std::shared_ptr<Slot> _slot;
                std::shared_ptr<State> _expected;
                MPIInfo _mpi;
                bool _collectivePreflight = true;
                Runtime::Array<T> _ghost;
                Runtime::TransferBinding _binding;
                friend class HaloRoute;
                Binding(std::shared_ptr<Slot> slot, std::shared_ptr<State> expected, Runtime::Array<T> ghost,
                        Runtime::TransferBinding binding, bool collectivePreflight)
                    : _slot(std::move(slot)), _expected(std::move(expected)), _mpi(_expected->space._mpi), _collectivePreflight(collectivePreflight),
                      _ghost(std::move(ghost)), _binding(std::move(binding)) {}
                void ValidateCurrent()
                {
                    Runtime::CommunicationDetail::LocalStage(_mpi.comm, _collectivePreflight, [&]
                                                             {
                        DNDS_check_throw_info(_slot && _expected && _slot->current == _expected,
                                              "entity field halo binding is moved or predates field publication");
                        _expected->Validate(_expected->space); }, "entity field halo binding predates field publication");
                }

            public:
                class Operation
                {
                    std::shared_ptr<State> _state;
                    Runtime::VersionTracker::ReadLease _mesh;
                    Runtime::TransferOperation _operation;
                    friend class Binding;
                    Operation(std::shared_ptr<State> state, Runtime::TransferBinding &binding, Runtime::ExecutionContext context, bool scheduled)
                        : _state(std::move(state)), _mesh(Retain(_state->space)),
                          _operation(scheduled ? binding.BeginScheduled(context) : binding.Begin(context)) {}

                public:
                    void Wait() { _operation.Wait(); }
                };
                Operation Begin()
                {
                    ValidateCurrent();
                    return Operation(_expected, _binding, _expected->HostContext(), false);
                }
                Operation Begin(Runtime::ExecutionContext context, bool scheduled = false)
                {
                    ValidateCurrent();
                    return Operation(_expected, _binding, context, scheduled);
                }
                Operation BeginScheduled(Runtime::ExecutionContext context) { return Begin(context, true); }
                Runtime::Array<T> GhostValues() const { return _ghost; }
            };

            Binding Bind(EntityField &field, const EntitySpace &expected, Runtime::TransferLayout destinationLayout = {}) const
            {
                _space.LocalStage([&]
                                  {
                    field.Validate(expected);
                    DNDS_check_throw_info(_space.SameOrder(expected), "entity field route belongs to another mesh/order"); });
                auto state = field._slot->current;
                auto mesh = Retain(_space);
                Runtime::PayloadPlan payload(_route, state->values.Layout(), state->schema, destinationLayout);
                std::optional<Runtime::Array<T>> ghost;
                _space.LocalStage([&]
                                  { ghost.emplace(payload.DestinationLayout(), state->values.GetDevice()); });
                auto binding = payload.Bind(state->values, *ghost, Runtime::TransferTransport::Direct);
                return Binding(field._slot, std::move(state), std::move(*ghost), std::move(binding), _route.Context()->ChecksCollectivePreflight());
            }
        };

        EntityField(EntitySpace space, Runtime::Array<T> values,
                    Runtime::ElementSchema schema = Runtime::ElementSchema::Arithmetic<T>(),
                    Runtime::ExecutionContext context = Runtime::ExecutionContext::Host())
            : _slot(std::make_shared<Slot>(Slot{std::make_shared<State>(std::move(space), std::move(values), std::move(schema), context)})) {}
        void Validate(const EntitySpace &space) const
        {
            DNDS_check_throw_info(_slot && _slot->current, "entity field was moved");
            _slot->current->Validate(space);
        }
        ReadScope Read(const EntitySpace &space, const Runtime::ExecutionStream &stream) const
        {
            auto borrow = stream.BorrowContext();
            auto result = Read(space, borrow.Context());
            result._borrow.emplace(std::move(borrow));
            return result;
        }
        ReadScope Read(const EntitySpace &space) const
        {
            Validate(space);
            return Read(space, _slot->current->HostContext());
        }
        ReadScope Read(const EntitySpace &space, Runtime::ExecutionContext context) const
        {
            Validate(space);
            return ReadScope(_slot->current, space, context);
        }
        HaloRoute BuildHaloRoute(const EntitySpace &space) const
        {
            space.LocalStage([&]
                             { Validate(space); });
            return HaloRoute(space);
        }
        /// Explicit generic escape. Retains values but loses Geom association;
        /// using it with arbitrary lower-level routes is a caller contract.
        Runtime::Array<T> UntypedValues(const EntitySpace &space) const
        {
            Validate(space);
            return _slot->current->values;
        }

        /// Register this association for detached migration. Compact CSR is the
        /// default; request PreserveStorage or an explicit table/padded shape
        /// when physical stride/capacity matters to subsequent accessors.
        CompanionEntry Register(std::string name, Runtime::TransferLayout destinationLayout = {},
                                std::optional<Runtime::ExecutionStream> stream = std::nullopt)
        {
            DNDS_check_throw_info(_slot && _slot->current, "entity field registration requires a live field");
            auto slot = _slot;
            auto state = slot->current;
            auto execution = stream ? std::make_shared<Runtime::ExecutionBorrow>(stream->BorrowContext()) : nullptr;
            const auto context = execution ? execution->Context() : state->HostContext();
            // Descriptor construction is local. Defer semantic checks to the
            // collectively guarded preparation stages, before any payload.
            auto entry = RegisterArray<T>(std::shared_ptr<Runtime::Array<T>>(state, &state->values), state->space.Kind(), std::move(name), state->schema, context,
                                          CompanionDisposition::TransferIfReordered, destinationLayout, state->space.Communication());
            entry.preflight = [state, execution, context](const PermutationTransfer &transfer)
            {
                state->Validate(state->space);
                DNDS_check_throw_info(state->values.RowCount() == size_t(transfer.size()), "entity field migration requires owned rows");
                auto read = state->values.Read(context);
            };
            entry.attachment = [slot, state, execution, context]
            {
                DNDS_check_throw_info(slot->current == state, "entity field registration predates publication");
                state->Validate(state->space);
                auto read = state->values.Read(context);
                return CompanionAttachment{state->schema.Identity() + ":" + Runtime::ElementSchema::TypeIdentity<T>() + ":entity-bound:" +
                                               std::to_string(int(state->values.GetDevice().space)),
                                           {},
                                           false,
                                           0,
                                           0,
                                           slot.get(),
                                           state->values.Identity(),
                                           state->values.Version(),
                                           state->values.ShapeVersion(),
                                           size_t(state->values.RowCount()),
                                           size_t(state->values.RowCount()),
                                           0};
            };
            auto prepare = entry.prepare;
            entry.prepare = [slot, state, prepare, execution, context](const PermutationTransfer &transfer, const MPIInfo &mpi)
            {
                std::shared_ptr<typename Runtime::Array<T>::ReadLease> retained;
                state->space.LocalStage([&]
                                        { retained = std::make_shared<typename Runtime::Array<T>::ReadLease>(state->values.Read(context)); });
                auto old = prepare(transfer, mpi);
                auto *array = dynamic_cast<FieldTransferDetail::Prepared<Runtime::Array<T>> *>(old.get());
                DNDS_check_throw_info(array, "entity field adapter preparation type mismatch");
                return std::make_unique<Prepared>(slot, std::move(array->next), state->schema, context, std::move(retained), execution);
            };
            entry.requiresEntityBinding = true;
            entry.validateSourceSpace = [slot, state](const EntitySpace &space)
            {
                DNDS_check_throw_info(slot->current == state, "entity field registration changed");
                state->Validate(space);
            };
            entry.prepareUnchanged = [slot, state, execution, context]
            {
                auto retained = std::make_shared<typename Runtime::Array<T>::ReadLease>(state->values.Read(context));
                return std::make_unique<Prepared>(slot, state->values, state->schema, context, std::move(retained), execution);
            };
            entry.fn = [](const PermutationTransfer &, const MPIInfo &)
            { DNDS_check_throw_info(false, "entity-bound fields require detached PreparedMeshTransition"); };
            return entry;
        }
    };
}
