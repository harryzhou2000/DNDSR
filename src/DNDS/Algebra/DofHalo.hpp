#pragma once

#include "DNDS/Algebra/DofField.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <optional>

namespace DNDS::Algebra
{
    /// CPU halo transport with an explicit completed-publication record.
    /// Epoch is a caller-defined, equality-comparable source context (for example
    /// a step/trial/stage/iteration tuple). All ranks must use a coherent epoch
    /// and exchange order. This facility never infers remote freshness from a
    /// local version and never skips an exchange. It adds no payload staging.
    ///
    /// The bound route is the exact owned-to-ghost mapping; its dependency
    /// stamps must cover mesh/order changes. Field aliases share publication
    /// state through this handle. Serialize Begin/Wait/RequireCurrent on the MPI
    /// controller; retained operations keep the binding and allocations alive.
    /// CUDA needs a separate stream-ready publication contract and is rejected.
    template <class T = real, class Epoch = uint64_t>
    class DofHalo
    {
        using Field = DofField<T>;
        struct Data
        {
            Field field;
            Runtime::RowTransferPlan route;
            Runtime::TransferBinding binding;
            uint64_t ownedShape, ghostShape, ownedIdentity, ghostIdentity;
            struct Publication
            {
                Epoch epoch;
                uint64_t ownedVersion, ghostVersion;
            };
            std::optional<Publication> publication;
            bool active = false;
            Data(Field field, Runtime::RowTransferPlan route, Runtime::TransferBinding binding)
                : field(std::move(field)), route(std::move(route)), binding(std::move(binding)),
                  ownedShape(this->field.Owned().ShapeVersion()), ghostShape(this->field.Ghost().ShapeVersion()),
                  ownedIdentity(this->field.Owned().Identity()), ghostIdentity(this->field.Ghost().Identity()) {}
            bool Matches(const Field &other) const
            {
                return route.Valid() && field.Space().Same(other.Space()) &&
                       field.Owned().SharedStorage() == other.Owned().SharedStorage() &&
                       field.Ghost().SharedStorage() == other.Ghost().SharedStorage() &&
                       other.Owned().ShapeVersion() == ownedShape && other.Ghost().ShapeVersion() == ghostShape &&
                       other.Owned().Identity() == ownedIdentity && other.Ghost().Identity() == ghostIdentity &&
                       other.Owned().IsValid() && other.Ghost().IsValid();
            }
        };
        std::shared_ptr<Data> _data;

    public:
        class Operation
        {
            std::shared_ptr<Data> _data;
            Runtime::TransferOperation _operation;
            Epoch _epoch;
            uint64_t _ownedVersion;
            bool _observed = false;
            friend class DofHalo;
            Operation(std::shared_ptr<Data> data, Runtime::TransferOperation operation, Epoch epoch)
                : _data(std::move(data)), _operation(std::move(operation)), _epoch(std::move(epoch)),
                  _ownedVersion(_data->field.Owned().Version()) {}
            void Publish()
            {
                DNDS_check_throw_info(_data->Matches(_data->field) && _data->field.Owned().Version() == _ownedVersion,
                                      "halo binding/source changed during transfer");
                _data->publication = typename Data::Publication{_epoch, _ownedVersion, _data->field.Ghost().Version()};
                _data->active = false;
                _observed = true;
            }

        public:
            Operation(const Operation &) = delete;
            Operation &operator=(const Operation &) = delete;
            Operation(Operation &&) noexcept = default;
            Operation &operator=(Operation &&) = delete;
            ~Operation()
            {
                if (_data && !_observed)
                {
                    // Drain for lifetime safety, but unobserved/failed work must
                    // not silently certify freshness. Explicit Wait/Test reports errors.
                    if (!std::uncaught_exceptions())
                    {
                        try
                        {
                            _operation.Wait();
                        }
                        catch (...)
                        {
                        }
                    }
                    _data->publication.reset();
                    _data->active = false;
                }
            }
            void Wait()
            {
                if (_observed)
                    return;
                _operation.Wait();
                Publish();
            }
            bool Test()
            {
                if (_observed)
                    return true;
                if (!_operation.Test())
                    return false;
                Publish();
                return true;
            }
        };

        /// Collective construction. Layouts remain AoS/SoA/AoSoA as bound;
        /// ordinary direct h-indexed transfer is the default transport.
        DofHalo(Field &field, Runtime::RowTransferPlan route, Runtime::TransferOptions options = {})
        {
            MPI::LocalStage(MPIInfo(route.Communicator()), [&]
                            {
                field.Validate();
                DNDS_check_throw_info(field.Device() == Runtime::Device::Host() && !route.HasCollisions() &&
                    route.SourceRows() == index(field.Space().OwnedRows()) && route.DestinationRows() == index(field.Space().GhostRows()),
                    "halo requires a CPU field and collision-free owned-to-ghost route"); });
            auto binding = Runtime::PayloadPlan(route, field.Owned().Layout(), field.Ghost().Layout(), Runtime::ElementSchema::Arithmetic<T>())
                               .Bind(field.Owned(), field.Ghost(), Runtime::TransferTransport::Direct, std::move(options));
            _data = std::make_shared<Data>(field, std::move(route), std::move(binding));
        }
        /// Conservatively exchange every call. Binding validity joins the
        /// transport's existing preflight. checks adds a separate diagnostic
        /// collective and defaults off. Checked streams
        /// observe their CPU producer prefix; caller-ordered eager streams require
        /// completed CPU producers. Core readers may overlap the ghost writer.
        Operation Begin(Epoch epoch, Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream, bool checks = false)
        {
            bool valid = false;
            try
            {
                valid = !_data->active && _data->Matches(_data->field) && stream.Context().device == Runtime::Device::Host();
            }
            catch (...)
            {
            } // Carry local failure into the existing MPI preflight.
            const auto validate = [&]
            {
                DNDS_check_throw_info(valid,
                                      "halo binding is stale, busy or not on CPU");
            };
            if (checks)
                MPI::LocalStage(MPIInfo(_data->route.Communicator()), validate);
            _data->publication.reset();
            auto operation = engine.ChecksAccess() ? _data->binding.BeginChecked(stream, valid) : _data->binding.BeginScheduled(stream.Context(), valid);
            _data->active = true;
            return Operation(_data, std::move(operation), std::move(epoch));
        }
        /// Local prologue check only. Epoch equality is necessary: unchanged
        /// local owned bytes cannot reveal a remote-only producer update.
        /// Read leases separately guarantee access safety, not source freshness.
        void RequireCurrent(const Field &field, const Epoch &epoch) const
        {
            DNDS_check_throw_info(!_data->active && _data->publication && _data->Matches(field) &&
                                      _data->publication->epoch == epoch && _data->publication->ownedVersion == field.Owned().Version() &&
                                      _data->publication->ghostVersion == field.Ghost().Version(),
                                  "halo has no completed publication for this field/source epoch");
        }
    };
}
