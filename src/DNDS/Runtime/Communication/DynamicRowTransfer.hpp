#pragma once

#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include <optional>

namespace DNDS::Runtime
{
    /// Collective preparation, then nonblocking payload progression. The new
    /// field preserves storage by default and stays private until Commit; source shape/read
    /// lease span both preparation and transfer. No partial destination escapes.
    /// @pre Every rank participates in matching construction/Commit order, even
    /// with zero rows. Caller supplies the real source element schema (native
    /// ArrayStorage is type-erased), a completed producer, and the correct device
    /// context. Drain before redistribution or MPI_Finalize; rebuild downstream
    /// payloads/stencils for the returned shape. Commit publishes once, not an
    /// atomic transaction across multiple separately migrated fields.
    class DynamicRowTransfer
    {
        ArrayStorage::Shared _source, _destination;
        ReadAccess _sourcePin;
        TransferOperation _operation;
        bool _committed = false;
        std::optional<RowTransferPlan> _routeOwner;
        std::exception_ptr _error;

    public:
        DynamicRowTransfer(RowTransferPlan route, ArrayStorage::Shared source,
                           ElementSchema schema, Device destinationDevice = Device::Host(),
                           TransferTransport transport = TransferTransport::Direct,
                           TransferOptions options = {TransferMode::Transient, {}, 0},
                           std::optional<Reducer> reducer = {}, ExecutionContext context = ExecutionContext::Host(),
                           TransferLayout destinationLayout = {})
            : _routeOwner(route)
        {
            bool valid = true;
            try
            {
                _source = ArrayStorage::Share(source);
                DNDS_check_throw_info(_source, "dynamic transfer requires live source storage");
                DNDS_check_throw_info(_source->ElementBytes() == schema.Extent(), "dynamic transfer element stride mismatch");
                _sourcePin = _source->Read(_source->Values().GetDevice() == Device::Host() ? ExecutionContext::Host() : context);
            }
            catch (...)
            {
                valid = false;
            }
            route.Context()->Preflight(valid, "dynamic transfer source reservation failed on a rank");
            PayloadPlan payload(route, _source->Layout(), schema, destinationLayout);
            try
            {
                _destination = ArrayStorage::Allocate(payload.DestinationLayout(), schema.Extent(), destinationDevice);
            }
            catch (...)
            {
                valid = false;
            }
            route.Context()->Preflight(valid, "dynamic transfer destination allocation failed on a rank");
            auto binding = reducer ? payload.Bind(_source->Values(), _destination->Values(), std::move(*reducer), transport, options)
                                   : payload.Bind(_source->Values(), _destination->Values(), transport, options);
            _operation = binding.Begin(context);
        }
        DynamicRowTransfer(const DynamicRowTransfer &) = delete;
        DynamicRowTransfer &operator=(const DynamicRowTransfer &) = delete;
        bool Test()
        {
            if (_error)
                std::rethrow_exception(_error);
            try
            {
                if (!_operation.Test())
                    return false;
            }
            catch (...)
            {
                _error = std::current_exception();
                throw;
            }
            _sourcePin.Wait();
            _source.reset();
            return true;
        }
        void Wait()
        {
            if (_error)
                std::rethrow_exception(_error);
            try
            {
                _operation.Wait();
            }
            catch (...)
            {
                _error = std::current_exception();
                throw;
            }
            _sourcePin.Wait();
            _source.reset();
        }
        ArrayStorage::Shared Commit()
        {
            DNDS_check_throw_info(!_committed, "dynamic transfer already committed");
            try
            {
                Wait();
            }
            catch (...)
            {
                _error = std::current_exception();
            }
            _routeOwner->Context()->Preflight(!_error, "dynamic transfer failed; destination was not published");
            _committed = true;
            return std::move(_destination);
        }
    };
}
