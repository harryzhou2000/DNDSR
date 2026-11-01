#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::Runtime
{
    class EagerEngine;
    class ExecutionStream;
    struct TransferPayload;
    struct TransferOptions;
    struct Reducer;
    enum class TransferTransport;

    namespace Detail
    {
        /// Provider-owned completion; it need not contain an MPI request.
        class TransferProviderOperation
        {
        public:
            virtual ~TransferProviderOperation() = default;
            virtual void WaitDestinationReady() = 0;
            virtual void Wait() = 0;
            virtual bool Test() = 0;
        };
        class TransferProviderBinding
        {
        public:
            virtual ~TransferProviderBinding() = default;
            virtual std::unique_ptr<TransferProviderOperation> Begin(ExecutionContext, bool scheduled) const = 0;
            /// Enqueue the whole transfer; stream Record/WaitFor exports readiness.
            /// Caller orders aliases, private scratch reuse and matching peers.
            virtual void Enqueue(EagerEngine &, const ExecutionStream &) const = 0;
        };
    }

    /// Explicit optional payload provider. Null TransferOptions::provider uses
    /// the existing MPI implementation. Binding may coordinate on MPI during
    /// setup; the provider defines payload progress and stream completion.
    class TransferProvider
    {
    public:
        virtual ~TransferProvider() = default;
        virtual const char *Name() const = 0;
        virtual std::shared_ptr<Detail::TransferProviderBinding> Bind(
            const TransferPayload &, Buffer source, Buffer destination,
            std::shared_ptr<const Reducer>, TransferTransport, const TransferOptions &, bool layoutsValid) const = 0;
    };
}
