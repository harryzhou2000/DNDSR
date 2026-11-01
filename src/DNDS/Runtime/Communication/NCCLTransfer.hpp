#pragma once

#include "DNDS/Runtime/Communication/RowTransfer.hpp"

namespace DNDS::Runtime
{
    /// One rank per CUDA device. MPI bootstraps only; prepared payload operations
    /// are stream ordered. Keep the provider alive through binding/engine work.
    /// Shared-communicator submissions must have matching per-peer call order
    /// across ranks. Use separate providers for independent communication lanes.
    /// Direct requires contiguous described fields; DevicePacked handles general
    /// supported layouts. Unsupported reducers fail at binding, never silently
    /// fall back to MPI. No global CUDA synchronization is implied.
    class NCCLTransferProvider final : public TransferProvider
    {
        struct State;
        std::shared_ptr<State> _state;
        explicit NCCLTransferProvider(std::shared_ptr<State>);

    public:
        /// Build capability only; Create validates runtime device availability.
        static bool Available();
        /// Collective setup on comm, outside hot execution/capture. Device
        /// ordinal is rank-local; each rank must select a distinct physical GPU.
        static std::shared_ptr<NCCLTransferProvider> Create(MPI_Comm comm, Runtime::Device device);
        /// Nonblocking error observation for caller-managed recovery/progress.
        /// Errors are rank-local; coordinate cancellation across participating
        /// ranks. An engine event alone cannot diagnose a stalled remote peer.
        void CheckAsyncError() const;
        /// Explicit cancellation of this communicator and all its bindings.
        /// Drain engine work before releasing arrays or their native streams.
        void Abort() const noexcept;
        const char *Name() const override { return "nccl"; }
        std::shared_ptr<Detail::TransferProviderBinding> Bind(
            const TransferPayload &, Buffer, Buffer, std::shared_ptr<const Reducer>,
            TransferTransport, const TransferOptions &, bool layoutsValid) const override;
    };
}
