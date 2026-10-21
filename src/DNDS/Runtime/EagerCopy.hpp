#pragma once

#include "EagerEngine.hpp"

namespace DNDS::Runtime
{
    /// Explicit byte copy in an eager stream. Host DMA buffers remain retained
    /// through actual completion. No event or host wait is inserted per copy.
    inline void CopyAsync(EagerEngine &engine, const ExecutionStream &stream, const Buffer &source, Buffer &destination)
    {
        const auto context = stream.Context();
        const auto srcDevice = source.GetDevice(), dstDevice = destination.GetDevice();
        const bool gpu = srcDevice != Device::Host() || dstDevice != Device::Host();
        DNDS_check_throw_info(source.Identity() && destination.Identity() && source.Identity() != destination.Identity() &&
                                  source.Bytes() == destination.Bytes(),
                              "eager copy requires distinct equal-size buffers");
        DNDS_check_throw_info(gpu == (context.device.space == MemorySpace::CUDA) &&
                                  (srcDevice == Device::Host() || srcDevice == context.device) &&
                                  (dstDevice == Device::Host() || dstDevice == context.device),
                              "eager copy placement mismatch");
        struct Pending
        {
            ScheduledAccess source, destination;
            const void *src;
            void *dst;
            bool handoffSource, handoffDestination;
            Pending(const Buffer &in, const Buffer &out, ExecutionContext context)
                : source(in.Reserve(AccessMode::Read, in.GetDevice())), destination(out.Reserve(AccessMode::Write, out.GetDevice())),
                  src(source.Pointer(1, 1)), dst(destination.Pointer(1, 1)),
                  handoffSource(in.GetDevice().space == MemorySpace::CUDA && in.CanReleaseOrdered(context)),
                  handoffDestination(out.GetDevice().space == MemorySpace::CUDA && out.CanReleaseOrdered(context)) {}
        };
        auto retained = std::make_shared<Pending>(source, destination, context);
        engine.SubmitOperation(stream, [retained, srcDevice, dstDevice, bytes = source.Bytes()](ExecutionContext execution)
                               {
                retained->source.Validate(); retained->destination.Validate();
                Detail::EnqueueCopyBytes(retained->src, srcDevice, retained->dst, dstDevice, bytes, execution);
                if (retained->handoffSource) retained->source.Handoff(execution);
                if (retained->handoffDestination) retained->destination.Handoff(execution); }, [retained](std::exception_ptr error) noexcept
                               {
                retained->source.Complete(!error);
                retained->destination.Complete(!error); });
    }
}
