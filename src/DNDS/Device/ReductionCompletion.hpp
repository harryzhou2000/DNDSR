#pragma once
#include <cuda/atomic>

namespace DNDS::Device
{
    /**
     * @brief Elect the last arriving block without a grid-wide wait.
     * @details Call once from each block's publishing thread, after writing its
     * partial. The device-scope acquire/release RMW chain publishes every earlier
     * block's writes to the last thread. That thread must use a block barrier to
     * make the acquired partials visible to its cooperating reduction threads.
     * Non-last blocks return; they never spin (which could prevent unscheduled
     * blocks from running). Reduce the partials by their fixed block index, not
     * arrival order, to preserve the floating-point reduction tree.
     *
     * The caller owns a distinct counter per concurrent operation, resets it to
     * zero in stream order before launch, and retains it and the partials until
     * completion. The counter is not a device global or a reusable grid barrier.
     */
    __device__ inline bool ArriveReductionBlock(unsigned int &completed, unsigned int blocks)
    {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> counter(completed);
        return counter.fetch_add(1, cuda::memory_order_acq_rel) == blocks - 1;
    }
}
