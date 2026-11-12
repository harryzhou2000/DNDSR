#pragma once

#include <cstddef>
#include <type_traits>

namespace DNDS::Runtime
{
    /// Conservative project budget for all parameters of one numerical CUDA
    /// launch, not a device's hardware limit. Large descriptors belong in a
    /// retained device buffer with an ordered upload and a small argument view.
    inline constexpr std::size_t KernelArgumentBudgetBytes = 2048;

    /// Include every __global__ argument (not the host execution context).
    /// Counting up to alignof(T)-1 padding for each parameter deliberately
    /// overestimates ordinary parameter alignment. NVCC still validates the
    /// actual ABI. This imposes no runtime or per-element cost.
    template <class... Args>
    constexpr void CheckKernelArgumentBudget()
    {
        static_assert((std::is_trivially_copyable_v<Args> && ...),
                      "kernel arguments must be trivially copyable descriptors");
        constexpr auto bytes = (std::size_t(0) + ... + (sizeof(Args) + alignof(Args) - 1));
        static_assert(bytes <= KernelArgumentBudgetBytes,
                      "aggregate kernel arguments exceed the project budget; place large descriptors in a retained device buffer");
    }
}
