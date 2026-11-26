#pragma once

#include "DNDS/Core/Errors.hpp"

#ifdef DNDS_USE_CUDA
#    include <cuda_runtime.h>
#endif

namespace DNDS::Device
{
#ifdef DNDS_USE_CUDA
    /// RAII guard that switches to a CUDA device ordinal and restores the
    /// previous active device on scope exit (never throwing from destructor).
    struct CUDADeviceScope
    {
        int previous = -1;
        bool switched = false;

        explicit CUDADeviceScope(int ordinal)
        {
            DNDS_check_throw_info(ordinal >= 0, "invalid CUDA device ordinal");
            const cudaError_t queryStatus = cudaGetDevice(&previous);
            DNDS_check_throw_info(queryStatus == cudaSuccess, cudaGetErrorString(queryStatus));
            if (previous == ordinal)
                return;
            const cudaError_t setStatus = cudaSetDevice(ordinal);
            DNDS_check_throw_info(setStatus == cudaSuccess, cudaGetErrorString(setStatus));
            switched = true;
        }

        CUDADeviceScope(const CUDADeviceScope &) = delete;
        CUDADeviceScope &operator=(const CUDADeviceScope &) = delete;

        ~CUDADeviceScope() noexcept
        {
            if (switched && previous >= 0)
                (void)cudaSetDevice(previous);
        }
    };
#endif
}
