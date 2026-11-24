#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Execution/Device.hpp"
#include <string_view>

namespace DNDS::Cauchy
{
    enum class CUDAKernelDeviceProfile
    {
        Other,
        A100,
        A800
    };
    CUDAKernelDeviceProfile ClassifyCUDAKernelDevice(std::string_view name, int major, int minor);

    /// @brief Query the selected device during setup, never inside a queued kernel.
    CUDAKernelDeviceProfile GetCUDAKernelDeviceProfile(Runtime::Device device);
    bool IsA100Device(Runtime::Device device);

    /** @brief Resolve automatic policies before constructing resident operators.
     * @details A100 and A800 share the compact FP64 V2 VR sweep preset.
     * The measured A100 P3 flow5 profile uses warp-per-cell PP with 512 threads.
     * Perfect-gas RHS selection is resolved separately when its workset is bound.
     * ACM retains its degree- and quadrature-dependent mapping.
     * The device profile is separate so policy tests need no CUDA device.
     */
    void ResolveAutomaticCUDAKernels(DomainSettings &settings, CUDAKernelDeviceProfile profile, bool withoutRANS);

    /// Resolve the automatic inverse provider once when binding a linear solver.
    std::string ResolveAutomaticCUDAInverse(const std::string &selection, Runtime::Device device);
}
