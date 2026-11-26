#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Execution/Device.hpp"
#include <string_view>

namespace DNDS::Cauchy
{
    enum class CUDAKernelDeviceProfile
    {
        Other,
        A100
    };
    /// Select a shared launch preset by product name. Device arithmetic has separate capability checks.
    CUDAKernelDeviceProfile ClassifyCUDAKernelDevice(std::string_view name);

    /// @brief Query the selected device during setup, never inside a queued kernel.
    CUDAKernelDeviceProfile GetCUDAKernelDeviceProfile(Runtime::Device device);
    bool UsesA100KernelDefaults(Runtime::Device device);

    /** @brief Resolve automatic policies before constructing resident operators.
     * @details RTX 30/40/50-series, A100/A800, H100/H200/H20, and B200/B300
     * share the A100 preset. Its P3 flow5 policy uses warp-per-cell PP with 512 threads.
     * Perfect-gas RHS selection is resolved separately when its workset is bound.
     * ACM retains its degree- and quadrature-dependent mapping.
     * The device profile is separate so policy tests need no CUDA device.
     */
    void ResolveAutomaticCUDAKernels(DomainSettings &settings, CUDAKernelDeviceProfile profile, bool withoutRANS);

    /// Resolve the automatic inverse provider once when binding a linear solver.
    std::string ResolveAutomaticCUDAInverse(const std::string &selection, Runtime::Device device);
}
