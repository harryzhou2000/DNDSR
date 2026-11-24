#include "Cauchy/Config/CUDAKernelDefaults.hpp"
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Cauchy
{
    CUDAKernelDeviceProfile ClassifyCUDAKernelDevice(std::string_view name, int major, int minor)
    {
        if (major == 8 && minor == 0)
        {
            if (name.find("A100") != std::string_view::npos)
                return CUDAKernelDeviceProfile::A100;
            if (name.find("A800") != std::string_view::npos)
                return CUDAKernelDeviceProfile::A800;
        }
        return CUDAKernelDeviceProfile::Other;
    }
    CUDAKernelDeviceProfile GetCUDAKernelDeviceProfile(Runtime::Device device)
    {
        if (device.space != Runtime::MemorySpace::CUDA)
            return CUDAKernelDeviceProfile::Other;
#ifdef DNDS_USE_CUDA
        cudaDeviceProp properties{};
        const auto status = cudaGetDeviceProperties(&properties, device.ordinal);
        DNDS_check_throw_info(status == cudaSuccess, "cannot resolve CUDA kernel defaults for selected device");
        return ClassifyCUDAKernelDevice(properties.name, properties.major, properties.minor);
#else
        DNDS_check_throw_info(false, "CUDA kernel defaults require a CUDA build");
        return CUDAKernelDeviceProfile::Other;
#endif
    }
    bool IsA100Device(Runtime::Device device)
    {
        return GetCUDAKernelDeviceProfile(device) == CUDAKernelDeviceProfile::A100;
    }
    void ResolveAutomaticCUDAKernels(DomainSettings &settings, CUDAKernelDeviceProfile profile, bool withoutRANS)
    {
        auto &execution = settings.execution;
        auto &kernels = execution.kernels;
        const auto &numerics = settings.numerics;
        const bool acm = settings.equation.system == "incompressibleACM";
        const bool coupledAoSPolynomial = numerics.reconstruction == "variational" &&
                                          execution.layout == "aos" && execution.fieldGrouping == "coupled";
        const bool a100Polynomial = profile == CUDAKernelDeviceProfile::A100 && coupledAoSPolynomial;
        const bool a100P3 = a100Polynomial && numerics.cfv.maxOrder == 3 && numerics.cfv.intOrder <= 5;
        const bool measuredFlow5 = a100P3 && !acm && withoutRANS && settings.mesh.dimension == 3;
        if (acm && kernels.rhsMapping == "auto")
        {
            const bool fp64 = kernels.polynomialArithmetic == "fp64" && kernels.dissipationArithmetic == "fp64" &&
                              kernels.fluxArithmetic == "fp64";
            // Low-work 3D profiles tie or favor thread-per-face. Require a
            // clear warmed-update gain before paying for CTA face packages.
            const int degree = numerics.cfv.maxOrder, rule = numerics.RHSIntegrationOrder();
            const bool acmCTA = a100Polynomial && settings.mesh.dimension == 3 && !numerics.firstOrderConvection &&
                                ((degree == 2 && (rule >= 4 || (rule == 3 && settings.equation.rans.model == "KOmegaSST"))) ||
                                 (degree == 3 && rule >= 2));
            kernels.rhsMapping = fp64 && acmCTA ? "ctaSegments" : "thread";
        }
        if (acm && kernels.rhsReduction == "auto")
            kernels.rhsReduction = "serial";
        if (execution.reconstructionKernel == "auto")
        {
            const bool v2Polynomial = coupledAoSPolynomial &&
                                      (profile == CUDAKernelDeviceProfile::A100 || profile == CUDAKernelDeviceProfile::A800);
            execution.reconstructionKernel = v2Polynomial && !numerics.reconstructionSolver.UsesAMG() ? "fp64V2" : "fp64";
        }
        if (!kernels.polynomialPPThreads)
            kernels.polynomialPPThreads = measuredFlow5 ? 512 : 128;
    }
    std::string ResolveAutomaticCUDAInverse(const std::string &selection, Runtime::Device device)
    {
        if (selection != "auto")
            return selection;
        // Guarded LU uses the existing condition guard and filtered SVD
        // fallback. It is an implementation choice, not a new inverse policy.
        return IsA100Device(device) ? "guardedLU" : "batchedSVD";
    }
}
