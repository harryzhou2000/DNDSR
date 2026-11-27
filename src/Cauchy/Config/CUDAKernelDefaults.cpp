#include "Cauchy/Config/CUDAKernelDefaults.hpp"
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Cauchy
{
    CUDAKernelDeviceProfile ClassifyCUDAKernelDevice(std::string_view name)
    {
        constexpr std::string_view separators = " \t-_()";
        bool afterRTX = false;
        for (auto start = name.find_first_not_of(separators); start != std::string_view::npos;
             start = name.find_first_not_of(separators))
        {
            name.remove_prefix(start);
            const auto product = name.substr(0, name.find_first_of(separators));
            if (product == "A100" || product == "A800" || product == "H100" || product == "H200" ||
                product == "H20" || product == "B200" || product == "B300")
                return CUDAKernelDeviceProfile::A100;
            // The x000 names belong to workstation lines, distinct from GeForce xx50–xx90 models.
            if (afterRTX && product.size() == 4 && product[0] >= '3' && product[0] <= '5' &&
                product[1] == '0' && product.substr(2) != "00" &&
                product.find_first_not_of("0123456789") == std::string_view::npos)
                return CUDAKernelDeviceProfile::A100;
            afterRTX = product == "RTX";
            name.remove_prefix(product.size());
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
        return ClassifyCUDAKernelDevice(properties.name);
#else
        DNDS_check_throw_info(false, "CUDA kernel defaults require a CUDA build");
        return CUDAKernelDeviceProfile::Other;
#endif
    }
    bool UsesA100KernelDefaults(Runtime::Device device)
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
        const bool measuredSAP2 = a100Polynomial && !acm && settings.mesh.dimension == 3 &&
                                  settings.equation.rans.model == "SA" && numerics.cfv.maxOrder == 2 &&
                                  numerics.RHSIntegrationOrder() == 3;
        // Coupled 3D SA P2/Q3 favors a larger CTA and PP block in complete
        // warmed updates. Preserve explicit mappings, arithmetic, staging and
        // launch overrides; the separate periodic RHS preset still applies.
        if (measuredSAP2 && !kernels.rhsBlockSize && !kernels.rhsDoubleBuffer &&
            !kernels.rhsHoistFaceRows.value_or(false) && kernels.rhsReduction == "auto" &&
            (kernels.rhsMapping == "auto" || kernels.rhsMapping == "ctaSegments") &&
            kernels.polynomialArithmetic == "fp64" && kernels.dissipationArithmetic == "fp64" &&
            kernels.fluxArithmetic == "fp64")
            kernels.rhsBlockSize = 256;
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
            execution.reconstructionKernel = a100Polynomial && !numerics.reconstructionSolver.UsesAMG() ? "fp64V2" : "fp64";
        }
        if (!kernels.polynomialPPThreads)
            kernels.polynomialPPThreads = measuredFlow5 || measuredSAP2 ? 512 : 128;
    }
    std::string ResolveAutomaticCUDAInverse(const std::string &selection, Runtime::Device device)
    {
        if (selection != "auto")
            return selection;
        // Guarded LU uses the existing condition guard and filtered SVD
        // fallback. It is an implementation choice, not a new inverse policy.
        return UsesA100KernelDefaults(device) ? "guardedLU" : "batchedSVD";
    }
}
