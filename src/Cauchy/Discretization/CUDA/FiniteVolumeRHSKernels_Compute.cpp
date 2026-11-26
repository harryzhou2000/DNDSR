#include "Cauchy/Discretization/CUDA/FiniteVolumeRHSKernels_Compute.hpp"

namespace DNDS::Cauchy::CUDA::Detail
{
    void LaunchComputeFluxFaces(Runtime::ExecutionContext context, const RHSGeometryPublication &geometry,
                                RHSComputeParametersView parameters, RHSKernelControls controls, RHSFields fields, RHSFaceOutputs output, RHSIndices selected)
    {
#ifdef DNDS_CAUCHY_CUDA_REDUCED_PRECISION
        using P = DNDS::Math::ComputePrecision;
        const auto precision = controls.polynomialPrecision != P::FP64 ? controls.polynomialPrecision : controls.fluxPrecision != P::FP64 ? controls.fluxPrecision
                                                                                                                                          : controls.dissipationPrecision;
        if (precision == P::FP32)
        {
            if (controls.fluxPrecision != P::FP64)
                LaunchFP32FluxFaces(context, geometry, parameters, controls, fields, output, selected);
            else if (controls.dissipationPrecision != P::FP64)
                LaunchFP32DissipationFluxFaces(context, geometry, parameters, controls, fields, output, selected);
            else
                LaunchFP32PolynomialFluxFaces(context, geometry, parameters, controls, fields, output, selected);
        }
#    ifdef DNDS_USE_CCCL_COMPUTE
        else if (precision == P::FloatFloat)
        {
            if (controls.fluxPrecision != P::FP64)
                LaunchFloatFloatFluxFaces(context, geometry, parameters, controls, fields, output, selected);
            else if (controls.dissipationPrecision != P::FP64)
                LaunchFloatFloatDissipationFluxFaces(context, geometry, parameters, controls, fields, output, selected);
            else
                LaunchFloatFloatPolynomialFluxFaces(context, geometry, parameters, controls, fields, output, selected);
        }
#    endif
        else
            DNDS_check_throw_info(false, "CUDA RHS compute precision is unavailable");
#else
        (void)context;
        (void)geometry;
        (void)parameters;
        (void)controls;
        (void)fields;
        (void)output;
        (void)selected;
        DNDS_check_throw_info(false, "CUDA RHS reduced-precision kernels were disabled at configure time (DNDS_CAUCHY_CUDA_REDUCED_PRECISION=OFF)");
#endif
    }
}
