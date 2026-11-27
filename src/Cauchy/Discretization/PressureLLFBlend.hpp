#pragma once
#include "Cauchy/Discretization/FiniteVolumeKernelPolicy.hpp"

namespace DNDS::Cauchy::FiniteVolumeDetail
{
    DNDS_DEVICE_CALLABLE inline real PressureRatioWeight(real minimum, real maximum, real start, real full) noexcept
    {
        const real ratio = maximum / minimum;
        if (ratio <= start)
            return 0;
        if (ratio >= full)
            return 1;
        const real x = (ratio - start) / (full - start);
        return x * x * (3 - 2 * x);
    }

    /// Combine packed [min,max] rows on the two sides of a face. The cap on
    /// the AUSMPW+ minimum does not change the physical extrema for blending.
    template <class PressureRows>
    DNDS_DEVICE_CALLABLE inline real ApplyFacePressureStencil(
        const FluxKernelPolicy &policy, bool useStencil, const PressureRows &bounds,
        index own, index other, Riemann::FluxSettings &settings) noexcept
    {
        real minimum = bounds(own, 0);
        if (other != UnInitIndex)
            minimum = DNDS::Math::ScalarMin(minimum, bounds(other, 0));
        if (useStencil && policy.pressureStencil)
            settings.pressureStencilMin = DNDS::Math::ScalarMin(settings.pressureStencilMin, minimum);
        if (!policy.pressureLLFBlend && settings.dissipationScale == 1)
            return 0;
        real maximum = bounds(own, 1);
        if (other != UnInitIndex)
            maximum = DNDS::Math::ScalarMax(maximum, bounds(other, 1));
        const real weight = PressureRatioWeight(minimum, maximum, policy.pressureRatioStart, policy.pressureRatioFull);
        settings.dissipationScale += weight * (1 - settings.dissipationScale);
        return policy.pressureLLFBlend ? weight : 0;
    }

    DNDS_DEVICE_CALLABLE inline real BlendFluxValue(real flux, real llf, real weight) noexcept
    {
        return weight == 1 ? llf : (1 - weight) * flux + weight * llf;
    }

    /// Blend after directional recombination, using the original face normal
    /// and the same admissible face states. The separate PP flux stays intact.
    DNDS_DEVICE_CALLABLE inline void BlendFlowToLLF(FlowState &flux, real weight,
                                                    const Riemann::AcousticState &left, const Riemann::AcousticState &right,
                                                    const Vector3 &normal, const Vector3 &grid) noexcept
    {
        if (weight == 0)
            return;
        const auto llf = Riemann::Rusanov(left, right, normal, grid);
        for (int j = 0; j < 5; ++j)
            flux[j] = BlendFluxValue(flux[j], llf[j], weight);
    }

}
