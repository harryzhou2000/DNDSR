#pragma once

#include "DNDS/Core/Defines.hpp"
#include <cmath>

namespace DNDS::CFV::Math
{
    /// Scalar legacy WBAP L2 biway, shared by polynomial/gradient CPU and
    /// device leaves. A zero neighbor suppresses the corresponding mode.
    DNDS_DEVICE_CALLABLE inline real WBAPBiway(real center, real other) noexcept
    {
        if (other == 0)
            return 0;
        const real epsilon = std::pow(verySmallReal, 0.25);
        const real ratio = center / (std::abs(other) + epsilon) * ((other > 0) - (other < 0));
        if (std::abs(ratio) > real(1e70))
            return center / ratio;
        const real r2 = ratio * ratio;
        return center * (1 + r2 * ratio) / (1 + r2 * r2);
    }

    /// Clamped WBAP ratio for multiway r^3 / r^4 accumulation without FP64 overflow.
    DNDS_DEVICE_CALLABLE inline real WBAPClampedRatio(real center, real value, real epsilon) noexcept
    {
        const real ratio = center / (std::abs(value) + epsilon) * ((value > 0) - (value < 0));
        return ratio > real(1e70) ? real(1e70) : (ratio < real(-1e70) ? real(-1e70) : ratio);
    }
}
