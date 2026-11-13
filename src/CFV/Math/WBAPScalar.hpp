#pragma once

#include "DNDS/Core/Defines.hpp"
#include <cmath>

namespace DNDS::CFV::Math
{
    /// Scalar legacy WBAP L2 biway, shared by polynomial/gradient CPU and
    /// device leaves. A zero neighbor suppresses the corresponding mode.
    DNDS_DEVICE_CALLABLE inline real WBAPBiway(real center, real other) noexcept
    {
        const real epsilon = std::pow(verySmallReal, 0.25);
        const real ratio = center / (std::abs(other) + epsilon) * ((other > 0) - (other < 0));
        const real r2 = ratio * ratio;
        return other == 0 ? 0 : center * (1 + r2 * ratio) / (1 + r2 * r2);
    }
}
