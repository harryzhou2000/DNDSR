#pragma once

#include "DNDS/Core/Defines.hpp"
#include <array>

namespace DNDS::Cauchy
{
    /// SSPRK convex-combination coefficients; residuals are cell integrated.
    struct ExplicitStageCoefficients
    {
        real dt = 0, historyWeight = 0, stageWeight = 1;
    };
    /// Rank-local flow5 conservative flux-blend decision. The controller must
    /// agree alpha/safe over its domain before applying one common factor.
    struct ExplicitBlendStatistics
    {
        real alpha = 1;
        index limited = 0;
        std::array<int, 2> safe{1, 1}; // high-order / low-order endpoint
    };
}
