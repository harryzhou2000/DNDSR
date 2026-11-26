#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>
#include <cmath>
#include <fmt/core.h>

namespace DNDS::Math
{
    struct BisectResult
    {
        real value = 0;
        real lower = 0;
        real upper = 0;
        int iterations = 0;
        bool converged = false;
    };

    /// Monotone increasing bisection on [lower, upper] for F(x) == target.
    /// Stops early when stop(midpoint, fMid, lower, upper) returns true.
    template <class TF, class TStop>
    BisectResult BisectMonotone(TF &&F, real lower, real upper, real target, int maxIter, TStop &&stop)
    {
        BisectResult result{upper, lower, upper, 0, false};
        if (!(lower <= upper) || maxIter <= 0)
            return result;
        for (int iter = 1; iter <= maxIter; ++iter)
        {
            const real mid = real(0.5) * (lower + upper);
            const real fMid = F(mid);
            result.value = mid;
            result.lower = lower;
            result.upper = upper;
            result.iterations = iter;
            if (stop(mid, fMid, lower, upper))
            {
                result.converged = true;
                return result;
            }
            if (fMid < target)
                lower = mid;
            else
                upper = mid;
        }
        result.lower = lower;
        result.upper = upper;
        result.value = real(0.5) * (lower + upper);
        return result;
    }

    /// Monotone increasing lower-bracket bisection preserving legacy
    /// DNDS::Scalar::BisectSolveLower semantics while fixing early exit.
    template <class TF>
    BisectResult BisectSolveLowerDetailed(TF &&F, real v0, real v1, real fTarget, int maxIter)
    {
        real f0 = F(v0);
        const real f1 = F(v1);
        DNDS_assert_info(f0 <= fTarget && f1 >= fTarget && v0 < v1,
                         fmt::format(" v0 {} v1 {} f0 {} f1 {} fT {}", v0, v1, f0, f1, fTarget));
        BisectResult result{v0, v0, v1, 0, f0 >= fTarget - verySmallReal};
        for (int iter = 1; iter <= maxIter; ++iter)
        {
            if (f0 >= fTarget - verySmallReal)
            {
                result.converged = true;
                break;
            }
            const real vm = real(0.5) * (v0 + v1);
            const real fm = F(vm);
            if (fm <= fTarget)
            {
                v0 = vm;
                f0 = fm;
            }
            else
                v1 = vm;
            result.value = v0;
            result.lower = v0;
            result.upper = v1;
            result.iterations = iter;
        }
        if (f0 >= fTarget - verySmallReal)
            result.converged = true;
        return result;
    }
}
