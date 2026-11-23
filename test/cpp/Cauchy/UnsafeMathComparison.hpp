#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace CauchyTest
{
    inline bool EqualWithUnsafeMath(double actual, double expected)
    {
#ifdef DNDS_TEST_UNSAFE_MATH_OPT
        const double scale = std::max({1., std::abs(actual), std::abs(expected)});
        return std::abs(actual - expected) <= 8 * std::numeric_limits<double>::epsilon() * scale;
#else
        return actual == expected;
#endif
    }
}
