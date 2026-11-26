#pragma once

#include "DNDS/Math/Bisection.hpp"
#include "DNDS/Math/HardEigen.hpp"
#include "DNDS/Core/Defines.hpp"
#include "fmt/core.h"

namespace DNDS::Scalar
{
    template <class TF>
    real BisectSolveLower(TF &&F, real v0, real v1, real fTarget, int maxIter)
    {
        return DNDS::Math::BisectSolveLowerDetailed(std::forward<TF>(F), v0, v1, fTarget, maxIter).value;
    }

}