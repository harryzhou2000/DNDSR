#pragma once

#include "CauchyBuildFeatures.hpp"
#include "DNDS/Core/Errors.hpp"

namespace DNDS::Cauchy
{
    inline constexpr bool BuiltForEquation(bool acm, int dimension) noexcept
    {
        return (acm ? DNDS_CAUCHY_ENABLE_ACM : DNDS_CAUCHY_ENABLE_COMPRESSIBLE) &&
               ((dimension == 2 && DNDS_CAUCHY_ENABLE_2D) || (dimension == 3 && DNDS_CAUCHY_ENABLE_3D));
    }

    inline void RequireBuiltEquation(bool acm, int dimension)
    {
        DNDS_check_throw_info(acm ? DNDS_CAUCHY_ENABLE_ACM : DNDS_CAUCHY_ENABLE_COMPRESSIBLE,
                              acm ? "Cauchy built without ACM; enable DNDS_CAUCHY_ENABLE_ACM" : "Cauchy built without compressible equations; enable DNDS_CAUCHY_ENABLE_COMPRESSIBLE");
        DNDS_check_throw_info(BuiltForEquation(acm, dimension),
                              "Unsupported Cauchy mesh dimension " + std::to_string(dimension) +
                                  "; check DNDS_CAUCHY_ENABLE_2D and DNDS_CAUCHY_ENABLE_3D");
    }
}
