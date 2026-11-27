#pragma once

#include "CauchyBuildFeatures.hpp"
#include "DNDS/Core/Errors.hpp"

namespace DNDS::Cauchy
{
    inline constexpr bool BuiltForEquation(bool acm, int dimension) noexcept
    {
        const bool enabledFamily = (acm && DNDS_CAUCHY_ENABLE_ACM != 0) ||
                                   (!acm && DNDS_CAUCHY_ENABLE_COMPRESSIBLE != 0);
        return enabledFamily &&
               ((dimension == 2 && DNDS_CAUCHY_ENABLE_2D != 0) ||
                (dimension == 3 && DNDS_CAUCHY_ENABLE_3D != 0));
    }

    inline void RequireBuiltEquation(bool acm, int dimension)
    {
        const bool enabledFamily = (acm && DNDS_CAUCHY_ENABLE_ACM != 0) ||
                                   (!acm && DNDS_CAUCHY_ENABLE_COMPRESSIBLE != 0);
        DNDS_check_throw_info(enabledFamily,
                              acm ? "Cauchy built without ACM; enable DNDS_CAUCHY_ENABLE_ACM"
                                  : "Cauchy built without compressible equations; enable DNDS_CAUCHY_ENABLE_COMPRESSIBLE");
        DNDS_check_throw_info(BuiltForEquation(acm, dimension),
                              "Unsupported Cauchy mesh dimension " + std::to_string(dimension) +
                                  "; check DNDS_CAUCHY_ENABLE_2D and DNDS_CAUCHY_ENABLE_3D");
    }
}
