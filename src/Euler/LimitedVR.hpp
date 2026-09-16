#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Defines.hpp"

#include <algorithm>
#include <cmath>

namespace DNDS::Euler
{
    struct LimitedVRSettings
    {
        static constexpr int O2ReferenceLimiterBarth = 0;
        static constexpr int O2ReferenceLimiterWBAP = 1;

        real pressureJumpStart = 0.10;
        real pressureJumpFull = 0.22;
        real compressionStart = 0.08;
        real compressionFull = 0.16;
        real alphaMax = 0.50;
        int o2ReferenceLimiter = O2ReferenceLimiterBarth;
        bool o2ReferenceUsePP = true;

        DNDS_DECLARE_CONFIG(LimitedVRSettings)
        {
            // clang-format off
            DNDS_FIELD(pressureJumpStart, "Pressure-jump ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(pressureJumpFull,  "Pressure-jump ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(compressionStart,  "Normal-compression ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(compressionFull,   "Normal-compression ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(alphaMax,         "Maximum O2 penalty fraction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(o2ReferenceLimiter, "Limited O2 reference limiter: 0=Barth, 1=WBAP", DNDS::Config::range(0, 1));
            DNDS_FIELD(o2ReferenceUsePP,   "Compress the limited O2 reference toward the cell mean to preserve facial positivity");
            // clang-format on
            config.check("limitedVR full thresholds must exceed start thresholds", [](const T &s)
                         { return s.pressureJumpFull > s.pressureJumpStart &&
                                  s.compressionFull > s.compressionStart; });
        }
    };

    struct LimitedVRFaceSensor
    {
        real pressureJump = 0;
        real compression = 0;
        real alpha = 0;
    };

    inline real LimitedVRCubicRamp(real value, real start, real full)
    {
        DNDS_check_throw_info(full > start, "limitedVR ramp full threshold must exceed start");
        if (value <= start)
            return 0;
        if (value >= full)
            return 1;
        real q = (value - start) / (full - start);
        return q * q * (3.0 - 2.0 * q);
    }

    inline LimitedVRFaceSensor EvaluateLimitedVRFaceSensor(
        real pressureLeft, real pressureRight,
        real normalVelocityLeft, real normalVelocityRight,
        real soundSpeedLeft, real soundSpeedRight,
        const LimitedVRSettings &settings)
    {
        LimitedVRFaceSensor result;
        real pressureScale = std::max({std::abs(pressureLeft), std::abs(pressureRight), real(1e-30)});
        real pressureDenominator = std::max(pressureLeft + pressureRight, smallReal * pressureScale);
        real acousticScale = std::max({std::abs(soundSpeedLeft), std::abs(soundSpeedRight), real(1e-30)});
        real acousticDenominator = std::max(0.5 * (soundSpeedLeft + soundSpeedRight), smallReal * acousticScale);
        result.pressureJump = 2.0 * std::abs(pressureRight - pressureLeft) / pressureDenominator;
        result.compression = std::max(real(0), normalVelocityLeft - normalVelocityRight) / acousticDenominator;
        result.alpha = settings.alphaMax *
                       LimitedVRCubicRamp(result.pressureJump, settings.pressureJumpStart, settings.pressureJumpFull) *
                       LimitedVRCubicRamp(result.compression, settings.compressionStart, settings.compressionFull);
        return result;
    }
}
