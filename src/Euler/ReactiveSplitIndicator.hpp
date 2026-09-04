#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Defines.hpp"

namespace DNDS::Euler
{
    struct ReactiveSplitIndicatorSettings
    {
        real coupledThreshold = 0.005;
        real transitionWidth = 0.001;
        real shockScale = 0.08;
        real strangBias = 1.0;
        real chiOverride = -1.0;

        DNDS_DECLARE_CONFIG(ReactiveSplitIndicatorSettings)
        {
            DNDS_FIELD(coupledThreshold, "Coupled-score midpoint", DNDS::Config::range(0.0));
            DNDS_FIELD(transitionWidth, "Smooth-switch width", DNDS::Config::range(0.0));
            DNDS_FIELD(shockScale, "Dimensionless pressure-jump scale", DNDS::Config::range(0.0));
            DNDS_FIELD(strangBias, "Multiplicative preference toward Strang", DNDS::Config::range(0.0));
            DNDS_FIELD(chiOverride, "Override chi in [0,1]; negative enables the indicator", DNDS::Config::range(-1.0, 1.0));
        }
    };

    inline real ReactiveSplitSaturate(real value)
    {
        value = std::max(value, real(0));
        return value / (1.0 + value);
    }

    inline real ReactiveSplitCoupledScore(real chemicalStep, real diffusiveStep, real shockSensor,
                                          const ReactiveSplitIndicatorSettings &settings)
    {
        DNDS_check_throw_info(settings.shockScale > 0, "reactive split shockScale must be positive");
        real shockRatio = std::max(shockSensor, real(0)) / settings.shockScale;
        real shockGate = 1.0 / (1.0 + std::pow(shockRatio, 4));
        return ReactiveSplitSaturate(chemicalStep) * ReactiveSplitSaturate(diffusiveStep) * shockGate;
    }

    inline real ReactiveSplitChi(real coupledScore, const ReactiveSplitIndicatorSettings &settings)
    {
        DNDS_check_throw_info(settings.transitionWidth > 0, "reactive split transitionWidth must be positive");
        DNDS_check_throw_info(settings.strangBias > 0, "reactive split strangBias must be positive");
        if (settings.chiOverride >= 0)
            return std::clamp(settings.chiOverride, real(0), real(1));
        real preferenceShift = std::log(settings.strangBias) * settings.transitionWidth;
        real argument = std::clamp(
            (coupledScore - settings.coupledThreshold - preferenceShift) / settings.transitionWidth,
            real(-60), real(60));
        return 1.0 - 1.0 / (1.0 + std::exp(-argument));
    }
}
