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
        int switchShape = 0;
        real hillExponent = 3.0;
        int spatialPasses = 0;
        real spatialDecay = 0.65;
        real chiOverride = -1.0;

        DNDS_DECLARE_CONFIG(ReactiveSplitIndicatorSettings)
        {
            DNDS_FIELD(coupledThreshold, "Coupled-score midpoint", DNDS::Config::range(0.0));
            DNDS_FIELD(transitionWidth, "Smooth-switch width", DNDS::Config::range(0.0));
            DNDS_FIELD(shockScale, "Dimensionless pressure-jump scale", DNDS::Config::range(0.0));
            DNDS_FIELD(strangBias, "Multiplicative preference toward Strang", DNDS::Config::range(0.0));
            DNDS_FIELD(switchShape, "Switch shape: 0=logistic, 1=compact-tail Hill", DNDS::Config::range(0, 1));
            DNDS_FIELD(hillExponent, "Positive Hill switch exponent", DNDS::Config::range(0.0));
            DNDS_FIELD(spatialPasses, "Decaying coupled-fraction neighbor passes", DNDS::Config::range(0, 8));
            DNDS_FIELD(spatialDecay, "Coupled-fraction retention per neighbor pass", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(chiOverride, "Override chi in [0,1]; negative enables the indicator", DNDS::Config::range(-1.0, 1.0));
        }
    };

    inline real ReactiveSplitSaturate(real value)
    {
        value = std::max(value, real(0));
        return value / (1.0 + value);
    }

    inline real ReactiveSplitShockGate(real shockSensor, const ReactiveSplitIndicatorSettings &settings)
    {
        DNDS_check_throw_info(settings.shockScale > 0, "reactive split shockScale must be positive");
        real shockRatio = std::max(shockSensor, real(0)) / settings.shockScale;
        return 1.0 / (1.0 + std::pow(shockRatio, 4));
    }

    inline real ReactiveSplitCoupledScore(real chemicalStep, real diffusiveStep, real shockSensor,
                                          const ReactiveSplitIndicatorSettings &settings)
    {
        return ReactiveSplitSaturate(chemicalStep) * ReactiveSplitSaturate(diffusiveStep) *
               ReactiveSplitShockGate(shockSensor, settings);
    }

    inline real ReactiveSplitCoupledFraction(real coupledScore, const ReactiveSplitIndicatorSettings &settings)
    {
        DNDS_check_throw_info(settings.strangBias > 0, "reactive split strangBias must be positive");
        if (settings.switchShape == 0)
        {
            DNDS_check_throw_info(settings.transitionWidth > 0, "reactive split transitionWidth must be positive");
            real preferenceShift = std::log(settings.strangBias) * settings.transitionWidth;
            real argument = std::clamp(
                (coupledScore - settings.coupledThreshold - preferenceShift) / settings.transitionWidth,
                real(-60), real(60));
            return 1.0 / (1.0 + std::exp(-argument));
        }
        DNDS_check_throw_info(settings.switchShape == 1, "reactive split switchShape must be 0 or 1");
        DNDS_check_throw_info(settings.hillExponent > 0, "reactive split hillExponent must be positive");
        real nonnegativeScore = std::max(coupledScore, real(0));
        if (nonnegativeScore == 0)
            return 0;
        real effectiveThreshold = settings.coupledThreshold * settings.strangBias;
        if (effectiveThreshold <= 0)
            return 1;
        real argument = std::clamp(
            settings.hillExponent * std::log(nonnegativeScore / effectiveThreshold), real(-60), real(60));
        return 1.0 / (1.0 + std::exp(-argument));
    }

    inline real ReactiveSplitExpandedCoupledFraction(real localCoupledFraction, real neighborCoupledFraction,
                                                     real shockSensor,
                                                     const ReactiveSplitIndicatorSettings &settings)
    {
        DNDS_check_throw_info(settings.spatialDecay >= 0 && settings.spatialDecay <= 1,
                              "reactive split spatialDecay must be in [0,1]");
        real propagatedFraction = settings.spatialDecay * std::max(neighborCoupledFraction, real(0)) *
                                  ReactiveSplitShockGate(shockSensor, settings);
        return std::clamp(std::max(localCoupledFraction, propagatedFraction), real(0), real(1));
    }

    inline real ReactiveSplitChi(real coupledScore, const ReactiveSplitIndicatorSettings &settings)
    {
        if (settings.chiOverride >= 0)
            return std::clamp(settings.chiOverride, real(0), real(1));
        return 1.0 - ReactiveSplitCoupledFraction(coupledScore, settings);
    }
}
