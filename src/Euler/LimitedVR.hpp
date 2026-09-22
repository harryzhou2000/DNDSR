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
        static constexpr int GateProductCubic = 0;
        static constexpr int GateUnionCubic = 1;
        static constexpr int GateCombinedRational = 2;
        static constexpr int GateCoupledRational = 3;
        static constexpr int GateCoupledRationalSaturated = 4;
        static constexpr int GateNeighborCoupledRationalSaturated = 5;
        static constexpr int GateNeighborhoodEnvelopeRationalSaturated = 6;
        static constexpr int GateNeighborhoodEnvelopeCubic = 7;
        static constexpr int GateProductCubicWithExtremumCorrection = 8;

        real pressureJumpStart = 0.10;
        real pressureJumpFull = 0.22;
        real compressionStart = 0.08;
        real compressionFull = 0.16;
        real alphaMax = 0.50;
        int gateMode = GateProductCubic;
        int gateHaloLayers = 0;
        bool gateHaloSaturatedOnly = false;
        real alphaUpdateRelaxation = 1.0;
        real oscillationExtremumStart = 0.001;
        real oscillationExtremumFull = 0.008;
        real oscillationSupportStart = 0.02;
        real oscillationSupportFull = 0.15;
        real oscillationRetention = 0.95;
        real oscillationTransverseTolerance = 1e-8;
        int oscillationMinTransverseNeighbours = 2;
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
            DNDS_FIELD(gateMode,         "LVR gate: 0=product cubic, 1=union cubic, 2=additive rational tail, 3=coupled rational tail, 4=coupled rational tail with full-threshold saturation, 5=cross-cell coupled rational tail, 6=one-neighbour indicator-envelope rational tail, 7=one-neighbour indicator-envelope cubic product, 8=product cubic with shock-supported cell-mean pressure-extremum correction", DNDS::Config::range(0, 8));
            DNDS_FIELD(gateHaloLayers,   "Number of face-neighbour cell layers added around the detected LVR gate", DNDS::Config::range(0, 4));
            DNDS_FIELD(gateHaloSaturatedOnly, "Expand only the saturated shock core and set its halo to alphaMax");
            DNDS_FIELD(alphaUpdateRelaxation, "Temporal relaxation applied when updating the LVR alpha field", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationExtremumStart, "Normalized cell-mean pressure-extremum ramp start for gate mode 8", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationExtremumFull, "Normalized cell-mean pressure-extremum ramp saturation for gate mode 8", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationSupportStart, "Adjacent product-gate alpha fraction ramp start for gate mode 8", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationSupportFull, "Adjacent product-gate alpha fraction ramp saturation for gate mode 8", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationRetention, "Per-update retention of mode-8 correction while adjacent product-gate support remains active", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationTransverseTolerance, "Relative pressure tolerance used to identify a transverse equal-state neighbour for gate mode 8", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationMinTransverseNeighbours, "Minimum equal-state transverse face neighbours required by gate mode 8", DNDS::Config::range(1, 4));
            DNDS_FIELD(o2ReferenceLimiter, "Limited O2 reference limiter: 0=Barth, 1=WBAP", DNDS::Config::range(0, 1));
            DNDS_FIELD(o2ReferenceUsePP,   "Compress the limited O2 reference toward the cell mean to preserve facial positivity");
            // clang-format on
            config.check("limitedVR full thresholds must exceed start thresholds", [](const T &s)
                         { return s.pressureJumpFull > s.pressureJumpStart &&
                                  s.compressionFull > s.compressionStart; });
            config.check("limitedVR alphaUpdateRelaxation must be positive", [](const T &s)
                         { return s.alphaUpdateRelaxation > 0; });
            config.check("limitedVR oscillation full thresholds must exceed start thresholds", [](const T &s)
                         { return s.oscillationExtremumFull > s.oscillationExtremumStart &&
                                  s.oscillationSupportFull > s.oscillationSupportStart; });
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

    inline real LimitedVRNormalizedExcess(real value, real start, real full)
    {
        DNDS_check_throw_info(full > start, "limitedVR ramp full threshold must exceed start");
        return std::max(real(0), (value - start) / (full - start));
    }

    inline real LimitedVRCoupledSaturatedResponse(real firstExcess, real secondExcess)
    {
        real coupledExcess = std::sqrt(firstExcess * secondExcess);
        return std::min(real(1), 2.0 * coupledExcess / (1.0 + coupledExcess));
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
        real pressureResponse = LimitedVRCubicRamp(
            result.pressureJump, settings.pressureJumpStart, settings.pressureJumpFull);
        real compressionResponse = LimitedVRCubicRamp(
            result.compression, settings.compressionStart, settings.compressionFull);
        real gateResponse = 0;
        switch (settings.gateMode)
        {
        case LimitedVRSettings::GateProductCubic:
            gateResponse = pressureResponse * compressionResponse;
            break;
        case LimitedVRSettings::GateUnionCubic:
            gateResponse = 1.0 - (1.0 - pressureResponse) * (1.0 - compressionResponse);
            break;
        case LimitedVRSettings::GateCombinedRational:
        {
            real combinedExcess =
                LimitedVRNormalizedExcess(result.pressureJump, settings.pressureJumpStart, settings.pressureJumpFull) +
                LimitedVRNormalizedExcess(result.compression, settings.compressionStart, settings.compressionFull);
            gateResponse = combinedExcess / (1.0 + combinedExcess);
            break;
        }
        case LimitedVRSettings::GateCoupledRational:
        {
            // Couple the two indicators before the long-tail map.  The geometric
            // mean retains an exact zero unless both shock signatures exceed
            // their starts, while q / (1 + q) avoids the weak near-start response
            // of multiplying two cubic ramps.
            real pressureExcess = LimitedVRNormalizedExcess(
                result.pressureJump, settings.pressureJumpStart, settings.pressureJumpFull);
            real compressionExcess = LimitedVRNormalizedExcess(
                result.compression, settings.compressionStart, settings.compressionFull);
            real coupledExcess = std::sqrt(pressureExcess * compressionExcess);
            gateResponse = coupledExcess / (1.0 + coupledExcess);
            break;
        }
        case LimitedVRSettings::GateCoupledRationalSaturated:
        case LimitedVRSettings::GateNeighborCoupledRationalSaturated:
        case LimitedVRSettings::GateNeighborhoodEnvelopeRationalSaturated:
        {
            real pressureExcess = LimitedVRNormalizedExcess(
                result.pressureJump, settings.pressureJumpStart, settings.pressureJumpFull);
            real compressionExcess = LimitedVRNormalizedExcess(
                result.compression, settings.compressionStart, settings.compressionFull);
            // This rational tail is continuous and reaches an exact plateau at
            // the declared full thresholds (coupledExcess == 1).
            gateResponse = LimitedVRCoupledSaturatedResponse(pressureExcess, compressionExcess);
            break;
        }
        case LimitedVRSettings::GateNeighborhoodEnvelopeCubic:
        case LimitedVRSettings::GateProductCubicWithExtremumCorrection:
            gateResponse = pressureResponse * compressionResponse;
            break;
        default:
            DNDS_assert_info(false, "invalid limitedVR gateMode");
        }
        result.alpha = settings.alphaMax * gateResponse;
        return result;
    }
}
