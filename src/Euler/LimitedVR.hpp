#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Defines.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::Euler
{
    /** @brief Cellwise A-weighted penalty toward a limited explicit GG O2 target.
     * @details The shock fraction alpha=lambda/(1+lambda) multiplies the target;
     * zero leaves VR unchanged and one fixes this cell's row to the O2 target.
     * See docs/guides/limited_variational_reconstruction.md for all controls and
     * input presets. Numeric keys remain compatible with the historical studies.
     */
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

        /// jp0: zero response when Jp=2|pR-pL|/(pR+pL) is at/below this value.
        real pressureJumpStart = 0.10;
        /// jp1>jp0: cubic pressure response reaches one at this value.
        real pressureJumpFull = 0.22;
        /// jc0: zero response for Jc=max(unL-unR,0)/mean(cL,cR) at/below this value.
        real compressionStart = 0.08;
        /// jc1>jc0: cubic compression response reaches one at this value.
        real compressionFull = 0.16;
        /// Shock-branch cap in [0,1]; lambda=alpha/(1-alpha), with alpha=1 the fixed-target limit.
        real alphaMax = 0.50;
        /// Additional weak-shock branch cap; zero disables its gradients and halo work.
        real weakAlphaMax = 0.0;
        /// Start of cubic ramp for |dp-dpGradL+dpGradR|/(|dp|+|dpGradL|+|dpGradR|).
        real weakMismatchStart = 0.05;
        /// Saturation of the dimensionless pressure-gradient mismatch ramp.
        real weakMismatchFull = 0.20;
        /// Start of cubic ramp for compressive mean normal-velocity jump times mean rho*c divided by |dp|.
        real weakAcousticStart = 0.10;
        /// Saturation of the acoustic compression ramp; expansion/contact pressure noise is rejected.
        real weakAcousticFull = 0.50;
        /// Positive background-relative pressure-jump floor, multiplied by max(|pL|,|pR|).
        real weakRelativeFloor = 1e-10;
        /// Synchronous face-neighbour maximum passes on the freshly detected weak branch (0..4).
        int weakHaloLayers = 2;
        /// theta in (0,1]: theta*fresh+(1-theta)*old, later inner evaluations only.
        real weakAlphaUpdateRelaxation = 1.0;
        /// q in [0,1): fresh+q*max(old-fresh,0); immediate rises; requires theta=1; zero disables.
        real weakAlphaDecayRetention = 0.0;
        /// Fresh gate on every stage start, including explicit-first INT_MAX; hold only within that stage.
        bool weakFreezeWithinImplicitStage = false;
        /// jrho0: density-only cubic ramp start for Jrho=2|rhoR-rhoL|/(rhoR+rhoL).
        real contactDensityJumpStart = 0.20;
        /// jrho1>jrho0: density-only cubic ramp saturation.
        real contactDensityJumpFull = 0.80;
        /// Contact-branch cap; zero disables it; final alpha=max(shock,contact,weak).
        real contactAlphaMax = 0.0;
        /// Use adjacent means for the base density jump; false uses finalized O2 face states.
        bool contactUseCellMeanJump = false;
        /// Enable supported strict density-extremum response and retention/halo below.
        bool contactUseExtremumCorrection = false;
        /// Use mean-density jumps for support only, independently of the base contact gate.
        bool contactExtremumUseCellMeanSupport = false;
        /// Start of normalized density excess beyond nearest unequal neighbour at a strict extremum.
        real contactExtremumStart = 0.001;
        /// Saturation of density-extremum response; active only with contactUseExtremumCorrection.
        real contactExtremumFull = 0.008;
        /// Start of adjacent contact-support alpha/contactAlphaMax cubic ramp.
        real contactSupportStart = 0.02;
        /// Saturation of support ramp; multiply it by density-extremum response.
        real contactSupportFull = 0.15;
        /// Fraction of previous contact alpha retained while adjacent support is present; per RHS evaluation.
        real contactExtremumRetention = 0.95;
        /// Face-neighbour maximum passes of density correction only; H2 uses two; boundaries excluded.
        int contactExtremumHaloLayers = 1;
        /// 0=product, 8=v2 supported pressure extrema; 1..7 preserve historical union/rational/envelope experiments.
        int gateMode = GateProductCubic;
        /// Extra synchronous face-neighbour maximum passes on shock branch, after mode-8 correction.
        int gateHaloLayers = 0;
        /// When expanding, propagate only alphaMax-saturated cells; local unsaturated cells remain.
        bool gateHaloSaturatedOnly = false;
        /// theta in (0,1] for shock/contact per-RHS update; theta=1 removes this temporal lag.
        real alphaUpdateRelaxation = 1.0;
        /// Mode 8: start of normalized pressure excess beyond nearest unequal neighbour at a strict extremum.
        real oscillationExtremumStart = 0.001;
        /// Mode 8: saturation of pressure-extremum response.
        real oscillationExtremumFull = 0.008;
        /// Mode 8: start of adjacent shock alpha/alphaMax support ramp.
        real oscillationSupportStart = 0.02;
        /// Mode 8: saturation of support ramp; multiply by pressure-extremum response.
        real oscillationSupportFull = 0.15;
        /// Mode 8: previous shock alpha retention while support is active; per RHS evaluation.
        real oscillationRetention = 0.95;
        /// Relative equality tolerance for pressure/density neighbours, scaled by max absolute field.
        real oscillationTransverseTolerance = 1e-8;
        /// Minimum nearly equal face-neighbour count (not halo depth); exclude boundary-adjacent extrema.
        int oscillationMinTransverseNeighbours = 2;
        /// 0=Barth, 1=characteristic WBAP on GG O2 modes; same finalized target drives ordinary face sensing.
        int o2ReferenceLimiter = O2ReferenceLimiterBarth;
        /// Use cell means as PP bases; enforce face/required volume-point density and sensible-energy positivity.
        bool o2ReferenceUsePP = true;

        DNDS_DECLARE_CONFIG(LimitedVRSettings)
        {
            // clang-format off
            DNDS_FIELD(pressureJumpStart, "Pressure-jump ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(pressureJumpFull,  "Pressure-jump ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(compressionStart,  "Normal-compression ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(compressionFull,   "Normal-compression ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(alphaMax,         "Maximum O2 penalty fraction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakAlphaMax, "Maximum additional pressure-gradient-consistency weak-shock penalty fraction; zero disables the branch", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakMismatchStart, "Relative mean-pressure versus unrestricted-O2-gradient disagreement ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(weakMismatchFull, "Relative pressure-gradient disagreement ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(weakAcousticStart, "Compressive mean-velocity jump times acoustic impedance divided by pressure jump: ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(weakAcousticFull, "Compressive acoustic consistency ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(weakRelativeFloor, "Background-relative pressure noise floor for the additional weak branch", DNDS::Config::range(0.0));
            DNDS_FIELD(weakHaloLayers, "Face-neighbour expansion layers for the additional weak branch only", DNDS::Config::range(0, 4));
            DNDS_FIELD(weakAlphaUpdateRelaxation, "Temporal relaxation of the weak-shock alpha on later inner evaluations only; each new stage starts with a fresh gate", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakAlphaDecayRetention, "Fraction of the previous weak-alpha excess retained when the freshly evaluated weak response decreases; rises remain immediate and zero disables one-sided decay", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakFreezeWithinImplicitStage, "Evaluate a fresh weak-shock alpha at each stage start, including one-shot explicit-first stages, and hold it fixed only for later inner evaluations of that stage");
            DNDS_FIELD(contactDensityJumpStart, "Facial density-jump contact-gate ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(contactDensityJumpFull, "Facial density-jump contact-gate ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(contactAlphaMax, "Maximum O2 penalty fraction contributed by the density-only contact gate; zero disables it", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactUseCellMeanJump, "Evaluate the contact density jump from adjacent cell averages instead of limited-O2 reconstructed facial states");
            DNDS_FIELD(contactUseExtremumCorrection, "Add a contact-supported cell-mean density-extremum correction using the contact-specific extremum, support, retention, and halo controls");
            DNDS_FIELD(contactExtremumUseCellMeanSupport, "Evaluate density-extremum support from adjacent cell averages while retaining the selected reconstructed-face or cell-mean base contact gate");
            DNDS_FIELD(contactExtremumStart, "Normalized cell-mean density-extremum correction ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(contactExtremumFull, "Normalized cell-mean density-extremum correction ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(contactSupportStart, "Adjacent density-gate support fraction ramp start for the density-extremum correction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactSupportFull, "Adjacent density-gate support fraction ramp saturation for the density-extremum correction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactExtremumRetention, "Per-update retention of the density-extremum correction while adjacent density-gate support remains active", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactExtremumHaloLayers, "Number of face-neighbour layers receiving the density-extremum correction", DNDS::Config::range(0, 4));
            DNDS_FIELD(gateMode,         "LVR gate: 0=product cubic, 1=union cubic, 2=additive rational tail, 3=coupled rational tail, 4=coupled rational tail with full-threshold saturation, 5=cross-cell coupled rational tail, 6=one-neighbour indicator-envelope rational tail, 7=one-neighbour indicator-envelope cubic product, 8=product cubic with shock-supported cell-mean pressure-extremum correction", DNDS::Config::range(0, 8));
            DNDS_FIELD(gateHaloLayers,   "Number of face-neighbour cell layers added around the detected LVR gate", DNDS::Config::range(0, 4));
            DNDS_FIELD(gateHaloSaturatedOnly, "Expand only the saturated shock core and set its halo to alphaMax");
            DNDS_FIELD(alphaUpdateRelaxation, "Temporal relaxation applied when updating the LVR alpha field", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationExtremumStart, "Normalized cell-mean pressure-extremum ramp start for gate mode 8", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationExtremumFull, "Normalized cell-mean pressure-extremum ramp saturation for gate mode 8", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationSupportStart, "Adjacent shock-gate alpha fraction ramp start for the mode-8 pressure correction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationSupportFull, "Adjacent shock-gate alpha fraction ramp saturation for the mode-8 pressure correction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationRetention, "Per-update retention of the mode-8 pressure correction while adjacent shock-gate support remains active", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationTransverseTolerance, "Relative field tolerance used by pressure- and density-extremum equal-state-neighbour tests", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationMinTransverseNeighbours, "Minimum equal-state transverse face neighbours required by pressure- and density-extremum corrections", DNDS::Config::range(1, 4));
            DNDS_FIELD(o2ReferenceLimiter, "Limited O2 reference limiter: 0=Barth, 1=WBAP", DNDS::Config::range(0, 1));
            DNDS_FIELD(o2ReferenceUsePP,   "Compress the limited O2 reference toward the cell mean to preserve facial positivity");
            // clang-format on
            config.check("limitedVR full thresholds must exceed start thresholds", [](const T &s)
                         { return s.pressureJumpFull > s.pressureJumpStart &&
                                  s.compressionFull > s.compressionStart &&
                                  s.contactDensityJumpFull > s.contactDensityJumpStart; });
            config.check("limitedVR alphaUpdateRelaxation must be positive", [](const T &s)
                         { return s.alphaUpdateRelaxation > 0; });
            config.check("limitedVR weak thresholds must be ordered and noise floor positive", [](const T &s)
                         { return s.weakMismatchFull > s.weakMismatchStart &&
                                  s.weakAcousticFull > s.weakAcousticStart && s.weakRelativeFloor > 0 &&
                                  s.weakAlphaUpdateRelaxation > 0 && s.weakAlphaDecayRetention < 1 &&
                                  (s.weakAlphaDecayRetention == 0 || s.weakAlphaUpdateRelaxation == 1); });
            config.check("limitedVR oscillation full thresholds must exceed start thresholds", [](const T &s)
                         { return s.oscillationExtremumFull > s.oscillationExtremumStart &&
                                  s.oscillationSupportFull > s.oscillationSupportStart; });
            config.check("limitedVR contact-extremum full thresholds must exceed start thresholds", [](const T &s)
                         { return s.contactExtremumFull > s.contactExtremumStart &&
                                  s.contactSupportFull > s.contactSupportStart; });
        }
    };

    /** @brief Expand an optional input-only preset before merging this input layer.
     * @details Presets select gate/target controls only: users still select
     * limiterProcedure=2, reconstruction order and derivative weights. Explicit
     * numeric keys in the same layer override the preset, and later layers win.
     * The input-only key is removed so saved/default JSON stays fully explicit.
     */
    inline void ExpandLimitedVRPreset(nlohmann::ordered_json &input)
    {
        if (!input.contains("limiterControl") || !input["limiterControl"].is_object() ||
            !input["limiterControl"].contains("limitedVR"))
            return;
        auto &section = input["limiterControl"]["limitedVR"];
        if (!section.is_object() || !section.contains("preset"))
            return;
        const auto name = section.at("preset").get<std::string>();
        DNDS_check_throw_info(name == "v1" || name == "v2" || name == "v3",
                              "limitedVR.preset must be v1, v2, or v3 (Average H2)");
        nlohmann::ordered_json values = LimitedVRSettings{};
        if (name != "v1")
        {
            values["o2ReferenceLimiter"] = LimitedVRSettings::O2ReferenceLimiterWBAP;
            values["gateMode"] = LimitedVRSettings::GateProductCubicWithExtremumCorrection;
            values["pressureJumpStart"] = 0.04;
            values["pressureJumpFull"] = 0.16;
            values["compressionStart"] = 0.03;
            values["compressionFull"] = 0.13;
        }
        if (name == "v3")
        {
            values["contactAlphaMax"] = 0.5;
            values["contactDensityJumpStart"] = 0.5;
            values["contactDensityJumpFull"] = 0.8;
            values["contactUseExtremumCorrection"] = true;
            values["contactExtremumUseCellMeanSupport"] = true;
            values["contactSupportStart"] = 0.005;
            values["contactSupportFull"] = 0.06;
            values["contactExtremumHaloLayers"] = 2;
        }
        section.erase("preset");
        values.merge_patch(section);
        section = std::move(values);
    }

    struct LimitedVRFaceSensor
    {
        real pressureJump = 0;
        real compression = 0;
        real alpha = 0;
    };

    struct LimitedVRContactFaceSensor
    {
        real densityJump = 0;
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

    inline LimitedVRContactFaceSensor EvaluateLimitedVRContactFaceSensor(
        real densityLeft, real densityRight,
        const LimitedVRSettings &settings)
    {
        LimitedVRContactFaceSensor result;
        real densityScale = std::max({std::abs(densityLeft), std::abs(densityRight), real(1e-30)});
        real densityDenominator = std::max(densityLeft + densityRight, smallReal * densityScale);
        result.densityJump = 2.0 * std::abs(densityRight - densityLeft) / densityDenominator;
        result.alpha = settings.contactAlphaMax * LimitedVRCubicRamp(
                                                      result.densityJump,
                                                      settings.contactDensityJumpStart,
                                                      settings.contactDensityJumpFull);
        return result;
    }

    inline real CombineLimitedVRAlpha(real shockAlpha, real contactAlpha)
    {
        return std::max(shockAlpha, contactAlpha);
    }

    inline real RelaxLimitedVRAlpha(real previousAlpha, real updatedAlpha, real relaxation)
    {
        DNDS_check_throw_info(relaxation > 0 && relaxation <= 1,
                              "limitedVR alpha update relaxation must lie in (0, 1]");
        return relaxation * updatedAlpha + (1.0 - relaxation) * previousAlpha;
    }

    inline bool IsLimitedVRStageStartIteration(int iter)
    {
        // Iterative stages start at one. Explicit-first ESDIRK stages use the
        // maximum integer as their one-shot RHS-evaluation sentinel.
        return iter <= 1 || iter == std::numeric_limits<int>::max();
    }

    inline real UpdateLimitedVRWeakAlpha(
        real previousAlpha, real updatedAlpha,
        bool refresh, bool reset,
        real relaxation, real decayRetention = 0)
    {
        DNDS_check_throw_info(decayRetention >= 0 && decayRetention < 1,
                              "limitedVR weak alpha decay retention must lie in [0, 1)");
        if (!refresh)
            return previousAlpha;
        if (reset)
            return updatedAlpha;
        if (decayRetention > 0)
            return updatedAlpha + decayRetention * std::max(previousAlpha - updatedAlpha, real(0));
        return RelaxLimitedVRAlpha(previousAlpha, updatedAlpha, relaxation);
    }

    /** @brief Additional weak-shock response from means and unrestricted explicit gradients.
     * pressureIncrementLeft/Right are each cell's primitive pressure-gradient
     * increment from its centroid to the shared face, in that cell's frame.
     * Exact linear/quadratic profiles on a uniform strip with centered GG
     * slopes have zero mismatch. Cubic and higher profiles can have a nonzero
     * third-difference mismatch. No derivative of the unknown high-order
     * polynomial enters this sensor.
     */
    inline real EvaluateLimitedVRWeakAlpha(
        real pressureLeft, real pressureRight,
        real pressureIncrementLeft, real pressureIncrementRight,
        real normalVelocityLeft, real normalVelocityRight,
        real acousticImpedance, const LimitedVRSettings &settings)
    {
        if (settings.weakAlphaMax == 0)
            return 0;
        real jump = pressureRight - pressureLeft;
        real floor = settings.weakRelativeFloor *
                     std::max({std::abs(pressureLeft), std::abs(pressureRight), real(1e-30)});
        if (std::abs(jump) <= floor || normalVelocityLeft <= normalVelocityRight)
            return 0;
        real mismatch = std::abs(jump - pressureIncrementLeft + pressureIncrementRight) /
                        std::max(std::abs(jump) + std::abs(pressureIncrementLeft) +
                                     std::abs(pressureIncrementRight),
                                 floor);
        real acoustic = (normalVelocityLeft - normalVelocityRight) * acousticImpedance /
                        std::abs(jump);
        return settings.weakAlphaMax *
               LimitedVRCubicRamp(mismatch, settings.weakMismatchStart, settings.weakMismatchFull) *
               LimitedVRCubicRamp(acoustic, settings.weakAcousticStart, settings.weakAcousticFull);
    }
}
