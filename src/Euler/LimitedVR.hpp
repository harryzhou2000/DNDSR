#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Config/ConfigEnum.hpp"
#include "DNDS/Defines.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace DNDS::Euler
{
    /// Reconstruction treatment selected by limiterControl.limiterProcedure.
    enum class ReconstructionLimiter
    {
        Unknown = -1,
        WBAP3 = 0,
        CWBAP = 1,
        LVR = 2,
    };
    DNDS_DEFINE_ENUM_JSON(ReconstructionLimiter, {{ReconstructionLimiter::Unknown, nullptr},
                                                  {ReconstructionLimiter::WBAP3, "3WBAP"},
                                                  {ReconstructionLimiter::CWBAP, "CWBAP"},
                                                  {ReconstructionLimiter::LVR, "LVR"}})

    /// Limiter applied to the explicit Green--Gauss linear reference polynomial.
    enum class O2ReferenceLimiter
    {
        Unknown = -1,
        Barth = 0,
        WBAP = 1,
    };
    DNDS_DEFINE_ENUM_JSON(O2ReferenceLimiter, {{O2ReferenceLimiter::Unknown, nullptr},
                                               {O2ReferenceLimiter::Barth, "Barth"},
                                               {O2ReferenceLimiter::WBAP, "WBAP"}})

    /// Both modes use the same face product gate; PressureExtremum adds the v2 correction.
    enum class LimitedVRGateMode
    {
        Unknown = -1,
        ProductCubic = 0,
        PressureExtremum = 8,
    };
    DNDS_DEFINE_ENUM_JSON(LimitedVRGateMode, {{LimitedVRGateMode::Unknown, nullptr},
                                              {LimitedVRGateMode::ProductCubic, "ProductCubic"},
                                              {LimitedVRGateMode::PressureExtremum, "PressureExtremum"}})

    namespace detail
    {
        /// Read only named choices or their exact legacy integer aliases.
        /// The standard enum macro silently maps unknown inputs to its first entry;
        /// these ordered-JSON overloads instead reject typos and retired experiments.
        template <typename Enum>
        void ReadReconstructionEnum(const nlohmann::ordered_json &input, Enum &value,
                                    std::initializer_list<Enum> choices, const char *message)
        {
            for (const auto choice : choices)
                if ((input.is_string() && input == nlohmann::ordered_json(choice)) ||
                    (input.is_number_integer() && input == static_cast<int>(choice)))
                {
                    value = choice;
                    return;
                }
            DNDS_check_throw_info(false, message);
        }
    }

    inline void from_json(const nlohmann::ordered_json &input, ReconstructionLimiter &value)
    {
        detail::ReadReconstructionEnum(input, value,
                                       {ReconstructionLimiter::WBAP3, ReconstructionLimiter::CWBAP, ReconstructionLimiter::LVR},
                                       "limiterProcedure must be 3WBAP, CWBAP, or LVR (legacy integers: 0, 1, 2)");
    }

    inline void from_json(const nlohmann::ordered_json &input, O2ReferenceLimiter &value)
    {
        detail::ReadReconstructionEnum(input, value, {O2ReferenceLimiter::Barth, O2ReferenceLimiter::WBAP},
                                       "o2ReferenceLimiter must be Barth or WBAP (legacy integers: 0, 1)");
    }

    inline void from_json(const nlohmann::ordered_json &input, LimitedVRGateMode &value)
    {
        detail::ReadReconstructionEnum(input, value, {LimitedVRGateMode::ProductCubic, LimitedVRGateMode::PressureExtremum},
                                       "gateMode must be ProductCubic or PressureExtremum (legacy integers: 0, 8); experimental modes 1..7 were removed");
    }

    /** @brief Cellwise A-weighted penalty toward a limited explicit GG O2 target.
     * @details For a fixed cell mean, A is the existing VR diagonal block and t
     * contains the limited linear coefficients, with all higher modes zero.
     * Adding lambda/2*(r-t)^T*A*(r-t) changes the block update to
     * (1-alpha)*r_VR+alpha*t, before the existing Jacobi/SOR relaxation, where
     * alpha=lambda/(1+lambda). Zero leaves VR unchanged; one fixes its row to t.
     * Final alpha is the maximum of the shock, contact and optional weak branches.
     * The target and gate are rebuilt per flow RHS evaluation, then held fixed
     * throughout that reconstruction solve: the system remains linear in r.
     *
     * Jp=2*|pR-pL|/(pR+pL), Jc=max(unL-unR,0)/mean(cL,cR), and
     * Jrho=2*|rhoR-rhoL|/(rhoR+rhoL) use finalized limited-O2 face states unless
     * a contact option explicitly selects means. un uses a common L-to-R normal;
     * c is sound speed. detectExpansion replaces max(unL-unR,0) by abs(unL-unR)
     * in Jc only; pressure gating is still required and the optional weak branch
     * remains compression-only. Every start/full pair defines a cubic 0-to-1 ramp.
     * For scalar z and thresholds z0<z1, that ramp is zero below z0, one above
     * z1, and q*q*(3-2*q) between them, q=(z-z0)/(z1-z0).
     * PressureExtremum adds supported cell-mean pressure extrema to the product
     * shock gate; density correction and the weak branch are independent options.
     * At a strict mean-field extremum, excess is the distance to the nearest
     * unequal face-neighbour value divided by the larger absolute value. The
     * correction requires enough nearly equal neighbours and no physical boundary
     * face. Support is the largest neighbouring base branch alpha divided by its
     * cap; mean-density support can be selected separately from the base gate.
     * Pressure correction spreads one face-neighbour layer; density correction
     * spreads contactExtremumHaloLayers. gateHaloLayers adds shock layers after
     * correction. All these are synchronous maxima over interior-face adjacency.
     * See docs/guides/limited_variational_reconstruction.md for full definitions,
     * presets and compatibility aliases. JSON writes selectors as strings.
     */
    struct LimitedVRSettings
    {
        /// jp0: zero response when Jp=2|pR-pL|/(pR+pL) is at/below this value.
        real pressureJumpStart = 0.04;
        /// jp1>jp0: cubic pressure response reaches one at this value.
        real pressureJumpFull = 0.16;
        /// Also admit expansion in the ordinary pressure/velocity product gate: use |unL-unR| instead of max(unL-unR,0).
        /// Does not change the contact or optional weak gate; pressure-extremum support follows this selected face gate.
        bool detectExpansion = false;
        /// jc0: zero response for selected Jc (compression, or absolute normal-velocity jump) at/below this value.
        real compressionStart = 0.03;
        /// jc1>jc0: selected normal-velocity-jump response reaches one at this value.
        real compressionFull = 0.13;
        /// Shock-branch cap in [0,1]; lambda=alpha/(1-alpha), with alpha=1 the fixed-target limit.
        real alphaMax = 1.0;
        /// Additional weak-shock branch cap; zero disables its gradients and halo work.
        real weakAlphaMax = 0.0;
        /// Weak branch only: mismatch-ramp start; dp=pR-pL from means, dL/R=grad(p)L/R dot (xf-xL/R).
        /// Mismatch is |dp-dL+dR|/max(|dp|+|dL|+|dR|,pressure floor), using unrestricted GG gradients.
        real weakMismatchStart = 0.05;
        /// Weak branch only: mismatch-ramp saturation; must exceed weakMismatchStart.
        real weakMismatchFull = 0.20;
        /// Weak branch only: ramp start for max(unL-unR,0)*mean(rho*c)/|dp|, using adjacent means.
        real weakAcousticStart = 0.10;
        /// Weak branch only: acoustic-ramp saturation; compression and a pressure jump above the floor are required.
        real weakAcousticFull = 0.50;
        /// Positive background-relative pressure-jump floor, multiplied by max(|pL|,|pR|).
        real weakRelativeFloor = 1e-10;
        /// Synchronous face-neighbour maximum passes on the freshly detected weak branch (0..4).
        int weakHaloLayers = 2;
        /// Weak branch only: theta*fresh+(1-theta)*old, theta in (0,1]; every new stage resets to fresh.
        real weakAlphaUpdateRelaxation = 1.0;
        /// Weak branch only: fresh+q*max(old-fresh,0), q in [0,1); immediate rises; requires weak relaxation=1.
        real weakAlphaDecayRetention = 0.0;
        /// Fresh gate on every stage start, including explicit-first INT_MAX; hold only within that stage.
        bool weakFreezeWithinImplicitStage = false;
        /// jrho0: density-only cubic ramp start for Jrho=2|rhoR-rhoL|/(rhoR+rhoL).
        real contactDensityJumpStart = 0.50;
        /// jrho1>jrho0: density-only cubic ramp saturation.
        real contactDensityJumpFull = 0.80;
        /// Contact-branch cap; zero disables it; final alpha=max(shock,contact,weak).
        real contactAlphaMax = 1.0;
        /// Use adjacent means for the base density jump; false uses finalized O2 face states.
        bool contactUseCellMeanJump = false;
        /// With contactAlphaMax>0, add supported cell-mean density extrema plus contact retention/halo below.
        bool contactUseExtremumCorrection = true;
        /// Contact correction only: use mean-density jumps for support, independently of the base contact gate.
        bool contactExtremumUseCellMeanSupport = true;
        /// Contact correction only: start of normalized cell-mean density excess defined in the class description.
        real contactExtremumStart = 0.001;
        /// Saturation of density-extremum response; active only with contactUseExtremumCorrection.
        real contactExtremumFull = 0.008;
        /// Start of adjacent contact-support alpha/contactAlphaMax cubic ramp.
        real contactSupportStart = 0.005;
        /// Saturation of support ramp; multiply it by density-extremum response.
        real contactSupportFull = 0.06;
        /// Contact correction only: fraction in [0,1] of previous contact alpha retained under support, per RHS evaluation.
        real contactExtremumRetention = 0.95;
        /// Contact correction only: synchronous maximum passes in [0,4]; H2 uses two. The base gate is not expanded.
        int contactExtremumHaloLayers = 2;
        /// ProductCubic: face product only; PressureExtremum: v2 supported pressure-extremum correction too.
        LimitedVRGateMode gateMode = LimitedVRGateMode::PressureExtremum;
        /// Extra synchronous face-neighbour maximum passes on the completed shock branch, in [0,4].
        int gateHaloLayers = 0;
        /// When expanding, propagate only alphaMax-saturated cells; local unsaturated cells remain.
        bool gateHaloSaturatedOnly = false;
        /// Shock/contact only: theta*fresh+(1-theta)*old per RHS evaluation, theta in (0,1]; independent of weak updates.
        real alphaUpdateRelaxation = 1.0;
        /// PressureExtremum only: start of normalized cell-mean pressure excess beyond the nearest unequal neighbour.
        real oscillationExtremumStart = 0.001;
        /// PressureExtremum only: saturation of normalized pressure-extremum response; must exceed its start.
        real oscillationExtremumFull = 0.008;
        /// PressureExtremum only: start of adjacent base-shock alpha/alphaMax support ramp, in [0,1].
        real oscillationSupportStart = 0.02;
        /// PressureExtremum only: saturation of support ramp; multiplies the extremum and retained responses.
        real oscillationSupportFull = 0.15;
        /// PressureExtremum only: fraction in [0,1] of previous shock alpha retained under support, per RHS evaluation.
        real oscillationRetention = 0.95;
        /// Relative equality tolerance for pressure/density neighbours, scaled by max absolute field.
        real oscillationTransverseTolerance = 1e-8;
        /// Minimum nearly equal face-neighbour count (not halo depth); exclude boundary-adjacent extrema.
        int oscillationMinTransverseNeighbours = 2;
        /// Barth or characteristic WBAP on GG linear modes; the finalized target also supplies ordinary face sensors.
        O2ReferenceLimiter o2ReferenceLimiter = O2ReferenceLimiter::WBAP;
        /// Use cell means as PP bases; enforce face/required volume-point density and sensible-energy positivity.
        bool o2ReferenceUsePP = true;

        DNDS_DECLARE_CONFIG(LimitedVRSettings)
        {
            // clang-format off
            DNDS_FIELD(pressureJumpStart, "jp0: start of cubic ramp for finalized-O2 face pressure jump 2*abs(pR-pL)/(pR+pL)", DNDS::Config::range(0.0));
            DNDS_FIELD(pressureJumpFull,  "jp1: pressure-jump ramp reaches one here; must exceed pressureJumpStart", DNDS::Config::range(0.0));
            DNDS_FIELD(detectExpansion, "Also detect expansion in the ordinary face gate: replace max(unL-unR,0) by abs(unL-unR), normalized by mean sound speed; pressure ramp is still required; contact/weak branches and halo exchanges are unchanged");
            DNDS_FIELD(compressionStart,  "jc0: start of cubic ramp for finalized-O2 normal-velocity jump; max(unL-unR,0)/mean(cL,cR), or abs(unL-unR)/mean(cL,cR) when detectExpansion=true", DNDS::Config::range(0.0));
            DNDS_FIELD(compressionFull,   "jc1: selected normal-velocity-jump ramp reaches one here; must exceed compressionStart", DNDS::Config::range(0.0));
            DNDS_FIELD(alphaMax,         "Shock-branch penalty fraction cap in [0,1]: zero disables this branch; one fixes a saturated reconstruction row to the O2 target", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakAlphaMax, "Maximum additional pressure-gradient-consistency weak-shock penalty fraction; zero disables the branch", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakMismatchStart, "Weak branch: ramp start for abs(dp-dL+dR)/max(abs(dp)+abs(dL)+abs(dR),floor); dp is the mean pressure difference, dL/R the unrestricted GG pressure increments from each centroid to the shared face", DNDS::Config::range(0.0));
            DNDS_FIELD(weakMismatchFull, "Relative pressure-gradient disagreement ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(weakAcousticStart, "Compressive mean-velocity jump times acoustic impedance divided by pressure jump: ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(weakAcousticFull, "Compressive acoustic consistency ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(weakRelativeFloor, "Weak branch: positive pressure-noise floor multiplier times max(abs(pL),abs(pR)); jumps at or below this floor give zero response", DNDS::Config::range(0.0));
            DNDS_FIELD(weakHaloLayers, "Face-neighbour expansion layers for the additional weak branch only", DNDS::Config::range(0, 4));
            DNDS_FIELD(weakAlphaUpdateRelaxation, "Temporal relaxation of the weak-shock alpha on later inner evaluations only; each new stage starts with a fresh gate", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakAlphaDecayRetention, "Fraction of the previous weak-alpha excess retained when the freshly evaluated weak response decreases; rises remain immediate and zero disables one-sided decay", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(weakFreezeWithinImplicitStage, "Evaluate a fresh weak-shock alpha at each stage start, including one-shot explicit-first stages, and hold it fixed only for later inner evaluations of that stage");
            DNDS_FIELD(contactDensityJumpStart, "jrho0: cubic ramp start for density-only jump 2*abs(rhoR-rhoL)/(rhoR+rhoL), from O2 face states or means selected by contactUseCellMeanJump", DNDS::Config::range(0.0));
            DNDS_FIELD(contactDensityJumpFull, "jrho1: density-jump ramp reaches one here; must exceed contactDensityJumpStart", DNDS::Config::range(0.0));
            DNDS_FIELD(contactAlphaMax, "Maximum O2 penalty fraction contributed by the density-only contact gate; zero disables it", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactUseCellMeanJump, "Evaluate the contact density jump from adjacent cell averages instead of limited-O2 reconstructed facial states");
            DNDS_FIELD(contactUseExtremumCorrection, "Add a contact-supported cell-mean density-extremum correction using the contact-specific extremum, support, retention, and halo controls");
            DNDS_FIELD(contactExtremumUseCellMeanSupport, "Evaluate density-extremum support from adjacent cell averages while retaining the selected reconstructed-face or cell-mean base contact gate");
            DNDS_FIELD(contactExtremumStart, "Normalized cell-mean density-extremum correction ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(contactExtremumFull, "Normalized cell-mean density-extremum correction ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(contactSupportStart, "Adjacent density-gate support fraction ramp start for the density-extremum correction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactSupportFull, "Adjacent density-gate support fraction ramp saturation for the density-extremum correction", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactExtremumRetention, "Contact correction only: fraction of previous contact alpha retained under adjacent density support, per RHS evaluation", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(contactExtremumHaloLayers, "Contact correction only: synchronous interior-face maximum passes of the correction seeds; base density gate remains local; H2 uses two", DNDS::Config::range(0, 4));
            DNDS_FIELD(gateMode,         "Shock gate: ProductCubic uses pressure/compression face ramps; PressureExtremum additionally suppresses supported cell-mean pressure extrema (v2/v3)", DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(LimitedVRGateMode)));
            DNDS_FIELD(gateHaloLayers,   "Extra synchronous interior-face maximum passes on the completed shock branch, after PressureExtremum's built-in one-pass correction; independent of contact and weak halos", DNDS::Config::range(0, 4));
            DNDS_FIELD(gateHaloSaturatedOnly, "Extra shock halos only: propagate alphaMax-saturated cells; preserve unsaturated local shock responses");
            DNDS_FIELD(alphaUpdateRelaxation, "Shock/contact per-RHS update theta*fresh+(1-theta)*old, theta in (0,1]; independent of weak-branch temporal controls", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationExtremumStart, "PressureExtremum: normalized cell-mean pressure excess ramp start", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationExtremumFull, "PressureExtremum: normalized cell-mean pressure excess ramp saturation", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationSupportStart, "PressureExtremum: adjacent base-shock alpha/alphaMax support ramp start", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationSupportFull, "PressureExtremum: adjacent base-shock alpha/alphaMax support ramp saturation", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationRetention, "PressureExtremum: fraction of previous shock alpha retained under adjacent support, per RHS evaluation", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(oscillationTransverseTolerance, "Pressure/density corrections: neighbours are nearly equal if their mean-field difference is at most this times the larger absolute value", DNDS::Config::range(0.0));
            DNDS_FIELD(oscillationMinTransverseNeighbours, "Pressure/density corrections: required count of nearly equal interior-face neighbours (not halo depth); physical-boundary-adjacent cells are excluded", DNDS::Config::range(1, 4));
            DNDS_FIELD(o2ReferenceLimiter, "Limiter of the GG linear target: Barth or characteristic WBAP; the finalized target also drives ordinary face sensing", DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(O2ReferenceLimiter)));
            DNDS_FIELD(o2ReferenceUsePP,   "Compress the limited GG O2 increment toward its cell mean for positive density and sensible internal energy at face and required volume quadrature points; applied before face sensing");
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
     * limiterProcedure=LVR, reconstruction order and derivative weights. Explicit
     * fields in the same layer override the preset, and later layers win.
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
        if (name != "v3")
        {
            // Historical presets remain explicit when the raw default changes.
            values["alphaMax"] = 0.5;
            values["contactAlphaMax"] = 0.0;
            values["contactDensityJumpStart"] = 0.2;
            values["contactUseExtremumCorrection"] = false;
            values["contactExtremumUseCellMeanSupport"] = false;
            values["contactSupportStart"] = 0.02;
            values["contactSupportFull"] = 0.15;
            values["contactExtremumHaloLayers"] = 1;
        }
        if (name == "v1")
        {
            values["o2ReferenceLimiter"] = O2ReferenceLimiter::Barth;
            values["gateMode"] = LimitedVRGateMode::ProductCubic;
            values["pressureJumpStart"] = 0.10;
            values["pressureJumpFull"] = 0.22;
            values["compressionStart"] = 0.08;
            values["compressionFull"] = 0.16;
        }
        section.erase("preset");
        values.merge_patch(section);
        section = std::move(values);
    }

    struct LimitedVRFaceSensor
    {
        real pressureJump = 0;
        /// Selected normal-velocity indicator: nonnegative compression, or absolute jump with detectExpansion.
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
        real velocityJump = normalVelocityLeft - normalVelocityRight;
        result.compression = (settings.detectExpansion ? std::abs(velocityJump) : std::max(real(0), velocityJump)) / acousticDenominator;
        real pressureResponse = LimitedVRCubicRamp(
            result.pressureJump, settings.pressureJumpStart, settings.pressureJumpFull);
        real compressionResponse = LimitedVRCubicRamp(
            result.compression, settings.compressionStart, settings.compressionFull);
        // Both retained modes share this local face response. PressureExtremum
        // adds its supported cell-mean correction in BuildLimitedVRO2.
        result.alpha = settings.alphaMax * (pressureResponse * compressionResponse);
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
