#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace DNDS::CFV
{
    /// Native v2/v3 controls. Caps and thresholds are explicit after config
    /// resolution. Variant defaults are selected before applying that layer's
    /// overrides; C++ callers can use ForVariant.
    struct LimitedVRSettings
    {
        std::string variant = "v3";
        real alphaMax = 1, contactAlphaMax = 1;
        std::array<real, 2> pressureJump{.04, .16}, compression{.03, .13}, densityJump{.50, .80};
        bool detectExpansion = false;
        std::string gateMode = "PressureExtremum";
        bool contactUseCellMeanJump = false, contactUseExtremumCorrection = true;
        bool contactExtremumUseCellMeanSupport = true;
        std::string o2ReferenceLimiter;
        bool o2ReferenceUsePP = true;
        int neighborLayers = 0;
        bool saturatedOnly = false;
        std::array<real, 2> extremumThreshold{.001, .008}, pressureSupport{.02, .15}, contactSupport{.005, .06};
        std::array<real, 2> contactExtremumThreshold{.001, .008};
        real retention = .95, equalityTolerance = 1e-8;
        real contactExtremumRetention = .95, alphaUpdateRelaxation = 1;
        int minimumEqualNeighbors = 2, contactLayers = 2;
        real weakAlphaMax = 0, weakRelativeFloor = 1e-10;
        std::array<real, 2> weakMismatch{.05, .20}, weakAcoustic{.10, .50};
        int weakHaloLayers = 2;
        real weakAlphaUpdateRelaxation = 1, weakAlphaDecayRetention = 0;
        bool weakFreezeWithinImplicitStage = false;

        bool Active() const { return alphaMax > 0 || contactAlphaMax > 0 || weakAlphaMax > 0; }
        index GateWidth() const { return weakAlphaMax > 0 ? 4 : 3; }
        index HistoryWidth() const { return weakAlphaMax > 0 ? 3 : 2; }
        static LimitedVRSettings ForVariant(const std::string &name)
        {
            DNDS_check_throw_info(name == "v1" || name == "v2" || name == "v3", "limitedVR variant must be v1, v2 or v3");
            LimitedVRSettings result;
            result.variant = name;
            if (name != "v3")
            {
                result.alphaMax = .5;
                result.contactAlphaMax = 0;
                result.densityJump = {.2, .8};
                result.contactSupport = {.02, .15};
                result.contactLayers = 1;
                result.contactUseExtremumCorrection = false;
                result.contactExtremumUseCellMeanSupport = false;
            }
            if (name == "v1")
            {
                result.gateMode = "ProductCubic";
                result.pressureJump = {.10, .22};
                result.compression = {.08, .16};
            }
            return result;
        }
        void Validate() const
        {
            DNDS_check_throw_info((variant == "v1" || variant == "v2" || variant == "v3") && std::isfinite(alphaMax) && alphaMax >= 0 && alphaMax <= 1 &&
                                      (gateMode == "ProductCubic" || gateMode == "PressureExtremum") &&
                                      std::isfinite(contactAlphaMax) && contactAlphaMax >= 0 && contactAlphaMax <= 1 &&
                                      neighborLayers >= 0 && neighborLayers <= 4 && contactLayers >= 0 && contactLayers <= 4 &&
                                      std::isfinite(retention) && retention >= 0 && retention <= 1 &&
                                      std::isfinite(contactExtremumRetention) && contactExtremumRetention >= 0 && contactExtremumRetention <= 1 &&
                                      std::isfinite(alphaUpdateRelaxation) && alphaUpdateRelaxation > 0 && alphaUpdateRelaxation <= 1 &&
                                      std::isfinite(weakAlphaMax) && weakAlphaMax >= 0 && weakAlphaMax <= 1 &&
                                      std::isfinite(weakRelativeFloor) && weakRelativeFloor > 0 &&
                                      weakHaloLayers >= 0 && weakHaloLayers <= 4 &&
                                      std::isfinite(weakAlphaUpdateRelaxation) && weakAlphaUpdateRelaxation > 0 && weakAlphaUpdateRelaxation <= 1 &&
                                      std::isfinite(weakAlphaDecayRetention) && weakAlphaDecayRetention >= 0 && weakAlphaDecayRetention < 1 &&
                                      (weakAlphaDecayRetention == 0 || weakAlphaUpdateRelaxation == 1) &&
                                      std::isfinite(equalityTolerance) && equalityTolerance >= 0 && minimumEqualNeighbors >= 1 && minimumEqualNeighbors <= 4,
                                  "invalid limitedVR controls");
            for (const auto &threshold : {pressureJump, compression, densityJump, extremumThreshold, contactExtremumThreshold, pressureSupport, contactSupport, weakMismatch, weakAcoustic})
                DNDS_check_throw_info(std::isfinite(threshold[0]) && std::isfinite(threshold[1]) && threshold[0] >= 0 && threshold[1] > threshold[0],
                                      "limitedVR thresholds must be finite, nonnegative and strictly increasing");
        }
        DNDS_DECLARE_CONFIG(LimitedVRSettings)
        {
            DNDS_FIELD(variant, "v2 pressure correction or v3 pressure/contact Average H2 correction");
            DNDS_FIELD(alphaMax, "Shock penalty fraction cap, including exact fixed-target limit one");
            DNDS_FIELD(contactAlphaMax, "Density/contact penalty fraction cap");
            DNDS_FIELD(pressureJump, "Cubic start/full pair for 2*abs(pR-pL)/(pR+pL)");
            DNDS_FIELD(compression, "Cubic start/full pair for selected normal-velocity jump over mean sound speed");
            DNDS_FIELD(densityJump, "Cubic start/full pair for 2*abs(rhoR-rhoL)/(rhoR+rhoL)");
            DNDS_FIELD(detectExpansion, "Use absolute velocity jump instead of compression only; pressure ramp is still required");
            DNDS_FIELD(gateMode, "ProductCubic face product or PressureExtremum with supported pressure extrema");
            DNDS_FIELD(contactUseCellMeanJump, "Use adjacent mean densities for the base contact gate");
            DNDS_FIELD(contactUseExtremumCorrection, "Add supported cell-mean density extrema to the contact gate");
            DNDS_FIELD(contactExtremumUseCellMeanSupport, "Use adjacent mean densities to support the contact correction");
            DNDS_FIELD(o2ReferenceLimiter, "Compatibility spelling for the shared Cauchy P1 limiter: Barth, WBAP or WBAP_Barth; empty uses numerics.p1Limiter");
            DNDS_FIELD(o2ReferenceUsePP, "Apply shared direct P1 positivity limiting before LVR target and face-sensor assembly");
            DNDS_FIELD(neighborLayers, "Extra synchronous face-neighbor shock maximum rounds after extremum correction,0..4");
            DNDS_FIELD(saturatedOnly, "Extra rounds propagate saturated shock cells only, retaining local unsaturated responses");
            DNDS_FIELD(extremumThreshold, "Normalized pressure-extremum excess start/full pair");
            DNDS_FIELD(contactExtremumThreshold, "Normalized density-extremum excess start/full pair");
            DNDS_FIELD(pressureSupport, "Neighboring base-shock support fraction start/full pair");
            DNDS_FIELD(contactSupport, "Neighboring mean-density support fraction start/full pair");
            DNDS_FIELD(retention, "Retain this fraction of previous shock alpha under current support");
            DNDS_FIELD(contactExtremumRetention, "Retain this fraction of previous contact alpha under current support");
            DNDS_FIELD(alphaUpdateRelaxation, "Per-evaluation update weight for fresh shock and contact penalties");
            DNDS_FIELD(equalityTolerance, "Relative transverse neighbor equality tolerance");
            DNDS_FIELD(minimumEqualNeighbors, "Equal face-incidence neighbors required for extremum correction");
            DNDS_FIELD(contactLayers, "v3 contact-correction seed expansion rounds; Average H2 uses two");
            DNDS_FIELD(weakAlphaMax, "Weak shock penalty cap; zero disables the branch");
            DNDS_FIELD(weakMismatch, "Weak shock pressure-slope mismatch cubic ramp start/full pair");
            DNDS_FIELD(weakAcoustic, "Weak shock acoustic compression cubic ramp start/full pair");
            DNDS_FIELD(weakRelativeFloor, "Weak shock background-relative pressure noise floor");
            DNDS_FIELD(weakHaloLayers, "Face-neighbor expansion passes for the weak shock branch");
            DNDS_FIELD(weakAlphaUpdateRelaxation, "Within-stage update weight for fresh weak shock penalties");
            DNDS_FIELD(weakAlphaDecayRetention, "Retention of old weak shock penalties during decreases; rises are immediate");
            DNDS_FIELD(weakFreezeWithinImplicitStage, "Freeze the weak shock penalty after the first evaluation of a stage");
        }
    };

    /// Immutable numerical controls. The variant string and orchestration's
    /// layer counts stay in LimitedVRSettings; no host configuration object is
    /// passed into a device kernel. Resolve/validate once at operator entry.
    struct LimitedVRKernelParameters
    {
        real alphaMax, contactAlphaMax;
        std::array<real, 2> pressureJump, compression, densityJump;
        bool detectExpansion, pressureCorrection, contactCorrection;
        bool contactUseCellMeanJump, contactExtremumUseCellMeanSupport, weakFirstMaximum;
        std::array<real, 2> extremumThreshold, contactExtremumThreshold, pressureSupport, contactSupport;
        real retention, contactExtremumRetention, equalityTolerance;
        int minimumEqualNeighbors;
        real weakAlphaMax, weakRelativeFloor;
        std::array<real, 2> weakMismatch, weakAcoustic;
        static LimitedVRKernelParameters FromSettings(const LimitedVRSettings &settings)
        {
            settings.Validate();
            return {settings.alphaMax, settings.contactAlphaMax, settings.pressureJump,
                    settings.compression, settings.densityJump, settings.detectExpansion,
                    settings.gateMode == "PressureExtremum", settings.contactUseExtremumCorrection,
                    settings.contactUseCellMeanJump, settings.contactExtremumUseCellMeanSupport,
                    settings.weakAlphaMax > 0 && settings.weakHaloLayers > 0 &&
                        ((settings.gateMode == "PressureExtremum" && settings.alphaMax > 0) ||
                         (settings.contactUseExtremumCorrection && settings.contactAlphaMax > 0)),
                    settings.extremumThreshold, settings.contactExtremumThreshold, settings.pressureSupport,
                    settings.contactSupport, settings.retention, settings.contactExtremumRetention, settings.equalityTolerance,
                    settings.minimumEqualNeighbors, settings.weakAlphaMax, settings.weakRelativeFloor,
                    settings.weakMismatch, settings.weakAcoustic};
        }
    };

    struct LimitedVRTemporalUpdate
    {
        real shockContactWeight = 1, weakWeight = 1, weakDecay = 0;
        bool newStage = true, refreshWeak = true;
    };

    namespace Math
    {
        /// Thresholds are validated at preparation, outside face/cell loops.
        DNDS_DEVICE_CALLABLE inline real LimitedVRCubicRamp(real value, const std::array<real, 2> &threshold)
        {
            if (value <= threshold[0])
                return 0;
            if (value >= threshold[1])
                return 1;
            const real q = (value - threshold[0]) / (threshold[1] - threshold[0]);
            return q * q * (3 - 2 * q);
        }
        struct LimitedVRFaceSensor
        {
            real pressureJump, compression, densityJump, shock, contact;
        };
        /// Equation supplies finalized O2 face density,p,normal speed and c.
        /// No EOS,energy convention or component ordering enters this operator.
        template <class Settings>
        DNDS_DEVICE_CALLABLE inline LimitedVRFaceSensor EvaluateLimitedVRFaceSensor(real rhoLeft, real rhoRight, real pressureLeft, real pressureRight,
                                                                                    real normalLeft, real normalRight, real soundLeft, real soundRight,
                                                                                    const Settings &settings)
        {
            const real pressureScale = std::max(std::max(std::abs(pressureLeft), std::abs(pressureRight)), real(1e-30));
            const real acousticScale = std::max(std::max(std::abs(soundLeft), std::abs(soundRight)), real(1e-30));
            const real densityScale = std::max(std::max(std::abs(rhoLeft), std::abs(rhoRight)), real(1e-30));
            const real jp = 2 * std::abs(pressureRight - pressureLeft) / std::max(pressureLeft + pressureRight, smallReal * pressureScale);
            const real dv = normalLeft - normalRight;
            const real jc = (settings.detectExpansion ? std::abs(dv) : std::max(real(0), dv)) / std::max(.5 * (soundLeft + soundRight), smallReal * acousticScale);
            const real jr = 2 * std::abs(rhoRight - rhoLeft) / std::max(rhoLeft + rhoRight, smallReal * densityScale);
            return {jp, jc, jr, settings.alphaMax * (LimitedVRCubicRamp(jp, settings.pressureJump) * LimitedVRCubicRamp(jc, settings.compression)),
                    settings.contactAlphaMax * LimitedVRCubicRamp(jr, settings.densityJump)};
        }
        DNDS_DEVICE_CALLABLE inline real LimitedVRWeakShock(
            real pressureLeft, real pressureRight, real incrementLeft, real incrementRight,
            real normalCompression, real impedance, const LimitedVRKernelParameters &settings)
        {
            const real jump = pressureRight - pressureLeft;
            const real floor = settings.weakRelativeFloor *
                               std::max(std::max(std::abs(pressureLeft), std::abs(pressureRight)), real(1e-30));
            if (std::abs(jump) <= floor || normalCompression <= 0)
                return 0;
            const real mismatch = std::abs(jump - incrementLeft + incrementRight) /
                                  std::max(std::abs(jump) + std::abs(incrementLeft) + std::abs(incrementRight), floor);
            return settings.weakAlphaMax * LimitedVRCubicRamp(mismatch, settings.weakMismatch) *
                   LimitedVRCubicRamp(normalCompression * impedance / std::abs(jump), settings.weakAcoustic);
        }
        template <class Gates, class History>
        DNDS_DEVICE_CALLABLE real PublishLimitedVRCell(Gates gates, History history, index cell, LimitedVRTemporalUpdate update)
        {
            real alpha = 0;
            for (int branch = 0; branch < 2; ++branch)
            {
                real fresh = gates(cell, branch);
                if (update.shockContactWeight < 1)
                    fresh = update.shockContactWeight * fresh + (1 - update.shockContactWeight) * history(cell, branch);
                history(cell, branch) = fresh;
                alpha = std::max(alpha, fresh);
            }
            if (history.Width() == 3)
            {
                if (update.refreshWeak)
                {
                    real fresh = gates(cell, 3);
                    if (!update.newStage && update.weakDecay > 0)
                        fresh += update.weakDecay * std::max(history(cell, 2) - fresh, real(0));
                    else if (!update.newStage && update.weakWeight < 1)
                        fresh = update.weakWeight * fresh + (1 - update.weakWeight) * history(cell, 2);
                    history(cell, 2) = fresh;
                }
                alpha = std::max(alpha, history(cell, 2));
            }
            return alpha;
        }
        /// One scalar extremum stencil, retaining incidence multiplicity (also
        /// on a one-cell periodic mesh). Boundaries exclude correction seeds.
        struct LimitedVRExtremumStencil
        {
            real lower = -std::numeric_limits<real>::infinity(), upper = std::numeric_limits<real>::infinity(), support = 0;
            int equal = 0;
            bool boundary = false;
            DNDS_DEVICE_CALLABLE void Observe(real value, real other, real neighborSupport, real tolerance)
            {
                const real scale = std::max(std::max(std::abs(value), std::abs(other)), real(1e-30));
                if (other < value - tolerance * scale)
                    lower = std::max(lower, other);
                else if (other > value + tolerance * scale)
                    upper = std::min(upper, other);
                else
                    ++equal;
                support = std::max(support, neighborSupport);
            }
            template <class Settings>
            DNDS_DEVICE_CALLABLE real Correction(real value, real previous, real cap, const std::array<real, 2> &supportThreshold,
                                                 const std::array<real, 2> &extremumThreshold, real retention,
                                                 const Settings &settings) const
            {
                if (cap == 0 || boundary || equal < settings.minimumEqualNeighbors)
                    return 0;
                real excess = 0;
                if (std::isfinite(lower) && !std::isfinite(upper))
                    excess = (value - lower) / std::max(std::max(std::abs(value), std::abs(lower)), real(1e-30));
                else if (std::isfinite(upper) && !std::isfinite(lower))
                    excess = (upper - value) / std::max(std::max(std::abs(value), std::abs(upper)), real(1e-30));
                const real supported = LimitedVRCubicRamp(support / std::max(cap, real(1e-30)), supportThreshold);
                const real response = LimitedVRCubicRamp(excess, extremumThreshold) * supported;
                const real retained = retention * previous / std::max(cap, real(1e-30)) * supported;
                return cap * std::max(response, retained);
            }
        };
    }
}
