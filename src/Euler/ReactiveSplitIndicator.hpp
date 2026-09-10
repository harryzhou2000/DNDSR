#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Defines.hpp"

namespace DNDS::Euler
{
    /**
     * @brief Configuration of the dimensionless mixed reactive-source selector.
     *
     * For cell @f$i@f$, the RRI constituent diagnostics are
     *
     * @f[
     * a_i=\Delta t_{\mathrm{phys}}r_{\mathrm{chem},i},\qquad
     * r_{\mathrm{chem},i}=\left[\sum_k\left(\frac{\dot\omega_{k,i}W_k}{\rho_i}\right)^2+
     * \left(\frac{|\dot q_i|}{\rho_i c_{v,i}T_{\mathrm{scale},i}}\right)^2\right]^{1/2},
     * @f]
     *
     * @f[
     * b_i=\Delta t_{\mathrm{phys}}\frac{D_{\max,i}}{L_{\mathrm{grad},i}^2},\qquad
     * h_i=\max_{j\in\mathcal N(i)}\frac{|p_j-p_i|}{\max(|p_i|,|p_j|,p_\epsilon)},\qquad
     * g_h(h_i)=\frac{1}{1+(h_i/h_0)^4}.
     * @f]
     *
     * Here @f$T_{\mathrm{scale},i}=\max(T_i,T_{\mathrm{floor}})@f$, @f$L_{\mathrm{grad},i}@f$ is bounded
     * below by the cell length, and all of @f$a_i@f$, @f$b_i@f$, @f$h_i@f$, and @f$g_h@f$ are
     * dimensionless. The local coupled score is
     *
     * @f[
     * C_i=\operatorname{sat}(a_i;a_0,p)\operatorname{sat}(b_i;b_0,p)g_h(h_i),\qquad
     * \operatorname{sat}(z;z_0,p)=\frac{(\max(z,0)/z_0)^p}{1+(\max(z,0)/z_0)^p}.
     * @f]
     *
     * The raw coupled fraction is @f$f_{c,i}=1-\chi_i^*@f$. Endpoint tolerances then snap
     * @f$\chi_i^*\leq\epsilon_c@f$ to exact coupled @f$\chi_i=0@f$ and
     * @f$1-\chi_i^*\leq\epsilon_s@f$ to exact Strang @f$\chi_i=1@f$. Thus @f$\chi_i=0@f$
     * selects fully coupled chemistry/flow integration and @f$\chi_i=1@f$ selects true Strang
     * splitting. The defaults @f$a_0=b_0=p=1@f$ recover the original mathematical saturation
     * @f$z/(1+z)@f$.
     * The logistic map uses midpoint @f$C_0+w\ln b_s@f$ and width @f$w@f$; the compact-tail Hill
     * map uses midpoint @f$C_0b_s@f$ and exponent @f$n@f$.
     */
    struct ReactiveSplitIndicatorSettings
    {
        real chemicalActivityThreshold = 1.0;  ///< Chemical saturation threshold @f$a_0@f$.
        real diffusiveActivityThreshold = 1.0; ///< Diffusion saturation threshold @f$b_0@f$.
        real activitySaturationExponent = 1.0; ///< Shared activity-saturation exponent @f$p@f$.
        real coupledThreshold = 0.005;         ///< Coupled-score switch midpoint @f$C_0@f$.
        real transitionWidth = 0.001;          ///< Logistic switch width @f$w@f$.
        real shockScale = 0.08;                ///< Pressure-jump scale @f$h_0@f$ in @f$g_h@f$.
        real strangBias = 1.0;                 ///< Dimensionless Strang-preference factor @f$b_s@f$.
        int switchShape = 1;                   ///< 0: logistic; 1: compact-tail Hill map (default).
        real hillExponent = 3.0;               ///< Final coupled-fraction Hill exponent @f$n@f$.
        int spatialPasses = 0;                 ///< Number of face-neighbor expansion passes.
        real spatialDecay = 0.65;              ///< Neighbor retention factor @f$\eta@f$ per pass.
        real strangSnapTolerance = 0.01;       ///< Snap to @f$\chi=1@f$ when @f$1-\chi^*\leq\epsilon_s@f$.
        real coupledSnapTolerance = 0.01;      ///< Snap to @f$\chi=0@f$ when @f$\chi^*\leq\epsilon_c@f$.
        real chiOverride = -1.0;               ///< Forced @f$\chi@f$ in [0,1]; negative enables selection.

        DNDS_DECLARE_CONFIG(ReactiveSplitIndicatorSettings)
        {
            // clang-format off
            DNDS_FIELD(chemicalActivityThreshold,  "Chemical activity threshold a0 in sat(a;a0,p)", DNDS::Config::range(0.0));
            DNDS_FIELD(diffusiveActivityThreshold, "Diffusion activity threshold b0 in sat(b;b0,p)", DNDS::Config::range(0.0));
            DNDS_FIELD(activitySaturationExponent, "Shared activity-saturation exponent p in sat(z;z0,p)", DNDS::Config::range(0.0));
            DNDS_FIELD(coupledThreshold,           "Coupled-score midpoint C0", DNDS::Config::range(0.0));
            DNDS_FIELD(transitionWidth,            "Logistic coupled-fraction width w", DNDS::Config::range(0.0));
            DNDS_FIELD(shockScale,                 "Pressure-jump scale h0 in shock gate gh", DNDS::Config::range(0.0));
            DNDS_FIELD(strangBias,                 "Dimensionless Strang-preference factor bs", DNDS::Config::range(0.0));
            DNDS_FIELD(switchShape,                "Coupled-fraction map: 0=logistic, 1=compact-tail Hill", DNDS::Config::range(0, 1));
            DNDS_FIELD(hillExponent,               "Final coupled-fraction Hill exponent n", DNDS::Config::range(0.0));
            DNDS_FIELD(spatialPasses,              "Number of decaying coupled-fraction neighbor passes", DNDS::Config::range(0, 8));
            DNDS_FIELD(spatialDecay,               "Neighbor coupled-fraction retention eta per pass", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(strangSnapTolerance,        "Endpoint tolerance epsilon_s for snapping the computed chi to exact Strang chi=1", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(coupledSnapTolerance,       "Endpoint tolerance epsilon_c for snapping the computed chi to exact coupled chi=0", DNDS::Config::range(0.0, 1.0));
            DNDS_FIELD(chiOverride,                "Override Strang fraction chi in [0,1]; negative enables the indicator", DNDS::Config::range(-1.0, 1.0));
            // clang-format on
            config.check("reactive split activity thresholds and exponent must be positive", [](const T &s)
                         { return s.chemicalActivityThreshold > 0 && s.diffusiveActivityThreshold > 0 &&
                                  s.activitySaturationExponent > 0; });
            config.check("reactive split shockScale and strangBias must be positive", [](const T &s)
                         { return s.shockScale > 0 && s.strangBias > 0; });
            config.check("reactive split transitionWidth must be positive for the logistic map", [](const T &s)
                         { return s.switchShape != 0 || s.transitionWidth > 0; });
            config.check("reactive split hillExponent must be positive for the Hill map", [](const T &s)
                         { return s.switchShape != 1 || s.hillExponent > 0; });
            config.check("reactive split endpoint snap tolerances must have sum less than one", [](const T &s)
                         { return s.strangSnapTolerance + s.coupledSnapTolerance < 1; });
        }
    };

    /** @brief Evaluate @f$\operatorname{sat}(z;z_0,p)@f$ with a stable positive-ratio form. */
    inline real ReactiveSplitSaturate(real value, real threshold = 1.0, real exponent = 1.0)
    {
        DNDS_check_throw_info(threshold > 0, "reactive split activity threshold must be positive");
        DNDS_check_throw_info(exponent > 0, "reactive split activity saturation exponent must be positive");
        value = std::max(value, real(0));
        if (value == 0)
            return 0;
        real logRatioPower = exponent * std::log(value / threshold);
        if (logRatioPower >= 0)
            return 1.0 / (1.0 + std::exp(-logRatioPower));
        real ratioPower = std::exp(logRatioPower);
        return ratioPower / (1.0 + ratioPower);
    }

    /** @brief Evaluate @f$g_h(h_i)=1/[1+(h_i/h_0)^4]@f$. */
    inline real ReactiveSplitShockGate(real shockSensor, const ReactiveSplitIndicatorSettings &settings)
    {
        DNDS_check_throw_info(settings.shockScale > 0, "reactive split shockScale must be positive");
        real shockRatio = std::max(shockSensor, real(0)) / settings.shockScale;
        return 1.0 / (1.0 + std::pow(shockRatio, 4));
    }

    /** @brief Form @f$C_i=\operatorname{sat}(a_i;a_0,p)\operatorname{sat}(b_i;b_0,p)g_h(h_i)@f$. */
    inline real ReactiveSplitCoupledScore(real chemicalActivity, real diffusionActivity, real shockSensor,
                                          const ReactiveSplitIndicatorSettings &settings)
    {
        return ReactiveSplitSaturate(chemicalActivity, settings.chemicalActivityThreshold,
                                     settings.activitySaturationExponent) *
               ReactiveSplitSaturate(diffusionActivity, settings.diffusiveActivityThreshold,
                                     settings.activitySaturationExponent) *
               ReactiveSplitShockGate(shockSensor, settings);
    }

    /** @brief Map @f$C_i@f$ to the local coupled fraction @f$f_{c,i}=1-\chi_i@f$. */
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

    /** @brief Expand @f$f_c@f$ from face neighbors with retention @f$\eta@f$ and local shock gate @f$g_h@f$. */
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

    /** @brief Snap an indicator-computed Strang fraction to exact coupled or Strang endpoints. */
    inline real ReactiveSplitSnapChi(real chi, const ReactiveSplitIndicatorSettings &settings)
    {
        DNDS_check_throw_info(settings.strangSnapTolerance >= 0 && settings.strangSnapTolerance <= 1,
                              "reactive split strangSnapTolerance must be in [0,1]");
        DNDS_check_throw_info(settings.coupledSnapTolerance >= 0 && settings.coupledSnapTolerance <= 1,
                              "reactive split coupledSnapTolerance must be in [0,1]");
        DNDS_check_throw_info(settings.strangSnapTolerance + settings.coupledSnapTolerance < 1,
                              "reactive split endpoint snap tolerances must have sum less than one");
        chi = std::clamp(chi, real(0), real(1));
        if (chi <= settings.coupledSnapTolerance)
            return 0;
        if (1.0 - chi <= settings.strangSnapTolerance)
            return 1;
        return chi;
    }

    /** @brief Return the snapped Strang fraction @f$\chi_i@f$ or the exact configured override. */
    inline real ReactiveSplitChi(real coupledScore, const ReactiveSplitIndicatorSettings &settings)
    {
        if (settings.chiOverride >= 0)
            return std::clamp(settings.chiOverride, real(0), real(1));
        return ReactiveSplitSnapChi(1.0 - ReactiveSplitCoupledFraction(coupledScore, settings), settings);
    }
}
