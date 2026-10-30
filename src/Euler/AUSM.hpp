#pragma once
/** @file AUSM.hpp
 *  @brief AUSMPW+, AUSM+-up and AUSM+-up2 face fluxes, independent of Roe splitting.
 *
 *  Kim, Kim & Rho, JCP 174 (2001), doi:10.1006/jcph.2001.6873;
 *  Liou, JCP 214 (2006), doi:10.1016/j.jcp.2005.09.020;
 *  Kitamura & Shima, JCP 245 (2013), doi:10.1016/j.jcp.2013.02.046.
 *  See docs/guides/ausm_fluxes.md for public equation sources and the explicitly
 *  identified unstructured pressure-stencil, frozen-thermodynamics and ALE extensions.
 */
#include "DNDS/Defines.hpp"
#include "DNDS/IdealGasPhysics.hpp"

namespace DNDS::Euler::Gas
{
    enum class AUSMVariant
    {
        PWPlus,  ///< Pressure-weighted AUSMPW+, quadratic Mach splitting.
        PlusUp,  ///< Liou AUSM+-up, fourth-order Mach splitting and p/u diffusion.
        PlusUp2, ///< AUSM+-up mass flux with Kitamura-Shima pressure flux.
    };

    /// @brief Positive/negative Mach polynomial; fourth-order variant uses beta=1/8.
    inline real AUSMMachSplit(real mach, bool positive, bool fourthOrder)
    {
        real s = positive ? real(1) : real(-1);
        if (std::abs(mach) >= 1)
            return 0.5 * (mach + s * std::abs(mach));
        real split = s * 0.25 * (mach + s) * (mach + s);
        if (fourthOrder)
            split += s * 0.125 * std::pow(mach * mach - 1, 2);
        return split;
    }

    /// @brief Pressure polynomial P^+/-_alpha; PW+ and up2 use the robust cubic alpha=0.
    inline real AUSMPressureSplit(real mach, bool positive, real alpha)
    {
        real s = positive ? real(1) : real(-1);
        if (std::abs(mach) >= 1)
            return 0.5 * (1 + s * (mach > 0 ? 1 : -1));
        return 0.25 * (mach + s) * (mach + s) * (2 - s * mach) +
               s * alpha * mach * std::pow(mach * mach - 1, 2);
    }

    /**
     * @brief AUSM flux in physical coordinates, with a unit normal pointing L to R.
     *
     * @param UL,UR Conservative states [rho,rho*v,rho*E,...].
     * @param vg Face-grid velocity; relative velocities enter all Mach polynomials.
     * @param gammaEqL,gammaEqR Pressure/energy closure gamma for each side.
     * @param gammaL,gammaR Acoustic gamma cp/cv for each side.
     * @param rhoEBaseL,rhoEBaseR Base energy density, excluded from sound-speed
     *        construction but retained in transported total enthalpy.
     * @param referenceMach Positive cutoff M_infinity for up/up2; ignored by PW+.
     * @param pressureStencilMin Minimum PW+ stencil pressure; veryLargeReal gives
     *        the exact 1-D factor 1. Euler supplies an unstructured neighbourhood minimum.
     * @param[out] F ALE gas flux in its first dim+2 entries; additional entries
     *        are untouched, matching the Roe/HLL kernel contract.
     * @param[out] lam0,lam123,lam4 Physical acoustic/contact magnitudes for the
     *        existing approximate preconditioner; these are not an AUSM Jacobian.
     *
     * The published perfect-gas formulas are recovered for gammaEq==gamma==constant.
     * Differing local gammas use a frozen critical-speed estimate, not a real-gas
     * sonic solve. ALE pressure work adds p_face*(vg dot n) to the energy flux;
     * transported velocity and total enthalpy remain in the lab frame.
     */
    template <int dim, typename TUL, typename TUR, typename TVecVG, typename TVecN,
              typename TF, typename TFdumpInfo>
    void AUSMFlux(AUSMVariant variant, const TUL &UL, const TUR &UR,
                  const TVecVG &vg, const TVecN &n,
                  real gammaEqL, real gammaEqR, real gammaL, real gammaR,
                  real rhoEBaseL, real rhoEBaseR,
                  real referenceMach, real pressureStencilMin, TF &F,
                  const TFdumpInfo &dumpInfo, real &lam0, real &lam123, real &lam4)
    {
        using TVec = Eigen::Vector<real, dim>;
        DNDS_assert(UL(0) > 0 && UR(0) > 0);
        TVec vL = UL.template segment<dim>(1) / UL(0);
        TVec vR = UR.template segment<dim>(1) / UR(0);
        TVec relativeVL = vL - vg;
        TVec relativeVR = vR - vg;
        real unL = relativeVL.dot(n), unR = relativeVR.dot(n);
        real pL, pR, asqrL, asqrR, HL, HR;
        IdealGas::IdealGasThermal(UL(dim + 1), UL(0), vL.squaredNorm(), gammaEqL, gammaL, pL, asqrL, HL, rhoEBaseL);
        IdealGas::IdealGasThermal(UR(dim + 1), UR(0), vR.squaredNorm(), gammaEqR, gammaR, pR, asqrR, HR, rhoEBaseR);
        if (!(pL > 0 && pR > 0 && asqrL > 0 && asqrR > 0))
            dumpInfo();
        DNDS_assert(pL > 0 && pR > 0 && asqrL > 0 && asqrR > 0);

        // Direct static enthalpy avoids subtracting large tangential kinetic energy
        // from total enthalpy when constructing the hypersonic PW+ normal sonic speed.
        real hL = gammaEqL / (gammaEqL - 1) * pL / UL(0);
        real hR = gammaEqR / (gammaEqR - 1) * pR / UR(0);
        real criticalFactorL = 2 * gammaL * (gammaEqL - 1) / (gammaEqL * (gammaL + 1));
        real criticalFactorR = 2 * gammaR * (gammaEqR - 1) / (gammaEqR * (gammaR + 1));
        real aFace;
        if (variant == AUSMVariant::PWPlus)
        {
            real criticalSqr = 0.5 * (criticalFactorL * (hL + 0.5 * unL * unL) +
                                      criticalFactorR * (hR + 0.5 * unR * unR));
            real critical = std::sqrt(criticalSqr);
            // At unL+unR==0 both |un| agree, so this tie preserves face reversal.
            real upstreamSpeed = unL + unR >= 0 ? std::abs(unL) : std::abs(unR);
            aFace = criticalSqr / std::max(critical, upstreamSpeed);
        }
        else
        {
            real criticalLSqr = criticalFactorL * (hL + 0.5 * relativeVL.squaredNorm());
            real criticalRSqr = criticalFactorR * (hR + 0.5 * relativeVR.squaredNorm());
            // Liou's one-sided sonic estimates use max(a*,uL) and max(a*,-uR).
            aFace = std::min(criticalLSqr / std::max(std::sqrt(criticalLSqr), unL),
                             criticalRSqr / std::max(std::sqrt(criticalRSqr), -unR));
        }
        DNDS_assert(aFace > 0 && std::isfinite(aFace));
        real machL = unL / aFace, machR = unR / aFace;
        real coefficientL, coefficientR, pressure;
        if (variant == AUSMVariant::PWPlus)
        {
            real mp = AUSMMachSplit(machL, true, false);
            real mm = AUSMMachSplit(machR, false, false);
            // PW+'s quadratic Mach split is paired with the robust cubic pressure
            // split. The less dissipative alpha=3/16 choice produces negative
            // acoustic diffusion near the Sod plateau (physical Mach ~0.93).
            pressure = AUSMPressureSplit(machL, true, 0) * pL +
                       AUSMPressureSplit(machR, false, 0) * pR;
            real ratio = std::min(pL / pR, pR / pL);
            real weight = 1 - ratio * ratio * ratio;
            real stencilFactor = std::min(real(1), pressureStencilMin / std::min(pL, pR));
            stencilFactor *= stencilFactor;
            real fL = 0, fR = 0;
            if (pressure > 0)
            {
                fL = (pL / pressure - 1) * stencilFactor;
                fR = (pR / pressure - 1) * stencilFactor;
            }
            if (mp + mm >= 0)
            {
                coefficientL = aFace * (mp + mm * ((1 - weight) * (1 + fR) - fL));
                coefficientR = aFace * mm * weight * (1 + fR);
            }
            else
            {
                coefficientL = aFace * mp * weight * (1 + fL);
                coefficientR = aFace * (mm + mp * ((1 - weight) * (1 + fL) - fR));
            }
        }
        else
        {
            DNDS_assert(referenceMach > 0 && std::isfinite(referenceMach));
            real meanMachSqr = 0.5 * (machL * machL + machR * machR);
            real m0 = std::sqrt(std::min(real(1), std::max(meanMachSqr, referenceMach * referenceMach)));
            real fa = m0 * (2 - m0);
            real rhoMean = 0.5 * (UL(0) + UR(0));
            real massMach = AUSMMachSplit(machL, true, true) + AUSMMachSplit(machR, false, true) -
                            0.25 / fa * std::max(1 - meanMachSqr, real(0)) * (pR - pL) / (rhoMean * aFace * aFace);
            coefficientL = massMach >= 0 ? aFace * massMach : real(0);
            coefficientR = massMach < 0 ? aFace * massMach : real(0);
            if (variant == AUSMVariant::PlusUp)
            {
                real alpha = 3. / 16 * (-4 + 5 * fa * fa);
                real pp = AUSMPressureSplit(machL, true, alpha);
                real pm = AUSMPressureSplit(machR, false, alpha);
                pressure = pp * pL + pm * pR - 0.75 * pp * pm * (UL(0) + UR(0)) * fa * aFace * (unR - unL);
            }
            else
            {
                real pp = AUSMPressureSplit(machL, true, 0);
                real pm = AUSMPressureSplit(machR, false, 0);
                real velocityMagnitude = std::sqrt(0.5 * (relativeVL.squaredNorm() + relativeVR.squaredNorm()));
                pressure = 0.5 * (pL + pR) + 0.5 * (pp - pm) * (pL - pR) +
                           velocityMagnitude * (pp + pm - 1) * rhoMean * aFace;
            }
        }

        F.template head<dim + 2>() = coefficientL * UL.template head<dim + 2>() +
                                     coefficientR * UR.template head<dim + 2>();
        F.template segment<dim>(1) += pressure * n;
        F(dim + 1) += coefficientL * pL + coefficientR * pR + pressure * vg.dot(n);

        real unMean = 0.5 * (unL + unR);
        real acousticBound = std::sqrt(std::max(asqrL, asqrR));
        lam0 = std::abs(unMean - acousticBound);
        lam123 = std::abs(unMean);
        lam4 = std::abs(unMean + acousticBound);
    }
}
