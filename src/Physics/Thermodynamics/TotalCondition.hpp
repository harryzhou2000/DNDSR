#pragma once

#include "Physics/ReferenceScales.hpp"
#include "Physics/Chemistry/ChemicalSource.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include <algorithm>
#include <cmath>

namespace DNDS::Physics::Thermodynamics
{
    /// Code-unit static state and accepted speed after Euler's kinetic-energy
    /// safeguard. No velocity direction, conservative layout or BC policy here.
    struct TotalConditionStaticState
    {
        real density = 0, pressure = 0, temperature = 0, speedSquared = 0;
    };

    namespace Detail
    {
        DNDS_DEVICE_CALLABLE inline bool ValidTotalConditionInput(real p, real t, real v2)
        {
            return std::isfinite(p) && p > 0 && std::isfinite(t) && t > 0 &&
                   std::isfinite(v2) && v2 >= 0;
        }
    }
    /// Calorically perfect closure: h(T)+|v|^2/2=h(T0), s(T,p)=s(T0,p0).
    /// Matches Euler's cap |v|^2 <= 0.95*2*cp*T0. Inputs use code units.
    /// Shared CPU/device arithmetic; no thermochemistry worker or exception.
    DNDS_DEVICE_CALLABLE inline NumericalResult<TotalConditionStaticState> TotalToStaticPerfectGas(
        real pressure, real temperature, real speedSquared, real gasConstant, real gamma) noexcept
    {
        if (!Detail::ValidTotalConditionInput(pressure, temperature, speedSquared) ||
            !(std::isfinite(gasConstant) && gasConstant > 0 && std::isfinite(gamma) && gamma > 1))
            return NumericalStatus{NumericalError::InvalidState, 1, -1};
        const real cp = gamma * gasConstant / (gamma - 1);
        const real v2 = std::min(speedSquared, .95 * 2 * cp * temperature);
        const real t = temperature - .5 * v2 / cp;
        const real p = pressure * std::pow(t / temperature, gamma / (gamma - 1));
        const real rho = p / (gasConstant * t);
        if (!Detail::ValidTotalConditionInput(p, t, v2) || !(std::isfinite(rho) && rho > 0))
            return NumericalStatus{NumericalError::InvalidState, 2, -1};
        return TotalConditionStaticState{rho, p, t, v2};
    }

    /// Existing Euler frozen-composition ideal-mixture closure. Uses one
    /// caller-owned CPU backend; no new worker or scratch allocation. Fractions
    /// contain all species, normalized by the caller, and remain unchanged.
    /// h(T)=h0-K is bisected with Euler's 60-step/1e-10 criterion; pressure
    /// follows constant entropy. The 0.95 enthalpy cap and mechanism Tmin rule
    /// match Euler. Cantera exceptions are intercepted at this physics adapter,
    /// never propagated into a numerical kernel or interpreted as recovery.
    NumericalResult<TotalConditionStaticState> TotalToStaticIdealMixture(
        Chemistry::ChemicalSource &source, const ReferenceScales &scales,
        real pressure, real temperature, real speedSquared,
        Chemistry::ConstSpeciesBufferView fractions) noexcept;
}
