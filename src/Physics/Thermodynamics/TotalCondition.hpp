#pragma once

#include "Physics/ReferenceScales.hpp"
#include "Physics/Chemistry/ChemicalSource.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Physics::Thermodynamics
{
    /// Code-unit static state and accepted speed after Euler's kinetic-energy
    /// safeguard. No velocity direction, conservative layout or BC policy here.
    struct TotalConditionStaticState
    {
        real density = 0, pressure = 0, temperature = 0, speedSquared = 0;
    };

    /// Calorically perfect closure: h(T)+|v|^2/2=h(T0), s(T,p)=s(T0,p0).
    /// Matches Euler's cap |v|^2 <= 0.95*2*cp*T0. Inputs use code units.
    NumericalResult<TotalConditionStaticState> TotalToStaticPerfectGas(
        real pressure, real temperature, real speedSquared, real gasConstant, real gamma) noexcept;

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
