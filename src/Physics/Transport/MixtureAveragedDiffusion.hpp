#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Physics::Transport
{
    /// Correct full-species raw normal mass fluxes J so their sum vanishes,
    /// then add -J to the independent species viscous flux and -sum(h*J) to
    /// its energy flux. F_total = F_inviscid - F_viscous. Full thermochemical
    /// species enthalpies are required, not bookkeeping base-energy offsets.
    ///
    /// fractions are the raw transported composition, not property-repaired Y.
    /// rhoInv carries the caller's established density-floor policy. Output is
    /// additive and must not alias input views. Independent output has Ns-1
    /// entries; all input views have Ns entries. Views preserve their strides.
    template <class TY, class TJ, class TH, class TFlux>
    void AddCorrectedMixtureAveragedNormalFlux(real rho, real rhoInv,
                                               const TY &fractions, const TJ &rawFlux,
                                               const TH &enthalpy, int species,
                                               TFlux &&independentFlux, real &energyFlux)
    {
        real sumRaw = 0;
        for (int k = 0; k < species; ++k)
            sumRaw += rawFlux[k];
        const real correctionVelocity = -sumRaw * rhoInv;
        for (int k = 0; k < species; ++k)
        {
            const real flux = -(rawFlux[k] + rho * fractions[k] * correctionVelocity);
            if (k < species - 1)
                independentFlux[k] += flux;
            energyFlux += enthalpy[k] * flux;
        }
    }
}
