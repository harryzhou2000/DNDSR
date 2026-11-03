#pragma once

#include "DNDS/Core/Defines.hpp"

namespace DNDS::Physics::Thermodynamics
{
    /// Exact directional temperature derivative of an ideal mixture. All inputs
    /// use one unit system. The caller supplies d(rho*e), after kinetic-energy
    /// differentiation, full species internal energies and Ns-1 conservative
    /// species derivatives. Views need only operator[]; no storage layout or
    /// momentum ordering is assumed here. Temporary property composition may be
    /// repaired, while conservative rhoY derivatives remain unprojected.
    template <class TY, class TE, class TDY>
    real IdealMixtureTemperatureDerivative(real rho, real internalEnergy, real cv,
                                           real densityDerivative, real internalEnergyDensityDerivative,
                                           const TY &fractions, const TE &speciesEnergy,
                                           const TDY &speciesDensityDerivative, int species)
    {
        real compositionTerm = 0;
        for (int k = 0; k < species - 1; ++k)
        {
            const real rhoGradY = speciesDensityDerivative[k] - fractions[k] * densityDerivative;
            compositionTerm += (speciesEnergy[k] - speciesEnergy[species - 1]) * rhoGradY;
        }
        return (internalEnergyDensityDerivative - internalEnergy * densityDerivative - compositionTerm) / (rho * cv);
    }
}
