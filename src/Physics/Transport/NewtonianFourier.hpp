#pragma once

#include "DNDS/Core/Defines.hpp"
#include <array>

namespace DNDS::Physics::Transport
{
    struct NewtonianFourierFlux
    {
        std::array<real, 3> traction{};
        real energy = 0;
    };
    /// Stokes/Newtonian stress and Fourier heat flux in a common Cartesian
    /// frame. Inputs are ready point values in one unit system; velocityGradient
    /// has (component,direction) indexing. No EOS, allocation, validation, mesh,
    /// normal normalization or communication. Caller supplies effective mu/k,
    /// including any explicitly prepared turbulence augmentation. This kernel
    /// implements the linear Newtonian law, not a QCR stress correction.
    template <class TVelocity, class TVelocityGradient, class TTemperatureGradient, class TNormal>
    DNDS_DEVICE_CALLABLE NewtonianFourierFlux NewtonianFourierNormalFlux(const TVelocity &velocity, const TVelocityGradient &velocityGradient,
                                                                         const TTemperatureGradient &temperatureGradient, const TNormal &normal,
                                                                         real viscosity, real conductivity)
    {
        real divergence = 0;
        for (int d = 0; d < 3; ++d)
            divergence += velocityGradient(d, d);
        NewtonianFourierFlux result;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
                result.traction[i] += viscosity * (velocityGradient(i, j) + velocityGradient(j, i) - (i == j ? (2. / 3) * divergence : 0)) * normal[j];
            result.energy += velocity[i] * result.traction[i];
        }
        for (int d = 0; d < 3; ++d)
            result.energy += conductivity * temperatureGradient[d] * normal[d];
        return result;
    }
}
