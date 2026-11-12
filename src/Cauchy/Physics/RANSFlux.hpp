#pragma once

#include "Cauchy/Physics/FlowState.hpp"

namespace DNDS::Cauchy
{
    /// Euler two-equation isotropic Reynolds stress, evaluated at the
    /// arithmetic conservative face state. This is separate from EOS pressure
    /// and Riemann acoustics. Energy work uses absolute normal velocity in the
    /// mesh-bound axes, including for ALE; grid velocity is not subtracted.
    DNDS_DEVICE_CALLABLE inline FlowState TurbulentNormalFlux(const FlowState &state, real rhoK, const Vector3 &normal)
    {
        FlowState flux{};
        const real pressure = (2. / 3.) * rhoK;
        for (int d = 0; d < 3; ++d)
        {
            flux[1 + d] = pressure * normal[d];
            flux[4] += flux[1 + d] * state[1 + d] / state[0];
        }
        return flux;
    }
    /// Directional conservative derivative of TurbulentNormalFlux. Callers
    /// map the scalar direction through the logical RANS block; it is not
    /// necessarily component five in physical storage. No gamma/EOS closure.
    DNDS_DEVICE_CALLABLE inline FlowState TurbulentNormalFluxDerivative(const FlowState &state, const FlowState &direction,
                                                                        real rhoK, real deltaRhoK, const Vector3 &normal)
    {
        FlowState flux{};
        for (int d = 0; d < 3; ++d)
        {
            const real velocity = state[1 + d] / state[0];
            flux[1 + d] = (2. / 3.) * deltaRhoK * normal[d];
            flux[4] += (2. / 3.) * normal[d] * (deltaRhoK * velocity + rhoK * (direction[1 + d] - velocity * direction[0]) / state[0]);
        }
        return flux;
    }
}
