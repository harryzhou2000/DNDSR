#pragma once

#include "Cauchy/Physics/FlowState.hpp"

namespace DNDS::Cauchy
{
    /// Physical compressible ALE flux derivative at a ready thermodynamic
    /// point. Pressure and its conservative directional derivative are explicit
    /// EOS/energy inputs. No acoustic approximation, gamma substitution, property
    /// mutation or numerical dissipation belongs here. Normal/grid velocity are
    /// fixed for this derivative; all three momentum components are retained.
    DNDS_DEVICE_CALLABLE inline FlowState PrepareCompressibleFluxDerivative(const FlowState &state, const FlowState &direction,
                                                                            real pressure, real pressureDerivative, const Vector3 &normal, const Vector3 &gridVelocity = {})
    {
        real normalVelocity = 0, normalGridVelocity = 0, velocityDerivative = 0;
        for (int d = 0; d < 3; ++d)
        {
            normalVelocity += state[1 + d] / state[0] * normal[d];
            normalGridVelocity += gridVelocity[d] * normal[d];
            velocityDerivative += (direction[1 + d] - state[1 + d] / state[0] * direction[0]) / state[0] * normal[d];
        }
        const real relativeVelocity = normalVelocity - normalGridVelocity;
        FlowState result{};
        result[0] = direction[0] * relativeVelocity + state[0] * velocityDerivative;
        for (int d = 0; d < 3; ++d)
            result[1 + d] = direction[1 + d] * relativeVelocity + state[1 + d] * velocityDerivative + pressureDerivative * normal[d];
        result[4] = direction[4] * relativeVelocity + (state[4] + pressure) * velocityDerivative + pressureDerivative * normalVelocity;
        return result;
    }
    /// Scalar conservative transport paired with the same flow linearization.
    /// Scalar is rho*q, never q; grid speed is frozen in relativeNormalVelocity.
    DNDS_DEVICE_CALLABLE inline real PrepareAdvectedScalarFluxDerivative(real scalar, real direction,
                                                                         real relativeNormalVelocity, real normalVelocityDerivative)
    {
        return direction * relativeNormalVelocity + scalar * normalVelocityDerivative;
    }
}
