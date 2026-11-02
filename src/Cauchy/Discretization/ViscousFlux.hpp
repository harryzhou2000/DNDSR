#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"

namespace DNDS::Cauchy
{
    /// Euler's polynomial face-gradient construction, expressed with explicit
    /// point values. Convective states set the thermodynamic linearization;
    /// viscous states/derivatives may come from a distinct unlimited polynomial.
    /// The caller maps both sides into one frame and supplies a unit normal,
    /// nonnegative inverse distance and admissible linearization states.
    /// No allocation or communication.
    inline FlowState PolynomialViscousFlux(const EquationSystem &equation,
                                           const FlowState &left, const FlowState &right,
                                           const FlowState &viscousLeft, const FlowState &viscousRight,
                                           const FlowGradient &gradientLeft, const FlowGradient &gradientRight,
                                           const Vector3 &normal, real inverseDistance,
                                           bool adiabatic)
    {
        FlowState mean{};
        for (int j = 0; j < 5; ++j)
            mean[j] = .5 * (left[j] + right[j]);
        FlowGradient gradient{};
        const auto &gas = equation.Gas();
        for (int j = 0; j < 5; ++j)
            for (int d = 0; d < 3; ++d)
                gradient[j][d] = .5 * (gradientLeft[j][d] + gradientRight[j][d]) + normal[d] * (viscousRight[j] - viscousLeft[j]) * inverseDistance;
        gradient = equation.PreparePrimitiveGradient(mean, gradient);
        if (adiabatic)
        {
            real normalTemperature = 0;
            for (int d = 0; d < 3; ++d)
                normalTemperature += gradient[4][d] * normal[d];
            for (int d = 0; d < 3; ++d)
                gradient[4][d] -= normalTemperature * normal[d];
        }
        return gas.ViscousFlux(gas.Primitive(mean), gradient, normal);
    }
}
