#pragma once

#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "Physics/Thermodynamics/ConservativeConversion.hpp"

namespace DNDS::Cauchy
{
    /** Perfect-gas specialization of PrepareViscousFlux, using the same
     * conservative chain rule as the generic CPU operator. Transport/EOS have
     * already been evaluated at state. Turbulent coefficients are explicit;
     * this function neither selects a RANS model nor performs a point repair.
     * Positive viscous flux is subtracted from the convective normal flux.
     */
    DNDS_DEVICE_CALLABLE inline FlowState PreparePerfectGasViscousFlux(
        const FlowState &state, const FlowGradient &gradient, const PerfectGasPoint &point,
        const Vector3 &normal, real turbulentViscosity, real turbulentConductivity,
        bool adiabatic, bool useQCR) noexcept
    {
        namespace T = Physics::Thermodynamics;
        const auto velocity = T::ConservativeVelocity<3>(state);
        const real internalEnergy = T::ConservativeSpecificInternalEnergy<3>(state);
        const real viscosity = point.transport.viscosity + turbulentViscosity;
        const real conductivity = adiabatic ? 0 : point.transport.conductivity + turbulentConductivity;
        struct VelocityGradient
        {
            std::array<Vector3, 3> values{};
            DNDS_DEVICE_CALLABLE real operator()(int i, int d) const { return values[i][d]; }
        };
        VelocityGradient derivative;
        Vector3 temperatureDerivative{};
        for (int d = 0; d < 3; ++d)
        {
            for (int i = 0; i < 3; ++i)
                derivative.values[i][d] = T::ConservativeSpecificDerivative(state[0], velocity[i], gradient[0][d], gradient[1 + i][d]);
            if (conductivity != 0)
            {
                FlowState direction;
                for (int j = 0; j < 5; ++j)
                    direction[j] = gradient[j][d];
                const real internalDerivative = T::ConservativeInternalEnergyDensityDerivative<3>(state, direction);
                temperatureDerivative[d] = (internalDerivative - internalEnergy * gradient[0][d]) / (state[0] * point.thermo.cv);
            }
        }
        const auto flux = Physics::Transport::NewtonianFourierNormalFlux(velocity, derivative, temperatureDerivative, normal,
                                                                         viscosity, conductivity, useQCR, turbulentViscosity / (viscosity + verySmallReal));
        return {0, flux.traction[0], flux.traction[1], flux.traction[2], flux.energy};
    }
}
