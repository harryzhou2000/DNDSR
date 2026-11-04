#pragma once

#include "DNDS/Core/Defines.hpp"
#include <array>

namespace DNDS::Physics::Thermodynamics
{
    /// Pure compressible kinematics, with no EOS/energy-reference assumption.
    /// Flow rows are [rho, momentum(Dim), rhoE]. Callers validate positive rho,
    /// sizes and access once; row types need only operator[]. Dim is 2 or 3.
    template <int Dim, class TFlow>
    std::array<real, Dim> ConservativeVelocity(const TFlow &flow)
    {
        static_assert(Dim == 2 || Dim == 3);
        std::array<real, Dim> velocity;
        for (int d = 0; d < Dim; ++d)
            velocity[d] = flow[1 + d] / flow[0];
        return velocity;
    }
    template <int Dim, class TFlow>
    real ConservativeSpecificInternalEnergy(const TFlow &flow)
    {
        const auto velocity = ConservativeVelocity<Dim>(flow);
        real speedSquared = 0;
        for (int d = 0; d < Dim; ++d)
            speedSquared += sqr(velocity[d]);
        return flow[Dim + 1] / flow[0] - .5 * speedSquared;
    }
    /// specificEnergy includes any thermochemical formation/reference energy.
    template <int Dim, class TVelocity>
    real ConservativeTotalEnergyDensity(real rho, const TVelocity &velocity, real specificEnergy)
    {
        static_assert(Dim == 2 || Dim == 3);
        real speedSquared = 0;
        for (int d = 0; d < Dim; ++d)
            speedSquared += sqr(velocity[d]);
        return rho * (specificEnergy + .5 * speedSquared);
    }
    /// One directional derivative d(rho*e), including all kinetic terms.
    template <int Dim, class TFlow, class TDerivative>
    real ConservativeInternalEnergyDensityDerivative(const TFlow &flow, const TDerivative &derivative)
    {
        const auto velocity = ConservativeVelocity<Dim>(flow);
        real speedSquared = 0, momentumDerivative = 0;
        for (int d = 0; d < Dim; ++d)
        {
            speedSquared += sqr(velocity[d]);
            momentumDerivative += velocity[d] * derivative[1 + d];
        }
        return derivative[Dim + 1] - momentumDerivative + .5 * speedSquared * derivative[0];
    }
    /// d(phi) from conservative d(rho*phi), e.g. velocity, species or RANS.
    inline real ConservativeSpecificDerivative(real rho, real phi, real densityDerivative, real conservedDerivative)
    {
        return (conservedDerivative - phi * densityDerivative) / rho;
    }
}
