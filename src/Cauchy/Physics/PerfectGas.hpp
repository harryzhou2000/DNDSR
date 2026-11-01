#pragma once

#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/Defines.hpp"
#include "Cauchy/Physics/FlowState.hpp"
#include <cmath>

namespace DNDS::Cauchy
{
    /// Pure five-component kernel parameters. Resolve physics/backend choices
    /// outside the loop; no mesh, communication, allocation, or callback occurs
    /// here. Future EOS/equation kernels may expose different local profiles
    /// behind the solver's coarse equation dispatch without virtual face calls.
    struct PerfectGas
    {
        real gamma = 1.4, gasConstant = 1., viscosity = 0., prandtl = 0.72;
        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(gamma) && gamma > 1 && std::isfinite(gasConstant) && gasConstant > 0 &&
                                      std::isfinite(viscosity) && viscosity >= 0 && std::isfinite(prandtl) && prandtl > 0,
                                  "invalid perfect-gas parameters");
        }
        DNDS_DEVICE_CALLABLE real Pressure(const FlowState &state) const
        {
            real momentumSquared = 0;
            for (int d = 0; d < 3; ++d)
                momentumSquared += state[1 + d] * state[1 + d];
            return (gamma - 1) * (state[4] - 0.5 * momentumSquared / state[0]);
        }
        DNDS_DEVICE_CALLABLE bool Admissible(const FlowState &state, real densityFloor = 0, real pressureFloor = 0) const
        {
            for (real value : state)
                if (!std::isfinite(value))
                    return false;
            return state[0] > densityFloor && Pressure(state) > pressureFloor;
        }
        DNDS_DEVICE_CALLABLE PrimitiveState Primitive(const FlowState &state) const
        {
            const real pressure = Pressure(state);
            PrimitiveState result{state[0], pressure, pressure / (state[0] * gasConstant), std::sqrt(gamma * pressure / state[0]), {}};
            for (int d = 0; d < 3; ++d)
                result.velocity[d] = state[1 + d] / state[0];
            return result;
        }
        /// Input is [rho,u,v,w,p]. Caller validates positive rho,p once at its
        /// data boundary; these pure math functions do not throw in hot loops.
        DNDS_DEVICE_CALLABLE FlowState Conservative(const FlowState &primitive) const
        {
            FlowState result{primitive[0], 0, 0, 0, primitive[4] / (gamma - 1)};
            for (int d = 0; d < 3; ++d)
            {
                result[1 + d] = primitive[0] * primitive[1 + d];
                result[4] += 0.5 * primitive[0] * primitive[1 + d] * primitive[1 + d];
            }
            return result;
        }
        /// Positive viscous flux; FV residual is -div(F_inviscid)+div(F_viscous).
        /// Stokes stress and Fourier heat conduction, with constant mu/Pr. The
        /// energy component is u dot (tau*n) + k grad(T) dot n.
        DNDS_DEVICE_CALLABLE FlowState ViscousFlux(const PrimitiveState &state, const FlowGradient &gradient, const Vector3 &normal) const
        {
            real divergence = 0;
            for (int d = 0; d < 3; ++d)
                divergence += gradient[1 + d][d];
            FlowState result{};
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    result[1 + i] += viscosity * (gradient[1 + i][j] + gradient[1 + j][i] - (i == j ? (2. / 3) * divergence : 0)) * normal[j];
                result[4] += state.velocity[i] * result[1 + i];
            }
            const real conductivity = viscosity * gamma * gasConstant / ((gamma - 1) * prandtl);
            for (int d = 0; d < 3; ++d)
                result[4] += conductivity * gradient[4][d] * normal[d];
            return result;
        }
    };
}
