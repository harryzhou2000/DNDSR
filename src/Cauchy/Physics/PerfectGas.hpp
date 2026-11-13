#pragma once

#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/Defines.hpp"
#include "Cauchy/Physics/FlowState.hpp"
#include "Physics/Transport/NewtonianFourier.hpp"
#include "Physics/Transport/MolecularViscosity.hpp"
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
        Physics::Transport::ViscosityModel viscosityModel = Physics::Transport::ViscosityModel::Constant;
        real viscosityReferenceTemperature = 1., sutherlandConstant = 0.;
        void Validate() const
        {
            DNDS_check_throw_info(std::isfinite(gamma) && gamma > 1 && std::isfinite(gasConstant) && gasConstant > 0 &&
                                      std::isfinite(viscosity) && viscosity >= 0 && std::isfinite(prandtl) && prandtl > 0,
                                  "invalid perfect-gas parameters");
            using Model = Physics::Transport::ViscosityModel;
            DNDS_check_throw_info((viscosityModel == Model::Constant || viscosityModel == Model::Sutherland || viscosityModel == Model::ConstantKinematic) &&
                                      std::isfinite(viscosityReferenceTemperature) && viscosityReferenceTemperature > 0 &&
                                      std::isfinite(sutherlandConstant) && sutherlandConstant >= 0,
                                  "invalid perfect-gas viscosity law");
        }
        /// Molecular coefficients use the actual point rho/T. Parameters are
        /// resolved once; the shared law is allocation-free and device-callable.
        DNDS_DEVICE_CALLABLE real MolecularViscosity(real density, real temperature) const
        {
            return Physics::Transport::MolecularViscosity(viscosityModel, viscosity, density, temperature,
                                                          viscosityReferenceTemperature, sutherlandConstant);
        }
        DNDS_DEVICE_CALLABLE real Pressure(const FlowState &state) const
        {
            real momentumSquared = 0;
            for (int d = 0; d < 3; ++d)
                momentumSquared += state[1 + d] * state[1 + d];
            return (gamma - 1) * (state[4] - 0.5 * momentumSquared / state[0]);
        }
        /// Conservative pressure derivative at a validated perfect-gas state.
        /// The energy convention excludes turbulent kinetic energy. RANS
        /// scalar derivatives are therefore zero and are appended by the
        /// equation-profile producer, independently of the rho*k stress term.
        DNDS_DEVICE_CALLABLE FlowState PressureDerivative(const FlowState &state) const noexcept
        {
            FlowState result{};
            real speedSquared = 0;
            for (int d = 0; d < 3; ++d)
            {
                const real velocity = state[1 + d] / state[0];
                speedSquared += velocity * velocity;
                result[1 + d] = -(gamma - 1) * velocity;
            }
            result[0] = .5 * (gamma - 1) * speedSquared;
            result[4] = gamma - 1;
            return result;
        }
        /// Column-major derivative of the physical normal flux with respect to
        /// all five conservative components. Thermodynamic derivatives belong
        /// to the model, independently of numerical Riemann dissipation.
        /// Caller checks state admissibility before evaluating a batch.
        DNDS_DEVICE_CALLABLE std::array<real, 25> FluxJacobian(const FlowState &state, const Vector3 &normal) const
        {
            std::array<real, 25> result{};
            Vector3 velocity{};
            real normalVelocity = 0, speedSquared = 0;
            for (int d = 0; d < 3; ++d)
            {
                velocity[d] = state[1 + d] / state[0];
                normalVelocity += velocity[d] * normal[d];
                speedSquared += velocity[d] * velocity[d];
            }
            const real pressure = Pressure(state);
            for (int column = 0; column < 5; ++column)
            {
                const real densityIncrement = column == 0 ? 1 : 0;
                const real energyIncrement = column == 4 ? 1 : 0;
                real pressureIncrement = (gamma - 1) * (energyIncrement + .5 * speedSquared * densityIncrement);
                real normalIncrement = 0, massIncrement = 0;
                for (int d = 0; d < 3; ++d)
                {
                    const real momentumIncrement = column == 1 + d ? 1 : 0;
                    pressureIncrement -= (gamma - 1) * velocity[d] * momentumIncrement;
                    normalIncrement += (momentumIncrement - velocity[d] * densityIncrement) / state[0] * normal[d];
                    massIncrement += momentumIncrement * normal[d];
                }
                result[column * 5] = massIncrement;
                for (int d = 0; d < 3; ++d)
                    result[column * 5 + 1 + d] = (column == 1 + d ? normalVelocity : 0) +
                                                 state[1 + d] * normalIncrement + pressureIncrement * normal[d];
                result[column * 5 + 4] = (energyIncrement + pressureIncrement) * normalVelocity + (state[4] + pressure) * normalIncrement;
            }
            return result;
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
        /// Chain rule from [rho,rho*u,rho*v,rho*w,rho*E] to [rho,u,v,w,T].
        /// State and gradient are point values, not conservative cell averages.
        /// The caller validates the linearization state outside this pure kernel.
        DNDS_DEVICE_CALLABLE FlowGradient PrimitiveGradient(const FlowState &state, const FlowGradient &gradient) const
        {
            FlowGradient result{};
            Vector3 velocity{};
            real speedSquared = 0;
            for (int j = 0; j < 3; ++j)
            {
                velocity[j] = state[1 + j] / state[0];
                speedSquared += velocity[j] * velocity[j];
            }
            const real pressure = Pressure(state);
            for (int d = 0; d < 3; ++d)
            {
                result[0][d] = gradient[0][d];
                real pressureGradient = gradient[4][d] + .5 * speedSquared * gradient[0][d];
                for (int j = 0; j < 3; ++j)
                {
                    result[1 + j][d] = (gradient[1 + j][d] - velocity[j] * gradient[0][d]) / state[0];
                    pressureGradient -= velocity[j] * gradient[1 + j][d];
                }
                pressureGradient *= gamma - 1;
                result[4][d] = (pressureGradient - pressure / state[0] * gradient[0][d]) / (state[0] * gasConstant);
            }
            return result;
        }
        /// Positive viscous flux; FV residual is -div(F_inviscid)+div(F_viscous).
        /// Stokes stress and Fourier heat conduction, with point mu and fixed Pr. The
        /// energy component is u dot (tau*n) + k grad(T) dot n.
        DNDS_DEVICE_CALLABLE FlowState ViscousFlux(const PrimitiveState &state, const FlowGradient &gradient, const Vector3 &normal) const
        {
            struct VelocityGradient
            {
                const FlowGradient &values;
                DNDS_DEVICE_CALLABLE real operator()(int i, int j) const { return values[1 + i][j]; }
            };
            const real mu = MolecularViscosity(state.density, state.temperature);
            const real conductivity = mu * gamma * gasConstant / ((gamma - 1) * prandtl);
            const auto flux = Physics::Transport::NewtonianFourierNormalFlux(state.velocity, VelocityGradient{gradient}, gradient[4], normal, mu, conductivity);
            FlowState result{};
            for (int i = 0; i < 3; ++i)
                result[1 + i] = flux.traction[i];
            result[4] = flux.energy;
            return result;
        }
    };
}
