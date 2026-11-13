#pragma once
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/FrozenFlowCharacteristics.hpp"

namespace DNDS::Cauchy
{
    /** @brief Perfect-gas mean-state eigenbasis with a bounded analytic inverse.
     * @details The basis uses the same Roe velocity/enthalpy, tangent convention
     * and column ordering as EquationSystem::PrepareCharacteristics. For a
     * conservative perturbation, pressure is
     * \f$dp=(\gamma-1)(dE-u\cdot dm+|u|^2 d\rho/2)\f$.
     * Acoustic coefficients are \f$(dp/c^2 \pm (dm_n-u_n d\rho)/c)/2\f$;
     * contact and shear coefficients follow by subtracting those waves.
     * Shared by the CPU and CUDA perfect-gas leaves, with no numerical matrix
     * inversion. Both products are checked at the existing fallback
     * tolerance; ill-conditioned valid states return identity maps. Invalid
     * physical inputs return status and must not be published as valid states.
     * No allocation, exception, lease or synchronization occurs here.
     * Thermodynamics is evaluated inline from the two means; no auxiliary
     * cache, gamma-equivalent substitution or reconstructed state is used.
     */
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowCharacteristics> TryPerfectGasCharacteristics(
        const PerfectGas &gas, const FlowState &left, const FlowState &right, const Vector3 &faceNormal) noexcept
    {
        if (!gas.Admissible(left) || !gas.Admissible(right))
            return NumericalStatus{NumericalError::InvalidState, 1};
        const real rl = std::sqrt(left[0]), rr = std::sqrt(right[0]);
        Eigen::Vector3d velocity;
        for (int d = 0; d < 3; ++d)
            velocity[d] = (left[1 + d] / rl + right[1 + d] / rr) / (rl + rr);
        const real enthalpy = ((left[4] + gas.Pressure(left)) / rl + (right[4] + gas.Pressure(right)) / rr) / (rl + rr);
        const real kinetic = .5 * velocity.squaredNorm();
        const real c2 = (gas.gamma - 1) * (enthalpy - kinetic);
        return TryFrozenFlowCharacteristics({velocity[0], velocity[1], velocity[2]},
                                            enthalpy, c2, gas.gamma - 1, kinetic, faceNormal);
    }
}
