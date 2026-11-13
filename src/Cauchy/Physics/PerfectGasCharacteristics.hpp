#pragma once
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/FlowCharacteristics.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include <Eigen/Geometry>

namespace DNDS::Cauchy
{
    /** @brief Perfect-gas mean-state eigenbasis with a bounded analytic inverse.
     * @details The basis uses the same Roe velocity/enthalpy, tangent convention
     * and column ordering as EquationSystem::PrepareCharacteristics. For a
     * conservative perturbation, pressure is
     * \f$dp=(\gamma-1)(dE-u\cdot dm+|u|^2 d\rho/2)\f$.
     * Acoustic coefficients are \f$(dp/c^2 \pm (dm_n-u_n d\rho)/c)/2\f$;
     * contact and shear coefficients follow by subtracting those waves.
     * This avoids a per-face device LU. CPU's existing Eigen inverse remains
     * the numerical anchor. Both products are checked at the CPU fallback
     * tolerance; ill-conditioned valid states return identity maps. Invalid
     * physical inputs return status and must not be published as valid states.
     * No allocation, exception, lease or synchronization occurs here.
     */
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowCharacteristics> TryPerfectGasCharacteristics(
        const PerfectGas &gas, const FlowState &left, const FlowState &right, const Vector3 &faceNormal) noexcept
    {
        if (!gas.Admissible(left) || !gas.Admissible(right))
            return NumericalStatus{NumericalError::InvalidState, 1};
        const Eigen::Vector3d normal{faceNormal[0], faceNormal[1], faceNormal[2]};
        if (!normal.allFinite() || std::abs(normal.squaredNorm() - 1) > 1e-10)
            return NumericalStatus{NumericalError::InvalidInput, 2};
        const real rl = std::sqrt(left[0]), rr = std::sqrt(right[0]);
        Eigen::Vector3d velocity;
        for (int d = 0; d < 3; ++d)
            velocity[d] = (left[1 + d] / rl + right[1 + d] / rr) / (rl + rr);
        const real enthalpy = ((left[4] + gas.Pressure(left)) / rl + (right[4] + gas.Pressure(right)) / rr) / (rl + rr);
        const real kinetic = .5 * velocity.squaredNorm();
        const real c2 = (gas.gamma - 1) * (enthalpy - kinetic);
        if (!(std::isfinite(c2) && c2 > 0))
            return NumericalStatus{NumericalError::InvalidState, 3};
        const real sound = std::sqrt(c2), un = velocity.dot(normal);
        const Eigen::Vector3d tangent = normal.unitOrthogonal(), second = normal.cross(tangent);
        FlowCharacteristics result{};
        result.soundSpeed = sound;
        for (int d = 0; d < 3; ++d)
            result.velocity[d] = velocity[d];
        auto &r = result.inverse;
        r[0] = r[15] = r[20] = 1;
        r[4] = kinetic;
        r[9] = velocity.dot(tangent);
        r[14] = velocity.dot(second);
        r[19] = enthalpy + sound * un;
        r[24] = enthalpy - sound * un;
        for (int d = 0; d < 3; ++d)
        {
            r[1 + d] = velocity[d];
            r[6 + d] = tangent[d];
            r[11 + d] = second[d];
            r[16 + d] = velocity[d] + sound * normal[d];
            r[21 + d] = velocity[d] - sound * normal[d];
        }
        for (int column = 0; column < 5; ++column)
        {
            const real dr = column == 0;
            const Eigen::Vector3d dm{real(column == 1), real(column == 2), real(column == 3)};
            const real dp = (gas.gamma - 1) * ((column == 4) - velocity.dot(dm) + kinetic * dr);
            const real acoustic = (normal.dot(dm) - un * dr) / sound;
            result.forward[5 * column] = dr - dp / c2;
            result.forward[5 * column + 1] = tangent.dot(dm) - velocity.dot(tangent) * dr;
            result.forward[5 * column + 2] = second.dot(dm) - velocity.dot(second) * dr;
            result.forward[5 * column + 3] = .5 * (dp / c2 + acoustic);
            result.forward[5 * column + 4] = .5 * (dp / c2 - acoustic);
        }
        real errors[2]{}, norms[2]{};
        bool valid = true;
        for (int j = 0; j < 25; ++j)
            valid &= std::isfinite(result.forward[j]) && std::isfinite(r[j]);
        for (int direction = 0; direction < 2; ++direction)
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                {
                    real value = 0;
                    for (int k = 0; k < 5; ++k)
                        value += direction ? r[i + 5 * k] * result.forward[k + 5 * j] : result.forward[i + 5 * k] * r[k + 5 * j];
                    norms[direction] += value * value;
                    value -= i == j;
                    errors[direction] += value * value;
                }
        for (int d = 0; d < 2; ++d)
            valid &= std::isfinite(errors[d]) && std::isfinite(norms[d]) && errors[d] <= 1e-22 * std::min(real(5), norms[d]);
        if (!valid)
        {
            for (int j = 0; j < 25; ++j)
                result.forward[j] = r[j] = (j % 6 == 0);
            result.componentwiseFallback = true;
        }
        return result;
    }
}
