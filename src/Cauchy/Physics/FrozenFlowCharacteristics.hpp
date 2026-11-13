#pragma once
#include "Cauchy/Physics/FlowCharacteristics.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include <algorithm>
#include <Eigen/Geometry>

namespace DNDS::Cauchy
{
    /** @brief Analytic flow-five basis at a supplied frozen-composition state.
     * @details Columns are contact, two shear waves, acoustic plus and minus.
     * With \f$q=H-a^2/\beta\f$, the contact energy is q and the pressure
     * differential is \f$dp=\beta[d(\rho E)-u\cdot d(\rho u)+
     * (|u|^2-q)d\rho]\f$. This retains thermochemical energy offsets without
     * a caloric-gamma substitution. Species/RANS remain separate scalars;
     * this is not the eigensystem of the full reacting species system.
     *
     * The caller supplies consistent face values. The perfect-gas adapter
     * supplies its historical Roe state; the mixture adapter supplies an
     * approximate averaged limiter state. No EOS call or matrix inversion is
     * performed. The historical ill-conditioned-basis identity fallback is
     * preserved, separately from optional diagnostic validation in CFV.
     */
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowCharacteristics> TryFrozenFlowCharacteristics(
        const Vector3 &flowVelocity, real enthalpy, real soundSquared, real beta,
        real contactEnergy, const Vector3 &faceNormal) noexcept
    {
        const Eigen::Vector3d velocity{flowVelocity[0], flowVelocity[1], flowVelocity[2]};
        const Eigen::Vector3d normal{faceNormal[0], faceNormal[1], faceNormal[2]};
        if (!normal.allFinite() || std::abs(normal.squaredNorm() - 1) > 1e-10)
            return NumericalStatus{NumericalError::InvalidInput, 2};
        if (!velocity.allFinite() || !std::isfinite(enthalpy) || !std::isfinite(contactEnergy) ||
            !(std::isfinite(soundSquared) && soundSquared > 0) || !(std::isfinite(beta) && beta > 0))
            return NumericalStatus{NumericalError::InvalidState, 3};
        const real sound = std::sqrt(soundSquared), un = velocity.dot(normal);
        const Eigen::Vector3d tangent = normal.unitOrthogonal(), second = normal.cross(tangent);
        FlowCharacteristics result{};
        result.soundSpeed = sound;
        result.velocity = flowVelocity;
        auto &r = result.inverse;
        r[0] = r[15] = r[20] = 1;
        r[4] = contactEnergy;
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
        const real densityCoefficient = velocity.squaredNorm() - contactEnergy;
        for (int column = 0; column < 5; ++column)
        {
            const real dr = column == 0;
            const Eigen::Vector3d dm{real(column == 1), real(column == 2), real(column == 3)};
            const real dp = beta * ((column == 4) - velocity.dot(dm) + densityCoefficient * dr);
            const real acoustic = (normal.dot(dm) - un * dr) / sound;
            result.forward[5 * column] = dr - dp / soundSquared;
            result.forward[5 * column + 1] = tangent.dot(dm) - velocity.dot(tangent) * dr;
            result.forward[5 * column + 2] = second.dot(dm) - velocity.dot(second) * dr;
            result.forward[5 * column + 3] = .5 * (dp / soundSquared + acoustic);
            result.forward[5 * column + 4] = .5 * (dp / soundSquared - acoustic);
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

    /** @brief Gather two cached cell thermodynamic records into one limiter basis.
     * @details Square-root-density weights average velocity, H, a^2 and beta.
     * This positive averaging is an approximate frozen-composition limiter
     * basis, not an exact mixture Roe linearization. Both matrices consume the
     * same tuple. Scalar thermodynamics is invariant under periodic rotations;
     * the caller rotates the neighbor momentum before entering this function.
     */
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowCharacteristics> TryCachedMeanCharacteristics(
        const FlowState &left, const FlowState &right,
        const CharacteristicThermodynamics &leftThermo, const CharacteristicThermodynamics &rightThermo,
        const Vector3 &faceNormal) noexcept
    {
        if (!(std::isfinite(left[0]) && left[0] > 0 && std::isfinite(right[0]) && right[0] > 0))
            return NumericalStatus{NumericalError::InvalidState, 1};
        for (int j = 1; j < 5; ++j)
            if (!std::isfinite(left[j]) || !std::isfinite(right[j]))
                return NumericalStatus{NumericalError::InvalidState, 1};
        if (!leftThermo.Valid() || !rightThermo.Valid())
            return NumericalStatus{NumericalError::InvalidState, 4};
        const real rl = std::sqrt(left[0]), rr = std::sqrt(right[0]);
        const real wl = rl / (rl + rr), wr = rr / (rl + rr);
        Vector3 velocity;
        for (int d = 0; d < 3; ++d)
            velocity[d] = (left[d + 1] / rl + right[d + 1] / rr) / (rl + rr);
        const real h = wl * leftThermo.totalEnthalpy + wr * rightThermo.totalEnthalpy;
        const real a2 = wl * leftThermo.soundSpeed * leftThermo.soundSpeed + wr * rightThermo.soundSpeed * rightThermo.soundSpeed;
        const real beta = wl * leftThermo.pressureInternalEnergyDerivative + wr * rightThermo.pressureInternalEnergyDerivative;
        return TryFrozenFlowCharacteristics(velocity, h, a2, beta, h - a2 / beta, faceNormal);
    }
}
