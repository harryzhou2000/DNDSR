#pragma once

#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"

namespace DNDS::Cauchy
{
    /// Value owner for optional model-prepared data. Input() creates borrowed
    /// pointers after this owner reaches its final location; copies/moves never
    /// leave embedded pointers referring to an old stack object.
    struct PreparedRiemannData
    {
        Riemann::AcousticState leftMean, rightMean;
        Riemann::RoeData roe;
        Riemann::WaveBounds outerWaves;
        Riemann::AUSMData ausm;
        Riemann::Requirements required;
        DNDS_DEVICE_CALLABLE Riemann::FluxInput Input(Riemann::AcousticState left, Riemann::AcousticState right,
                                                      Vector3 normal, Vector3 gridVelocity = {}) const
        {
            return {left, right, normal, gridVelocity, required.meanStates ? &leftMean : nullptr, required.meanStates ? &rightMean : nullptr,
                    required.roe ? &roe : nullptr, required.outerWaveBounds ? &outerWaves : nullptr, required.ausmFaceSpeed ? &ausm : nullptr};
        }
    };
    /** @brief Perfect-gas Roe, shock and AUSM physical input producer.
     * @details Host/device shared arithmetic. Inputs are already admissible,
     * in code units and one common Cartesian frame. This is the physics side
     * of the gas-agnostic Riemann boundary: it supplies enthalpy, wave estimates
     * and the acoustic scale, without allocating or acquiring any service.
     * Keep the returned owner alive while its Input() view is consumed.
     */
    DNDS_DEVICE_CALLABLE inline PreparedRiemannData PreparePerfectGasRiemann(
        const PerfectGas &gas, Riemann::Solver solver,
        const Riemann::AcousticState &left, const Riemann::AcousticState &right,
        const Riemann::AcousticState &leftMeanData, const Riemann::AcousticState &rightMeanData,
        const Vector3 &normal, const Vector3 &gridVelocity = {}) noexcept
    {
        PreparedRiemannData data{};
        data.required = Riemann::RequiredInputs(solver);
        if (!data.required.meanStates)
            return data;
        data.leftMean = leftMeanData;
        data.rightMean = rightMeanData;
        const auto &leftMean = leftMeanData.flow.conservative, &rightMean = rightMeanData.flow.conservative;
        const auto velocity = [](const FlowState &state)
        { Vector3 value; for (int d=0;d<3;++d) value[d]=state[1+d]/state[0]; return value; };
        const auto dot = [](const Vector3 &a, const Vector3 &b)
        { real sum=0; for(int d=0;d<3;++d) sum+=a[d]*b[d];return sum; };
        if (data.required.roe)
        {
            const real rootL = std::sqrt(leftMean[0]), rootR = std::sqrt(rightMean[0]), sum = rootL + rootR;
            auto &roe = data.roe;
            const auto vL = velocity(leftMean), vR = velocity(rightMean);
            for (int d = 0; d < 3; ++d)
                roe.velocity[d] = (rootL * vL[d] + rootR * vR[d]) / sum;
            const real hL = (leftMean[4] + data.leftMean.flow.pressure) / leftMean[0], hR = (rightMean[4] + data.rightMean.flow.pressure) / rightMean[0];
            roe.enthalpy = (rootL * hL + rootR * hR) / sum;
            const real speedSquared = dot(roe.velocity, roe.velocity), denominator = roe.enthalpy - .5 * speedSquared;
            roe.soundSpeed = std::sqrt((gas.gamma - 1) * denominator);
            roe.contactEnergy = .5 * speedSquared;
            roe.contactProjection = {(roe.enthalpy - speedSquared) / denominator, roe.velocity[0] / denominator, roe.velocity[1] / denominator, roe.velocity[2] / denominator, -1 / denominator};
        }
        if (data.required.outerWaveBounds)
        {
            const auto vL = velocity(leftMean), vR = velocity(rightMean);
            const real unL = dot(vL, normal) - dot(gridVelocity, normal), unR = dot(vR, normal) - dot(gridVelocity, normal);
            const real rho = std::sqrt(leftMean[0] * rightMean[0]);
            const real star = std::max(real(0), .5 * (data.leftMean.flow.pressure + data.rightMean.flow.pressure) - .5 * (unL - unR) * rho * data.roe.soundSpeed);
            const auto shock = [&](real pressure)
            { return star <= pressure ? real(1) : std::sqrt(1 + (gas.gamma + 1) / (2 * gas.gamma) * (star / pressure - 1)); };
            data.outerWaves = {unL - data.leftMean.soundSpeed * shock(data.leftMean.flow.pressure), unR + data.rightMean.soundSpeed * shock(data.rightMean.flow.pressure)};
        }
        if (data.required.ausmFaceSpeed)
        {
            auto vL = velocity(left.flow.conservative), vR = velocity(right.flow.conservative);
            for (int d = 0; d < 3; ++d)
            {
                vL[d] -= gridVelocity[d];
                vR[d] -= gridVelocity[d];
            }
            const real unL = dot(vL, normal), unR = dot(vR, normal), factor = 2 * (gas.gamma - 1) / (gas.gamma + 1);
            const real hL = gas.gamma / (gas.gamma - 1) * left.flow.pressure / left.flow.conservative[0], hR = gas.gamma / (gas.gamma - 1) * right.flow.pressure / right.flow.conservative[0];
            if (solver == Riemann::Solver::AUSMPWPlus)
            {
                const real aSquared = .5 * factor * (hL + .5 * unL * unL + hR + .5 * unR * unR);
                data.ausm.faceSoundSpeed = aSquared / std::max(std::sqrt(aSquared), unL + unR >= 0 ? std::abs(unL) : std::abs(unR));
            }
            else
            {
                const real aLSquared = factor * (hL + .5 * dot(vL, vL)), aRSquared = factor * (hR + .5 * dot(vR, vR));
                data.ausm.faceSoundSpeed = std::min(aLSquared / std::max(std::sqrt(aLSquared), unL), aRSquared / std::max(std::sqrt(aRSquared), -unR));
            }
        }
        return data;
    }
}
