#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy
{
    EquationSystem::EquationSystem(PerfectGas gas, bool viscous) : _gas(gas), _viscous(viscous)
    {
        _gas.Validate();
        _schema = {"compressible", "sensible-total-energy", "conservative-volume", {{"flow", {{"rho", "mass/volume", "mass"}, {"rhou", "mass/(area*time)", "momentum"}, {"rhov", "mass/(area*time)", "momentum"}, {"rhow", "mass/(area*time)", "momentum"}, {"rhoE", "energy/volume", "total-energy"}}, {1}}}};
    }
    EquationSystem EquationSystem::Resolve(const EquationSettings &settings)
    {
        DNDS_check_throw_info(settings.system == "compressible" && settings.eos == "perfectGas" && settings.source == "none" &&
                                  (settings.transport == "inviscid" || settings.transport == "NewtonianFourier"),
                              "unavailable Cauchy equation profile");
        const bool viscous = settings.transport == "NewtonianFourier";
        DNDS_check_throw_info(viscous || settings.viscosity == 0, "unused inviscid viscosity");
        return EquationSystem({settings.gamma, settings.gasConstant, settings.viscosity, settings.prandtl}, viscous);
    }
    void EquationSystem::ValidateStateSchema(const Algebra::DofBundle &state) const
    {
        state.Validate();
        DNDS_check_throw_info(state.BlockNumber() == index(_schema.blocks.size()), "equation state block count differs");
        for (const auto &block : _schema.blocks)
            DNDS_check_throw_info(state.Block(block.name).Width() == index(block.components.size()), "equation component count differs");
    }
    PreparedRiemannData EquationSystem::PrepareRiemann(const Riemann::RiemannFlux &flux,
                                                       const Riemann::AcousticState &left, const Riemann::AcousticState &right,
                                                       const FlowState &leftMean, const FlowState &rightMean,
                                                       const Vector3 &normal, const Vector3 &gridVelocity) const
    {
        PreparedRiemannData data{};
        data.required = flux.RequiredInputs();
        if (!data.required.meanStates)
            return data;
        data.leftMean = PrepareFluxState(leftMean);
        data.rightMean = PrepareFluxState(rightMean);
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
            roe.soundSpeed = std::sqrt((_gas.gamma - 1) * denominator);
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
            { return star <= pressure ? real(1) : std::sqrt(1 + (_gas.gamma + 1) / (2 * _gas.gamma) * (star / pressure - 1)); };
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
            const real unL = dot(vL, normal), unR = dot(vR, normal), factor = 2 * (_gas.gamma - 1) / (_gas.gamma + 1);
            const real hL = _gas.gamma / (_gas.gamma - 1) * left.flow.pressure / left.flow.conservative[0], hR = _gas.gamma / (_gas.gamma - 1) * right.flow.pressure / right.flow.conservative[0];
            if (flux.Kind() == Riemann::Solver::AUSMPWPlus)
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
