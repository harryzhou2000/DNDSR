#pragma once

#include "Cauchy/Physics/PerfectGas.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy
{
    enum class FlowProfileKind
    {
        Constant,
        IsentropicVortex,
        DoubleMachReflection,
        CylinderWake,
        Unsupported
    };
    /// Immutable analytic-profile parameters, resolved once from host config.
    /// Runtime expressions remain explicit host input/setup producers; no
    /// string comparison, expression engine or callback runs on the device.
    struct FlowProfileParameters
    {
        FlowProfileKind kind = FlowProfileKind::Constant;
        real strength = 5, period = 10;
        Vector3 center{5, 5, 0};
        real perturbation = 0;
    };
    /// Shared analytic initial/BC/oracle mathematics. Inputs and time are in
    /// code units. Prescribed state uses [rho,u,v,w,p]. Failures are returned
    /// explicitly; no mutation of solver fields or fallback is performed.
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowState> TryProfileValue(const FlowProfileParameters &profile, FlowState primitive,
                                                                           const PerfectGas &gas, const Vector3 &point, real time) noexcept
    {
        constexpr real pi = 3.1415926535897932384626433832795;
        if (profile.kind == FlowProfileKind::IsentropicVortex)
        {
            real x = point[0] - primitive[1] * time - profile.center[0];
            real y = point[1] - primitive[2] * time - profile.center[1];
            x -= profile.period * std::floor(x / profile.period + .5);
            y -= profile.period * std::floor(y / profile.period + .5);
            const real r2 = x * x + y * y;
            const real temperature = 1 - (gas.gamma - 1) * profile.strength * profile.strength / (8 * gas.gamma * pi * pi) * std::exp(1 - r2);
            if (!(temperature > 0))
                return NumericalStatus{NumericalError::InvalidState, 1};
            primitive[0] = std::pow(temperature, 1 / (gas.gamma - 1));
            primitive[4] = std::pow(temperature, gas.gamma / (gas.gamma - 1));
            const real rotation = profile.strength / (2 * pi) * std::exp(.5 * (1 - r2));
            primitive[1] -= rotation * y;
            primitive[2] += rotation * x;
        }
        else if (profile.kind == FlowProfileKind::DoubleMachReflection)
        {
            if (!(std::abs(gas.gamma - 1.4) < 1e-14))
                return NumericalStatus{NumericalError::InvalidInput, 2};
            // Woodward-Colella Mach-10 shock, angle 60 degrees, foot x=1/6.
            primitive = point[0] - 10 / std::sin(pi / 3) * time - 1. / 6 - point[1] / std::tan(pi / 3) > 0
                            ? FlowState{1.4, 0, 0, 0, 1}
                            : FlowState{8, 57.157676649772960 / 8, -33. / 8, 0, 116.5};
        }
        else if (profile.kind == FlowProfileKind::CylinderWake)
            primitive[2] += profile.perturbation * std::exp(-((point[0] - 2) * (point[0] - 2) + point[1] * point[1]));
        else if (profile.kind != FlowProfileKind::Constant)
            return NumericalStatus{NumericalError::UnsupportedRequest, 3};
        return gas.Conservative(primitive);
    }
}
