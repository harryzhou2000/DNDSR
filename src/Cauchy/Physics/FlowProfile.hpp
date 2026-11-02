#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"

namespace DNDS::Cauchy
{
    /// Prescribed physical input evaluated outside reconstruction kernels. The
    /// same provider supplies initial cell averages, stage-time boundary values
    /// and the analytic accuracy oracle; it never owns solver state or clocks.
    inline FlowState ProfileValue(const FlowProfileSettings &profile, FlowState primitive,
                                  const PerfectGas &gas, const Vector3 &point, real time)
    {
        constexpr real pi = 3.1415926535897932384626433832795;
        if (profile.kind == "isentropicVortex")
        {
            real x = point[0] - primitive[1] * time - profile.center[0];
            real y = point[1] - primitive[2] * time - profile.center[1];
            x -= profile.period * std::floor(x / profile.period + .5);
            y -= profile.period * std::floor(y / profile.period + .5);
            const real r2 = x * x + y * y;
            const real temperature = 1 - (gas.gamma - 1) * profile.strength * profile.strength / (8 * gas.gamma * pi * pi) * std::exp(1 - r2);
            DNDS_check_throw_info(temperature > 0, "isentropic vortex has nonpositive temperature");
            primitive[0] = std::pow(temperature, 1 / (gas.gamma - 1));
            primitive[4] = std::pow(temperature, gas.gamma / (gas.gamma - 1));
            const real rotation = profile.strength / (2 * pi) * std::exp(.5 * (1 - r2));
            primitive[1] -= rotation * y;
            primitive[2] += rotation * x;
        }
        else if (profile.kind == "doubleMachReflection")
        {
            DNDS_check_throw_info(std::abs(gas.gamma - 1.4) < 1e-14, "canonical DMR input requires gamma=1.4");
            // Woodward-Colella Mach-10 shock, angle 60 degrees, foot x=1/6.
            primitive = point[0] - 10 / std::sin(pi / 3) * time - 1. / 6 - point[1] / std::tan(pi / 3) > 0
                            ? FlowState{1.4, 0, 0, 0, 1}
                            : FlowState{8, 57.157676649772960 / 8, -33. / 8, 0, 116.5};
        }
        else if (profile.kind == "cylinderWake")
            primitive[2] += profile.perturbation * std::exp(-((point[0] - 2) * (point[0] - 2) + point[1] * point[1]));
        else
            DNDS_check_throw_info(profile.kind == "constant", "unknown physical input profile " + profile.kind);
        return gas.Conservative(primitive);
    }
}
