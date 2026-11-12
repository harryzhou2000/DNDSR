#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Input/FlowProfileMath.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy
{
    /// Prescribed physical input evaluated outside reconstruction kernels. The
    /// same provider supplies initial cell averages, stage-time boundary values
    /// and the analytic accuracy oracle; it never owns solver state or clocks.
    /// Kernel entry: profile failures are explicit values, including analytic
    /// inputs that are valid at setup but inadmissible at this point/time.
    inline FlowProfileParameters PrepareFlowProfile(const FlowProfileSettings &profile)
    {
        FlowProfileKind kind = FlowProfileKind::Unsupported;
        if (profile.kind == "constant")
            kind = FlowProfileKind::Constant;
        else if (profile.kind == "isentropicVortex")
            kind = FlowProfileKind::IsentropicVortex;
        else if (profile.kind == "doubleMachReflection")
            kind = FlowProfileKind::DoubleMachReflection;
        else if (profile.kind == "cylinderWake")
            kind = FlowProfileKind::CylinderWake;
        return {kind, profile.strength, profile.period, profile.center, profile.perturbation};
    }
    inline NumericalResult<FlowState> TryProfileValue(const FlowProfileSettings &profile, FlowState primitive,
                                                      const PerfectGas &gas, const Vector3 &point, real time) noexcept
    {
        return TryProfileValue(PrepareFlowProfile(profile), primitive, gas, point, time);
    }

    /// Checked setup/oracle convenience; numerical kernels use TryProfileValue.
    inline FlowState ProfileValue(const FlowProfileSettings &profile, FlowState primitive,
                                  const PerfectGas &gas, const Vector3 &point, real time)
    {
        const auto result = TryProfileValue(profile, primitive, gas, point, time);
        DNDS_check_throw_info(result, "physical input profile evaluation failed");
        return result.value;
    }
}
