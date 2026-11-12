#pragma once

#include "Cauchy/Physics/PerfectGas.hpp"
#include "Physics/Thermodynamics/Properties.hpp"
#include "Physics/Transport/Properties.hpp"
#include "Physics/EvaluationStatus.hpp"

namespace DNDS::Cauchy
{
    /// Value-only point output for the perfect-gas specialization. Unlike the
    /// mixture worker result this owns no borrowed composition/Jacobian views.
    struct PerfectGasPoint
    {
        Physics::Thermodynamics::Properties thermo;
        Physics::Transport::Properties transport;
        real diffusionBound = 0;
    };

    /// Shared CPU/device producer, preserving PointEvaluator's code-unit
    /// perfect-gas convention. The operator selects transport explicitly; an
    /// inviscid/convective-only request does not evaluate the viscosity law.
    /// Parameters have been validated at setup. Failure is a status value and
    /// does not initiate clipping, mean fallback, logging or exception handling.
    DNDS_DEVICE_CALLABLE inline NumericalResult<PerfectGasPoint> TryPerfectGasPoint(
        const PerfectGas &gas, const FlowState &flow, bool transport) noexcept
    {
        if (!gas.Admissible(flow))
            return Physics::EvaluationError(NumericalError::InvalidState, Physics::EvaluationFailure::Properties);
        const auto primitive = gas.Primitive(flow);
        const real cv = gas.gasConstant / (gas.gamma - 1), cp = gas.gamma * cv;
        const real energy = primitive.pressure / (flow[0] * (gas.gamma - 1));
        const real mu = transport ? gas.MolecularViscosity(flow[0], primitive.temperature) : 0;
        return PerfectGasPoint{{flow[0], primitive.temperature, primitive.pressure, primitive.soundSpeed,
                                energy, energy + primitive.pressure / flow[0], cp, cv, gas.gasConstant},
                               {mu, mu * cp / gas.prandtl},
                               mu / flow[0] * std::max(real(4. / 3), gas.gamma / gas.prandtl)};
    }
}
