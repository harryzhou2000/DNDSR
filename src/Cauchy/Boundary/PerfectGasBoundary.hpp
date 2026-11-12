#pragma once

#include "Cauchy/Boundary/BoundaryKind.hpp"
#include "Cauchy/Input/FlowProfileMath.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "Physics/Thermodynamics/TotalCondition.hpp"

namespace DNDS::Cauchy
{
    /// Compact immutable zone metadata. No vectors, names, callbacks or physics
    /// workers are uploaded. RANS fields are kept as resolved code-unit values;
    /// flow targets and scalar targets are produced separately by the operator.
    struct PerfectGasBoundaryParameters
    {
        BoundaryKind kind = BoundaryKind::Interior;
        FlowState farfield{}, prescribedPrimitive{};
        Vector3 velocity{};
        real temperature = 1;
        bool isothermal = false, wallMovesWithMesh = true;
        FlowProfileParameters profile;
        std::array<real, 2> prescribedRANS{}, totalRANSPrimitive{};
        std::array<real, 5> totalCondition{};
    };
    static_assert(std::is_trivially_copyable_v<PerfectGasBoundaryParameters>);

    DNDS_DEVICE_CALLABLE inline BoundaryKind ResolveBoundaryKind(
        BoundaryKind kind, const Vector3 &point, const RigidFrameParameters &frame) noexcept
    {
        if (kind != BoundaryKind::DoubleMachBottom)
            return kind;
        return frame.WorldPoint(point)[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
    }
    /// Absolute wall velocity in the local mesh axes. A moving wall attached
    /// to the mesh adds grid velocity; a prescribed ground vector rotates only.
    DNDS_DEVICE_CALLABLE inline Vector3 PrepareWallVelocity(
        const PerfectGasBoundaryParameters &condition, const Vector3 &point, const RigidFrameParameters &frame) noexcept
    {
        if (!frame.moving)
            return condition.velocity;
        if (!condition.wallMovesWithMesh)
            return frame.LocalVector(condition.velocity);
        auto velocity = frame.GridVelocity(point);
        for (int d = 0; d < 3; ++d)
            velocity[d] += condition.velocity[d];
        return velocity;
    }

    /// Prescribed-in-reconstruction physical face or reflected exterior flow
    /// state. exterior=true requests the state for flux/mirrored reconstruction;
    /// false gives the physical face target. Isothermal exterior density uses
    /// prescribed wall T itself, matching Euler, not 2*Twall-Tinside.
    ///
    /// Inputs are setup-validated, point-admissible and use code units. This
    /// producer performs no leases, exchange, EOS worker calls or host fallback.
    /// Interface values must be taken from the supplied incoming packet by the
    /// caller, never recomputed as physical BCs. No branch is injected into CFV.
    DNDS_DEVICE_CALLABLE inline NumericalResult<FlowState> TryPerfectGasBoundaryValue(
        const PerfectGasBoundaryParameters &condition, const PerfectGas &gas,
        const FlowState &inside, const Vector3 &normal, const Vector3 &point,
        real time, const RigidFrameParameters &frame, bool exterior) noexcept
    {
        const auto kind = ResolveBoundaryKind(condition.kind, point, frame);
        if (kind == BoundaryKind::Interface || kind == BoundaryKind::Interior)
            return NumericalStatus{NumericalError::InvalidInput, 2};
        if (kind == BoundaryKind::Extrapolate)
            return inside;
        if (kind == BoundaryKind::Analytic)
        {
            const auto result = TryProfileValue(condition.profile, condition.prescribedPrimitive, gas, frame.WorldPoint(point), time);
            if (!result)
                return result.status;
            return frame.LocalState(result.value);
        }
        if (kind == BoundaryKind::TotalConditionInlet)
        {
            const auto &total = condition.totalCondition;
            const auto direction = frame.LocalVector({total[2], total[3], total[4]});
            real speedSquared = 0, inward = 0;
            for (int d = 0; d < 3; ++d)
            {
                speedSquared += sqr(inside[1 + d] / inside[0]);
                inward += direction[d] * normal[d];
            }
            if (!(inward < -1e-14))
                return NumericalStatus{NumericalError::InvalidInput, 1};
            const auto result = Physics::Thermodynamics::TotalToStaticPerfectGas(total[0], total[1], speedSquared, gas.gasConstant, gas.gamma);
            if (!result)
                return result.status;
            const auto &state = result.value;
            const real speed = std::sqrt(state.speedSquared);
            return gas.Conservative({state.density, direction[0] * speed, direction[1] * speed, direction[2] * speed, state.pressure});
        }
        const auto primitive = gas.Primitive(inside);
        if (kind == BoundaryKind::Farfield)
        {
            const auto farfield = frame.LocalState(condition.farfield);
            const auto far = gas.Primitive(farfield);
            const auto grid = frame.GridVelocity(point);
            real gridSpeed = 0;
            for (int d = 0; d < 3; ++d)
                gridSpeed += grid[d] * normal[d];
            real speed = -gridSpeed;
            for (int d = 0; d < 3; ++d)
                speed += primitive.velocity[d] * normal[d];
            if (speed > primitive.soundSpeed)
                return inside;
            if (speed <= -primitive.soundSpeed)
                return farfield;
            const auto &source = speed > 0 ? primitive : far;
            return gas.Conservative({source.density, source.velocity[0], source.velocity[1], source.velocity[2],
                                     speed > 0 ? far.pressure : primitive.pressure});
        }
        const auto wall = PrepareWallVelocity(condition, point, frame);
        FlowState result{primitive.density, primitive.velocity[0], primitive.velocity[1], primitive.velocity[2], primitive.pressure};
        if (kind == BoundaryKind::SlipWall)
        {
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += (primitive.velocity[d] - wall[d]) * normal[d];
            for (int d = 0; d < 3; ++d)
                result[1 + d] -= exterior ? 2 * speed * normal[d] : speed * normal[d];
        }
        else if (kind == BoundaryKind::NoSlipWall)
        {
            for (int d = 0; d < 3; ++d)
                result[1 + d] = exterior ? 2 * wall[d] - primitive.velocity[d] : wall[d];
            if (condition.isothermal)
                result[0] = primitive.pressure / (gas.gasConstant * condition.temperature);
        }
        else
            return NumericalStatus{NumericalError::UnsupportedRequest};
        return gas.Conservative(result);
    }
}
