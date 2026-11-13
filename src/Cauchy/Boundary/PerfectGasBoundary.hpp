#pragma once

#include "Cauchy/Boundary/BoundaryKind.hpp"
#include "Cauchy/Input/FlowProfileMath.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "Physics/Thermodynamics/TotalCondition.hpp"
#include "Cauchy/Physics/PerfectGasPoint.hpp"
#include "Cauchy/Boundary/RANSWallTarget.hpp"

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

    /** Combined flow5/SA/two-equation physical boundary state, in local axes.
     * This is the value-only counterpart of BoundaryPointOperator::TryEvaluate.
     * The caller supplies point and mean separately: wall omega uses mean
     * viscosity (wall-temperature viscosity when isothermal), whereas its
     * exterior reflection uses the reconstructed interior scalar.
     * @pre Interface packets are handled by the enclosing point producer.
     * @note Reflected RANS scalars may be negative by construction. Do not
     * apply a mean-state positivity repair to this exterior boundary state.
     */
    template <int Components>
    DNDS_DEVICE_CALLABLE NumericalStatus TryPerfectGasBoundaryState(
        const PerfectGasBoundaryParameters &condition, const PerfectGas &gas,
        Physics::RANS::ModelKind model, std::array<real, Components> &point,
        const std::array<real, Components> &mean, real meanWallDistance,
        const Vector3 &normal, const Vector3 &location, real time,
        const RigidFrameParameters &frame, bool exterior) noexcept
    {
        static_assert(Components >= 5 && Components <= 7);
        const FlowState inside{point[0], point[1], point[2], point[3], point[4]};
        if (const auto checked = TryPerfectGasPoint(gas, inside, false); !checked)
            return checked.status;
        const auto result = TryPerfectGasBoundaryValue(condition, gas, inside, normal, location, time, frame, exterior);
        if (!result)
            return result.status;
        const auto &flow = result.value;
        if (const auto checked = TryPerfectGasPoint(gas, flow, false); !checked)
            return checked.status;
        for (int j = 0; j < 5; ++j)
            point[j] = flow[j];
        if constexpr (Components > 5)
        {
            const auto kind = ResolveBoundaryKind(condition.kind, location, frame);
            if (kind == BoundaryKind::TotalConditionInlet)
                for (int j = 0; j < Components - 5; ++j)
                    point[5 + j] = flow[0] * condition.totalRANSPrimitive[j];
            else if (kind == BoundaryKind::NoSlipWall)
            {
                std::array<real, 2> target{};
                if constexpr (Components == 7)
                {
                    if (!(std::isfinite(meanWallDistance) && meanWallDistance > 0))
                        return NumericalStatus{NumericalError::InvalidState, 3};
                    const FlowState viscosityFlow = condition.isothermal ? flow : FlowState{mean[0], mean[1], mean[2], mean[3], mean[4]};
                    const auto properties = TryPerfectGasPoint(gas, viscosityFlow, true);
                    if (!properties)
                        return properties.status;
                    Physics::RANS::Point wall;
                    for (int j = 0; j < Components; ++j)
                        wall.state[j] = mean[j];
                    wall.molecularViscosity = properties.value.transport.viscosity;
                    wall.wallDistance = meanWallDistance;
                    target = PrepareRANSWallTarget(model, wall, inside[0]);
                }
                for (int j = 0; j < Components - 5; ++j)
                {
                    real reflected = point[5 + j];
                    if (condition.isothermal && (Components == 6 || j == 0))
                        reflected *= flow[0] / inside[0];
                    point[5 + j] = exterior ? 2 * target[j] - reflected : target[j];
                }
            }
            else if (kind == BoundaryKind::Farfield || kind == BoundaryKind::Analytic)
            {
                const auto grid = frame.GridVelocity(location);
                real speed = 0;
                for (int d = 0; d < 3; ++d)
                    speed += (inside[1 + d] / inside[0] - grid[d]) * normal[d];
                if (kind == BoundaryKind::Analytic || speed <= 0)
                    for (int j = 0; j < Components - 5; ++j)
                        point[5 + j] = condition.prescribedRANS[j];
            }
        }
        return {};
    }
}
