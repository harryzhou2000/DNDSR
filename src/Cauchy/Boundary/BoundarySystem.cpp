#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include <set>

namespace DNDS::Cauchy
{
    namespace
    {
        Vector3 WallVelocity(const BoundaryCondition &condition, const Vector3 &point, const RigidFrame *frame)
        {
            if (!frame || !frame->moving)
                return condition.velocity;
            if (!condition.wallMovesWithMesh)
                return frame->LocalVector(condition.velocity);
            auto velocity = frame->GridVelocity(point);
            for (int d = 0; d < 3; ++d)
                velocity[d] += condition.velocity[d];
            return velocity;
        }
        FlowState AnalyticValue(const BoundaryCondition &condition, const PerfectGas &gas, const Vector3 &point, real time, const RigidFrame *frame)
        {
            const auto value = ProfileValue(condition.profile, condition.prescribedPrimitive, gas, frame ? frame->WorldPoint(point) : point, time);
            return frame ? frame->LocalState(value) : value;
        }
        real GridNormalSpeed(const Vector3 &normal, const Vector3 &point, const RigidFrame *frame)
        {
            if (!frame || !frame->moving)
                return 0;
            const auto velocity = frame->GridVelocity(point);
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += velocity[d] * normal[d];
            return speed;
        }
        FlowState FarfieldValue(const PerfectGas &gas, const FlowState &inside, const FlowState &farfield, const Vector3 &normal, real gridNormalSpeed)
        {
            const auto local = gas.Primitive(inside), far = gas.Primitive(farfield);
            real normalSpeed = -gridNormalSpeed;
            for (int d = 0; d < 3; ++d)
                normalSpeed += local.velocity[d] * normal[d];
            if (normalSpeed > local.soundSpeed)
                return inside;
            if (normalSpeed <= -local.soundSpeed)
                return farfield;
            // Match the incoming characteristic count: prescribe pressure at
            // subsonic outflow, density/velocity at subsonic inflow.
            const auto &source = normalSpeed > 0 ? local : far;
            return gas.Conservative({source.density, source.velocity[0], source.velocity[1], source.velocity[2], normalSpeed > 0 ? far.pressure : local.pressure});
        }
    }
    Vector3 BoundaryCondition::WallVelocityAt(const Vector3 &point, const RigidFrame *frame) const
    {
        return WallVelocity(*this, point, frame);
    }
    FlowState BoundaryCondition::FaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point, real time, const RigidFrame *frame) const
    {
        DNDS_check_throw_info(hasPerfectGasState, "boundary needs its equation-specific value producer");
        DNDS_check_throw_info(kind != BoundaryKind::Interface, "interface boundary needs a ready incoming packet");
        if (kind == BoundaryKind::Analytic)
            return AnalyticValue(*this, gas, point, time, frame);
        if (kind == BoundaryKind::DoubleMachBottom)
        {
            auto condition = *this;
            condition.kind = (frame ? frame->WorldPoint(point) : point)[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
            return condition.FaceValue(gas, inside, normal, point, time, frame);
        }
        if (kind == BoundaryKind::Farfield)
            return FarfieldValue(gas, inside, frame ? frame->LocalState(farfield) : farfield, normal, GridNormalSpeed(normal, point, frame));
        if (kind == BoundaryKind::Extrapolate)
            return inside;
        const auto p = gas.Primitive(inside);
        const auto wallVelocity = WallVelocity(*this, point, frame);
        FlowState face{p.density, p.velocity[0], p.velocity[1], p.velocity[2], p.pressure};
        if (kind == BoundaryKind::SlipWall)
        {
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += (p.velocity[d] - wallVelocity[d]) * normal[d];
            for (int d = 0; d < 3; ++d)
                face[1 + d] -= speed * normal[d];
        }
        else
        {
            for (int d = 0; d < 3; ++d)
                face[1 + d] = wallVelocity[d];
            if (isothermal)
                face[0] = p.pressure / (gas.gasConstant * temperature);
        }
        return gas.Conservative(face);
    }
    FlowState BoundaryCondition::Exterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point, real time, const RigidFrame *frame) const
    {
        DNDS_check_throw_info(hasPerfectGasState, "boundary needs its equation-specific exterior producer");
        DNDS_check_throw_info(kind != BoundaryKind::Interface, "interface boundary needs a ready incoming packet");
        if (kind == BoundaryKind::Analytic)
            return AnalyticValue(*this, gas, point, time, frame);
        if (kind == BoundaryKind::DoubleMachBottom)
        {
            auto condition = *this;
            condition.kind = (frame ? frame->WorldPoint(point) : point)[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
            return condition.Exterior(gas, inside, normal, point, time, frame);
        }
        if (kind == BoundaryKind::Farfield)
            return FarfieldValue(gas, inside, frame ? frame->LocalState(farfield) : farfield, normal, GridNormalSpeed(normal, point, frame));
        if (kind == BoundaryKind::Extrapolate)
            return inside;
        const auto p = gas.Primitive(inside);
        const auto wallVelocity = WallVelocity(*this, point, frame);
        FlowState exterior{p.density, p.velocity[0], p.velocity[1], p.velocity[2], p.pressure};
        if (kind == BoundaryKind::SlipWall)
        {
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += (p.velocity[d] - wallVelocity[d]) * normal[d];
            for (int d = 0; d < 3; ++d)
                exterior[1 + d] -= 2 * speed * normal[d];
        }
        else
        {
            for (int d = 0; d < 3; ++d)
                exterior[1 + d] = 2 * wallVelocity[d] - p.velocity[d];
            // Euler imposes the wall temperature on the reflected exterior
            // itself (also the VR target), keeping its pressure and velocity.
            if (isothermal)
                exterior[0] = p.pressure / (gas.gasConstant * temperature);
        }
        return gas.Conservative(exterior);
    }
    std::array<real, 25> BoundaryCondition::ExteriorJacobian(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point, real time, const RigidFrame *frame) const
    {
        DNDS_check_throw_info(hasPerfectGasState, "boundary needs its equation-specific derivative producer");
        DNDS_check_throw_info(kind != BoundaryKind::Interface, "interface boundary Jacobian needs a frozen incoming packet");
        std::array<real, 25> result{};
        if (kind == BoundaryKind::Analytic)
            return result;
        if (kind == BoundaryKind::DoubleMachBottom)
        {
            auto condition = *this;
            condition.kind = (frame ? frame->WorldPoint(point) : point)[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
            return condition.ExteriorJacobian(gas, inside, normal, point, time, frame);
        }
        for (int j = 0; j < 5; ++j)
            result[j * 5 + j] = 1;
        if (kind == BoundaryKind::Extrapolate)
            return result;
        const auto wallVelocity = WallVelocity(*this, point, frame);
        if (kind == BoundaryKind::SlipWall)
        {
            // Reflection about the moving wall's normal velocity. Absolute
            // total energy changes only by the reflected kinetic energy.
            real wallNormal = 0;
            for (int d = 0; d < 3; ++d)
                wallNormal += wallVelocity[d] * normal[d];
            result[4] += 2 * wallNormal * wallNormal;
            for (int d = 0; d < 3; ++d)
            {
                result[1 + d] += 2 * wallNormal * normal[d];
                result[(1 + d) * 5 + 4] -= 2 * wallNormal * normal[d];
            }
            for (int column = 0; column < 3; ++column)
                for (int row = 0; row < 3; ++row)
                    result[(1 + column) * 5 + 1 + row] -= 2 * normal[row] * normal[column];
            return result;
        }
        if (kind == BoundaryKind::NoSlipWall)
        {
            if (isothermal)
            {
                const auto primitive = gas.Primitive(inside);
                const real density = primitive.pressure / (gas.gasConstant * temperature);
                Vector3 reflected{};
                real speedSquared = 0, reflectedSquared = 0;
                for (int d = 0; d < 3; ++d)
                {
                    reflected[d] = 2 * wallVelocity[d] - primitive.velocity[d];
                    speedSquared += primitive.velocity[d] * primitive.velocity[d];
                    reflectedSquared += reflected[d] * reflected[d];
                }
                for (int column = 0; column < 5; ++column)
                {
                    const real dp = (gas.gamma - 1) * (column == 0 ? .5 * speedSquared : column == 4 ? 1
                                                                                                     : -primitive.velocity[column - 1]);
                    const real drho = dp / (gas.gasConstant * temperature);
                    result[column * 5] = drho;
                    result[column * 5 + 4] = dp / (gas.gamma - 1) + .5 * reflectedSquared * drho;
                    for (int d = 0; d < 3; ++d)
                    {
                        const real dv = (column == 0 ? primitive.velocity[d] : column == 1 + d ? -1
                                                                                               : 0) /
                                        inside[0];
                        result[column * 5 + 1 + d] = drho * reflected[d] + density * dv;
                        result[column * 5 + 4] += density * reflected[d] * dv;
                    }
                }
                return result;
            }
            // m' = 2*rho*v_wall-m; E' = E+2*rho*|v_wall|^2-2*v_wall.m.
            for (int d = 0; d < 3; ++d)
            {
                result[1 + d] = 2 * wallVelocity[d];
                result[(1 + d) * 5 + 1 + d] = -1;
                result[4] += 2 * wallVelocity[d] * wallVelocity[d];
                result[(1 + d) * 5 + 4] = -2 * wallVelocity[d];
            }
            return result;
        }
        DNDS_check_throw_info(kind == BoundaryKind::Farfield, "boundary Jacobian requires a resolved physical condition");
        const auto local = gas.Primitive(inside);
        real normalSpeed = -GridNormalSpeed(normal, point, frame), speedSquared = 0;
        for (int d = 0; d < 3; ++d)
        {
            normalSpeed += local.velocity[d] * normal[d];
            speedSquared += local.velocity[d] * local.velocity[d];
        }
        if (normalSpeed > local.soundSpeed)
            return result;
        if (normalSpeed <= -local.soundSpeed)
            return {};
        if (normalSpeed > 0)
        {
            // Prescribed pressure with unchanged rho,m: E'=p_inf/(gamma-1)+KE.
            result[4] = -.5 * speedSquared;
            result[24] = 0;
            for (int d = 0; d < 3; ++d)
                result[(1 + d) * 5 + 4] = local.velocity[d];
        }
        else
        {
            // Prescribed rho,velocity with interior pressure: only E' varies.
            result.fill(0);
            result[4] = .5 * speedSquared;
            result[24] = 1;
            for (int d = 0; d < 3; ++d)
                result[(1 + d) * 5 + 4] = -local.velocity[d];
        }
        return result;
    }
    std::array<real, 25> BoundaryCondition::ApproximateExteriorJacobian(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal,
                                                                        const Vector3 &point, real time, const RigidFrame *frame) const
    {
        // Euler's linMode=1 returns homogeneous wall reflection before its
        // thermal/scalar wall treatment, and zero for prescribed/open BCs.
        auto wallKind = kind;
        if (kind == BoundaryKind::DoubleMachBottom)
            wallKind = (frame ? frame->WorldPoint(point) : point)[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
        if (wallKind != BoundaryKind::SlipWall && wallKind != BoundaryKind::NoSlipWall)
            return {};
        BoundaryCondition wall;
        wall.kind = wallKind;
        wall.velocity = velocity;
        wall.wallMovesWithMesh = wallMovesWithMesh;
        wall.isothermal = false;
        return wall.ExteriorJacobian(gas, inside, normal, point, time, frame);
    }
    BoundarySystem::BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas &gas)
        : BoundarySystem(settings, &gas) {}
    BoundarySystem::BoundarySystem(const std::map<std::string, BoundarySettings> &settings)
        : BoundarySystem(settings, nullptr) {}
    BoundarySystem::BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const EquationSystem &equation)
        : BoundarySystem(settings, equation.HasMixture() ? nullptr : &equation.Gas())
    {
        for (const auto &[zone, source] : settings)
        {
            auto &condition = _conditions.at(std::stoi(zone));
            const bool prescribed = condition.kind == BoundaryKind::Farfield || condition.kind == BoundaryKind::Analytic;
            DNDS_check_throw_info(prescribed || source.massFractions.empty(), "mass fractions supplied to a non-prescribed boundary");
            if (prescribed)
                condition.prescribedFractions = equation.IndependentMassFractions(source.massFractions);
            DNDS_check_throw_info((equation.HasRANS() && prescribed) || source.ransConservative.empty(),
                                  "RANS values require a prescribed RANS boundary");
            if (equation.HasRANS() && prescribed)
                condition.prescribedRANS = ResolveRANSValues(equation, source.ransConservative);
        }
    }
    BoundarySystem::BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas *gas)
    {
        for (const auto &[zone, source] : settings)
        {
            DNDS_check_throw_info(!gas || source.massFractions.empty(), "perfect-gas boundary has unused mass fractions");
            DNDS_check_throw_info(source.state.is_null(), "resolve tagged boundary state before constructing BoundarySystem directly");
            BoundaryCondition condition;
            if (source.kind == "extrapolate")
                condition.kind = BoundaryKind::Extrapolate;
            else if (source.kind == "farfield")
                condition.kind = BoundaryKind::Farfield;
            else if (source.kind == "slipWall")
                condition.kind = BoundaryKind::SlipWall;
            else if (source.kind == "noSlipWall")
                condition.kind = BoundaryKind::NoSlipWall;
            else if (source.kind == "analytic")
                condition.kind = BoundaryKind::Analytic;
            else if (source.kind == "doubleMachBottom")
                condition.kind = BoundaryKind::DoubleMachBottom;
            else if (source.kind == "interface")
                condition.kind = BoundaryKind::Interface;
            else
                DNDS_check_throw_info(false, "unsupported boundary family " + source.kind);
            condition.hasPerfectGasState = bool(gas);
            if (gas)
                condition.farfield = gas->Conservative(source.primitive);
            condition.prescribedPrimitive = source.primitive;
            condition.profile = source.profile;
            condition.velocity = source.velocity;
            DNDS_check_throw_info(source.wallVelocityFrame == "mesh" || source.wallVelocityFrame == "ground", "unknown wall velocity frame");
            condition.wallMovesWithMesh = source.wallVelocityFrame == "mesh";
            condition.temperature = source.temperature;
            condition.isothermal = source.thermal == "isothermal";
            _conditions.emplace(std::stoi(zone), condition);
        }
    }
    const BoundaryCondition &BoundarySystem::At(int zone) const
    {
        const auto found = _conditions.find(zone);
        DNDS_check_throw_info(found != _conditions.end(), "missing Cauchy boundary condition for zone " + std::to_string(zone));
        return found->second;
    }
    void BoundarySystem::ValidateUsed(const std::set<int> &zones) const
    {
        for (int zone : zones)
            At(zone);
        for (const auto &[zone, condition] : _conditions)
            DNDS_check_throw_info(zones.count(zone), "unused Cauchy boundary condition for zone " + std::to_string(zone));
    }
}
