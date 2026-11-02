#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include <set>

namespace DNDS::Cauchy
{
    namespace
    {
        FlowState FarfieldValue(const PerfectGas &gas, const FlowState &inside, const FlowState &farfield, const Vector3 &normal)
        {
            const auto local = gas.Primitive(inside), far = gas.Primitive(farfield);
            real normalSpeed = 0;
            for (int d = 0; d < 3; ++d)
                normalSpeed += local.velocity[d] * normal[d];
            if (normalSpeed >= local.soundSpeed)
                return inside;
            if (normalSpeed <= -local.soundSpeed)
                return farfield;
            // Match the incoming characteristic count: prescribe pressure at
            // subsonic outflow, density/velocity at subsonic inflow.
            const auto &source = normalSpeed > 0 ? local : far;
            return gas.Conservative({source.density, source.velocity[0], source.velocity[1], source.velocity[2], normalSpeed > 0 ? far.pressure : local.pressure});
        }
    }
    FlowState BoundaryCondition::FaceValue(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point, real time) const
    {
        if (kind == BoundaryKind::Analytic)
            return ProfileValue(profile, prescribedPrimitive, gas, point, time);
        if (kind == BoundaryKind::DoubleMachBottom)
        {
            auto condition = *this;
            condition.kind = point[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
            return condition.FaceValue(gas, inside, normal, point, time);
        }
        if (kind == BoundaryKind::Farfield)
            return FarfieldValue(gas, inside, farfield, normal);
        if (kind == BoundaryKind::Extrapolate)
            return inside;
        const auto p = gas.Primitive(inside);
        FlowState face{p.density, p.velocity[0], p.velocity[1], p.velocity[2], p.pressure};
        if (kind == BoundaryKind::SlipWall)
        {
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += p.velocity[d] * normal[d];
            for (int d = 0; d < 3; ++d)
                face[1 + d] -= speed * normal[d];
        }
        else
        {
            for (int d = 0; d < 3; ++d)
                face[1 + d] = velocity[d];
            if (isothermal)
                face[0] = p.pressure / (gas.gasConstant * temperature);
        }
        return gas.Conservative(face);
    }
    FlowState BoundaryCondition::Exterior(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point, real time) const
    {
        if (kind == BoundaryKind::Analytic)
            return ProfileValue(profile, prescribedPrimitive, gas, point, time);
        if (kind == BoundaryKind::DoubleMachBottom)
        {
            auto condition = *this;
            condition.kind = point[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
            return condition.Exterior(gas, inside, normal, point, time);
        }
        if (kind == BoundaryKind::Farfield)
            return FarfieldValue(gas, inside, farfield, normal);
        if (kind == BoundaryKind::Extrapolate)
            return inside;
        const auto p = gas.Primitive(inside);
        FlowState exterior{p.density, p.velocity[0], p.velocity[1], p.velocity[2], p.pressure};
        if (kind == BoundaryKind::SlipWall)
        {
            real speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += p.velocity[d] * normal[d];
            for (int d = 0; d < 3; ++d)
                exterior[1 + d] -= 2 * speed * normal[d];
        }
        else
        {
            for (int d = 0; d < 3; ++d)
                exterior[1 + d] = 2 * velocity[d] - p.velocity[d];
            // Enforce T_wall through the viscous gradient, not an extrapolated
            // negative exterior temperature. Inviscid wall uses equal rho,p.
        }
        return gas.Conservative(exterior);
    }
    std::array<real, 25> BoundaryCondition::ExteriorJacobian(const PerfectGas &gas, const FlowState &inside, const Vector3 &normal, const Vector3 &point, real time) const
    {
        std::array<real, 25> result{};
        if (kind == BoundaryKind::Analytic)
            return result;
        if (kind == BoundaryKind::DoubleMachBottom)
        {
            auto condition = *this;
            condition.kind = point[0] < 1. / 6 ? BoundaryKind::Analytic : BoundaryKind::SlipWall;
            return condition.ExteriorJacobian(gas, inside, normal, point, time);
        }
        for (int j = 0; j < 5; ++j)
            result[j * 5 + j] = 1;
        if (kind == BoundaryKind::Extrapolate)
            return result;
        if (kind == BoundaryKind::SlipWall)
        {
            // Orthogonal momentum reflection preserves kinetic and total
            // energy. The pressure recovery cancels in this conservative map.
            for (int column = 0; column < 3; ++column)
                for (int row = 0; row < 3; ++row)
                    result[(1 + column) * 5 + 1 + row] -= 2 * normal[row] * normal[column];
            return result;
        }
        if (kind == BoundaryKind::NoSlipWall)
        {
            // m' = 2*rho*v_wall-m; E' = E+2*rho*|v_wall|^2-2*v_wall.m.
            // Wall temperature belongs to the viscous face law, not Exterior.
            for (int d = 0; d < 3; ++d)
            {
                result[1 + d] = 2 * velocity[d];
                result[(1 + d) * 5 + 1 + d] = -1;
                result[4] += 2 * velocity[d] * velocity[d];
                result[(1 + d) * 5 + 4] = -2 * velocity[d];
            }
            return result;
        }
        DNDS_check_throw_info(kind == BoundaryKind::Farfield, "boundary Jacobian requires a resolved physical condition");
        const auto local = gas.Primitive(inside);
        real normalSpeed = 0, speedSquared = 0;
        for (int d = 0; d < 3; ++d)
        {
            normalSpeed += local.velocity[d] * normal[d];
            speedSquared += local.velocity[d] * local.velocity[d];
        }
        if (normalSpeed >= local.soundSpeed)
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
    BoundarySystem::BoundarySystem(const std::map<std::string, BoundarySettings> &settings, const PerfectGas &gas)
    {
        for (const auto &[zone, source] : settings)
        {
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
            else
                DNDS_check_throw_info(false, "unsupported boundary family " + source.kind);
            condition.farfield = gas.Conservative(source.primitive);
            condition.prescribedPrimitive = source.primitive;
            condition.profile = source.profile;
            condition.velocity = source.velocity;
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
