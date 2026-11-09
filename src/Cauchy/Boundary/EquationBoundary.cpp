#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"

namespace DNDS::Cauchy
{
    void BoundaryCondition::ValidateEquationProducer() const
    {
        DNDS_check_throw_info(kind == BoundaryKind::Extrapolate || kind == BoundaryKind::Farfield ||
                                  kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall ||
                                  (kind == BoundaryKind::Analytic && profile.kind == "constant"),
                              "requested equation-state boundary producer is not connected");
    }
    namespace
    {
        FlowState EquationValue(const BoundaryCondition &condition, EquationPointEvaluator &evaluator,
                                const FlowState &inside, Physics::Chemistry::SpeciesBufferView rhoY,
                                const Vector3 &normal, const Vector3 &point, const RigidFrame *frame, bool exterior)
        {
            if (condition.kind == BoundaryKind::Extrapolate)
                return inside;
            if (condition.kind == BoundaryKind::Farfield || condition.kind == BoundaryKind::Analytic)
            {
                DNDS_check_throw_info(condition.prescribedFractions.size() == size_t(rhoY.nSpecies), "boundary composition extent differs from point state");
                auto primitive = condition.prescribedPrimitive;
                if (frame)
                {
                    const auto velocity = frame->LocalVector({primitive[1], primitive[2], primitive[3]});
                    for (int d = 0; d < 3; ++d)
                        primitive[1 + d] = velocity[d];
                }
                if (condition.kind == BoundaryKind::Farfield)
                {
                    const auto properties = evaluator.Evaluate(inside, rhoY, 0, {});
                    const auto grid = frame ? frame->GridVelocity(point) : Vector3{};
                    real speed = 0;
                    for (int d = 0; d < 3; ++d)
                        speed += (inside[1 + d] / inside[0] - grid[d]) * normal[d];
                    if (speed > properties.thermo.soundSpeed)
                        return inside;
                    if (speed > 0)
                    {
                        // Subsonic outflow preserves rho, velocity and raw
                        // composition; only pressure is prescribed.
                        primitive[0] = inside[0];
                        for (int d = 0; d < 3; ++d)
                            primitive[1 + d] = inside[1 + d] / inside[0];
                        return evaluator.FromPrimitive(primitive, rhoY);
                    }
                    if (speed > -properties.thermo.soundSpeed)
                        primitive[4] = properties.thermo.pressure;
                }
                // Supersonic inflow or a prescribed value replaces all data;
                // subsonic inflow retains the interior pressure above.
                for (int k = 0; k < rhoY.nSpecies; ++k)
                    rhoY[k] = primitive[0] * condition.prescribedFractions[k];
                return evaluator.FromPrimitive(primitive, rhoY);
            }
            DNDS_check_throw_info(condition.kind == BoundaryKind::SlipWall || condition.kind == BoundaryKind::NoSlipWall,
                                  "equation boundary requires a ready supported physical producer");
            const auto wall = condition.WallVelocityAt(point, frame);
            const real rho = inside[0], factor = exterior ? 2 : 1;
            DNDS_check_throw_info(rho > 0, "wall producer requires positive density");
            Vector3 velocity{inside[1] / rho, inside[2] / rho, inside[3] / rho}, target = velocity;
            if (condition.kind == BoundaryKind::SlipWall)
            {
                real speed = 0;
                for (int d = 0; d < 3; ++d)
                    speed += (velocity[d] - wall[d]) * normal[d];
                for (int d = 0; d < 3; ++d)
                    target[d] -= factor * speed * normal[d];
            }
            else
                for (int d = 0; d < 3; ++d)
                    target[d] += factor * (wall[d] - velocity[d]);
            if (condition.kind == BoundaryKind::NoSlipWall && condition.isothermal)
            {
                const auto properties = evaluator.Evaluate(inside, rhoY, 0, {});
                const real density = properties.thermo.pressure / (properties.thermo.gasConstant * condition.temperature);
                for (int k = 0; k < rhoY.nSpecies; ++k)
                    rhoY[k] *= density / rho;
                return evaluator.FromPrimitive({density, target[0], target[1], target[2], properties.thermo.pressure}, rhoY, condition.temperature);
            }
            // Adiabatic reflection retains internal energy and composition.
            auto result = inside;
            for (int d = 0; d < 3; ++d)
            {
                result[1 + d] = rho * target[d];
                result[4] += .5 * rho * (target[d] * target[d] - velocity[d] * velocity[d]);
            }
            return result;
        }
    }
    FlowState BoundaryCondition::FaceValue(EquationPointEvaluator &evaluator, const FlowState &inside,
                                           Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                           const Vector3 &point, real, const RigidFrame *frame) const
    {
        return EquationValue(*this, evaluator, inside, rhoY, normal, point, frame, false);
    }
    FlowState BoundaryCondition::Exterior(EquationPointEvaluator &evaluator, const FlowState &inside,
                                          Physics::Chemistry::SpeciesBufferView rhoY, const Vector3 &normal,
                                          const Vector3 &point, real, const RigidFrame *frame) const
    {
        return EquationValue(*this, evaluator, inside, rhoY, normal, point, frame, true);
    }
}
