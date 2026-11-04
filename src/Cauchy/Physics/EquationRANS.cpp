#include "Cauchy/Physics/EquationRANS.hpp"

namespace DNDS::Cauchy
{
    RANSStateView PrepareRANSStatePolicy(const EquationSystem &equation, const std::array<real, 2> &farfield,
                                         real saUpperLimit, real lowerFactor)
    {
        const auto &model = equation.RANS();
        DNDS_check_throw_info(std::isfinite(saUpperLimit) && saUpperLimit > 0 &&
                                  std::isfinite(lowerFactor) && lowerFactor >= 0,
                              "invalid RANS state limits");
        RANSStateView policy{model.Kind(), model.ComponentNumber(), saUpperLimit, lowerFactor * farfield[1]};
        DNDS_check_throw_info(policy.Admissible(farfield) && std::isfinite(policy.omegaLowerLimit), "invalid far-field RANS state");
        return policy;
    }
    std::array<real, 2> RANSWallTarget(const Physics::RANS::Model &model,
                                       const Physics::RANS::Point &mean, real interiorDensity)
    {
        using Kind = Physics::RANS::ModelKind;
        std::array<real, 2> values{};
        if (model.Kind() == Kind::KOmegaWilcox || model.Kind() == Kind::KOmegaSST)
            values[1] = mean.molecularViscosity / sqr(mean.wallDistance) * Physics::RANS::kWallOmegaCoeff;
        else if (model.Kind() == Kind::RealizableKE)
            values[1] = 2 * mean.molecularViscosity / mean.state[0] *
                        (mean.state[5] / mean.state[0]) / sqr(mean.wallDistance) * interiorDensity;
        return values;
    }
}
