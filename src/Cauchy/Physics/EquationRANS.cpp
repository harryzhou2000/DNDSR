#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        struct RANSReference
        {
            std::array<real, 2> values{};
            std::optional<real> density;
        };
        RANSReference ResolveReference(const EquationSystem &equation, const RANSStateSettings &settings)
        {
            RANSReference result;
            auto values = settings.referenceConservative;
            const auto rho = values.find("rho");
            if (rho != values.end())
            {
                result.density = rho->second;
                values.erase(rho);
            }
            DNDS_check_throw_info(!result.density || (std::isfinite(*result.density) && *result.density > 0), "RANS reference rho must be finite and positive");
            const auto kind = equation.RANS().Kind();
            const bool omega = kind == Physics::RANS::ModelKind::KOmegaWilcox || kind == Physics::RANS::ModelKind::KOmegaSST;
            if (!values.empty() || (omega && settings.omegaLowerFactor != 0))
                result.values = ResolveRANSValues(equation, values);
            return result;
        }
    }
    std::array<real, 2> ResolveRANSValues(const EquationSystem &equation, const std::map<std::string, real> &values)
    {
        std::array<real, 2> result{};
        if (!equation.HasRANS())
        {
            DNDS_check_throw_info(values.empty(), "RANS values supplied without a RANS model");
            return result;
        }
        const auto &model = equation.RANS();
        DNDS_check_throw_info(values.size() == size_t(model.ComponentNumber()), "all named conservative RANS values are required");
        for (int j = 0; j < model.ComponentNumber(); ++j)
        {
            const auto entry = values.find(model.ComponentNames()[j]);
            DNDS_check_throw_info(entry != values.end(), "missing or unknown conservative RANS component");
            result[j] = entry->second;
        }
        DNDS_check_throw_info((RANSStateView{model.Kind(), model.ComponentNumber()}.Admissible(result)), "inadmissible conservative RANS values");
        return result;
    }
    RANSStateView ResolveRANSStatePolicy(const EquationSystem &equation, const RANSStateSettings &settings)
    {
        return PrepareRANSStatePolicy(equation, ResolveReference(equation, settings).values, settings.saUpperLimit, settings.omegaLowerFactor);
    }
    RANSReconstructionView ResolveRANSReconstructionPolicy(const EquationSystem &equation)
    {
        return {equation.RANS().Kind(), equation.RANS().ComponentNumber()};
    }
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
