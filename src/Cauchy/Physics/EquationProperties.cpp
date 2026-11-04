#include "Cauchy/Physics/EquationProperties.hpp"

namespace DNDS::Cauchy
{
    EquationPointEvaluator EquationSystem::MakePointEvaluator(Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                                                              Physics::Thermodynamics::MixturePropertyRequest request) const
    {
        return EquationPointEvaluator(*this, std::move(worker), request);
    }
    EquationPointEvaluator::EquationPointEvaluator(const EquationSystem &equation,
                                                   Physics::Chemistry::ChemicalWorkerPool::Lease worker, Physics::Thermodynamics::MixturePropertyRequest request)
        : _viscous(equation.Viscous()), _request(request)
    {
        if (equation.HasMixture())
        {
            request.diffusivity = _viscous && request.diffusivity;
            request.transport = _viscous && (request.transport || request.diffusivity);
            _request = request;
            _mixture.emplace(equation.Mixture().Borrow(std::move(worker), request));
        }
        else
        {
            DNDS_check_throw_info(!worker, "perfect gas requires no chemistry worker");
            DNDS_check_throw_info(!request.chemicalSource && !request.sourceJacobian, "perfect gas has no chemical source profile");
            _gas = equation.Gas();
        }
    }
    Physics::Chemistry::SpeciesBufferView EquationPointEvaluator::SpeciesInput() const
    {
        return _mixture ? _mixture->SpeciesInput() : Physics::Chemistry::SpeciesBufferView{};
    }
    FlowState EquationPointEvaluator::FromPrimitive(const FlowState &primitive,
                                                    Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperature)
    {
        DNDS_check_throw_info(std::isfinite(temperature) && temperature >= 0 && primitive[0] > 0 &&
                                  std::all_of(primitive.begin(), primitive.end(), [](real value)
                                              { return std::isfinite(value); }) &&
                                  (temperature > 0 || primitive[4] > 0),
                              "invalid equation primitive input");
        if (!_mixture)
        {
            DNDS_check_throw_info(!rhoY.nSpecies, "perfect-gas primitive input has species");
            auto value = primitive;
            if (temperature > 0)
                value[4] = value[0] * _gas.gasConstant * temperature;
            return _gas.Conservative(value);
        }
        const auto point = temperature > 0 ? _mixture->FromRhoT(primitive[0], temperature, rhoY)
                                           : _mixture->FromRhoP(primitive[0], primitive[4], rhoY);
        const Vector3 velocity{primitive[1], primitive[2], primitive[3]};
        return {primitive[0], primitive[0] * velocity[0], primitive[0] * velocity[1], primitive[0] * velocity[2],
                Physics::Thermodynamics::ConservativeTotalEnergyDensity<3>(primitive[0], velocity, point.thermo.internalEnergy)};
    }
    EquationPointProperties EquationPointEvaluator::Evaluate(const FlowState &flow,
                                                             Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess)
    {
        return Evaluate(flow, rhoY, temperatureGuess, _request);
    }
    EquationPointProperties EquationPointEvaluator::Evaluate(const FlowState &flow,
                                                             Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess,
                                                             Physics::Thermodynamics::MixturePropertyRequest request)
    {
        request.diffusivity = _viscous && request.diffusivity;
        request.transport = _viscous && (request.transport || request.diffusivity);
        DNDS_check_throw_info((!request.transport || _request.transport) && (!request.diffusivity || _request.diffusivity) &&
                                  (!request.speciesEnthalpy || _request.speciesEnthalpy) && (!request.sourceJacobian || _request.sourceJacobian) &&
                                  (!(request.chemicalSource || request.sourceJacobian) || _request.chemicalSource || _request.sourceJacobian),
                              "point output was not reserved by this equation evaluator");
        EquationPointProperties result;
        if (!_mixture)
        {
            DNDS_check_throw_info(!rhoY.nSpecies && _gas.Admissible(flow), "perfect-gas property evaluation needs an admissible flow5 state");
            const auto primitive = _gas.Primitive(flow);
            const real cv = _gas.gasConstant / (_gas.gamma - 1), cp = _gas.gamma * cv;
            const real energy = primitive.pressure / (flow[0] * (_gas.gamma - 1));
            result.thermo = {flow[0], primitive.temperature, primitive.pressure, primitive.soundSpeed,
                             energy, energy + primitive.pressure / flow[0], cp, cv, _gas.gasConstant};
            const real mu = _viscous ? _gas.viscosity : 0;
            result.transport = {mu, mu * cp / _gas.prandtl};
            result.diffusionBound = _viscous ? _gas.viscosity / flow[0] * std::max(real(4. / 3), _gas.gamma / _gas.prandtl) : 0;
            return result;
        }
        for (const real value : flow)
            DNDS_check_throw_info(std::isfinite(value), "mixture property evaluation requires finite conservative values");
        DNDS_check_throw_info(flow[0] > 0, "mixture property evaluation requires positive density");
        const auto point = _mixture->FromConservative(flow[0], flow[4], {flow[1], flow[2], flow[3]}, rhoY, temperatureGuess, request);
        result.thermo = point.thermo;
        result.transport = _viscous ? point.transport : Physics::Transport::Properties{0, 0};
        result.fractions = point.fractions;
        result.diffusivity = point.diffusivity;
        result.speciesEnthalpy = point.speciesEnthalpy;
        result.speciesGasConstant = point.speciesGasConstant;
        result.chemicalSource = point.chemicalSource;
        result.sourceJacobian = point.sourceJacobian;
        if (!_viscous)
            result.diffusionBound = 0;
        if (_viscous && request.diffusivity)
        {
            result.diffusionBound = std::max(real(4. / 3) * result.transport.viscosity / flow[0],
                                             result.transport.conductivity / (flow[0] * result.thermo.cv));
            for (int k = 0; k < result.diffusivity.nSpecies; ++k)
                result.diffusionBound = std::max(result.diffusionBound, result.diffusivity[k]);
        }
        return result;
    }
}
