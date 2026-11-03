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
            _gas = equation.Gas();
        }
    }
    Physics::Chemistry::SpeciesBufferView EquationPointEvaluator::SpeciesInput() const
    {
        return _mixture ? _mixture->SpeciesInput() : Physics::Chemistry::SpeciesBufferView{};
    }
    EquationPointProperties EquationPointEvaluator::Evaluate(const FlowState &flow,
                                                             Physics::Chemistry::ConstSpeciesBufferView rhoY, real temperatureGuess)
    {
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
        real speedSquared = 0;
        for (int d = 0; d < 3; ++d)
            speedSquared += sqr(flow[1 + d] / flow[0]);
        const auto point = _mixture->FromRhoE(flow[0], flow[4] / flow[0] - .5 * speedSquared, rhoY, temperatureGuess);
        result.thermo = point.thermo;
        result.transport = _viscous ? point.transport : Physics::Transport::Properties{0, 0};
        result.fractions = point.fractions;
        result.diffusivity = point.diffusivity;
        result.speciesEnthalpy = point.speciesEnthalpy;
        if (!_viscous)
            result.diffusionBound = 0;
        if (_viscous && _request.diffusivity)
        {
            result.diffusionBound = std::max(real(4. / 3) * result.transport.viscosity / flow[0],
                                             result.transport.conductivity / (flow[0] * result.thermo.cv));
            for (int k = 0; k < result.diffusivity.nSpecies; ++k)
                result.diffusionBound = std::max(result.diffusionBound, result.diffusivity[k]);
        }
        return result;
    }
}
