#include "Physics/Thermodynamics/IdealGasMixture.hpp"

namespace DNDS::Physics::Thermodynamics
{
    struct IdealGasMixture::State
    {
        ReferenceScales scales;
        std::shared_ptr<const void> identity;
        std::vector<std::string> species;
        std::vector<real> baseEnergy;
        real temperatureFloor;
        State(const Chemistry::ChemicalSource &prototype, ReferenceScales reference)
            : scales(reference), identity(prototype.ModelIdentity())
        {
            scales.Validate();
            DNDS_check_throw_info(identity, "ideal-gas mixture needs an initialized backend");
            DNDS_check_throw_info(prototype.isIdealGas(), "this material requires an ideal-gas EOS");
            DNDS_check_throw_info(scales.U0 == prototype.velScale() && scales.rho0 == prototype.rhoScale(),
                                  "mixture scales differ from the chemical backend's energy/source convention");
            species = prototype.speciesNames();
            auto base = prototype.mixtureBaseInternalRhoESpecies();
            baseEnergy.assign(base.data, base.data + base.nSpecies);
            temperatureFloor = prototype.baseTemperature() / scales.T0;
        }
    };
    IdealGasMixture::IdealGasMixture(const Chemistry::ChemicalSource &prototype, ReferenceScales scales)
        : _state(std::make_shared<const State>(prototype, scales)) {}
    int IdealGasMixture::SpeciesNumber() const { return int(_state->species.size()); }
    const std::vector<std::string> &IdealGasMixture::SpeciesNames() const { return _state->species; }
    const ReferenceScales &IdealGasMixture::Scales() const { return _state->scales; }
    real IdealGasMixture::BaseTemperature() const { return _state->temperatureFloor; }
    std::shared_ptr<const void> IdealGasMixture::ModelIdentity() const { return _state->identity; }
    real IdealGasMixture::BaseInternalEnergy(real rho, Chemistry::ConstSpeciesBufferView rhoY) const
    {
        DNDS_assert(rhoY.nSpecies == SpeciesNumber() - 1);
        const auto &base = _state->baseEnergy;
        real result = rho * base.back();
        for (int k = 0; k < rhoY.nSpecies; ++k)
            result += rhoY[k] * (base[k] - base.back());
        return result;
    }
    IdealGasMixture::PointEvaluator IdealGasMixture::Borrow(Chemistry::ChemicalWorkerPool::Lease worker,
                                                            MixturePropertyRequest request) const
    {
        DNDS_check_throw_info(worker && worker.Source().ModelIdentity() == _state->identity,
                              "chemical worker belongs to a different loaded material");
        return PointEvaluator(_state, std::move(worker), request);
    }
    IdealGasMixture::PointEvaluator::PointEvaluator(std::shared_ptr<const State> state,
                                                    Chemistry::ChemicalWorkerPool::Lease worker,
                                                    MixturePropertyRequest request)
        : _state(std::move(state)), _worker(std::move(worker)), _request(request) {}
    Chemistry::SpeciesBufferView IdealGasMixture::PointEvaluator::SpeciesInput() const
    {
        auto &input = _worker.Buffers().rhoY;
        return {input.data(), int(input.size()) - 1};
    }
    void IdealGasMixture::PointEvaluator::PrepareComposition(real rho, Chemistry::ConstSpeciesBufferView rhoY)
    {
        auto &fractions = _worker.Buffers().fractions;
        Chemistry::RepairMassFractions(rho, rhoY, {fractions.data(), int(fractions.size())});
    }
    MixturePoint IdealGasMixture::PointEvaluator::Evaluate(real rho, real temperature)
    {
        DNDS_check_throw_info(std::isfinite(temperature) && temperature > 0, "mixture temperature must be finite and positive");
        const auto &scales = _state->scales;
        auto &source = _worker.Source();
        auto &buffers = _worker.Buffers();
        const int species = int(buffers.fractions.size());
        const Chemistry::ConstSpeciesBufferView fractions{buffers.fractions.data(), species};
        const real T = temperature * scales.T0;
        const real p = rho * scales.rho0 * source.mixtureR(fractions) * T;
        MixturePoint result;
        Chemistry::SpeciesBufferView diffusion, enthalpy;
        if (_request.diffusivity)
            diffusion = {buffers.diffusivity.data(), species};
        if (_request.speciesEnthalpy)
            enthalpy = {buffers.enthalpy.data(), species};
        source.EvaluateProperties(T, p, fractions, result.thermo, _request.transport ? &result.transport : nullptr,
                                  diffusion, enthalpy);
        auto &thermo = result.thermo;
        thermo.density /= scales.rho0;
        thermo.temperature /= scales.T0;
        thermo.pressure /= scales.Pressure();
        thermo.soundSpeed /= scales.U0;
        thermo.internalEnergy /= scales.SpecificEnergy();
        thermo.enthalpy /= scales.SpecificEnergy();
        thermo.cp /= scales.HeatCapacity();
        thermo.cv /= scales.HeatCapacity();
        thermo.gasConstant /= scales.HeatCapacity();
        if (_request.transport)
        {
            result.transport.viscosity /= scales.Viscosity();
            result.transport.conductivity /= scales.Conductivity();
        }
        for (int k = 0; k < species; ++k)
        {
            if (diffusion.data)
                diffusion[k] /= scales.Diffusivity();
            if (enthalpy.data)
                enthalpy[k] /= scales.SpecificEnergy();
        }
        result.fractions = fractions;
        result.diffusivity = diffusion;
        result.speciesEnthalpy = enthalpy;
        return result;
    }
    MixturePoint IdealGasMixture::PointEvaluator::FromRhoE(real rho, real internalEnergy,
                                                           Chemistry::ConstSpeciesBufferView rhoY,
                                                           real temperatureGuess, real uvTolerance)
    {
        DNDS_check_throw_info(std::isfinite(internalEnergy) && std::isfinite(uvTolerance) && uvTolerance > 0,
                              "mixture UV inversion requires finite energy and positive tolerance");
        PrepareComposition(rho, rhoY);
        auto &fractions = _worker.Buffers().fractions;
        const auto &scales = _state->scales;
        const real guess = std::isfinite(temperatureGuess) && temperatureGuess > 0 ? temperatureGuess * scales.T0 : 0;
        const real temperature = _worker.Source().temperatureFromUV(internalEnergy * scales.SpecificEnergy(),
                                                                    1 / (rho * scales.rho0),
                                                                    {fractions.data(), int(fractions.size())}, guess, uvTolerance);
        return Evaluate(rho, temperature / scales.T0);
    }
    MixturePoint IdealGasMixture::PointEvaluator::FromRhoT(real rho, real temperature,
                                                           Chemistry::ConstSpeciesBufferView rhoY)
    {
        PrepareComposition(rho, rhoY);
        return Evaluate(rho, temperature);
    }
    MixturePoint IdealGasMixture::PointEvaluator::FromRhoP(real rho, real pressure,
                                                           Chemistry::ConstSpeciesBufferView rhoY)
    {
        DNDS_check_throw_info(std::isfinite(pressure) && pressure > 0, "mixture pressure must be finite and positive");
        PrepareComposition(rho, rhoY);
        const auto &fractions = _worker.Buffers().fractions;
        const auto &scales = _state->scales;
        const real R = _worker.Source().mixtureR({fractions.data(), int(fractions.size())});
        return Evaluate(rho, pressure * scales.Pressure() / (rho * scales.rho0 * R * scales.T0));
    }
}
