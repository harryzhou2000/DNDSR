#include "Cauchy/Physics/StateInput.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"

namespace DNDS::Cauchy
{
    Physics::ReferenceScales ResolveReferenceScales(const EquationSettings &settings)
    {
        const auto &old = settings.mixture;
        Physics::ReferenceScales result{old.rho0, old.U0, old.L0, old.T0};
        if (!settings.referenceScales.is_null())
        {
            const auto &input = settings.referenceScales;
            DNDS_check_throw_info(input.is_object() && input.size() == 4 && input.contains("rho0") && input.contains("U0") && input.contains("L0") && input.contains("T0"),
                                  "equation.referenceScales requires exactly rho0,U0,L0,T0");
            result = {input.at("rho0").get<real>(), input.at("U0").get<real>(), input.at("L0").get<real>(), input.at("T0").get<real>()};
            DNDS_check_throw_info((old.rho0 == 1 && old.U0 == 1 && old.L0 == 1 && old.T0 == 1) ||
                                      (old.rho0 == result.rho0 && old.U0 == result.U0 && old.L0 == result.L0 && old.T0 == result.T0),
                                  "conflicting equation.referenceScales and mixture scale fields");
        }
        result.Validate("equation.referenceScales");
        return result;
    }
    PhysicalStateConverter::PhysicalStateConverter(const EquationSystem &equation, const Physics::ReferenceScales &scales,
                                                   Physics::Chemistry::ChemicalWorkerPool::Lease worker)
        : _equation(equation), _scales(scales), _evaluator(equation.MakePointEvaluator(std::move(worker), {})),
          _rans(equation.HasRANS() ? equation.RANS().ComponentNumber() : 0),
          _species(equation.HasMixture() ? equation.Mixture().SpeciesNumber() : 0)
    {
        scales.Validate();
        _scalarScales.assign(_rans + (_species ? _species - 1 : 0), 1);
        if (_rans == 2)
        {
            _scalarScales[0] = scales.SpecificEnergy();
            _scalarScales[1] = (equation.RANS().Kind() == Physics::RANS::ModelKind::RealizableKE ? scales.SpecificEnergy() : 1) / scales.Time();
        }
        if (_species)
        {
            const auto &material = equation.Mixture().Scales();
            DNDS_check_throw_info(scales.rho0 == material.rho0 && scales.U0 == material.U0 && scales.L0 == material.L0 && scales.T0 == material.T0,
                                  "state input scales differ from equation material");
        }
    }
    std::vector<real> PhysicalStateConverter::Conservative(Physics::StateInput parsed)
    {
        using Form = Physics::StateInput::Form;
        const auto &equation = _equation;
        const int nr = _rans, ns = _species, ni = ns ? ns - 1 : 0;
        parsed.ToCodeUnits(_scales, _scalarScales);
        auto &u = parsed.values;
        DNDS_check_throw_info(std::all_of(u.begin(), u.end(), [](real v)
                                          { return std::isfinite(v); }),
                              "physical state input is not finite");
        const bool conservative = parsed.form == Form::Conservative || parsed.form == Form::Sensible;
        DNDS_check_throw_info(u[0] > 0, "physical state requires positive rho (or T for primTP)");
        std::vector<real> fractions(ns), rhoY(ni);
        if (ns)
        {
            // Here 1 denotes mass-fraction normalization, not a missing density.
            Physics::Chemistry::RepairMassFractions(conservative ? u[0] : 1,
                                                    {u.data() + 5 + nr, ni}, {fractions.data(), ns});
        }
        if (!conservative)
        {
            real gasConstant = equation.HasMixture() ? 0 : equation.Gas().gasConstant;
            if (ns)
                for (int k = 0; k < ns; ++k)
                    gasConstant += fractions[k] * equation.Mixture().SpeciesGasConstants()[k];
            DNDS_check_throw_info(u[4] > 0, "primitive input requires positive pressure or temperature");
            if (parsed.form == Form::TP)
                u[0] = u[4] / (gasConstant * u[0]);
            for (int k = 0; k < ni; ++k)
                u[5 + nr + k] = u[0] * fractions[k];
        }
        for (int k = 0; k < ni; ++k)
            rhoY[k] = u[5 + nr + k];
        FlowState flow;
        std::copy_n(u.begin(), 5, flow.begin());
        if (!conservative)
        {
            flow = _evaluator.FromPrimitive(flow, {rhoY.data(), ni}, parsed.form == Form::RhoT ? u[4] : 0);
            for (int j = 0; j < nr; ++j)
                u[5 + j] *= flow[0];
        }
        else if (parsed.form == Form::Sensible && ns)
        {
            // Euler's I/O reference energy uses repaired mass fractions; the
            // transported conservative species themselves are not rewritten.
            for (int k = 0; k < ns; ++k)
                flow[4] += flow[0] * fractions[k] * equation.Mixture().BaseSpecificInternalEnergies()[k];
        }
        std::copy(flow.begin(), flow.end(), u.begin());
        const auto properties = _evaluator.Evaluate(flow, {rhoY.data(), ni});
        DNDS_check_throw_info(std::isfinite(properties.thermo.temperature) && properties.thermo.temperature > 0 &&
                                  std::isfinite(properties.thermo.pressure) && properties.thermo.pressure > 0 &&
                                  std::all_of(u.begin(), u.end(), [](real v)
                                              { return std::isfinite(v); }),
                              "resolved physical state is not finite and thermodynamically admissible");
        return std::move(u);
    }
    Physics::StateInput PhysicalStateConverter::Represent(const std::vector<real> &conservative, Physics::StateInput::Form form, bool physical)
    {
        using Form = Physics::StateInput::Form;
        DNDS_check_throw_info(conservative.size() == 5 + _scalarScales.size() && conservative[0] > 0 &&
                                  std::all_of(conservative.begin(), conservative.end(), [](real v)
                                              { return std::isfinite(v); }),
                              "state representation requires a finite complete conservative row with positive density");
        Physics::StateInput result{form, false, conservative};
        const int ni = _species ? _species - 1 : 0;
        FlowState flow;
        std::copy_n(conservative.begin(), 5, flow.begin());
        const auto point = _evaluator.Evaluate(flow, {conservative.data() + 5 + _rans, ni});
        auto &values = result.values;
        if (form == Form::Sensible && _species)
            for (int k = 0; k < _species; ++k)
                values[4] -= flow[0] * point.fractions[k] * _equation.Mixture().BaseSpecificInternalEnergies()[k];
        if (form != Form::Conservative && form != Form::Sensible)
        {
            for (index j = 1; j < index(values.size()); ++j)
                values[j] /= flow[0];
            values[4] = form == Form::RhoT ? point.thermo.temperature : point.thermo.pressure;
            if (form == Form::TP)
                values[0] = point.thermo.temperature;
        }
        if (physical)
            result.ToPhysicalUnits(_scales, _scalarScales);
        return result;
    }
    ResolvedPhysicalState ResolvePhysicalState(const Json &input, const EquationSystem &equation,
                                               const Physics::ReferenceScales &scales, Physics::Chemistry::ChemicalWorkerPool::Lease worker)
    {
        PhysicalStateConverter converter(equation, scales, std::move(worker));
        auto u = converter.Conservative(Physics::StateInput::Parse(input));
        const auto primitive = converter.Represent(u, Physics::StateInput::Form::RhoP, false);
        const auto temperature = converter.Represent(u, Physics::StateInput::Form::RhoT, false).values[4];
        FlowState flow;
        std::copy_n(primitive.values.begin(), 5, flow.begin());
        ResolvedPhysicalState result{u, flow, temperature, {}, {}};
        const int nr = equation.HasRANS() ? equation.RANS().ComponentNumber() : 0;
        const int ns = equation.HasMixture() ? equation.Mixture().SpeciesNumber() : 0;
        if (ns)
        {
            std::vector<real> fractions(ns);
            Physics::Chemistry::RepairMassFractions(u[0], {u.data() + 5 + nr, ns - 1}, {fractions.data(), ns});
            for (int k = 0; k < ns - 1; ++k)
                result.massFractions[equation.Mixture().SpeciesNames()[k]] = fractions[k];
        }
        for (int j = 0; j < nr; ++j)
            result.ransConservative[equation.RANS().ComponentNames()[j]] = u[5 + j];
        ResolveRANSValues(equation, result.ransConservative);
        return result;
    }
    ResolvedPhysicalInputs ResolvePhysicalStateInputs(const DomainSettings &settings, const EquationSystem &equation,
                                                      const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers)
    {
        ResolvedPhysicalInputs prepared{settings, {}, {}};
        auto &result = prepared.settings;
        const auto scales = ResolveReferenceScales(settings.equation);
        const auto resolve = [&](const Json &value)
        {
            return ResolvePhysicalState(value, equation, scales, workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
        };
        if (!settings.initialState.is_null())
        {
            DNDS_check_throw_info(settings.initialPrimitive == DomainSettings{}.initialPrimitive && settings.initialTemperature == 0 && settings.initialMassFractions.empty() && settings.initialRANS.empty(),
                                  "initialState conflicts with legacy initial fields");
            DNDS_check_throw_info(settings.initialProfile.kind == "constant", "tagged initialState currently requires a constant profile");
            auto state = resolve(settings.initialState);
            prepared.initialConservative = std::move(state.conservative);
            result.initialPrimitive = state.primitive;
            result.initialMassFractions = std::move(state.massFractions);
            result.initialRANS = std::move(state.ransConservative);
            result.initialState = nullptr;
        }
        if (!settings.referenceState.is_null())
        {
            DNDS_check_throw_info(settings.numerics.ransState.referenceConservative.empty(), "referenceState conflicts with partial referenceConservative");
            auto state = resolve(settings.referenceState);
            prepared.referenceConservative = state.conservative;
            if (equation.HasRANS())
            {
                result.numerics.ransState.referenceConservative = std::move(state.ransConservative);
                result.numerics.ransState.referenceConservative["rho"] = state.conservative[0];
            }
            result.referenceState = nullptr;
        }
        for (auto &[zone, boundary] : result.boundaries)
            if (!boundary.state.is_null())
            {
                DNDS_check_throw_info(boundary.kind == "farfield" || boundary.kind == "analytic", "tagged boundary state requires a prescribed boundary");
                DNDS_check_throw_info(boundary.primitive == BoundarySettings{}.primitive && boundary.massFractions.empty() && boundary.ransConservative.empty(),
                                      "boundary state conflicts with legacy primitive/scalar fields");
                auto state = resolve(boundary.state);
                DNDS_check_throw_info(state.temperature >= settings.numerics.temperatureFloor,
                                      "prescribed boundary state lies below configured temperature floor");
                boundary.primitive = state.primitive;
                boundary.massFractions = std::move(state.massFractions);
                boundary.ransConservative = std::move(state.ransConservative);
                boundary.state = nullptr;
            }
        return prepared;
    }
}
