#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Physics/Thermodynamics/ConservativeConversion.hpp"
#include "Physics/Thermodynamics/IdealMixtureGradient.hpp"
#include <limits>

namespace DNDS::Cauchy
{
    /// Scalars own their values. Species views borrow the evaluator's worker
    /// scratch until its next point or destruction; no layout/Array assumption.
    struct EquationPointProperties
    {
        Physics::Thermodynamics::Properties thermo;
        Physics::Transport::Properties transport;
        Physics::Chemistry::ConstSpeciesBufferView fractions, diffusivity, speciesEnthalpy, speciesGasConstant;
        Physics::Chemistry::ConstSpeciesBufferView chemicalSource;
        Physics::Chemistry::ConstJacobianBufferView sourceJacobian;
        /// NaN when viscous mixture diffusivity was not requested.
        real diffusionBound = std::numeric_limits<real>::quiet_NaN();
    };

    struct EquationPrimitiveDerivative
    {
        /// Directional derivative of [rho,u,v,w,T], in code units.
        FlowState rhoVelocityTemperature;
        real pressure;
    };
    /// Pure directional chain rule at a ready EOS point. Input/output can be
    /// gathered/scattered through any layout. Species derivatives are Ns-1
    /// raw d(rhoY), with operator[] (a strided row span is sufficient). Mixtures
    /// require full species enthalpies requested when making the evaluator.
    /// Retain its worker/material and consume the borrowed point before the next
    /// evaluation. Caller supplies a point from this state and prevalidates the
    /// species extent; no implicit property refresh, repair or gamma closure.
    template <class TSpeciesDerivative>
    EquationPrimitiveDerivative PreparePrimitiveDerivative(const FlowState &state, const FlowState &derivative,
                                                           const EquationPointProperties &point, const TSpeciesDerivative &speciesDerivative)
    {
        namespace T = Physics::Thermodynamics;
        const auto velocity = T::ConservativeVelocity<3>(state);
        const real internalEnergy = T::ConservativeSpecificInternalEnergy<3>(state);
        const real internalDerivative = T::ConservativeInternalEnergyDensityDerivative<3>(state, derivative);
        const int species = point.fractions.nSpecies;
        DNDS_check_throw_info(!species || (point.speciesEnthalpy.nSpecies == species && point.speciesGasConstant.nSpecies == species),
                              "primitive mixture derivative needs full species enthalpies and gas constants");
        struct SpeciesEnergy
        {
            const EquationPointProperties &point;
            real operator[](int k) const { return point.speciesEnthalpy[k] - point.speciesGasConstant[k] * point.thermo.temperature; }
        };
        const real temperatureDerivative = T::IdealMixtureTemperatureDerivative(state[0], internalEnergy, point.thermo.cv,
                                                                                derivative[0], internalDerivative, point.fractions, SpeciesEnergy{point}, speciesDerivative, species);
        EquationPrimitiveDerivative result;
        result.rhoVelocityTemperature[0] = derivative[0];
        for (int d = 0; d < 3; ++d)
            result.rhoVelocityTemperature[1 + d] = T::ConservativeSpecificDerivative(state[0], velocity[d], derivative[0], derivative[1 + d]);
        result.rhoVelocityTemperature[4] = temperatureDerivative;
        result.pressure = T::IdealMixturePressureDerivative(state[0], point.thermo.temperature, point.thermo.gasConstant,
                                                            derivative[0], temperatureDerivative, point.fractions, point.speciesGasConstant, speciesDerivative, species);
        return result;
    }

    /// CPU point producer shared by explicit derived refresh and fused/tiled
    /// operator work. Owns one exclusive chemistry worker for mixtures. Names,
    /// backend, units and material compatibility are resolved before the loop.
    /// Caller serializes use; move this evaluator into asynchronous CPU work.
    class EquationPointEvaluator
    {
        friend class EquationSystem;
        PerfectGas _gas;
        bool _viscous;
        std::optional<Physics::Thermodynamics::IdealGasMixture::PointEvaluator> _mixture;
        Physics::Thermodynamics::MixturePropertyRequest _request;
        EquationPointEvaluator(const EquationSystem &equation, Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                               Physics::Thermodynamics::MixturePropertyRequest request);

    public:
        EquationPointEvaluator(const EquationPointEvaluator &) = delete;
        EquationPointEvaluator &operator=(const EquationPointEvaluator &) = delete;
        EquationPointEvaluator(EquationPointEvaluator &&) noexcept = default;
        EquationPointEvaluator &operator=(EquationPointEvaluator &&) noexcept = default;
        Physics::Chemistry::SpeciesBufferView SpeciesInput() const;
        /// Input flow is [rho,rhou,rhov,rhow,rhoE], with full thermal energy for
        /// mixtures. rhoY has Ns-1 independent conservative species; it may be
        /// worker scratch gathered from any physical layout or allocation.
        /// A stale positive T guess seeds UV inversion, never its final answer.
        EquationPointProperties Evaluate(const FlowState &flow, Physics::Chemistry::ConstSpeciesBufferView rhoY = {},
                                         real temperatureGuess = 0);
        /// Per-point subset of the reserved outputs; intended for fused FV
        /// face/source phases sharing one explicit worker context.
        EquationPointProperties Evaluate(const FlowState &flow, Physics::Chemistry::ConstSpeciesBufferView rhoY,
                                         real temperatureGuess, Physics::Thermodynamics::MixturePropertyRequest request);
        /// Numerical fluxes consume these actual EOS values; no gamma closure.
        static Riemann::AcousticState FluxState(const FlowState &flow, const EquationPointProperties &point)
        {
            return {{flow, point.thermo.pressure}, point.thermo.soundSpeed};
        }
    };
}
