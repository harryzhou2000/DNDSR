#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include <limits>

namespace DNDS::Cauchy
{
    /// Scalars own their values. Species views borrow the evaluator's worker
    /// scratch until its next point or destruction; no layout/Array assumption.
    struct EquationPointProperties
    {
        Physics::Thermodynamics::Properties thermo;
        Physics::Transport::Properties transport;
        Physics::Chemistry::ConstSpeciesBufferView fractions, diffusivity, speciesEnthalpy;
        /// NaN when viscous mixture diffusivity was not requested.
        real diffusionBound = std::numeric_limits<real>::quiet_NaN();
    };

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
        /// Numerical fluxes consume these actual EOS values; no gamma closure.
        static Riemann::AcousticState FluxState(const FlowState &flow, const EquationPointProperties &point)
        {
            return {{flow, point.thermo.pressure}, point.thermo.soundSpeed};
        }
    };
}
