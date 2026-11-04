#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"

namespace DNDS::Cauchy
{
    /// Small rank-local policy over already globally reduced component norms.
    /// All components remain observable; selection only controls acceptance.
    /// One instance belongs to one solve, including a finest FAS solve. It
    /// performs no MPI and never changes residuals, corrections or tolerances.
    template <class Norms>
    class BasicResidualConvergence
    {
        const ImplicitSettings &_settings;
        Norms _initial{}, _reference{};
        bool _started = false, _accepted = false;
        real _ratio = 0;

    public:
        /// Settings must be validated and outlive this policy.
        explicit BasicResidualConvergence(const ImplicitSettings &settings, Norms prototype = {}) : _settings(settings), _initial(prototype), _reference(prototype) {}
        void Observe(const Norms &residual)
        {
            DNDS_check_throw_info(residual.size() == _initial.size(), "residual component count changed");
            for (auto value : residual)
                DNDS_check_throw_info(std::isfinite(value) && value >= 0, "invalid global residual norm");
            if (!_started)
                _initial = _reference = residual;
            else if (_settings.residualReference == "runningMaximum")
                for (index j = 0; j < index(residual.size()); ++j)
                    _reference[j] = std::max(_reference[j], residual[j]);
            _started = true;
            _accepted = true;
            _ratio = 0;
            for (const auto j : _settings.residualComponents)
            {
                const real target = std::max(_settings.absoluteTolerance, _settings.relativeTolerance * _reference[j]);
                _accepted = _accepted && residual[j] <= target;
                // A zero requested tolerance means exact zero. Keep diagnostics
                // finite rather than serializing an infinity as JSON null.
                const real ratio = target > 0 ? residual[j] / target : residual[j] == 0 ? 0
                                                                                        : std::numeric_limits<real>::max();
                _ratio = std::max(_ratio, std::min(ratio, std::numeric_limits<real>::max()));
            }
        }
        const Norms &Initial() const { return _initial; }
        const Norms &Reference() const { return _reference; }
        bool Accepted() const { return _started && _accepted; }
        real TargetRatio() const { return _ratio; }
    };
    using ResidualConvergence = BasicResidualConvergence<FlowState>;
}
