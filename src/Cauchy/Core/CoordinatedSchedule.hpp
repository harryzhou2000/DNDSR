#pragma once

#include "Cauchy/Core/DomainPlacement.hpp"
#include "Cauchy/Core/StateEpoch.hpp"
#include <functional>
#include <cmath>

namespace DNDS::Cauchy
{
    /// One physical clock for all fluid domains. A round consumes frozen traces
    /// and returns fresh candidates; iteration order does not implement block GS.
    /// Domain work owns its collectives. Unexpected exceptions propagate to a
    /// fail-fast application boundary without a control agreement/recovery.
    /// Nonfinite returned candidates are agreed only after every callback returns.
    class CoordinatedSchedule
    {
        DomainPlacement _placement;
        StateEpoch _epoch;
        real _time = 0;
        uint64_t _nextTrial = 0;
        std::vector<real> _traces;

    public:
        explicit CoordinatedSchedule(DomainPlacement placement)
            : _placement(std::move(placement)), _traces(size_t(_placement.DomainNumber()), 0) {}
        real Time() const { return _time; }
        const StateEpoch &Epoch() const { return _epoch; }
        const std::vector<real> &Traces() const { return _traces; }
        /// Mock/scalar connection protocol used to validate placement and clock
        /// semantics independently of PDE code. Only domain roots publish each
        /// result; domain reduction values are not counted once per member.
        void Advance(real dt, const std::function<real(index, const std::vector<real> &, const StateEpoch &)> &domain)
        {
            int valid = std::isfinite(dt) && dt > 0;
            valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _placement.Control());
            DNDS_check_throw_info(valid, "Cauchy timestep is invalid on a rank");
            real minimum = dt, maximum = dt;
            minimum = MPI::AllreduceValue<MPI::Reduction::Min>(minimum, _placement.Control());
            maximum = MPI::AllreduceValue<MPI::Reduction::Max>(maximum, _placement.Control());
            DNDS_check_throw_info(std::isfinite(dt) && minimum > 0 && minimum == maximum, "Cauchy timestep differs between ranks or is invalid");
            auto trial = _epoch;
            trial.trial = ++_nextTrial;
            std::vector<real> candidates(_traces.size(), 0);
            valid = 1;
            for (auto iDomain : _placement.LocalDomains())
            {
                const auto value = domain(iDomain, _traces, trial);
                valid &= std::isfinite(value);
                if (_placement.DomainRank(iDomain) == 0)
                    candidates[iDomain] = value;
            }
            valid = MPI::AllreduceValue<MPI::Reduction::Min>(valid, _placement.Control());
            DNDS_check_throw_info(valid, "Cauchy domain round failed; clock and traces were not committed");
            MPI::AllreduceInPlace<MPI::Reduction::Sum>(
                MPI::ContiguousView<real>{candidates.data(), candidates.size()}, _placement.Control());
            _traces = std::move(candidates);
            _time += dt;
            _epoch = trial;
            ++_epoch.acceptedStep;
        }
    };
}
