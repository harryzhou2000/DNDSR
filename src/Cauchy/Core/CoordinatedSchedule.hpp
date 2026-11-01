#pragma once

#include "Cauchy/Core/DomainPlacement.hpp"
#include <functional>
#include <cmath>

namespace DNDS::Cauchy
{
    struct StateEpoch
    {
        uint64_t acceptedStep = 0, trial = 0, stage = 0, couplingIteration = 0;
        uint64_t geometry = 0, schema = 0;
    };
    /// One physical clock for all fluid domains. A round consumes frozen traces
    /// and returns fresh candidates; iteration order does not implement block GS.
    /// Domain-local work must complete/coordinate its own collectives before
    /// returning or throwing. MPI process failure recovery is not provided.
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
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
            DNDS_check_throw_info(valid, "Cauchy timestep is invalid on a rank");
            real minimum = dt, maximum = dt;
            MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, _placement.Control());
            MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, _placement.Control());
            DNDS_check_throw_info(std::isfinite(dt) && minimum > 0 && minimum == maximum, "Cauchy timestep differs between ranks or is invalid");
            auto trial = _epoch;
            trial.trial = ++_nextTrial;
            std::vector<real> candidates(_traces.size(), 0);
            valid = 1;
            for (auto iDomain : _placement.LocalDomains())
                try
                {
                    const auto value = domain(iDomain, _traces, trial);
                    valid &= std::isfinite(value);
                    if (_placement.DomainRank(iDomain) == 0)
                        candidates[iDomain] = value;
                }
                catch (...)
                {
                    valid = 0;
                }
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, _placement.Control());
            DNDS_check_throw_info(valid, "Cauchy domain round failed; clock and traces were not committed");
            MPI_Allreduce(MPI_IN_PLACE, candidates.data(), int(candidates.size()), MPI_DOUBLE, MPI_SUM, _placement.Control());
            _traces = std::move(candidates);
            _time += dt;
            _epoch = trial;
            ++_epoch.acceptedStep;
        }
    };
}
