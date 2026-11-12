#pragma once

#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/IO/OutputPlan.hpp"

namespace DNDS::Cauchy
{
    /// Production output/progress owner. It installs and removes Domain
    /// observers, owns CSV/logger state and serializes accepted/internal output
    /// decisions without leaking file/control logic into the solver frontend.
    class OutputCoordinator
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        OutputCoordinator(Simulation &simulation, const SimulationConfig &config,
                          MPIInfo mpi, real started);
        ~OutputCoordinator();
        OutputCoordinator(const OutputCoordinator &) = delete;
        OutputCoordinator &operator=(const OutputCoordinator &) = delete;

        OutputPlan::StepTarget SnapStep(real proposedDt) const;
        void BeginAttempt(real dt, index retries, real targetTime);
        void FlushPending();
        void WarnRetry(const std::string &reason);
        /// Record one accepted step, emit due snapshots and return the global
        /// configured steady/residual stopping decision.
        bool CompleteAcceptedStep(
            real dt, index retries,
            const std::vector<PhysicalStepLimit> &limits,
            bool useLimits, bool printConsole);
        /// Always write final checkpoint+visualization and summary. Returns 2
        /// for wall/step caps, otherwise zero.
        int Finalize(bool wallStop, bool reached);
    };
}
