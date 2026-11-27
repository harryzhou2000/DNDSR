#pragma once

#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/IO/OutputPlan.hpp"
#include "Cauchy/IO/RunControl.hpp"

namespace DNDS::Cauchy
{
    enum class RunStop
    {
        None,
        WallLimit,
        Requested
    };

    /// Production output/progress owner. It installs and removes Domain
    /// observers, owns CSV/logger state and serializes accepted/internal output
    /// decisions without leaking file/control logic into the solver frontend.
    /// The caller retains `simulation` and `config` through destruction and
    /// serializes all coordinator calls with Simulation stepping. Destruction
    /// withdraws every installed observer before releasing the sink state.
    class OutputCoordinator
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        OutputCoordinator(Simulation &simulation, const SimulationConfig &config,
                          MPIInfo mpi, real started, RunControl *control = nullptr,
                          std::function<void()> pollInput = {});
        ~OutputCoordinator();
        OutputCoordinator(const OutputCoordinator &) = delete;
        OutputCoordinator &operator=(const OutputCoordinator &) = delete;

        /// Apply exact output/end-time snapping to one proposed physical step.
        OutputPlan::StepTarget SnapStep(real proposedDt) const;
        /// Start one retryable attempt in every local domain's progress sink.
        void BeginAttempt(real dt, index retries, real targetTime);
        /// Drain buffered nonlinear rows after the collective solve returns.
        void FlushPending();
        /// Emit an operational retry diagnostic; this performs no solver work.
        void WarnRetry(const std::string &reason);
        /// Record one accepted step, emit due snapshots and return the global
        /// configured steady/residual stopping decision.
        bool CompleteAcceptedStep(
            real dt, index retries,
            const std::vector<PhysicalStepLimit> &limits,
            bool useLimits, bool printConsole);
        /// Collective check at a step boundary. A queued stop takes precedence
        /// over a simultaneous wall cap so it is guaranteed a checkpoint.
        RunStop PollStop(bool wallLimitReached = false);
        /// Publish summary, flush output, and force a restart archive for a
        /// requested stop. Returns 2 for wall/step caps, otherwise zero.
        int Finalize(RunStop stop, bool reached);
        int Finalize(bool wallStop, bool reached)
        {
            return Finalize(wallStop ? RunStop::WallLimit : RunStop::None, reached);
        }
    };
}
