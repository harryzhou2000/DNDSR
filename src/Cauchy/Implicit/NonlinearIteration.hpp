#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <chrono>
#include <tuple>
#include <type_traits>

namespace DNDS::Cauchy::NonlinearDetail
{
    /// Accumulate numerical work without treating coarse corrections as finest
    /// nonlinear iterations or overwriting the finest convergence observation.
    template <class Report>
    void AccumulateWork(Report &total, const Report &step)
    {
        total.linearIterations += step.linearIterations;
        total.rhsLimited += step.rhsLimited;
        total.incrementLimited += step.incrementLimited;
        total.minimumRHSAlpha = std::min(total.minimumRHSAlpha, step.minimumRHSAlpha);
        total.minimumIncrementAlpha = std::min(total.minimumIncrementAlpha, step.minimumIncrementAlpha);
        total.lastLinear = step.lastLinear;
        total.multigridCycles += step.multigridCycles;
        if (total.levelCorrections.size() < step.levelCorrections.size())
            total.levelCorrections.resize(step.levelCorrections.size());
        for (index level = 0; level < index(step.levelCorrections.size()); ++level)
            total.levelCorrections[level] += step.levelCorrections[level];
    }
    inline LinearSolveSettings ResolveLinear(const ImplicitSettings &settings)
    {
        LinearSolveSettings result;
        result.method = settings.linear.method;
        result.preconditioner = settings.linear.preconditioner;
        result.stopping = settings.linear.stopping;
        result.maxIterations = settings.linear.iterations;
        result.subspace = settings.linear.subspace;
        result.relativeTolerance = settings.linear.relativeTolerance;
        result.absoluteTolerance = settings.linear.absoluteTolerance;
        result.sgsSweeps = settings.sgs.sweeps;
        result.jacobiLocalSweeps = settings.linear.jacobiLocalSweeps;
        result.exchangeBetweenSweeps = settings.sgs.exchangeBetweenSweeps;
        result.iluFill = settings.ilu.fill;
        result.iluPreconditionerCorrections = settings.ilu.preconditionerCorrections;
        result.iluCoreFirst = settings.ilu.coreFirst;
        result.iluOrdering = settings.ilu.ordering;
        return result;
    }
    /// Exact controller agreement, including component lists larger than a
    /// machine-word mask. Callers coordinate serialization before entering.
    inline void AgreeControls(const MPIInfo &mpi, const std::string &serialized)
    {
        int length = mpi.rank == 0 ? int(serialized.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string reference;
        MPI::LocalStage(mpi, [&]
                        { reference = mpi.rank == 0 ? serialized : std::string(size_t(length), '\0'); });
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, mpi.comm);
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(reference == serialized, "implicit controls differ between domain ranks"); });
    }
    /// Optional local timing only. Scope unwinding preserves partial evidence.
    class PhaseTimer
    {
        using Clock = std::chrono::steady_clock;
        real *_seconds = nullptr;
        Clock::time_point _start{};

    public:
        PhaseTimer(NonlinearProfile *profile, NonlinearProfile::Phase phase)
            : _seconds(profile ? &profile->seconds[phase] : nullptr)
        {
            if (_seconds)
                _start = Clock::now();
        }
        PhaseTimer(OptionalRef<NonlinearProfile> profile,
                   NonlinearProfile::Phase phase)
            : PhaseTimer(profile.Get(), phase) {}
        PhaseTimer(const PhaseTimer &) = delete;
        PhaseTimer &operator=(const PhaseTimer &) = delete;
        ~PhaseTimer()
        {
            if (_seconds)
                *_seconds += std::chrono::duration<real>(Clock::now() - _start).count();
        }
    };

    /// One iteration policy for field and logical-equation adapters. Residual
    /// and correction callbacks own their numerical access/MPI contracts;
    /// measure returns globally reduced norms without modifying their order.
    /// No physical acceptance, rollback or extra collective is introduced here.
    enum class ResidualObservation
    {
        BeforeUpdate,
        AfterUpdate
    };
    template <class Report, class Evaluate, class Measure, class Reconstruction, class Advance>
    Report Iterate(const ImplicitSettings &settings, const EquationSchema &schema, int rank, int stage, const NonlinearSolveControl &control,
                   Report report, real reconstructionTolerance, Evaluate &&evaluate, Measure &&measure,
                   Reconstruction &&reconstruction, Advance &&advance,
                   ResidualObservation observation = ResidualObservation::BeforeUpdate)
    {
        (void)rank; // Output is emitted only through control.progress.
        // A default logical-equation report has no compile-time width. Resolve
        // it once from the schema before constructing the convergence policy;
        // production callers may still provide a populated report to retain
        // accumulated multigrid work and controller diagnostics.
        if constexpr (std::is_same_v<std::decay_t<decltype(report.residual)>, std::vector<real>>)
        {
            index components = 0;
            for (const auto &block : schema.blocks)
                components += index(block.components.size());
            if (report.residual.empty())
                report.residual.assign(size_t(components), 0);
        }
        BasicResidualConvergence<decltype(report.residual)> convergence(settings, schema, report.residual);
        const index limit = control.correctionLimit < 0 ? settings.iterations : control.correctionLimit;
        const bool afterUpdate = observation == ResidualObservation::AfterUpdate;
        DNDS_check_throw_info(!afterUpdate || limit > 0, "post-update residual observation requires at least one update");
        index firstIteration = afterUpdate ? 1 : 0;
        if (control.resume)
        {
            DNDS_check_throw_info(control.state && control.state->schemaVersion == 1 &&
                                      control.state->iteration >= 0 && !control.state->rejected,
                                  "nonlinear resume needs a valid live controller state");
            const auto &state = *control.state;
            DNDS_check_throw_info(state.stage == stage && state.residual.size() == report.residual.size() &&
                                      state.initialResidual.size() == report.residual.size() &&
                                      state.referenceResidual.size() == report.residual.size() &&
                                      std::isfinite(state.pseudoCFL) && state.pseudoCFL > 0,
                                  "nonlinear resume state differs from the active equation");
            report.iterations = state.iteration;
            report.residual = state.residual;
            report.initialResidual = state.initialResidual;
            report.referenceResidual = state.referenceResidual;
            report.residualTargetRatio = state.residualTargetRatio;
            report.pseudoCFL = state.pseudoCFL;
            report.linearIterations = state.linearIterations;
            report.multigridCycles = state.multigridCycles;
            report.levelCorrections = state.levelCorrections;
            report.rhsLimited = state.rhsLimited;
            report.incrementLimited = state.incrementLimited;
            report.minimumRHSAlpha = state.minimumRHSAlpha;
            report.minimumIncrementAlpha = state.minimumIncrementAlpha;
            report.reconstructionIterations = state.reconstructionIterations;
            report.reconstructionRelativeUpdate = state.reconstructionRelativeUpdate;
            report.reconstructionUpdateMeasured = state.reconstructionUpdateMeasured;
            convergence.Restore(report.initialResidual, report.referenceResidual);
            firstIteration = state.iteration + 1;
            DNDS_check_throw_info(state.terminal || firstIteration <= limit,
                                  "restored nonlinear cursor exceeds its correction cap");
            if (state.terminal)
            {
                report.converged = state.converged;
                return report;
            }
        }
        else
        {
            report.pseudoCFL = control.initialPseudoCFL == 0 ? settings.pseudoCFL : control.initialPseudoCFL;
            if (control.state)
            {
                auto &state = *control.state;
                state.stage = stage;
                state.coupledStage = -1;
                state.iteration = 0;
                state.converged = state.terminal = state.capped = state.rejected = false;
                state.pseudoCFL = report.pseudoCFL;
            }
        }
        for (index iteration = firstIteration; iteration <= limit; ++iteration)
        {
            // A time problem may update several coupledStages here, with one
            // shared CFL/reference/cap policy and one subsequent observation.
            if (afterUpdate)
                advance(report);
            {
                const PhaseTimer timer(control.profile, NonlinearProfile::Residual);
                evaluate();
                if (control.profile)
                    ++control.profile->residualEvaluations;
            }
            std::tie(report.reconstructionIterations, report.reconstructionRelativeUpdate) = reconstruction();
            report.reconstructionUpdateMeasured = std::isfinite(report.reconstructionRelativeUpdate);
            if (!report.reconstructionUpdateMeasured)
                report.reconstructionRelativeUpdate = 0;
            {
                const PhaseTimer timer(control.profile, NonlinearProfile::Convergence);
                measure(report.residual);
                convergence.Observe(report.residual);
                report.initialResidual = convergence.Initial();
                report.referenceResidual = convergence.Reference();
                report.residualTargetRatio = convergence.TargetRatio();
                report.iterations = iteration;
                const bool converged = iteration >= std::min(settings.minIterations, limit) &&
                                       (reconstructionTolerance <= 0 || (report.reconstructionUpdateMeasured && report.reconstructionRelativeUpdate <= reconstructionTolerance)) && convergence.Accepted();
                // Console iteration numbers are one-based. Before-update mode
                // observes the initial residual at internal index zero; omit
                // that pre-iteration line so the first reported nonlinear
                // iteration is the first completed correction.
                const index reportedIteration = std::max(iteration, index(1));
                const bool initialBeforeUpdate = !afterUpdate && iteration == 0;
                const auto copy = [](const auto &values)
                { return std::vector<real>(values.begin(), values.end()); };
                const NonlinearProgress progress{stage,
                                                 -1,
                                                 reportedIteration,
                                                 converged,
                                                 converged || iteration == limit,
                                                 copy(report.residual),
                                                 copy(report.initialResidual),
                                                 copy(report.referenceResidual),
                                                 report.residualTargetRatio,
                                                 report.pseudoCFL,
                                                 report.linearIterations,
                                                 report.rhsLimited,
                                                 report.incrementLimited,
                                                 report.minimumRHSAlpha,
                                                 report.minimumIncrementAlpha,
                                                 report.reconstructionIterations,
                                                 report.reconstructionRelativeUpdate,
                                                 report.reconstructionUpdateMeasured};
                if (control.state)
                {
                    auto &state = *control.state;
                    state.iteration = iteration;
                    state.converged = progress.converged;
                    state.terminal = progress.terminal;
                    state.capped = progress.terminal && !progress.converged;
                    state.residual = progress.residual;
                    state.initialResidual = progress.initialResidual;
                    state.referenceResidual = progress.referenceResidual;
                    state.residualTargetRatio = progress.residualTargetRatio;
                    state.pseudoCFL = progress.pseudoCFL;
                    state.linearIterations = progress.linearIterations;
                    state.multigridCycles = report.multigridCycles;
                    state.levelCorrections = report.levelCorrections;
                    state.rhsLimited = progress.rhsLimited;
                    state.incrementLimited = progress.incrementLimited;
                    state.minimumRHSAlpha = progress.minimumRHSAlpha;
                    state.minimumIncrementAlpha = progress.minimumIncrementAlpha;
                    state.reconstructionIterations = progress.reconstructionIterations;
                    state.reconstructionRelativeUpdate = progress.reconstructionRelativeUpdate;
                    state.reconstructionUpdateMeasured = progress.reconstructionUpdateMeasured;
                }
                if (control.progress && !initialBeforeUpdate)
                    control.progress(progress);
                if (converged || iteration == limit)
                {
                    report.converged = converged;
                    if (control.quiescent)
                    {
                        DNDS_check_throw_info(control.state, "quiescent nonlinear observer needs live controller state");
                        control.quiescent(*control.state);
                    }
                    return report;
                }
            }
            if (!afterUpdate)
                advance(report);
            report.pseudoCFL = std::min(settings.maxPseudoCFL, report.pseudoCFL * settings.pseudoGrowth);
            if (control.state)
                control.state->pseudoCFL = report.pseudoCFL;
            if (control.quiescent)
            {
                DNDS_check_throw_info(control.state, "quiescent nonlinear observer needs live controller state");
                control.quiescent(*control.state);
            }
        }
        return report;
    }
}
