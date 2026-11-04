#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <chrono>
#include <iostream>
#include <tuple>

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
        result.maxIterations = settings.linear.iterations;
        result.subspace = settings.linear.subspace;
        result.relativeTolerance = settings.linear.relativeTolerance;
        result.absoluteTolerance = settings.linear.absoluteTolerance;
        result.sgsSweeps = settings.sgs.sweeps;
        result.exchangeBetweenSweeps = settings.sgs.exchangeBetweenSweeps;
        result.iluFill = settings.ilu.fill;
        result.iluCoreFirst = settings.ilu.coreFirst;
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
        BasicResidualConvergence<decltype(report.residual)> convergence(settings, schema, report.residual);
        report.pseudoCFL = control.initialPseudoCFL == 0 ? settings.pseudoCFL : control.initialPseudoCFL;
        const index limit = control.correctionLimit < 0 ? settings.iterations : control.correctionLimit;
        const bool afterUpdate = observation == ResidualObservation::AfterUpdate;
        DNDS_check_throw_info(!afterUpdate || limit > 0, "post-update residual observation requires at least one update");
        for (index iteration = afterUpdate ? 1 : 0; iteration <= limit; ++iteration)
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
            {
                const PhaseTimer timer(control.profile, NonlinearProfile::Convergence);
                measure(report.residual);
                convergence.Observe(report.residual);
                report.initialResidual = convergence.Initial();
                report.referenceResidual = convergence.Reference();
                report.residualTargetRatio = convergence.TargetRatio();
                report.iterations = iteration;
                const bool converged = iteration >= std::min(settings.minIterations, limit) &&
                                       (reconstructionTolerance <= 0 || report.reconstructionRelativeUpdate <= reconstructionTolerance) && convergence.Accepted();
                if (control.correctionLimit < 0 && settings.consoleInterval > 0 && rank == 0 &&
                    (iteration % settings.consoleInterval == 0 || converged || iteration == limit))
                    std::cout << "Cauchy nonlinear stage " << stage << " iteration " << iteration << " residual " << Json(report.residual).dump()
                              << " CFL " << report.pseudoCFL << " VR " << report.reconstructionIterations << ':' << report.reconstructionRelativeUpdate << std::endl;
                if (converged || iteration == limit)
                {
                    report.converged = converged;
                    return report;
                }
            }
            if (!afterUpdate)
                advance(report);
            report.pseudoCFL = std::min(settings.maxPseudoCFL, report.pseudoCFL * settings.pseudoGrowth);
        }
        return report;
    }
}
