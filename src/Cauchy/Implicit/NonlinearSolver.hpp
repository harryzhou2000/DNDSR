#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"

namespace DNDS::Cauchy
{
    struct NonlinearIterationReport
    {
        bool converged = false;
        index iterations = 0, linearIterations = 0, rhsLimited = 0, incrementLimited = 0;
        FlowState initialResidual{}, residual{};
        real minimumRHSAlpha = 1, minimumIncrementAlpha = 1, pseudoCFL = 0;
        Linear::LinearIterationReport lastLinear;
    };
    /// CPU pseudo-time-continuation stage solver. Time equations supply explicit
    /// source/mass/RHS coefficients; physics/discretization, sparse algebra and
    /// physical history remain separate. Each instance binds to one stable trial
    /// state/workspace. Caller serializes all accesses and coordinates retries.
    class NonlinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        NonlinearSolver(const FiniteVolume &discrete, CFV::FVField &state, std::shared_ptr<FiniteVolume::Workspace> workspace,
                        Runtime::EagerEngine &engine, Runtime::ExecutionStream stream, const DomainSettings &settings);
        /// Solve mass*U-rhsScale*(R(U)/V)=source. Source is owned and immutable;
        /// all domain ranks enter with identical controls. Returns unconverged
        /// on a correction cap; the caller must reject that physical trial.
        /// Nonfinite/unusable linear corrections and inadmissible states throw.
        /// PP uses Euler alpha compression/flow5 repair only when requested.
        NonlinearIterationReport Solve(CFV::FVField &state, const CFV::FVField &source,
                                       real mass, real rhsScale, real time, int stage = 0);
    };
}
