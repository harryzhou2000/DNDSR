#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"

namespace DNDS::Cauchy
{
    struct NonlinearSolveControl
    {
        index correctionLimit = -1; // -1 uses the configured stage cap
        real initialPseudoCFL = 0;  // 0 starts at the configured stage CFL
    };
    struct NonlinearIterationReport
    {
        bool converged = false;
        index iterations = 0, linearIterations = 0, rhsLimited = 0, incrementLimited = 0;
        index multigridCycles = 0;
        std::vector<index> levelCorrections;
        /// All component norms remain visible even for a selected-component
        /// stop. Reference may be a running maximum; initial never changes.
        FlowState initialResidual{}, referenceResidual{}, residual{};
        real residualTargetRatio = 0;
        real minimumRHSAlpha = 1, minimumIncrementAlpha = 1, pseudoCFL = 0;
        /// Final equation evaluation's reconstruction, not an observation
        /// workspace or a sum/max over every inner reconstruction evaluation.
        int reconstructionIterations = 0;
        real reconstructionRelativeUpdate = 0;
        Linear::LinearIterationReport lastLinear;
    };
    /// Initial compressible-flow5 CPU stage adapter. Its physical PP, norms
    /// and identity accumulation do not implement other equation profiles.
    /// The underlying time interface and coupled linear algebra are generic.
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
        /// on a correction cap; the caller chooses whether to advance that
        /// finite iterate or reject the physical trial. The report never labels
        /// a capped, unconverged solve as converged.
        /// Nonfinite/unusable linear corrections and inadmissible states throw.
        /// PP uses Euler alpha compression/flow5 repair only when requested.
        /// Strict reconstruction acceptance, when requested, additionally checks
        /// the latest global VR update against its positive configured tolerance.
        /// Nonnegative control.correctionLimit selects a bounded FAS smoother;
        /// that cap reports work and does not imply physical-stage acceptance.
        /// All ranks agree the override and initial pseudo-CFL before iteration.
        NonlinearIterationReport Solve(CFV::FVField &state, const CFV::FVField &source,
                                       real mass, real rhsScale, real time, int stage = 0, NonlinearSolveControl control = {});
        /// FAS operator in the same normalized convention as Solve. No input
        /// mean mutation; Evaluate may refresh ghosts and reconstruction scratch.
        void EvaluateEquation(CFV::FVField &state, CFV::FVField &value, real mass, real rhsScale, real time);
        /// Rank-local snapshot after the most recent equation evaluation.
        std::pair<int, real> ReconstructionStats() const;
        /// Already agreed by the FV reconstruction's global norm. No MPI here.
        /// With strict acceptance disabled, fixed-sweep Euler behavior remains.
        bool ReconstructionConverged() const;
        /// Apply a separate increment using the same Euler flow5 PP policy as
        /// the smoother. Returns global (limited cells, minimum fraction).
        /// Candidate admissibility is agreed before any state values change.
        std::pair<index, real> ApplyCorrection(CFV::FVField &state, const CFV::FVField &correction, real scale = 1);
    };
}
