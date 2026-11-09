#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"

namespace DNDS::Cauchy
{
    /// Optional synchronous CPU profile. Values are local wall seconds, not
    /// reduced across ranks. Total encloses the disjoint named sections;
    /// its remainder includes controller work and timer overhead. The caller
    /// owns this output through Solve, including unwinding, and serializes it.
    struct NonlinearProfile
    {
        enum Phase
        {
            Entry,
            Residual,
            Convergence,
            PseudoTimePP,
            Jacobian,
            LinearSolve,
            Correction,
            Total,
            PhaseNumber
        };
        std::array<real, PhaseNumber> seconds{};
        index residualEvaluations = 0;
    };
    struct NonlinearSolveControl
    {
        index correctionLimit = -1; // -1 uses the configured stage cap
        real initialPseudoCFL = 0;  // 0 starts at the configured stage CFL
        /// Null means no clock queries or extra communication. A provided
        /// output resets per call and retains partial timings on exceptions.
        NonlinearProfile *profile = nullptr;
        /// Empty updates all components. A subset solves the principal
        /// linear system with all other means frozen, including through PP.
        /// This never changes the separately configured convergence selection.
        std::vector<index> updateComponents;
    };
    template <class Norms>
    struct BasicNonlinearIterationReport
    {
        bool converged = false;
        index iterations = 0, linearIterations = 0, rhsLimited = 0, incrementLimited = 0;
        index multigridCycles = 0;
        std::vector<index> levelCorrections;
        /// All component norms remain visible even for a selected-component
        /// stop. Reference may be a running maximum; initial never changes.
        Norms initialResidual{}, referenceResidual{}, residual{};
        real residualTargetRatio = 0;
        real minimumRHSAlpha = 1, minimumIncrementAlpha = 1, pseudoCFL = 0;
        /// Final equation evaluation's reconstruction, not an observation
        /// workspace or a sum/max over every inner reconstruction evaluation.
        int reconstructionIterations = 0;
        real reconstructionRelativeUpdate = 0;
        Linear::LinearIterationReport lastLinear;
    };
    using NonlinearIterationReport = BasicNonlinearIterationReport<FlowState>;
    using EquationNonlinearIterationReport = BasicNonlinearIterationReport<std::vector<real>>;
    /// Initial compressible-flow5 CPU stage adapter. Its physical PP, norms
    /// and identity accumulation do not implement other equation profiles.
    /// The underlying time interface and coupled linear algebra are generic.
    /// CPU pseudo-time-continuation stage solver. Time equations supply explicit
    /// source/mass/RHS coefficients; physics/discretization, sparse algebra and
    /// physical history remain separate. Each instance binds to one stable trial
    /// state/workspace. Caller serializes all accesses and coordinates retries.
    /// Copies share mutable implementation/workspace state; they are not
    /// independent solvers. The engine is borrowed: it must remain alive at its
    /// original address through all solver/hierarchy copies and queued work.
    /// Drain work before destroying the final owner; do not move a bound engine.
    /// Optional local phases omit successful safety agreement when checks are
    /// disabled and abort on failure. Solve-entry, numerical norms and candidate
    /// acceptance remain coordinated; off mode does not permit rank-local retry.
    class NonlinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;
        NonlinearIterationReport Execute(CFV::FVField &state, const CFV::FVField &source,
                                         real mass, real rhsScale, real time, int stage,
                                         NonlinearSolveControl control, bool singleCorrection, const CFV::FVField *cachedRate);

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
        /// With collectiveChecks enabled ranks agree controls before iteration;
        /// otherwise the caller must supply identical controls/order on all
        /// ranks and fail fast on local errors. Local validation stays active.
        NonlinearIterationReport Solve(CFV::FVField &state, const CFV::FVField &source,
                                       real mass, real rhsScale, real time, int stage = 0, NonlinearSolveControl control = {});
        /// Exactly one pseudo-time/Jacobian/linear/PP update, with no convergence
        /// loop or norm reduction. The caller owns CFL progression and stopping.
        /// Accepts stale input halo; returns owned updates with unspecified halo.
        /// report.converged is false: this operation makes no acceptance decision.
        /// Optional cachedRate is a method-owned F=dU/dt (owned values only).
        /// Caller guarantees its time/stage provenance; it is consumed verbatim,
        /// with fresh auxiliary preparation but no duplicate reconstruction/RHS.
        NonlinearIterationReport CorrectOnce(CFV::FVField &state, const CFV::FVField &source,
                                             real mass, real rhsScale, real time, int stage = 0, NonlinearSolveControl control = {},
                                             const CFV::FVField *cachedRate = nullptr);
        /// FAS operator in the same normalized convention as Solve. No input
        /// mean mutation; Evaluate may refresh ghosts and reconstruction scratch.
        /// Optional rate receives the unscaled spatial dU/dt from this same
        /// evaluation, avoiding cancellation when mass*U dominates the defect.
        /// It must be distinct from state/value and have no conflicting leases.
        void EvaluateEquation(CFV::FVField &state, CFV::FVField &value, real mass, real rhsScale, real time,
                              CFV::FVField *rate = nullptr);
        /// Rank-local snapshot after the most recent equation evaluation.
        std::pair<int, real> ReconstructionStats() const;
        /// Already agreed by the FV reconstruction's global norm. No MPI here.
        /// With strict acceptance disabled, fixed-sweep Euler behavior remains.
        bool ReconstructionConverged() const;
        real ReconstructionTolerance() const;
        /// Apply a separate increment using the same Euler flow5 PP policy as
        /// the smoother. Returns global (limited cells, minimum fraction).
        /// Candidate admissibility is agreed before any state values change.
        /// Optional updateComponents masks before PP; frozen means remain
        /// bit-identical. Ranks must agree selection and scale before entering.
        std::pair<index, real> ApplyCorrection(CFV::FVField &state, const CFV::FVField &correction, real scale = 1,
                                               const std::vector<index> &updateComponents = {});
    };

    /// CPU whole-equation stage adapter over the shared nonlinear iteration
    /// policy, equation FV and coupled linear driver. The explicit residual
    /// selection is resolved by the equation frontend; it cannot silently use
    /// the historical flow5 defaults for a larger system. Every component's
    /// norm is reported. Storage grouping/layout may differ between operands.
    ///
    /// The equation profile matches EquationWorkspace: no Roe fallback.
    /// Polynomial FAS uses CorrectOnce as its bounded smoother. A capped finite solve
    /// remains unconverged and leaves acceptance to the time orchestrator.
    /// Callers serialize bound state/workspace/engine access, coordinate all
    /// domain ranks and keep owners on the controller through worker joins.
    /// Copies share mutable implementation/workspace state. The borrowed engine
    /// must outlive every solver/hierarchy copy and queued operation at a stable
    /// address. Drain work before destroying the final owner; do not move it.
    /// Optional local-phase failures follow the flow5 adapter's same policy:
    /// off mode aborts; candidate acceptance and solve-entry remain coordinated.
    class EquationNonlinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;
        EquationNonlinearIterationReport Execute(EquationDofs &state, const EquationDofs &source,
                                                 real mass, real rhsScale, real time, int stage,
                                                 NonlinearSolveControl control, bool singleCorrection, const EquationDofs *cachedRate);

    public:
        EquationNonlinearSolver(const FiniteVolume &discrete, EquationDofs &state, std::shared_ptr<FiniteVolume::EquationWorkspace> workspace,
                                Runtime::EagerEngine &engine, Runtime::ExecutionStream stream, const DomainSettings &settings,
                                const Algebra::ComponentSelection &residualSelection,
                                std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt);
        EquationNonlinearIterationReport Solve(EquationDofs &state, const EquationDofs &source, real mass, real rhsScale, real time,
                                               int stage = 0, NonlinearSolveControl control = {});
        /// Whole-equation counterpart of NonlinearSolver::CorrectOnce, retaining
        /// split allocation/layout and EOS/source/PP dispatch. No nested solve.
        EquationNonlinearIterationReport CorrectOnce(EquationDofs &state, const EquationDofs &source, real mass, real rhsScale, real time,
                                                     int stage = 0, NonlinearSolveControl control = {}, const EquationDofs *cachedRate = nullptr);
        /// Evaluate dU/dt with aux preparation before reconstruction/RHS and
        /// retain that publication for a subsequent cached CorrectOnce.
        /// Pointwise volume normalization writes owned values only. State/rate
        /// must be distinct; caller coordinates state/time/cache usage on ranks.
        void EvaluateRate(EquationDofs &state, EquationDofs &rate, real time);
        /// Frozen auxiliary publication from the last rate/operator evaluation.
        /// A subsequent cached correction preserves it; after a mean update it
        /// describes the previous linearization, not the updated state. Read-only
        /// diagnostics may inspect its buffers/generations after joining work.
        const AuxiliaryValues &PreparedAuxValues() const;
        /// Same normalized stage operator as the field adapter, without
        /// changing owned means. Polynomial FAS uses this same operator.
        /// Optional rate follows the field adapter's distinct-output contract.
        void EvaluateEquation(EquationDofs &state, EquationDofs &value, real mass, real rhsScale, real time,
                              EquationDofs *rate = nullptr);
        /// Caller agrees scale/selection and releases conflicting views before
        /// entry. Candidate EOS/raw-species checks precede any state mutation.
        /// With PP enabled, use Euler relaxed/common increment compression,
        /// final energy repair and transported mean-species repair. A repair
        /// which would alter a frozen component rejects instead of unfreezing.
        /// RANS always applies its Euler exponential scalar update and final
        /// model bounds after the optional common compression. Old RANS means
        /// must be admissible, and nonfinite corrections are never repaired.
        std::pair<index, real> ApplyCorrection(EquationDofs &state, const EquationDofs &correction, real scale = 1,
                                               const std::vector<index> &updateComponents = {});
        std::pair<int, real> ReconstructionStats() const;
        bool ReconstructionConverged() const;
        real ReconstructionTolerance() const;
    };
}
