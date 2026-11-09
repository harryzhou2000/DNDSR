#pragma once

#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Solver/FieldGMRES.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"

namespace DNDS::Cauchy
{
    /// Resolved numerical choices, independent of frontend configuration layout.
    /// sgs and ilu are stationary iterations; only gmres constructs Krylov
    /// history and uses preconditioner. ILU uses frozen rank-local factors and
    /// x_next=x+M^-1(b-A*x), with a normal halo before evaluating A*x.
    /// Norms use the integrated RHS, without volume normalization. The default
    /// metric is Euclidean; Solve may supply explicit component scales.
    struct LinearSolveSettings
    {
        std::string method = "sgs", preconditioner = "sgs";
        /// residual: tolerance checks/reporting. fixedCount: SGS/ILU execute
        /// exactly maxIterations, omit diagnostic residuals and check only the
        /// returned update's finiteness. Tolerances are ignored in fixedCount.
        std::string stopping = "residual";
        index maxIterations = 10;
        int subspace = 20, sgsSweeps = 1, iluFill = 0;
        /// Fixed zero-started ILU defect corrections per GMRES application.
        /// More than one uses the full operator/halo between local solves.
        /// The count is frozen for Solve; stationary ILU uses maxIterations.
        int iluPreconditionerCorrections = 1;
        bool iluCoreFirst = true, exchangeBetweenSweeps = false;
        real relativeTolerance = .1, absoluteTolerance = 1e-12;
        std::string iluOrdering = "natural";
    };
    struct LinearExecution
    {
        bool overlap = true;
        /// Checked mode coordinates local kernel failures before more MPI.
        /// False is caller-ordered eager mode: callers own distributed failure
        /// handling and identical per-call controls/order. Setup agreement stays
        /// enabled; repeated solve-entry checks follow this switch.
        /// A nonfinite fixed-count result aborts the communicator in unchecked
        /// mode; checked mode returns a globally agreed NonFinite report.
        bool collectiveChecks = true;
    };
    /// Domain-local CPU linear orchestration. Owns reusable native vectors and
    /// persistent bindings; numerical matrices and kernels remain independent.
    /// All domain ranks construct and call Solve together with agreed controls.
    /// The caller serializes solver/operand access and releases debug views
    /// before reuse. No physical state, time history or PP policy lives here.
    class LinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        using Field = Algebra::DofField<real>;
        using HaloBinder = std::function<Runtime::TransferBinding(Field &)>;
        LinearSolver(std::shared_ptr<LinearProblemContext> problem, const Field &prototype, const HaloBinder &bindHalo,
                     LinearSolveSettings settings = {}, LinearExecution execution = {});
        LinearSolver(const LinearSolver &) = delete;
        LinearSolver &operator=(const LinearSolver &) = delete;
        LinearSolver(LinearSolver &&) noexcept = default;
        LinearSolver &operator=(LinearSolver &&) noexcept = default;
        /// Frozen matrix for this solve. Stationary methods default to zero initial
        /// iterate; a warm guess is explicit. Every GMRES preconditioner call
        /// resets its iterate to zero and uses a fixed number of SGS sweeps or
        /// the same local ILU factors. Standard GMRES must not vary that policy.
        /// Optional positive logical component scales define a frozen inner
        /// product sum(x_j/s_j*y_j/s_j). Empty keeps the Euclidean metric.
        /// Caller supplies agreed scales across ranks. Their width and
        /// positivity are checked once per solve;
        /// no physics or mean-state reduction occurs inside the linear solver.
        Linear::LinearIterationReport Solve(const Field &rhs, const Field *initialGuess = nullptr,
                                            const std::vector<real> &componentScales = {});
        /// Borrowed result, overwritten by the next Solve. Always inspect its
        /// report; finite fixed-count or iteration-limited results are usable
        /// without being certified converged. Unevaluated residual norms are NaN.
        const Field &Solution() const;
    };

    /// The same domain-local CPU solve policy on a complete logical equation
    /// state. Flow/RANS/species may share an array or occupy separate fields in
    /// any supported table layout. Dense blocks couple every logical component;
    /// no full-state repacking or implicit device fallback occurs.
    ///
    /// All domain ranks construct/call together. The prototype's physical
    /// grouping and schema must agree across ranks before bindHalo is invoked.
    /// The binder prepares each physical field in that order using the domain's
    /// shared row mapping; it must coordinate its own distributed failures.
    /// Callers serialize access, retain MPI owners on the controller, and release
    /// views before Solve. Results and fixed-preconditioner rules match
    /// LinearSolver. Numerical scopes and transfer requests remain separate.
    class EquationLinearSolver
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        using HaloBinder = LinearSolver::HaloBinder;
        EquationLinearSolver(std::shared_ptr<LinearProblemContext> problem, const EquationDofs &prototype,
                             const HaloBinder &bindHalo, LinearSolveSettings settings = {}, LinearExecution execution = {});
        EquationLinearSolver(const EquationLinearSolver &) = delete;
        EquationLinearSolver &operator=(const EquationLinearSolver &) = delete;
        EquationLinearSolver(EquationLinearSolver &&) noexcept = default;
        EquationLinearSolver &operator=(EquationLinearSolver &&) noexcept = default;
        /// Same optional frozen component-scale metric as LinearSolver.
        Linear::LinearIterationReport Solve(const EquationDofs &rhs, const EquationDofs *initialGuess = nullptr,
                                            const std::vector<real> &componentScales = {});
        const EquationDofs &Solution() const;
    };
}
