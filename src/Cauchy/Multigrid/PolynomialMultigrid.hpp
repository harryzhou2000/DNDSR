#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"

namespace DNDS::Cauchy
{
    /// Same-cell FAS hierarchy: high-order VR -> direct-gradient O2 -> first
    /// order convection. All levels retain viscous gradients/physical BCs.
    /// Normalized state, residual and correction maps are identity on means.
    /// Different meshes/coarsening belong to a different transfer adapter.
    /// CPU flow5 and logical equation states support full or selected-component
    /// principal-system cycles through the same hierarchy and FAS recursion.
    /// Unselected means remain frozen; all finest residuals stay observable.
    /// Fine state/workspace remain bound; caller retains the finest solver and
    /// engine, serializes calls and owns physical trial rollback.
    class PolynomialMultigrid
    {
        struct Data;
        struct EquationData;
        std::shared_ptr<Data> _data;
        std::shared_ptr<EquationData> _equationData;

    public:
        /// Zero-based coarse-level controls, also usable to reproduce a level
        /// operator independently. Level 0 uses direct-gradient O2 convection;
        /// subsequent levels use mean convection with the same gradients for
        /// diffusion and sources. No mesh/state ownership is changed.
        static DomainSettings CoarseSettings(const DomainSettings &fine, index level);
        PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, CFV::FVField &state,
                            NonlinearSolver &fineSolver, Runtime::EagerEngine &engine, Runtime::ExecutionStream stream,
                            const DomainSettings &settings);
        /// Logical-state counterpart. Preserve each physical allocation's
        /// layout and every block's component map on all levels; no packing.
        /// Coarse operators retain the same immutable equation and worker pool.
        PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, EquationDofs &state,
                            EquationNonlinearSolver &fineSolver, Runtime::EagerEngine &engine, Runtime::ExecutionStream stream,
                            const DomainSettings &settings, std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = std::nullopt);
        /// Accept only the original finest uncompressed equation, never the
        /// coarse FAS forcing. Caps return unconverged. Coarse levels have no
        /// mutable VR warm guesses; they are reset by restriction every cycle.
        NonlinearIterationReport Solve(const CFV::FVField &source, real mass, real rhsScale, real time, int stage = 0);
        EquationNonlinearIterationReport Solve(const EquationDofs &source, real mass, real rhsScale, real time, int stage = 0);
        /// One finest update, with bounded CorrectOnce smoothing at each level.
        /// No convergence loop/reference reset; pseudoCFL comes from the caller.
        /// Other-stage inputs in source must remain frozen throughout the cycle.
        /// Optional rate is the caller's already evaluated finest dU/dt, as in
        /// CorrectOnce. Caller guarantees current owned means/time/stage and
        /// reconstruction provenance. It is consumed only by the first finest
        /// smoother; further corrections evaluate their updated states normally.
        NonlinearIterationReport ApplyCycle(const CFV::FVField &source, real mass, real rhsScale, real time,
                                            int stage = 0, real pseudoCFL = 0, const CFV::FVField *rate = nullptr);
        EquationNonlinearIterationReport ApplyCycle(const EquationDofs &source, real mass, real rhsScale, real time,
                                                    int stage = 0, real pseudoCFL = 0, const EquationDofs *rate = nullptr);
        /// Frozen stage trace shared with coarse levels after checking their
        /// canonical boundary quadrature order. Local alias binding only; no MPI.
        void SetBoundaryInputs(const BoundaryInputs &input);
    };
}
