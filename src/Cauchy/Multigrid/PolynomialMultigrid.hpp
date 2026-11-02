#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Solver/FAS.hpp"

namespace DNDS::Cauchy
{
    /// Same-cell FAS hierarchy: high-order VR -> direct-gradient O2 -> first
    /// order convection. All levels retain viscous gradients/physical BCs.
    /// Normalized state, residual and correction maps are identity on means.
    /// Different meshes/coarsening belong to a different transfer adapter.
    /// This CPU flow5 profile updates the complete coupled block. Fine state
    /// and workspace must remain bound; caller owns physical trial rollback.
    class PolynomialMultigrid
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        PolynomialMultigrid(ssp<Geom::UnstructuredMesh> mesh, const FiniteVolume &fine, CFV::FVField &state,
                            NonlinearSolver &fineSolver, Runtime::EagerEngine &engine, Runtime::ExecutionStream stream,
                            const DomainSettings &settings);
        /// Accept only the original finest uncompressed equation, never the
        /// coarse FAS forcing. Caps return unconverged. Coarse levels have no
        /// mutable VR warm guesses; they are reset by restriction every cycle.
        NonlinearIterationReport Solve(const CFV::FVField &source, real mass, real rhsScale, real time, int stage = 0);
    };
}
