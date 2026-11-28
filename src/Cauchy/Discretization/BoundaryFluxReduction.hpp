#pragma once

#include "Cauchy/Discretization/BoundaryFluxIntegral.hpp"
#include "Cauchy/IO/BoundaryOutputGeometry.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

#include <memory>
#include <vector>

namespace DNDS::Cauchy
{
    /// Reusable per-zone boundary conservative-flux, tangential-force, and area
    /// reduction over only the opted-in boundary faces on CPU or CUDA.
    class BoundaryFluxReduction
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        explicit BoundaryFluxReduction(Runtime::ExecutionContext context);

        /// Evaluate rank-local sums for each requested zone in order.
        /// Caller performs the domain MPI sum reduction across the returned vector.
        std::vector<real> EvaluateLocal(Runtime::EagerEngine &engine,
                                        const Runtime::ExecutionStream &stream,
                                        const std::vector<BoundaryOutputRow> &allBoundaryRows,
                                        const Runtime::Array<real> &faceFlux,
                                        const Runtime::Array<real> &boundaryMomentum,
                                        const Runtime::Array<real> &faceQuadrature,
                                        const std::vector<int> &zones);
    };
}
