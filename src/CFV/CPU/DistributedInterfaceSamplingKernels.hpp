#pragma once

#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::CFV::CPU
{
    /// Prepared direct constant/gradient donor evaluation. The plan facade
    /// retains all leases and validates participation before calling this loop.
    struct DirectDonorSamplingRequest
    {
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            cells;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            means;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            points, gradients;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            output;
        FiniteVolumeHostView geometry;
        int dimension = 0;
        bool gradient = false;
    };

    void EvaluateDirectDonors(const DirectDonorSamplingRequest &request);
}
