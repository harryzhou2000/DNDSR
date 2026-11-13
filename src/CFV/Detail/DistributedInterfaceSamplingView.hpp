#pragma once
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::CFV::Detail
{
    /** Borrowed direct donor sampling inputs, prepared by the plan facade.
     * Each query refers to an owned cell. Constant sampling reads only cells
     * and means; gradient sampling also reads points, gradients and cell metrics.
     * Gradient columns are component*dimension+direction; metric columns 1:4
     * hold the reference-frame centroid. There is no frame rotation or MPI here.
     * The enclosing operation retains the arrays and the geometry read scope.
     */
    struct DirectDonorSamplingView
    {
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> cells;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> means, points, gradients;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> geometry;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> output;
        int dimension = 0;
        bool gradient = false;

        /// Whole-array residency is checked once at the C++ launcher boundary.
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, cells, means, output);
            if (gradient)
                Runtime::ValidateViewDevices(device, points, gradients, geometry);
        }
    };
}
