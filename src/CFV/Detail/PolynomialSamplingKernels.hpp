#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "CFV/Math/TaylorBasis.hpp"

namespace DNDS::CFV::Detail
{
    using SamplingIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using SamplingTable = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using SamplingField = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using SamplingOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    template <index Width>
    using SamplingFixedInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<Width>>;
    template <index Width>
    using SamplingFixedOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<Width>>;

    /// Fused bounded CPU specialization for the common P3/2D/SA source
    /// gradient queries. Optional values have a separate compile-time body;
    /// callers validate extents and aliases before supplying borrowed views.
    void EvaluateCachedOwnedValueGradient(SamplingIndices cells, SamplingFixedInput<6> means,
                                          SamplingFixedInput<54> coefficients, SamplingFixedInput<27> basis,
                                          SamplingFixedOutput<6> values, SamplingFixedOutput<6> dx, SamplingFixedOutput<6> dy, bool includeValues);

    /// Explicit numerical I/O; retained access and backend dispatch stay above.
    /// Cached basis is [basisRows[point], mode*cachedDerivatives+derivative];
    /// empty basisRows means identity. outputRows similarly selects unique
    /// output destinations, preserving unrelated rows in a combined PP buffer.
    /// Direct inputs
    /// have zero rows when all allowed derivatives are cached. Modes omit the
    /// constant polynomial; moments include it at column zero.
    struct PolynomialSamplingView
    {
        SamplingIndices cells, basisRows, outputRows;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> points;
        SamplingTable basis, moments;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<8>> geometry;
        Math::TaylorBasisParameters polynomial;
        int cachedDerivatives = 0;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, cells, basisRows, outputRows, points, basis, moments, geometry);
        }
    };
    /// Independent CPU point rows, with bounded thread-local basis scratch.
    /// Validated shapes and derivative ordinal are preconditions.
    void EvaluatePolynomialSamples(PolynomialSamplingView plan, SamplingField mean, SamplingField coefficients,
                                   SamplingOutput output, int derivative);
    /// CUDA host launcher: validates residency and parameter budget once, then
    /// enqueues cached/direct specialized kernels on the supplied stream.
    void LaunchPolynomialSamples(Runtime::ExecutionContext context, PolynomialSamplingView plan,
                                 SamplingField mean, SamplingField coefficients, SamplingOutput output, int derivative);
    /// Optional bounded CPU value kernel. All views borrow enclosing host
    /// leases; tables have unit component stride and may have row padding.
    /// Returns false for an unsupported width/mode count without writing.
    /// Keep this out of the generic derivative sampler's translation unit:
    /// value specializations must not enlarge its numerical loop body.
    bool TryEvaluateCachedOwnedTable(
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> cells,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> mean,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> coefficients,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> basis,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> output);
}
