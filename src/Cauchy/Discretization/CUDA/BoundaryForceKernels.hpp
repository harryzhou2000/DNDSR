#pragma once
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"

namespace DNDS::Cauchy::CUDA
{
    using ForceSamples = Runtime::ArrayView<const FiniteVolumeDetail::SampleRecord, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using ForceEntries = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using ForceInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>>;
    using ForceOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>>;
    using ForceCounter = Runtime::ArrayView<unsigned int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    index BoundaryForceScratchRows(index points);
    /// One fused fixed-tree reduction, including a last-block tail. Views and
    /// selected-zone metadata must be resident; output/counter/partials distinct.
    /// Setup guarantees every boundary entry indexes samples and its ordinal
    /// indexes momentum. The wrapper checks shape/residency once, not per point.
    void AccumulateBoundaryForce(Runtime::ExecutionContext, ForceSamples, ForceEntries, ForceInput,
                                 SpanView<const int> zones, ForceOutput partial, ForceOutput result, ForceCounter);
}
