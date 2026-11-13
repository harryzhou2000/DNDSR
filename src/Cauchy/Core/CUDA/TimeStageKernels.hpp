#pragma once

#include "Cauchy/Core/TimeStageValues.hpp"
#include "Cauchy/Physics/Positivity.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy::CUDA
{
    using TimeStageInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using TimeStageOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using TimeStageCells = Runtime::ArrayView<const real, false, Runtime::FixedRows<8>>;
    /// Bounded deterministic reduction; status invalidates the blend values.
    struct TimeStageSummary
    {
        ExplicitBlendStatistics blend;
        NumericalStatus status;
    };
    using TimeStageSummaryOutput = Runtime::ArrayView<TimeStageSummary, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    index TimeStageScratchRows(index rows);
    /// Read-only thermal forecast; alpha restricts a common dt, not a local
    /// conservative update. No RANS bounds participate in the factor.
    void LimitPhysicalStep(Runtime::ExecutionContext, TimeStageCells, TimeStageInput mean, TimeStageInput rhs,
                           ThermalView, real dt, real relax, real scale, int compressionPower,
                           TimeStageSummaryOutput partial, TimeStageSummaryOutput result);
    void MeasureExplicitBlend(Runtime::ExecutionContext, TimeStageCells, TimeStageInput initial,
                              TimeStageInput stage, TimeStageInput high, TimeStageInput low, ThermalBounds, ExplicitStageCoefficients,
                              TimeStageSummaryOutput partial, TimeStageSummaryOutput result);
    void UpdateExplicitFlow(Runtime::ExecutionContext, TimeStageCells, TimeStageInput initial,
                            TimeStageInput rhs, TimeStageInput low, TimeStageOutput stage, real alpha, ExplicitStageCoefficients);
    void UpdateExplicitExtended(Runtime::ExecutionContext, TimeStageCells, TimeStageInput initial,
                                TimeStageInput rhs, TimeStageOutput stage, RANSStateView, ExplicitStageCoefficients,
                                TimeStageSummaryOutput partial, TimeStageSummaryOutput result);
}
