#pragma once
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Input/FlowProfileMath.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include <limits>

namespace DNDS::Cauchy::CUDA
{
    /// Rank-local reduction result. Admissibility returns value=1/0; CFL returns
    /// the minimum step. firstInvalid==-1 denotes success. Empty partitions use
    /// true or max(real), matching the CPU reductions. No MPI is hidden here.
    struct StateReductionValue
    {
        real value;
        index firstInvalid;
    };
    using StateInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using StateReductionInput = Runtime::ArrayView<const StateReductionValue, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using StateReductionOutput = Runtime::ArrayView<StateReductionValue, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    inline constexpr StateReductionValue StableStepIdentity{std::numeric_limits<real>::max(), -1};
    /// Reserve this many rows once in an operator workspace, on the execution
    /// device. The result has one row. Both buffers remain owned by the caller.
    index StateReductionScratchRows(index ownedRows);
    using LimiterCountOutput = Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using LimiterMinimumOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Deterministic rank-local count(beta<1) and min(beta), shared by modal
    /// and gradient PP. Two launches, no allocation/host synchronization or
    /// atomic floating-point accumulation. Scratch uses StateReductionScratchRows
    /// rows (at most 256); scalar outputs have one row. Buffers must be disjoint.
    /// Numerical validity is checked by the separate mandatory status segment.
    void ReduceLimiterStatistics(Runtime::ExecutionContext, StateInput beta,
                                 LimiterCountOutput partialCount, LimiterMinimumOutput partialMinimum,
                                 LimiterCountOutput count, LimiterMinimumOutput minimum);
    /// Count nonzero byte flags into one exact integer. Same bounded scratch
    /// and stream-order contract; controller decides all/any after its drain.
    void CountFlags(Runtime::ExecutionContext,
                    Runtime::ArrayView<const uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>,
                    LimiterCountOutput partialCount, LimiterCountOutput count);
    using StatusInput = Runtime::ArrayView<const NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using StatusOutput = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Begin a numerical segment with successful status, ordered on the same
    /// stream as its producers. This resets diagnostics, never solution data.
    void ClearNumericalStatus(Runtime::ExecutionContext, StatusOutput);
    /// Reduce two status tables into one device result without a host wait.
    /// Success is the identity; failures are ordered lexicographically by
    /// row, point, code and detail so scheduling does not choose the diagnostic.
    /// Scratch has at least StateReductionScratchRows(first.rows + second.rows)
    /// rows (at most 256), permitting reuse for different producer sizes. This
    /// mandatory error publication is independent of optional diagnostic counts.
    /// Result may alias a one-row input accumulator: all input reads complete
    /// in the first launch before the second overwrites the result. Scratch
    /// must be disjoint from every operand. This supports sticky segment errors.
    void ReduceNumericalStatus(Runtime::ExecutionContext, StatusInput first, StatusInput second,
                               StatusOutput scratch, StatusOutput result);
    /// One block-reduced atomic minimum launch, without allocation or host sync.
    /// An ordered memset seeds unsigned identities; at least one block publishes
    /// value=1/0 and the first invalid row, including empty selected sets.
    /// Only owned input is read, so a ghost exchange may proceed independently.
    /// Complete combined AoS, perfect-gas flow5/SA/2EQ policy only; no mixture
    /// coefficient pointers or host policy owners enter the kernel.
    void AdmissibleMeans(Runtime::ExecutionContext, StateInput means, AdmissibilityView,
                         StateReductionOutput result,
                         Runtime::ArrayView<const uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> activeRows = {});
    /// One reduction launch with block-local minima and global atomic minima.
    /// identity is an immutable one-row device buffer containing StableStepIdentity;
    /// it resets result by stream-ordered device copy on every invocation. No
    /// partial array or fixed partial-count ceiling. Geometry volumes must be
    /// finite and positive (validated at setup); invalid radii report their first
    /// row. Empty input returns the identity. Caller retains all buffers through
    /// completion and serializes reuse of result across streams.
    void MinimumStableStep(Runtime::ExecutionContext, StateInput spectralSum,
                           Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells, real cfl,
                           StateReductionInput identity, StateReductionOutput result);

    /// Prepared reference-coordinate quadrature, in the CPU oracle's exact
    /// element/quadrature order. points rows are [x,y,z,Jacobian*weight]. These
    /// are setup-owned immutable arrays, not reconstructed solution samples.
    /// The enclosing operator retains the geometry version and all allocations.
    /// Use INT_ORDER_MAX when comparing with CPU::MeasureFlowError, independently
    /// of the lower order optionally chosen for spatial residual integration.
    struct FlowProfileQuadrature
    {
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<4>> points;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, offsets, points, cells);
        }
    };
    /// Unnormalized rank-local L1/L2-squared/Linf contributions. A host reporting
    /// boundary performs MPI aggregation and final division/square root. A failed
    /// status invalidates the entire summary; failed cells are never silently
    /// accepted as zero error. Empty ranks produce zero contributions.
    struct FlowErrorReduction
    {
        std::array<FlowState, 3> errors{};
        real volume = 0;
        NumericalStatus status;
    };
    using FlowErrorOutput = Runtime::ArrayView<FlowErrorReduction, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using FlowErrorCounter = Runtime::ArrayView<unsigned int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Preallocated scratch and operation-owned counter. The last arriving
    /// block performs the fixed-index tail in the same kernel; no grid spinning.
    /// Measures the first flow-five components of combined AoS 5/6/7 means.
    /// Auxiliary RANS components do not participate in the analytic flow oracle.
    /// No mesh access, quadrature construction, MPI or host wait in this wrapper.
    void MeasureFlowError(Runtime::ExecutionContext, FlowProfileQuadrature, StateInput means,
                          FlowProfileParameters, FlowState primitive, PerfectGas, RigidFrameParameters,
                          real time, FlowErrorOutput scratch, FlowErrorOutput result, FlowErrorCounter completed);
}
