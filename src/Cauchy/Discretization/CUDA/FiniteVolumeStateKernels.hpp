#pragma once
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Input/FlowProfileMath.hpp"
#include "Cauchy/Motion/RigidFrameParameters.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

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
    using StateReductionOutput = Runtime::ArrayView<StateReductionValue, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Reserve this many rows once in an operator workspace, on the execution
    /// device. The result has one row. Both buffers remain owned by the caller.
    index StateReductionScratchRows(index ownedRows);
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
    /// Two stream-ordered reduction launches, without allocation or host sync.
    /// Only owned input is read, so a ghost exchange may proceed independently.
    /// Complete combined AoS, perfect-gas flow5/SA/2EQ policy only; no mixture
    /// coefficient pointers or host policy owners enter the kernel.
    void AdmissibleMeans(Runtime::ExecutionContext, StateInput means, AdmissibilityView,
                         StateReductionOutput scratch, StateReductionOutput result);
    void MinimumStableStep(Runtime::ExecutionContext, StateInput spectralSum,
                           Runtime::ArrayView<const real, false, Runtime::FixedRows<8>> cells, real cfl,
                           StateReductionOutput scratch, StateReductionOutput result);

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
    /// Same two-launch, preallocated scratch contract as StateReductionValue.
    /// Measures the first flow-five components of combined AoS 5/6/7 means.
    /// Auxiliary RANS components do not participate in the analytic flow oracle.
    /// No mesh access, quadrature construction, MPI or host wait in this wrapper.
    void MeasureFlowError(Runtime::ExecutionContext, FlowProfileQuadrature, StateInput means,
                          FlowProfileParameters, FlowState primitive, PerfectGas, RigidFrameParameters,
                          real time, FlowErrorOutput scratch, FlowErrorOutput result);
}
