#pragma once
#include "Cauchy/Discretization/LimitedVRFace.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "CFV/Math/LimitedVR.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy::CUDA
{
    using LVRInput = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using LVROutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using LVRMeans = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    using LVRIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using LVRStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    /// Pure-flow5 O2 target guard, matching the current CPU LVR contract.
    /// It is separate from final equation-aware reconstruction PP.
    void LimitLVRTarget(Runtime::ExecutionContext, LVRInput means, LVRInput points, LVROutput coefficients,
                        LVRIndices offsets, ThermalBounds bounds, int betaPower, real betaCutoff, LVRStatus);
    void FillLVRScalars(Runtime::ExecutionContext, LVRInput means, LVROutput scalars, PerfectGas, LVRStatus);
    /// Per-face finalized-target physical sensor. Sampling/transfer are explicit
    /// preceding operators; row 2*f/2*f+1 is in each parent cell's own frame.
    void BuildLVRFaceSensors(Runtime::ExecutionContext, LVRInput samples, LVRMeans means,
                             Runtime::ArrayView<const LimitedVRFace, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces,
                             PerfectGas, ThermalBounds, CFV::LimitedVRKernelParameters,
                             LVROutput sensors, LVRStatus);
    /// Deterministic owned-cell maximum over a setup-time face-incidence CSR.
    /// No face-scatter atomics. Duplicated/self-periodic incidences are safe.
    void GatherLVRBranches(Runtime::ExecutionContext, LVRIndices offsets, LVRIndices faceIndices, LVRInput sensors,
                           Runtime::ArrayView<const NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faceStatus,
                           LVROutput branches, Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base, LVRStatus);
    void MergeLVRBase(Runtime::ExecutionContext, Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base,
                      LVROutput seeds, int column);
    void PublishLVRPenalty(Runtime::ExecutionContext, LVRInput branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> history,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> penalty);
}
