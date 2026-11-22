#pragma once
#include "Cauchy/Discretization/LimitedVREvaluation.hpp"
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
    using LVRStencil = Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using LVRStatus = Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    void PrepareLVRScalarsAndTarget(Runtime::ExecutionContext, CFV::Detail::GradientCellGeometry,
                                    LVRInput means, LVRInput gradients, LVROutput scalars, LVROutput target,
                                    PerfectGas, int dimension, index modes, index flowFirst, LVRStatus);
    void BuildLVRFaceSensors(Runtime::ExecutionContext, LVRStencil, LVRMeans means, LVRMeans gradients, LVRMeans raw,
                             int dimension, index flowFirst, PerfectGas, ThermalBounds,
                             CFV::LimitedVRKernelParameters, LVROutput sensors, LVRStatus,
                             Runtime::RowSelection);
    /// Deterministic cell gather using the shared face routing and incidences.
    void GatherLVRBranches(Runtime::ExecutionContext, LVRStencil, LVRInput sensors,
                           Runtime::ArrayView<const NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faceStatus,
                           LVROutput branches, Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base, LVRStatus);
    void MergeLVRBase(Runtime::ExecutionContext, Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base,
                      LVROutput seeds, int column);
    void PublishLVRPenalty(Runtime::ExecutionContext, LVRInput branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> history,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> penalty, CFV::LimitedVRTemporalUpdate update = {});
}
