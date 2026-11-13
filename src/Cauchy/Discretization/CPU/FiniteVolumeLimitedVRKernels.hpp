#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Discretization/LimitedVRFace.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "CFV/Plans/LimitedVRPlan.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus LimitLVRTarget(RHSScalarView means, RHSScalarView points, RHSOutputView coefficients,
                                   Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets,
                                   const ThermalBounds &bounds, real betaPower, real betaCutoff) noexcept;
    void FillLVRScalars(RHSScalarView input, RHSOutputView output, const PerfectGas &gas);
    /// Independent face sensors: [shock, contact, mean-density support].
    /// Samples 2*f and 2*f+1 are in their parent-cell frames. The caller
    /// supplies a face-count by 3 output Array and checks status before gather.
    NumericalStatus BuildLVRFaceSensors(RHSScalarView samples, RHSFieldView means,
                                        Runtime::ArrayView<const LimitedVRFace, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces,
                                        const PerfectGas &, const ThermalBounds &, const CFV::LimitedVRSettings &,
                                        RHSOutputView sensors) noexcept;
    /// Owned-cell maximum using the same setup incidence CSR as CUDA.
    /// Independent rows replace the serial face scatter; duplicate and
    /// self-periodic incidences preserve the maximum exactly.
    void GatherLVRBranches(Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> offsets,
                           Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faceIndices,
                           RHSScalarView sensors, RHSOutputView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base);
    NumericalStatus BuildLVRBranches(RHSScalarView samples, RHSFieldView means,
                                     Runtime::ArrayView<const LimitedVRFace, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> faces,
                                     const PerfectGas &gas, const ThermalBounds &bounds,
                                     const CFV::LimitedVRSettings &settings, RHSOutputView branches,
                                     Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base) noexcept;
    void MergeLVRBase(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base,
                      RHSOutputView seeds, int column);
    void PublishLVRPenalty(RHSScalarView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> history,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> penalty);
}
