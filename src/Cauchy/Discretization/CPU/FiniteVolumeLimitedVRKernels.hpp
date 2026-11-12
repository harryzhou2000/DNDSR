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
