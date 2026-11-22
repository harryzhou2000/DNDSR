#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Discretization/LimitedVREvaluation.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/Physics/ReconstructionPositivity.hpp"
#include "CFV/Plans/LimitedVRPlan.hpp"
namespace DNDS::Cauchy::CPU
{
    void PrepareLVRScalarsAndTarget(CFV::Detail::GradientCellGeometry, RHSScalarView means,
                                    RHSScalarView gradients, RHSOutputView scalars, RHSOutputView target,
                                    const PerfectGas &, int dimension, index modes, index flowFirst);
    NumericalStatus BuildLVRFaceSensors(FiniteVolumeStencilView stencil, RHSFieldView means,
                                        RHSFieldView gradients, RHSFieldView raw, int dimension, index flowFirst,
                                        const PerfectGas &, const ThermalBounds &, const CFV::LimitedVRKernelParameters &,
                                        RHSOutputView sensors, Runtime::RowSelection) noexcept;
    /// Deterministic cell gather through canonical cell-face incidences.
    /// Repeated self-periodic incidences retain their semantic multiplicity.
    void GatherLVRBranches(FiniteVolumeStencilView stencil, RHSScalarView sensors, RHSOutputView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base);
    void MergeLVRBase(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> base,
                      RHSOutputView seeds, int column);
    void PublishLVRPenalty(RHSScalarView branches,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> history,
                           Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> penalty, CFV::LimitedVRTemporalUpdate update = {});
}
