#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Discretization/FiniteVolumeViews.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Physics/Positivity.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
namespace DNDS::Cauchy::CPU
{
    /// One SSPRK block update. Only RANS may use scalar exponential repair;
    /// flow and species preserve the raw conservative stage increment.
    /// state is explicitly in-out; flow is read-only and used only for RANS.
    /// The controller updates flow first, then passes that updated stage
    /// density to the RANS block under the same enclosing write reservation.
    NumericalStatus UpdateExplicitBlock(EquationBlockArrayView<const real> base, EquationBlockArrayView<const real> rhs,
                                        EquationBlockArrayView<real> state, EquationBlockArrayView<const real> flow,
                                        CFV::FiniteVolumeHostView metric, RANSStateView ransState, bool rans,
                                        real dt, real historyWeight, real stageWeight) noexcept;
    struct ExplicitBlendStatistics
    {
        real alpha = 1;
        index limited = 0;
        std::array<int, 2> safe{1, 1};
    };
    NumericalResult<ExplicitBlendStatistics> MeasureExplicitBlend(RHSScalarView initial, RHSScalarView state, RHSScalarView high,
                                                                  RHSScalarView low, CFV::FiniteVolumeHostView metrics, ThermalBounds bounds,
                                                                  real dt, real historyWeight, real stageWeight) noexcept;
    void UpdateExplicitFlow(RHSScalarView initial, RHSScalarView rhs, RHSScalarView low, RHSOutputView stage,
                            CFV::FiniteVolumeHostView metrics, real dt, real historyWeight, real stageWeight, real alpha);
}
