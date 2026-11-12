#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Input/FlowProfile.hpp"
#include "Cauchy/Motion/RigidMotion.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Geom/Access/MeshView.hpp"
namespace DNDS::Cauchy::CPU
{
    struct FlowErrorStatistics
    {
        std::array<FlowState, 3> errors{};
        real volume = 0;
    };
    NumericalStatus InitializeFlowProfile(Geom::UnstructuredMeshView mesh, EquationBlockArrayView<real> flow,
                                          const FlowProfileSettings &profile, const FlowState &primitive, const PerfectGas &gas, const RigidFrame &frame) noexcept;
    NumericalResult<FlowErrorStatistics> MeasureFlowError(Geom::UnstructuredMeshView mesh, CFV::FiniteVolumeHostView metric,
                                                          EquationBlockArrayView<const real> flow, const FlowProfileSettings &profile,
                                                          const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame) noexcept;
    bool AdmissibleMeans(Algebra::DofComponentView<const real> values, AdmissibilityView bounds);
    NumericalResult<real> MinimumStableStep(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> sums,
                                            CFV::FiniteVolumeHostView metric, real cfl) noexcept;
}
