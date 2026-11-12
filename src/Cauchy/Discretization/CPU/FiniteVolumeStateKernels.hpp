#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include "Cauchy/Physics/InitialField.hpp"
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
    void InitializeFlowProfile(Geom::UnstructuredMeshView mesh, EquationBlockArrayView<real> flow,
                               const FlowProfileSettings &profile, const FlowState &primitive, const PerfectGas &gas, const RigidFrame &frame);
    FlowErrorStatistics MeasureFlowError(Geom::UnstructuredMeshView mesh, CFV::FiniteVolumeHostView metric,
                                         EquationBlockArrayView<const real> flow, const FlowProfileSettings &profile,
                                         const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame);
    bool AdmissibleMeans(Algebra::DofComponentView<const real> values, EquationAdmissibilityView bounds);
    NumericalResult<real> MinimumStableStep(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> sums,
                                            CFV::FiniteVolumeHostView metric, real cfl) noexcept;
    NumericalStatus InitializeProgrammedField(Geom::UnstructuredMeshView mesh, Algebra::DofComponentView<real> rows,
                                              const RigidFrame &frame, InitialFieldEvaluator &evaluator) noexcept;
}
