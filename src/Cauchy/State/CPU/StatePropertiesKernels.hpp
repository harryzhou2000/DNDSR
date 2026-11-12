#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/State/DerivedFields.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus RefreshDerivedRows(EquationBlockArrayView<const real> flow, EquationBlockArrayView<const real> species,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> output,
                                       std::array<int, DerivedQuantityNumber> columns, bool useTemperatureGuess, PointEvaluator &) noexcept;
    NumericalStatus AccumulateStateDiagnostics(EquationBlockReadView flow, std::optional<EquationBlockReadView> species,
                                               Algebra::DofComponentView<const real> rhs, CFV::FiniteVolumeHostView metric,
                                               PointEvaluator &, SpanView<real> result) noexcept;
}
