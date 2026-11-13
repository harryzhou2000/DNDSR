#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/State/DerivedFields.hpp"
#include "Cauchy/State/EquationViews.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
namespace DNDS::Cauchy::CPU
{
    /// Same complete-equation mean predicate for implicit and explicit acceptance.
    /// Empty activeRows checks every owned row; otherwise zero entries skip
    /// unused rows (e.g. physical boundaries in an interface-only packet).
    NumericalStatus ValidateMeanState(EquationLayoutView<const real>, AdmissibilityView,
                                      Runtime::ArrayView<const uint8_t, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> activeRows = {}) noexcept;
    NumericalStatus RefreshDerivedRows(EquationBlockArrayView<const real> flow, EquationBlockArrayView<const real> species,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> output,
                                       std::array<int, DerivedQuantityNumber> columns, bool useTemperatureGuess, PointEvaluator &) noexcept;
    NumericalStatus AccumulateStateDiagnostics(EquationBlockReadView flow, std::optional<EquationBlockReadView> species,
                                               Algebra::DofComponentView<const real> rhs, CFV::FiniteVolumeHostView metric,
                                               PointEvaluator &, SpanView<real> result) noexcept;
}
