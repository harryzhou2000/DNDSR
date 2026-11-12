#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "Cauchy/Physics/EquationSource.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "DNDS/Core/Span.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus AdvanceChemicalRows(SpanView<const EquationBlockArrayView<const real>> from,
                                        SpanView<const EquationBlockArrayView<real>> to, EquationBlockArrayView<const real> flow,
                                        EquationBlockArrayView<const real> species, EquationBlockArrayView<real> target,
                                        Physics::Thermodynamics::IdealGasMixture::PointEvaluator &evaluator, int independent,
                                        real dt, const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale) noexcept;
    NumericalStatus EvaluateChemicalRows(EquationBlockArrayView<const real> flow, EquationBlockArrayView<const real> rhoY,
                                         EquationBlockArrayView<real> sourceSpecies, SpanView<const EquationBlockArrayView<real>> nonchemical,
                                         SpanView<const index> columns, index components, index speciesFirst, int species, real sourceScale,
                                         std::optional<Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>> matrix,
                                         std::optional<EquationBlockArrayView<const real>> oldT, EquationPointEvaluator &evaluator) noexcept;
}
