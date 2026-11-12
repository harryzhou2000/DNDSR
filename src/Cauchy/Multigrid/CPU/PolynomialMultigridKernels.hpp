#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"

namespace DNDS::Cauchy::CPU
{
    NumericalStatus CheckFrozenComponents(Algebra::DofComponentView<const real> values,
                                          Algebra::DofComponentView<const real> saved, SpanView<const uint8_t> active) noexcept;
    void ProjectSelectedCorrection(Algebra::DofComponentView<real> values, SpanView<const uint8_t> active);
    void AddSelectedCorrection(Algebra::DofComponentView<real> values,
                               Algebra::DofComponentView<const real> delta, SpanView<const index> components);
    void CopyPolynomialLevel(Algebra::DofComponentView<const real> source, Algebra::DofComponentView<real> destination);
    void AccumulateFASNorm(Algebra::DofComponentView<const real> source, Algebra::DofComponentView<const real> value,
                           CFV::FiniteVolumeHostView metric, int normOrder, SpanView<real> sum);
}
