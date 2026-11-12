#include "Cauchy/Multigrid/CPU/PolynomialMultigridKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include <cstring>
namespace DNDS::Cauchy::CPU
{
    NumericalStatus CheckFrozenComponents(Algebra::DofComponentView<const real> values,
                                          Algebra::DofComponentView<const real> saved, SpanView<const uint8_t> active) noexcept
    {
        for (index iRow = 0; iRow < values.RowCount(); ++iRow)
            for (index j = 0; j < values.ComponentNumber(); ++j)
                if (!active[j])
                    if (!(std::memcmp(&values(iRow, j), &saved(iRow, j), sizeof(real)) == 0))
                        return NumericalStatus{NumericalError::InvalidState, 1, iRow};
        return {};
    }
    void ProjectSelectedCorrection(Algebra::DofComponentView<real> values, SpanView<const uint8_t> active)
    {
        OMP::ForRows(0, values.RowCount(), [&](index iRow)
                     {
            for (index j = 0; j < values.ComponentNumber(); ++j)
                if (!active[j])
                    values(iRow, j) = 0; });
    }
    void AddSelectedCorrection(Algebra::DofComponentView<real> values,
                               Algebra::DofComponentView<const real> delta, SpanView<const index> components)
    {
        OMP::ForRows(0, values.RowCount(), [&](index iRow)
                     {
            for (const auto j : components)
                values(iRow, j) += delta(iRow, j); });
    }
    void CopyPolynomialLevel(Algebra::DofComponentView<const real> source, Algebra::DofComponentView<real> destination)
    {
        OMP::ForRows(0, source.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < source.ComponentNumber(); ++j)
                destination(iCell, j) = source(iCell, j); });
    }
    void AccumulateFASNorm(Algebra::DofComponentView<const real> source, Algebra::DofComponentView<const real> value,
                           CFV::FiniteVolumeHostView metric, int normOrder, SpanView<real> sum)
    {
        const index components = value.ComponentNumber();
        for (index iCell = 0; iCell < value.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            sum[components] += volume;
            for (index j = 0; j < components; ++j)
            {
                const real difference = source(iCell, j) - value(iCell, j);
                sum[j] += normOrder == 1 ? volume * std::abs(difference) : volume * difference * difference;
            }
        }
    }
}
