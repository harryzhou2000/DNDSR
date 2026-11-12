#pragma once
#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/Span.hpp"
#include "DNDS/Core/OMP.hpp"
#include <cmath>
#include <array>
namespace DNDS::Cauchy::CPU
{
    /// Simple owned-row kernels. The operator reserves access and passes the
    /// logical width explicitly; physical pitch remains part of each view.
    template <class Input, class Output>
    void CopyRows(Input input, Output output, index components)
    {
        OMP::ForRows(0, output.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < components; ++j)
                output(iCell, j) = input(iCell, j); });
    }
    template <class Output>
    void FillRows(Output output, index components, real value)
    {
        OMP::ForRows(0, output.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < components; ++j)
                output(iCell, j) = value; });
    }
    /// Return the first invalid row; error emission belongs to the caller.
    template <class Input>
    NumericalStatus RequireFiniteRows(Input input, index components) noexcept
    {
        for (index iCell = 0; iCell < input.RowCount(); ++iCell)
            for (index j = 0; j < components; ++j)
                if (!(std::isfinite(input(iCell, j))))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
        return {};
    }
    template <class Input>
    NumericalStatus RequireCappedRows(Input input, SpanView<const real> caps) noexcept
    {
        for (index iCell = 0; iCell < input.RowCount(); ++iCell)
            for (index j = 0; j < caps.size(); ++j)
                if (!(std::isfinite(input(iCell, j)) && input(iCell, j) >= 0 && input(iCell, j) <= caps[j]))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
        return {};
    }
    template <class Values, class Metric>
    void AccumulateMeanScale(Values values, const Metric &metric, index components, SpanView<real> sum)
    {
        for (index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            sum[components] += volume;
            for (index j = 0; j < components; ++j)
                sum[j] += volume * std::abs(values(iCell, j));
        }
    }

    template <class Rows>
    bool FiniteRows(Rows values, index components)
    {
        bool valid = true;
        for (index iRow = 0; iRow < values.RowCount(); ++iRow)
            for (index j = 0; j < components; ++j)
                valid &= std::isfinite(values(iRow, j));
        return valid;
    }
    template <class Input, class Output, class Metric>
    void DivideRowsByVolume(Input input, Output output, const Metric &metric, index components)
    {
        OMP::ForRows(0, input.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < components; ++j)
                output(iCell, j) = input(iCell, j) / metric.GetCellVol(iCell); });
    }
    template <class Input, class Metric>
    NumericalStatus AccumulateVolumeNorm(Input values, const Metric &metric, index components, int normOrder, SpanView<real> sum) noexcept
    {
        for (index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            sum[components] += volume;
            for (index j = 0; j < components; ++j)
            {
                const real value = values(iCell, j);
                if (!(std::isfinite(value)))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                sum[j] += volume * (normOrder == 1 ? std::abs(value) : value * value);
            }
        }
        return {};
    }
    template <class Input>
    void AccumulateReconstructionUpdateNorm(Input a, Input b, SpanView<real> norms)
    {
        for (index iCell = 0; iCell < a.RowCount(); ++iCell)
            for (index j = 0; j < a.Width(); ++j)
            {
                const real delta = b(iCell, j) - a(iCell, j);
                norms[0] += delta * delta;
                norms[1] += b(iCell, j) * b(iCell, j);
            }
    }

    template <class Output>
    void FillConstantRows(Output rows, SpanView<const real> values)
    {
        OMP::ForRows(0, rows.RowCount(), [&](index iCell)
                     {
            for (index j = 0; j < values.size(); ++j)
                rows(iCell, j) = values[j]; });
    }
}
