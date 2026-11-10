#pragma once

#include "DNDS/Core/Errors.hpp"

#include <cmath>

namespace DNDS::Cauchy::Detail
{
    /// Scale an already evaluated cell rate by cell volume. The caller owns
    /// every lease and validates the row/component shapes before entering.
    template <class Input, class Output, class Metric>
    void ScaleRateByVolume(Input input, Output output, const Metric &metric, index components)
    {
        for (index iCell = 0; iCell < input.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            for (index j = 0; j < components; ++j)
                output(iCell, j) = volume * input(iCell, j);
        }
    }

    /// Divide a cell-integrated residual by cell volume. This is the CPU leaf
    /// used by rate-only evaluation; it performs no ownership or MPI work.
    template <class Input, class Output, class Metric>
    void NormalizeRateByVolume(Input input, Output output, const Metric &metric, index components)
    {
        for (index iCell = 0; iCell < input.RowCount(); ++iCell)
        {
            const real inverseVolume = 1 / metric.GetCellVol(iCell);
            for (index j = 0; j < components; ++j)
                output(iCell, j) = inverseVolume * input(iCell, j);
        }
    }

    /// Assemble the implicit stage defect and, when requested, its local norm
    /// contribution. Reduction and stopping decisions remain with the caller.
    template <class Mean, class Residual, class Constant, class Output, class Metric, class Active, class Norm>
    void AssembleDefectRows(Mean mean, Residual residual, Constant constant, Output output,
                            const Metric &metric, index components, real mass, real rhsScale, const Active &active,
                            int normOrder, bool measureNorm, Norm &norm, const char *invalidMessage)
    {
        for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            if (measureNorm)
                norm[components] += volume;
            for (index j = 0; j < components; ++j)
            {
                const real value = rhsScale * residual(iCell, j) + volume * (constant(iCell, j) - mass * mean(iCell, j));
                DNDS_check_throw_info(std::isfinite(value), invalidMessage);
                output(iCell, j) = active(j) ? value : 0;
                if (measureNorm)
                    norm[j] += normOrder == 1 ? std::abs(value) : value * value / volume;
            }
        }
    }

    /// Assemble mass*u-rhsScale*R(u)/volume and optionally publish the spatial
    /// rate. The two entry points avoid a nullable output inside the hot loop.
    template <class Mean, class Residual, class Output, class Metric>
    void AssembleEquationRows(Mean mean, Residual residual, Output output, const Metric &metric,
                              index components, real mass, real rhsScale)
    {
        for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            for (index j = 0; j < components; ++j)
                output(iCell, j) = mass * mean(iCell, j) - rhsScale * residual(iCell, j) / volume;
        }
    }

    template <class Mean, class Residual, class Output, class Rate, class Metric>
    void AssembleEquationAndRateRows(Mean mean, Residual residual, Output output, Rate rate,
                                     const Metric &metric, index components, real mass, real rhsScale)
    {
        for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
        {
            const real volume = metric.GetCellVol(iCell);
            for (index j = 0; j < components; ++j)
            {
                const real rateValue = residual(iCell, j) / volume;
                output(iCell, j) = mass * mean(iCell, j) - rhsScale * residual(iCell, j) / volume;
                rate(iCell, j) = rateValue;
            }
        }
    }
}
