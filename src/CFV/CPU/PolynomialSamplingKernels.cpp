#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV::Detail
{
    void EvaluatePolynomialSamples(PolynomialSamplingView plan, SamplingField mean, SamplingField coefficients,
                                   SamplingOutput output, int derivative)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), plan, mean, coefficients, output);
        const int modes = plan.polynomial.size - 1;
        OMP::ForRows(0, plan.cells.RowCount(), [&](index iPoint)
                     {
            const index iCell = plan.cells(iPoint, 0);
            std::array<real, 19> values;
            if (derivative < plan.cachedDerivatives)
                for (int mode = 0; mode < modes; ++mode)
                    values[mode] = plan.basis(iPoint, mode * plan.cachedDerivatives + derivative);
            else
            {
                const Geom::tPoint point{plan.points(iPoint, 0), plan.points(iPoint, 1), plan.points(iPoint, 2)};
                const Geom::tPoint center{plan.geometry(iCell, 1), plan.geometry(iCell, 2), plan.geometry(iCell, 3)};
                Math::SmallMatrix scratch;
                Math::PolynomialDerivativesUnchecked(scratch, plan.polynomial.dimension, plan.polynomial.size, plan.polynomial.size,
                                                       point, center, plan.geometry(iCell, 7));
                for (int mode = 0; mode < modes; ++mode)
                {
                    values[mode] = scratch(derivative, mode + 1);
                    if (derivative == 0)
                        values[mode] -= plan.moments(iCell, mode + 1);
                }
            }
            const auto meanRow = mean[iCell], coefficientRow = coefficients[iCell];
            const auto result = output[iPoint];
            for (index component = 0; component < result.size(); ++component)
            {
                real value = derivative == 0 ? meanRow[component] : 0;
                for (int mode = 0; mode < modes; ++mode)
                    value += values[mode] * coefficientRow[component * modes + mode];
                result[component] = value;
            } });
    }
}
