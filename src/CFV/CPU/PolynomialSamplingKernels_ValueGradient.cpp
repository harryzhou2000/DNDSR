#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include <type_traits>

namespace DNDS::CFV::Detail
{
    void EvaluateCachedOwnedValueGradient(SamplingIndices cells, SamplingFixedInput<6> means,
                                          SamplingFixedInput<54> coefficientsView, SamplingFixedInput<27> basis,
                                          SamplingFixedOutput<6> values, SamplingFixedOutput<6> dx, SamplingFixedOutput<6> dy, bool includeValues)
    {
        Runtime::ValidateViewDevices(Runtime::Device::Host(), cells, means, coefficientsView, basis, values, dx, dy);
        // Source-center Jacobians request gradients only. Compile the
        // value arithmetic/store out, without inventing a temporary value
        // output or changing either derivative's accumulation order.
        const auto sample = [&](auto includeValues)
        {
            constexpr bool valueRequested = decltype(includeValues)::value;
            OMP::ForRows(0, cells.RowCount(), [&](index iPoint)
                         {
                const index iCell = cells(iPoint, 0);
                const auto coefficientRow = coefficientsView[iCell];
                const auto basisRow = basis[iPoint];
                const auto dxRow = dx[iPoint], dyRow = dy[iPoint];
                for (index component = 0; component < 6; ++component)
                {
                real value = 0, gradientX = 0, gradientY = 0;
                if constexpr (valueRequested)
                    value = means(iCell, component);
                for (index mode = 0; mode < 9; ++mode)
                {
                    const real coefficient = coefficientRow[component * 9 + mode];
                    // Cached basis is derivative-major within each mode.
                    // Each sum retains the reference evaluator's order.
                    if constexpr (valueRequested)
                        value += basisRow[mode * 3] * coefficient;
                    gradientX += basisRow[mode * 3 + 1] * coefficient;
                    gradientY += basisRow[mode * 3 + 2] * coefficient;
                }
                if constexpr (valueRequested)
                    values(iPoint, component) = value;
                dxRow[component] = gradientX;
                dyRow[component] = gradientY;
                } });
        };
        if (includeValues)
            sample(std::true_type{});
        else
            sample(std::false_type{});
    }
}
