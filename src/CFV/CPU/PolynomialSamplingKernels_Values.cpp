#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include <type_traits>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;

    bool TryEvaluateCachedOwnedTable(
        R::ArrayView<const index, R::AccessBoundsChecks, R::FixedRows<1>> cells,
        R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> mean,
        R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> coefficients,
        R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> basis,
        R::ArrayView<real, R::AccessBoundsChecks, R::TableRows> output)
    {
        R::ValidateViewDevices(R::Device::Host(), cells, mean, coefficients, basis, output);
        const auto evaluate = [&](auto componentCount, auto modeCount)
        {
            constexpr int width = decltype(componentCount)::value, modes = decltype(modeCount)::value;
            DNDS_check_throw_info(mean.Width() == width && coefficients.Width() == width * modes &&
                                      output.Width() == width && basis.Width() == modes &&
                                      mean.RowCount() == coefficients.RowCount() && cells.RowCount() == basis.RowCount() &&
                                      cells.RowCount() == output.RowCount(),
                                  "cached sample kernel shape mismatch");
            // Rebind access shape only after the prologue check; allocation,
            // physical pitch and access ownership remain with the caller.
            const R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<width>> means{
                mean.data, nullptr, mean.rows, width, mean.stride, nullptr, R::RowLayoutKind::Table, mean.device};
            const R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<width * modes>> rec{
                coefficients.data, nullptr, coefficients.rows, width * modes, coefficients.stride, nullptr, R::RowLayoutKind::Table, coefficients.device};
            const R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<modes>> bases{
                basis.data, nullptr, basis.rows, modes, basis.stride, nullptr, R::RowLayoutKind::Table, basis.device};
            const R::ArrayView<real, R::AccessBoundsChecks, R::FixedRows<width>> values{
                output.data, nullptr, output.rows, width, output.stride, nullptr, R::RowLayoutKind::Table, output.device};
            OMP::ForRows(0, cells.RowCount(), [&](index iPoint)
                         {
                const index iCell = cells(iPoint, 0);
                const auto meanRow = means[iCell];
                const auto coefficientRow = rec[iCell];
                const auto basisRow = bases[iPoint];
                const auto outputRow = values[iPoint];
                for (index component = 0; component < width; ++component)
                {
                    real value = meanRow[component];
                    for (index mode = 0; mode < modes; ++mode)
                        value += basisRow[mode] * coefficientRow[component * modes + mode];
                    outputRow[component] = value;
                } });
        };
        const auto dispatch = [&](auto modes)
        {
            switch (mean.Width())
            {
            case 5:
                evaluate(std::integral_constant<int, 5>{}, modes);
                return true;
            case 6:
                evaluate(std::integral_constant<int, 6>{}, modes);
                return true;
            case 7:
                evaluate(std::integral_constant<int, 7>{}, modes);
                return true;
            default:
                return false;
            }
        };
        return (basis.Width() == 2 && dispatch(std::integral_constant<int, 2>{})) ||
               (basis.Width() == 9 && dispatch(std::integral_constant<int, 9>{}));
    }
}
