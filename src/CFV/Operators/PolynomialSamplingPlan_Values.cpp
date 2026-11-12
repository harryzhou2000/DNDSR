#include "CFV/Detail/PolynomialSamplingKernels.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
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
            for (index iPoint = 0; iPoint < cells.RowCount(); ++iPoint)
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
                }
            }
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

namespace DNDS::CFV
{
    void PolynomialSamplingPlan::EvaluateValues(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &output) const
    {
        namespace R = Runtime;
        const index components = mean.Width(), modes = _reconstruction.NRecDOF();
        if (_needsGhost || _cachedDerivatives != 1 || (components != 5 && components != 6 && components != 7) ||
            (modes != 2 && modes != 9) || mean.Owned().Layout().Kind() != R::RowLayoutKind::Table ||
            coefficients.Owned().Layout().Kind() != R::RowLayoutKind::Table || output.Layout().Kind() != R::RowLayoutKind::Table)
        {
            Evaluate(mean, coefficients, output);
            return;
        }
        // Same entry checks and lifetime discipline as Evaluate; no ghost
        // lease or source/shape ownership is hidden in the numerical helper.
        auto geometry = _reconstruction.Metrics().AcquireRead();
        Validate();
        _reconstruction.Metrics().ValidateField(mean, components);
        _reconstruction.Metrics().ValidateField(coefficients, components * modes);
        DNDS_check_throw_info(output.GetDevice() == R::Device::Host() && output.RowCount() == Size() &&
                                  output.Layout().Width() == components,
                              "polynomial value sample output shape/placement mismatch");
        auto meanRead = mean.Owned().Read(), recRead = coefficients.Owned().Read();
        auto idsRead = _cells.Read();
        auto basisRead = _basis.Read();
        auto write = output.Write();
        const bool evaluated = Detail::TryEvaluateCachedOwnedTable(idsRead.View<R::FixedRows<1>>(),
                                                                   meanRead.View<R::TableRows>(), recRead.View<R::TableRows>(), basisRead.View<R::TableRows>(), write.View<R::TableRows>());
        DNDS_assert(evaluated);
    }
}
