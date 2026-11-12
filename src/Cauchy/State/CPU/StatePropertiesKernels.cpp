#include "Cauchy/State/CPU/StatePropertiesKernels.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus RefreshDerivedRows(EquationBlockArrayView<const real> rows, EquationBlockArrayView<const real> species,
                                       Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> values,
                                       std::array<int, DerivedQuantityNumber> columns, bool useTemperatureGuess, PointEvaluator &evaluator) noexcept
    {
        auto rhoY = evaluator.SpeciesInput();
        for (index iRow = 0; iRow < values.RowCount(); ++iRow)
        {
            FlowState point;
            for (int component = 0; component < 5; ++component)
                point[component] = rows(iRow, component);
            for (int k = 0; k < rhoY.nSpecies; ++k)
                rhoY[k] = species(iRow, k);
            const real temperatureGuess = useTemperatureGuess ? values(iRow, columns[int(DerivedQuantity::Temperature)]) : 0;
            const auto properties = TryEvaluateDerivedPoint(evaluator, point, rhoY, temperatureGuess);
            if (!properties)
                return properties.status.At(iRow);
            for (int quantity = 0; quantity < DerivedQuantityNumber; ++quantity)
                if (columns[quantity] >= 0)
                    values(iRow, columns[quantity]) = properties.value[quantity];
        }
        return {};
    }
    NumericalStatus AccumulateStateDiagnostics(EquationBlockReadView flow, std::optional<EquationBlockReadView> species,
                                               Algebra::DofComponentView<const real> rhs, CFV::FiniteVolumeHostView metric,
                                               PointEvaluator &evaluator, SpanView<real> result) noexcept
    {
        const index width = rhs.ComponentNumber();
        const int independent = species ? int(species->ComponentNumber()) : 0;
        auto rhoY = evaluator.SpeciesInput();
        for (index iCell = 0; iCell < rhs.RowCount(); ++iCell)
        {
            FlowState value;
            for (int j = 0; j < 5; ++j)
                value[j] = flow(iCell, j);
            if (independent)
            {
                for (int k = 0; k < independent; ++k)
                    rhoY[k] = (*species)(iCell, k);
            }
            PointProperties point;
            if (const auto status = evaluator.TryEvaluate(value, rhoY, 0, point); !status)
                return status.At(iCell);
            const real volume = metric.GetCellVol(iCell);
            result[0] += volume * value[0];
            result[1] += volume * value[4];
            result[2] = std::min(result[2], value[0]);
            result[3] = std::min(result[3], point.thermo.pressure);
            result[13] = std::min(result[13], point.thermo.temperature);
            for (index j = 0; j < width; ++j)
                result[4] += rhs(iCell, j) * rhs(iCell, j) / volume;
        }
        return {};
    }
}
