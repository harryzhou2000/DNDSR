#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
namespace DNDS::Cauchy::CPU
{
    namespace
    {
        NumericalResult<FlowState> CellAverage(const Geom::UnstructuredMeshView &mesh, index iCell, const FlowProfileSettings &profile,
                                               const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame) noexcept
        {
            if (profile.kind == "constant")
                return frame.LocalState(gas.Conservative(primitive));
            Geom::tSmallCoords coords;
            mesh.GetCoordsOnCell(iCell, coords);
            const auto element = mesh.GetCellElement(iCell);
            constexpr int order = Geom::Elem::INT_ORDER_MAX;
            Geom::Elem::Quadrature quadrature(element, order);
            FlowState average{};
            real volume = 0;
            for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
            {
                const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                const Geom::tPoint point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                const real weight = Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape) * quadrature.GetWeight(iQ);
                const auto valueResult = TryProfileValue(profile, primitive, gas, frame.WorldPoint({point[0], point[1], point[2]}), time);
                if (!valueResult)
                    return valueResult.status.At(iCell, iQ);
                const auto value = frame.LocalState(valueResult.value);
                for (int j = 0; j < 5; ++j)
                    average[j] += weight * value[j];
                volume += weight;
            }
            for (auto &value : average)
                value /= volume;
            return average;
        }
    }

    NumericalStatus InitializeFlowProfile(Geom::UnstructuredMeshView mesh, EquationBlockArrayView<real> flow,
                                          const FlowProfileSettings &profile, const FlowState &primitive, const PerfectGas &gas, const RigidFrame &frame) noexcept
    {
        for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            const auto average = CellAverage(mesh, iCell, profile, primitive, gas, 0, frame);
            if (!average)
                return average.status;
            for (int j = 0; j < 5; ++j)
                flow(iCell, j) = average.value[j];
        }
        return {};
    }
    NumericalResult<FlowErrorStatistics> MeasureFlowError(Geom::UnstructuredMeshView mesh, CFV::FiniteVolumeHostView metric,
                                                          EquationBlockArrayView<const real> flow, const FlowProfileSettings &profile,
                                                          const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame) noexcept
    {
        FlowErrorStatistics result;
        auto &errors = result.errors;
        auto &volume = result.volume;
        for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            const auto exact = CellAverage(mesh, iCell, profile, primitive, gas, time, frame);
            if (!exact)
                return exact.status;
            const real weight = metric.GetCellVol(iCell);
            volume += weight;
            for (int j = 0; j < 5; ++j)
            {
                const real error = std::abs(flow(iCell, j) - exact.value[j]);
                errors[0][j] += weight * error;
                errors[1][j] += weight * error * error;
                errors[2][j] = std::max(errors[2][j], error);
            }
        }
        return result;
    }
    bool AdmissibleMeans(Algebra::DofComponentView<const real> values, AdmissibilityView bounds)
    {
        bool valid = true;
        for (index iCell = 0; iCell < values.RowCount(); ++iCell)
            valid &= bounds.Admissible(ThermalAdmissibilityDetail::PointRow<decltype(values)>{values, iCell});
        return valid;
    }
    NumericalResult<real> MinimumStableStep(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> sums,
                                            CFV::FiniteVolumeHostView metric, real cfl) noexcept
    {
        real dt = std::numeric_limits<real>::max();
        for (index iCell = 0; iCell < sums.RowCount(); ++iCell)
        {
            const real radius = sums(iCell, 0);
            if (!(std::isfinite(radius) && radius > 0))
                return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            dt = std::min(dt, cfl * metric.GetCellVol(iCell) / radius);
        }
        return dt;
    }

}
