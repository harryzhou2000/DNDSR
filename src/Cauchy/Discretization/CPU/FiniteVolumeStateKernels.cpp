#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
namespace DNDS::Cauchy::CPU
{
    namespace
    {
        FlowState CellAverage(const Geom::UnstructuredMeshView &mesh, index iCell, const FlowProfileSettings &profile,
                              const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame)
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
                const auto value = frame.LocalState(ProfileValue(profile, primitive, gas, frame.WorldPoint({point[0], point[1], point[2]}), time));
                for (int j = 0; j < 5; ++j)
                    average[j] += weight * value[j];
                volume += weight;
            }
            for (auto &value : average)
                value /= volume;
            return average;
        }
    }

    void InitializeFlowProfile(Geom::UnstructuredMeshView mesh, EquationBlockArrayView<real> flow,
                               const FlowProfileSettings &profile, const FlowState &primitive, const PerfectGas &gas, const RigidFrame &frame)
    {
        for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            const auto average = CellAverage(mesh, iCell, profile, primitive, gas, 0, frame);
            for (int j = 0; j < 5; ++j)
                flow(iCell, j) = average[j];
        }
    }
    FlowErrorStatistics MeasureFlowError(Geom::UnstructuredMeshView mesh, CFV::FiniteVolumeHostView metric,
                                         EquationBlockArrayView<const real> flow, const FlowProfileSettings &profile,
                                         const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame)
    {
        FlowErrorStatistics result;
        auto &errors = result.errors;
        auto &volume = result.volume;
        for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            const auto exact = CellAverage(mesh, iCell, profile, primitive, gas, time, frame);
            const real weight = metric.GetCellVol(iCell);
            volume += weight;
            for (int j = 0; j < 5; ++j)
            {
                const real error = std::abs(flow(iCell, j) - exact[j]);
                errors[0][j] += weight * error;
                errors[1][j] += weight * error * error;
                errors[2][j] = std::max(errors[2][j], error);
            }
        }
        return result;
    }
    bool AdmissibleMeans(Algebra::DofComponentView<const real> values, EquationAdmissibilityView bounds)
    {
        bool valid = true;
        for (index iCell = 0; iCell < values.RowCount(); ++iCell)
            valid &= bounds.Admissible(EquationPositivityDetail::PointRow<decltype(values)>{values, iCell});
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

    NumericalStatus InitializeProgrammedField(Geom::UnstructuredMeshView mesh, Algebra::DofComponentView<real> rows,
                                              const RigidFrame &frame, InitialFieldEvaluator &evaluator) noexcept
    {
        const index width = rows.ComponentNumber();
        std::vector<real> background(width), average(width);
        for (index iCell = 0; iCell < rows.RowCount(); ++iCell)
        {
            for (index j = 0; j < width; ++j)
                background[j] = rows(iCell, j);
            FlowState flow;
            std::copy_n(background.begin(), 5, flow.begin());
            flow = frame.WorldState(flow);
            std::copy(flow.begin(), flow.end(), background.begin());
            std::fill(average.begin(), average.end(), 0);
            Geom::tSmallCoords coords;
            mesh.GetCoordsOnCell(iCell, coords);
            const auto element = mesh.GetCellElement(iCell);
            constexpr int order = Geom::Elem::INT_ORDER_MAX;
            const Geom::Elem::Quadrature quadrature(element, order);
            real volume = 0;
            for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
            {
                const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                const Geom::tPoint point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                const real jacobian = Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape);
                if (!(std::isfinite(jacobian) && jacobian > 0))
                    return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                const real weight = jacobian * quadrature.GetWeight(iQ);
                const auto value = evaluator.Evaluate(frame.WorldPoint({point[0], point[1], point[2]}), 0, iCell, background);
                for (index j = 0; j < width; ++j)
                    average[j] += weight * value[j];
                volume += weight;
            }
            if (!(std::isfinite(volume) && volume > 0))
                return NumericalStatus{NumericalError::InvalidState, 2, iCell};
            for (auto &value : average)
                value /= volume;
            std::copy_n(average.begin(), 5, flow.begin());
            flow = frame.LocalState(flow);
            std::copy(flow.begin(), flow.end(), average.begin());
            for (index j = 0; j < width; ++j)
                rows(iCell, j) = average[j];
        }
        return {};
    }
}
