#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "DNDS/Core/OMP.hpp"
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
        return OMP::ForRowsStatus(0, flow.RowCount(), [&](index iCell) -> NumericalStatus
                                  {
            const auto average = CellAverage(mesh, iCell, profile, primitive, gas, 0, frame);
            if (!average)
                return average.status;
            for (int j = 0; j < 5; ++j)
                flow(iCell, j) = average.value[j];
            return {}; });
    }
    NumericalResult<FlowErrorStatistics> MeasureFlowError(Geom::UnstructuredMeshView mesh, CFV::FiniteVolumeHostView metric,
                                                          EquationBlockArrayView<const real> flow, const FlowProfileSettings &profile,
                                                          const FlowState &primitive, const PerfectGas &gas, real time, const RigidFrame &frame,
                                                          SpanView<NumericalResult<FlowErrorStatistics>> workers) noexcept
    {
        const auto accumulate = [&](index first, index end) -> NumericalResult<FlowErrorStatistics>
        {
            FlowErrorStatistics result;
            auto &errors = result.errors;
            auto &volume = result.volume;
            for (index iCell = first; iCell < end; ++iCell)
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
        };
        if (workers.empty())
            return accumulate(0, flow.RowCount());
        const index quotient = flow.RowCount() / workers.size(), remainder = flow.RowCount() % workers.size();
        OMP::ForRows(0, workers.size(), [&](index iWorker)
                     {
            const index first = iWorker * quotient + std::min(iWorker, remainder);
            workers[iWorker] = accumulate(first, first + quotient + (iWorker < remainder)); });
        if (!workers[0])
            return workers[0].status;
        auto result = workers[0].value;
        for (index iWorker = 1; iWorker < workers.size(); ++iWorker)
        {
            if (!workers[iWorker])
                return workers[iWorker].status;
            const auto &part = workers[iWorker].value;
            result.volume += part.volume;
            for (int j = 0; j < 5; ++j)
            {
                result.errors[0][j] += part.errors[0][j];
                result.errors[1][j] += part.errors[1][j];
                result.errors[2][j] = std::max(result.errors[2][j], part.errors[2][j]);
            }
        }
        return result;
    }
    bool AdmissibleMeans(Algebra::DofComponentView<const real> values, AdmissibilityView bounds)
    {
        bool valid = true;
        const int threads = OMP::DistributedThreads();
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1) reduction(&& : valid)
#endif
        for (index iCell = 0; iCell < values.RowCount(); ++iCell)
            valid &= bounds.Admissible(ThermalAdmissibilityDetail::PointRow<decltype(values)>{values, iCell});
        return valid;
    }
    NumericalResult<real> MinimumStableStep(Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> sums,
                                            CFV::FiniteVolumeHostView metric, real cfl) noexcept
    {
        real dt = std::numeric_limits<real>::max();
        index firstInvalid = sums.RowCount();
        const int threads = OMP::DistributedThreads();
        // Minima and the first failing row are order-independent. Unlike an
        // FP sum this reduction introduces no scheduling-dependent rounding.
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1) reduction(min : dt, firstInvalid)
#endif
        for (index iCell = 0; iCell < sums.RowCount(); ++iCell)
        {
            const real radius = sums(iCell, 0);
            if (!(std::isfinite(radius) && radius > 0))
                firstInvalid = std::min(firstInvalid, iCell);
            else
                dt = std::min(dt, cfl * metric.GetCellVol(iCell) / radius);
        }
        if (firstInvalid != sums.RowCount())
            return NumericalStatus{NumericalError::InvalidState, 1, firstInvalid};
        return dt;
    }

}
