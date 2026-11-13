#include "Cauchy/Core/CPU/TimeStageKernels.hpp"
#include "DNDS/Core/OMP.hpp"
namespace DNDS::Cauchy::CPU
{
    NumericalStatus UpdateExplicitBlock(EquationBlockArrayView<const real> base, EquationBlockArrayView<const real> rhs,
                                        EquationBlockArrayView<real> state, EquationBlockArrayView<const real> flow,
                                        CFV::FiniteVolumeHostView metric, RANSStateView ransState, bool rans,
                                        real dt, real historyWeight, real stageWeight) noexcept
    {
        return OMP::ForRowsStatus(0, state.RowCount(), [&](index iCell) -> NumericalStatus
                                  {
            if (rans)
            {
                // PP POLICY -- explicit extended equations retain
                // raw flow/species stage increments and reject the
                // whole step when their accepted mean is invalid.
                // Euler's scalar-only RANS repair is permitted here.
                // It must never rescale the flow5/species update.
                std::array<real, 2> old{}, change{}, next{};
                for (int j = 0; j < ransState.components; ++j)
                {
                    old[j] = state(iCell, j);
                    change[j] = historyWeight * base(iCell, j) + stageWeight * (old[j] + dt * rhs(iCell, j) / metric.GetCellVol(iCell)) - old[j];
                    if (!(std::isfinite(change[j])))
                        return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                }
                ransState.RepairIncrement(old, change);
                for (int j = 0; j < ransState.components; ++j)
                    next[j] = old[j] + change[j];
                ransState.RepairMean(flow(iCell, 0), next);
                for (int j = 0; j < ransState.components; ++j)
                    state(iCell, j) = next[j];
                return {};
            }
            // PP POLICY -- no cellwise thermal increment compression:
            // flow5
            // and species retain their conservative residuals.
            // Do not clip/normalize species means here either;
            // the full raw candidate must pass Admissible below.
            for (index j = 0; j < state.ComponentNumber(); ++j)
                state(iCell, j) = historyWeight * base(iCell, j) + stageWeight * (state(iCell, j) + dt * rhs(iCell, j) / metric.GetCellVol(iCell));
            return {}; });
    }
    NumericalResult<ExplicitBlendStatistics> MeasureExplicitBlend(RHSScalarView initial, RHSScalarView state, RHSScalarView high,
                                                                  RHSScalarView low, CFV::FiniteVolumeHostView metrics, ThermalBounds bounds,
                                                                  real dt, real historyWeight, real stageWeight) noexcept
    {
        real alpha = 1;
        index limited = 0, firstInvalid = state.RowCount();
        int highSafe = 1, lowSafeAll = 1;
        const int threads = OMP::DistributedThreads();
        // Only minima, exact counts and flags: no order-dependent FP sum.
#if defined(DNDS_USE_OMP) && defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1) reduction(min : alpha, firstInvalid) reduction(+ : limited) reduction(& : highSafe, lowSafeAll)
#endif
        for (index iCell = 0; iCell < state.RowCount(); ++iCell)
        {
            FlowState base, target;
            bool finite = true;
            for (int j = 0; j < 5; ++j)
            {
                const real constant = historyWeight * initial(iCell, j) + stageWeight * state(iCell, j);
                base[j] = constant + stageWeight * dt * low(iCell, j) / metrics.GetCellVol(iCell);
                target[j] = constant + stageWeight * dt * high(iCell, j) / metrics.GetCellVol(iCell);
                finite &= std::isfinite(base[j]) && std::isfinite(target[j]);
            }
            if (!finite)
            {
                firstInvalid = std::min(firstInvalid, iCell);
                continue;
            }
            highSafe &= ThermallyAdmissible(target, bounds);
            const bool lowSafe = ThermallyAdmissible(base, bounds);
            lowSafeAll &= lowSafe;
            const auto fraction = lowSafe ? AdmissibleFraction(base, target, bounds) : real(1);
            if (fraction < 1)
            {
                ++limited;
                alpha = std::min(alpha, fraction);
            }
        }
        if (firstInvalid != state.RowCount())
            return NumericalStatus{NumericalError::InvalidState, 1, firstInvalid};
        return ExplicitBlendStatistics{alpha, limited, {highSafe, lowSafeAll}};
    }
    void UpdateExplicitFlow(RHSScalarView initial, RHSScalarView rhs, RHSScalarView low, RHSOutputView stage,
                            CFV::FiniteVolumeHostView metrics, real dt, real historyWeight, real stageWeight, real alpha)
    {
        OMP::ForRows(0, stage.RowCount(), [&](index iCell)
                     {
            for (int j = 0; j < 5; ++j)
            {
                const real residual = alpha == 1 ? rhs(iCell, j) : (1 - alpha) * low(iCell, j) + alpha * rhs(iCell, j);
                stage(iCell, j) = historyWeight * initial(iCell, j) + stageWeight * (stage(iCell, j) + dt * residual / metrics.GetCellVol(iCell));
            } });
    }
}
