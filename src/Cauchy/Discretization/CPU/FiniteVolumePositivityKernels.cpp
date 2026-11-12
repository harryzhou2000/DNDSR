#include "Cauchy/Discretization/CPU/FiniteVolumePositivityKernels.hpp"

namespace DNDS::Cauchy::CPU
{
    namespace
    {
        struct RepeatedMean
        {
            PPInput values;
            index cell;
            real operator()(index, index component) const { return values(cell, component); }
        };
        bool ReferenceGood(PPInput values, index q, const EquationThermalView &bounds)
        {
            const EquationPositivityDetail::PointRow<PPInput> point{values, q};
            // PP POLICY -- reference selection must support BOTH the raw-affine
            // beta calculation and the final clipped EOS/T acceptance predicate.
            // No independent simplex condition is imposed on raw polynomial rhoY.
            return bounds.RawThermallyAdmissible(point) && bounds.ClippedReferenceAdmissible(point);
        }
    }
    NumericalStatus ValidateReconstructionMeans(const PolynomialPositivityRequest &r) noexcept
    {
        for (index iCell = 0; iCell < r.means.ownedRows; ++iCell)
            if (!(r.bounds.Admissible(EquationPositivityDetail::PointRow<PPInput>{r.means, iCell})))
                return NumericalStatus{NumericalError::InvalidState, 1, iCell};
        return {};
    }
    NumericalResult<bool> MarkAdmissibleDegrees(const PolynomialPositivityRequest &r, PPInput samples, PPIndexOutput degrees) noexcept
    {
        for (index q = 0; q < samples.RowCount(); ++q)
            for (index j = 0; j < samples.ComponentNumber(); ++j)
                if (!(std::isfinite(samples(q, j))))
                    return NumericalStatus{NumericalError::InvalidState, 1, q};
        bool allGood = true;
        for (index iCell = 0; iCell < r.means.ownedRows; ++iCell)
        {
            bool good = true;
            for (index q = r.offsets(iCell, 0); q < r.offsets(iCell + 1, 0); ++q)
                good = good && ReferenceGood(samples, q, r.bounds);
            degrees(iCell, 0) = good ? r.maxOrder : -1;
            allGood = allGood && good;
            // Flow acceptance excludes RANS; the fast exit must prove its
            // independent scalar repair unnecessary as well.
            if (allGood && r.rans.components)
                allGood = r.rans.Compute(RepeatedMean{r.means, iCell}, samples, r.ransFirst,
                                         r.offsets(iCell, 0), r.offsets(iCell + 1, 0)) == 1;
        }
        return allGood;
    }
    void TruncateReferenceModes(const PolynomialPositivityRequest &r, PPIndexInput degrees, PPOutput coefficients, int keep)
    {
        for (index iCell = 0; iCell < r.means.ownedRows; ++iCell)
            if (degrees(iCell, 0) < 0)
                for (index j = 0; j < r.means.ComponentNumber(); ++j)
                    if (r.bounds.CompressesReconstructionComponent(j))
                        for (int mode = keep; mode < r.modes; ++mode)
                            coefficients(iCell, j * r.modes + mode) = 0;
    }
    void SelectReferenceDegree(const PolynomialPositivityRequest &r, PPInput samples, PPInput reference,
                               PPIndexOutput degrees, PPOutput coefficients, int order)
    {
        for (index iCell = 0; iCell < r.means.ownedRows; ++iCell)
            if (degrees(iCell, 0) < 0)
            {
                bool good = true;
                for (index q = r.offsets(iCell, 0); q < r.offsets(iCell + 1, 0); ++q)
                    good = good && ReferenceGood(samples, q, r.bounds);
                if (good)
                    degrees(iCell, 0) = order;
                else
                    for (index component = 0; component < r.means.ComponentNumber(); ++component)
                        if (r.bounds.CompressesReconstructionComponent(component))
                            for (int mode = 0; mode < r.modes; ++mode)
                                coefficients(iCell, component * r.modes + mode) = reference(iCell, component * r.modes + mode);
            }
    }
    NumericalStatus CompressPolynomial(const PolynomialPositivityRequest &r, PPInput higher, std::optional<PPInput> base,
                                       std::optional<PPInput> reference, PPIndexInput degrees, PPOutput coefficients, PPScalarOutput cellBeta) noexcept
    {
        for (index iCell = 0; iCell < r.means.ownedRows; ++iCell)
        {
            cellBeta(iCell, 0) = 1;
            const EquationPositivityDetail::PointRow<PPInput> mean{r.means, iCell};
            const RepeatedMean meanBase{r.means, iCell};
            const index begin = r.offsets(iCell, 0), end = r.offsets(iCell + 1, 0);
            // PP POLICY -- RANS is excluded from flow base tests/writes. Its
            // original polynomial receives an independent scalar compression.
            const real scale = r.rans.Compute(meanBase, higher, r.ransFirst, begin, end);
            if (!(std::isfinite(scale) && scale >= 0 && scale <= 1))
                return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            if (scale < 1)
                for (int component = 0; component < r.rans.components; ++component)
                    for (int mode = 0; mode < r.modes; ++mode)
                        coefficients(iCell, (r.ransFirst + component) * r.modes + mode) *= scale;
            if (degrees(iCell, 0) == r.maxOrder)
                continue;
            real beta = 1;
            if (r.segmentBounds)
            {
                const auto flowMean = r.bounds.Flow(mean);
                for (index q = begin; q < end; ++q)
                    beta = std::min(beta, AdmissibleFraction(flowMean,
                                                             r.bounds.Flow(EquationPositivityDetail::PointRow<PPInput>{higher, q}), *r.segmentBounds));
                if (beta < r.betaCutoff)
                    beta = 0;
            }
            else
                beta = base ? r.bounds.ReconstructionFraction(mean, *base, higher, begin, end, r.betaPower, r.betaCutoff, 1, r.clippedPolicy)
                            : r.bounds.ReconstructionFraction(mean, meanBase, higher, begin, end, r.betaPower, r.betaCutoff, 1, r.clippedPolicy);
            if (!(std::isfinite(beta) && beta >= 0 && beta <= 1))
                return NumericalStatus{NumericalError::InvalidState, 2, iCell};
            cellBeta(iCell, 0) = beta;
            if (beta < 1)
                for (index component = 0; component < r.means.ComponentNumber(); ++component)
                    if (r.bounds.CompressesReconstructionComponent(component))
                        for (int mode = 0; mode < r.modes; ++mode)
                        {
                            const index j = component * r.modes + mode;
                            const real value = reference ? (*reference)(iCell, j) : 0;
                            coefficients(iCell, j) = value + beta * (coefficients(iCell, j) - value);
                        }
        }
        return {};
    }
    NumericalResult<bool> MarkPolynomialRecovery(const PolynomialPositivityRequest &r, PPInput samples, PPScalarOutput beta,
                                                 PPFlagOutput recover, PPScalarOutput recoveryScale) noexcept
    {
        bool any = false;
        for (index iCell = 0; iCell < r.means.ownedRows; ++iCell)
        {
            bool clippedGood = true;
            for (index q = r.offsets(iCell, 0); q < r.offsets(iCell + 1, 0); ++q)
                clippedGood = clippedGood && r.bounds.ClippedReferenceAdmissible(
                                                 EquationPositivityDetail::PointRow<PPInput>{samples, q});
            recover(iCell, 0) = uint8_t(!clippedGood);
            if (!recover(iCell, 0))
                continue;
            any = true;
            const real previous = beta(iCell, 0);
            beta(iCell, 0) = r.clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase ? 0 : previous * .9;
            if (beta(iCell, 0) < r.betaCutoff)
                beta(iCell, 0) = 0;
            if (!(previous > 0))
                return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            recoveryScale(iCell, 0) = beta(iCell, 0) / previous;
        }
        return any;
    }
    void RecoverPolynomial(const PolynomialPositivityRequest &r, std::optional<PPInput> reference,
                           PPOutput coefficients, PPFlagInput recover, PPScalarInput recoveryScale)
    {
        for (index iCell = 0; iCell < r.means.ownedRows; ++iCell)
            if (recover(iCell, 0))
                for (index component = 0; component < r.means.ComponentNumber(); ++component)
                    if (r.bounds.CompressesReconstructionComponent(component))
                        for (int mode = 0; mode < r.modes; ++mode)
                        {
                            const index j = component * r.modes + mode;
                            const real value = reference ? (*reference)(iCell, j) : 0;
                            coefficients(iCell, j) = value + recoveryScale(iCell, 0) * (coefficients(iCell, j) - value);
                        }
    }
    NumericalStatus VerifyReconstructedPoints(const PolynomialPositivityRequest &r, PPInput samples) noexcept
    {
        // PP POLICY -- raw rhoY remains in the polynomial and fluxes. This
        // final predicate uses only the clipped EOS/T composition. Sampling
        // actual post-layout storage makes the guarantee about published values.
        for (index q = 0; q < samples.RowCount(); ++q)
            if (!(r.bounds.ClippedReferenceAdmissible(
                    EquationPositivityDetail::PointRow<PPInput>{samples, q})))
                return NumericalStatus{NumericalError::InvalidState, 1, q};
        return {};
    }
    std::pair<index, real> PolynomialPositivityStatistics(PPScalarInput beta)
    {
        index limited = 0;
        real minimum = 1;
        for (index iCell = 0; iCell < beta.RowCount(); ++iCell)
            if (beta(iCell, 0) < 1)
            {
                ++limited;
                minimum = std::min(minimum, beta(iCell, 0));
            }
        return {limited, minimum};
    }

    NumericalResult<std::pair<index, real>> LimitPhysicalStepRows(PPInput mean, PPInput rhs, CFV::FiniteVolumeHostView metric,
                                                                  EquationThermalView thermal, real dt, real relax, real scale, int compressionPower) noexcept
    {
        index limited = 0;
        real minimum = 1;
        std::vector<real> increment(size_t(mean.ComponentNumber()));
        for (index iCell = 0; iCell < mean.RowCount(); ++iCell)
        {
            const EquationPositivityDetail::PointRow<decltype(mean)> row{mean, iCell};
            if (!(thermal.Admissible(row)))
                return NumericalStatus{NumericalError::InvalidState, 1, iCell};
            for (index j = 0; j < mean.ComponentNumber(); ++j)
            {
                increment[j] = (dt * scale) * rhs(iCell, j) / metric.GetCellVol(iCell);
                if (!(std::isfinite(increment[j])))
                    return NumericalStatus{NumericalError::InvalidState, 2, iCell};
            }
            // PP POLICY -- physical-dt forecast. The complete coupled trial
            // enters one clipped-reference thermal estimate; no species
            // simplex factor is formed and no state is changed. The scalar
            // restricts the common dt only. Explicit stage acceptance and
            // retry remain the actual update guarantee.
            const real alpha = thermal.IncrementFraction(row, increment, relax, 1, compressionPower);
            if (!(std::isfinite(alpha) && alpha >= 0 && alpha <= 1))
                return NumericalStatus{NumericalError::InvalidState, 3, iCell};
            if (alpha < 1)
            {
                ++limited;
                minimum = std::min(minimum, alpha);
            }
        }
        return std::pair<index, real>{limited, minimum};
    }
}
