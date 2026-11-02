#pragma once

#include "Cauchy/Physics/ImplicitPositivity.hpp"

namespace DNDS::Cauchy
{
    namespace PositivityDetail
    {
        template <class Values>
        DNDS_DEVICE_CALLABLE inline FlowState LoadQuadratureState(const Values &values, index iPoint)
        {
            FlowState result{};
            for (int j = 0; j < 5; ++j)
                result[j] = values(iPoint, j);
            return result;
        }
    }
    /// Euler EvaluateURecBeta's common density fraction, scheme-one energy
    /// fraction and beta power/safety, applied to explicit quadrature values.
    /// Base may be a lower-order polynomial; all its points and the mean must
    /// be admissible. Higher points must be finite, but need not be admissible.
    /// Views supply (point,component); no allocation, MPI, EOS call or mutation.
    /// The caller selects/truncates the reference order, validates controls and
    /// applies baseCoefficients+beta*(highCoefficients-baseCoefficients).
    /// The initial thermal model is the intersection of two affine energy
    /// constraints, as in FlowIncrementFraction. No equivalent gamma is used.
    template <class BaseValues, class HigherValues>
    DNDS_DEVICE_CALLABLE real ReconstructionFraction(const FlowState &mean, BaseValues baseValues, HigherValues higherValues,
                                                     index begin, index end, const ThermalBounds &bounds, int power = 1, real cutoff = 1e-3)
    {
        using namespace PositivityDetail;
        bool good = true;
        for (index iPoint = begin; iPoint < end; ++iPoint)
            good = good && ThermallyAdmissible(LoadQuadratureState(higherValues, iPoint), bounds);
        if (good)
            return 1; // Legacy early exit: do not perturb a good polynomial.
        real density = 1;
        for (index iPoint = begin; iPoint < end; ++iPoint)
        {
            const real base = baseValues(iPoint, 0), delta = higherValues(iPoint, 0) - base;
            if (delta < 0)
                density = std::min(density, (base - bounds.densityFloor) / (-delta + verySmallReal));
        }
        real energy = 1;
        for (int constraint = 0; constraint < 2; ++constraint)
        {
            if (constraint == 0 && !std::isfinite(bounds.energyDensityFloor))
                continue;
            const real floor = constraint == 0 ? bounds.energyDensityFloor : 0;
            const real meanReference = constraint == 0 ? 0 : mean[0] * bounds.specificEnergyAtMinimumTemperature;
            if (InternalEnergy(mean) - meanReference <= 2 * floor)
            {
                energy = 0;
                continue;
            }
            for (index iPoint = begin; iPoint < end; ++iPoint)
            {
                const auto base = LoadQuadratureState(baseValues, iPoint), high = LoadQuadratureState(higherValues, iPoint);
                FlowState delta{};
                for (int j = 0; j < 5; ++j)
                    delta[j] = density * (high[j] - base[j]);
                const auto next = Add(base, delta);
                const real baseOld = constraint == 0 ? 0 : base[0] * bounds.specificEnergyAtMinimumTemperature;
                const real baseNew = constraint == 0 ? 0 : next[0] * bounds.specificEnergyAtMinimumTemperature;
                if (InternalEnergy(next) - baseNew < 2 * floor)
                {
                    const real fraction = InternalEnergyCompression(base, delta, floor, baseOld, baseNew);
                    if (!std::isfinite(fraction))
                        return fraction;
                    energy = std::min(energy, fraction);
                }
            }
        }
        const real beta = density * energy;
        const real compressed = beta < 1 ? beta * std::pow(beta, power) * (1 - 1e-5) : 1;
        return compressed < cutoff ? 0 : compressed;
    }
}
