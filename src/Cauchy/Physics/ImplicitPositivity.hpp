#pragma once

#include "Cauchy/Physics/Positivity.hpp"

namespace DNDS::Cauchy
{
    namespace PositivityDetail
    {
        DNDS_DEVICE_CALLABLE inline FlowState Add(const FlowState &state, const FlowState &increment, real alpha = 1)
        {
            auto value = state;
            for (int j = 0; j < 5; ++j)
                value[j] += alpha * increment[j];
            return value;
        }
        DNDS_DEVICE_CALLABLE inline real InternalEnergy(const FlowState &state)
        {
            real momentum2 = 0;
            for (int d = 0; d < 3; ++d)
                momentum2 += state[1 + d] * state[1 + d];
            return state[4] - .5 * momentum2 / (state[0] + verySmallReal);
        }
    }
    /// Euler Gas::IdealGasGetCompressionRatioPressure scheme=1, expressed in
    /// conservative flow5 and explicit internal-energy reference values. The
    /// reference must be affine on this segment (or a supplied safe affine
    /// upper bound). No EOS/gamma substitution belongs in this numerical rule.
    /// Caller supplies finite states with positive density and an admissible
    /// old endpoint. The 1e-5 safety and .99 repair are legacy constants.
    DNDS_DEVICE_CALLABLE inline real InternalEnergyCompression(const FlowState &state, const FlowState &increment,
                                                               real floorInput, real baseOld = 0, real baseNew = 0)
    {
        using namespace PositivityDetail;
        const real oldEnergy = InternalEnergy(state) - baseOld;
        const real newEnergy = InternalEnergy(Add(state, increment)) - baseNew;
        const real floor = std::max(floorInput, smallReal * std::max(oldEnergy, real(0)));
        real alpha = newEnergy < floor ? (oldEnergy - floor) / std::max(oldEnergy - newEnergy, verySmallReal) : 1;
        alpha = std::max(real(0), std::min(real(1), alpha)) * (1 - 1e-5);
        if (alpha < smallReal)
            alpha = 0;
        for (int iteration = 0; iteration < 1000; ++iteration)
        {
            const auto value = Add(state, increment, alpha);
            if (InternalEnergy(value) - (baseOld + alpha * (baseNew - baseOld)) >= floor)
                break;
            alpha *= .99;
        }
        return InternalEnergy(Add(state, increment, alpha)) - (baseOld + alpha * (baseNew - baseOld)) >= floor
                   ? alpha
                   : std::numeric_limits<real>::quiet_NaN();
    }
    /// Euler EvaluateCellRHSAlpha's density/energy relaxation, power compression
    /// and final safety factor. The caller performs field/global reductions and
    /// applies RHS compression or dTau compression; this pure rule does neither.
    /// Initial ThermalBounds is the intersection of two affine energy bounds:
    /// rho*e(Tmin) and the pressure-derived energy-density floor. Applying the
    /// same concave Euler rule to both preserves the temperature constraint even
    /// when the controlling bound changes along the segment. Future EOS must
    /// provide valid energy constraints rather than an equivalent gamma.
    DNDS_DEVICE_CALLABLE inline real FlowIncrementFraction(const FlowState &state, const FlowState &increment,
                                                           const ThermalBounds &bounds, real relax = 1, int compression = 2, int power = 1)
    {
        using namespace PositivityDetail;
        const real relaxedDensity = bounds.densityFloor * relax + state[0] * (1 - relax);
        const real alphaRho = increment[0] < 0 ? std::min(real(1), (state[0] - relaxedDensity) / (-increment[0] - smallReal * increment[0])) : 1;
        auto limited = increment;
        for (auto &value : limited)
            value *= alphaRho;
        const auto next = Add(state, limited);
        real alphaEnergy = 1;
        for (int constraint = 0; constraint < 2; ++constraint)
        {
            if (constraint == 0 && !std::isfinite(bounds.energyDensityFloor))
                continue;
            const real baseOld = constraint == 0 ? 0 : state[0] * bounds.specificEnergyAtMinimumTemperature;
            const real baseNew = constraint == 0 ? 0 : next[0] * bounds.specificEnergyAtMinimumTemperature;
            const real floor = constraint == 0 ? bounds.energyDensityFloor : 0;
            const real oldEnergy = InternalEnergy(state) - baseOld, newEnergy = InternalEnergy(next) - baseNew;
            const real relaxedEnergy = newEnergy < oldEnergy ? floor + (oldEnergy - floor) * (1 - relax) : floor;
            if (newEnergy < relaxedEnergy)
            {
                const real fraction = InternalEnergyCompression(state, limited, relaxedEnergy, baseOld, baseNew);
                if (!std::isfinite(fraction))
                    return fraction;
                alphaEnergy = std::min(alphaEnergy, fraction);
            }
        }
        const real alpha = alphaRho * alphaEnergy;
        return alpha < 1 ? std::pow(alpha, compression * power) * (1 - 1e-5) : 1;
    }
    /// Euler CompressInc flow5 repair, after the separately selected relaxed
    /// delta compression. Density is corrected first; each affine internal-
    /// energy bound uses scheme=1, .99 extra safety and the legacy .9 pull-up.
    /// No component clipping and no arbitrary pressure/temperature reset occurs.
    DNDS_DEVICE_CALLABLE inline FlowState FixFlowIncrement(const FlowState &state, FlowState increment, const ThermalBounds &bounds)
    {
        using namespace PositivityDetail;
        if (state[0] + increment[0] <= bounds.densityFloor)
        {
            const real fraction = (bounds.densityFloor - state[0]) / (increment[0] - verySmallReal);
            for (auto &value : increment)
                value *= fraction;
        }
        for (int constraint = 0; constraint < 2; ++constraint)
        {
            if (constraint == 0 && !std::isfinite(bounds.energyDensityFloor))
                continue;
            const real baseOld = constraint == 0 ? 0 : state[0] * bounds.specificEnergyAtMinimumTemperature;
            const real floor = constraint == 0 ? bounds.energyDensityFloor : 0;
            auto next = Add(state, increment);
            const real baseNew = constraint == 0 ? 0 : next[0] * bounds.specificEnergyAtMinimumTemperature;
            if (InternalEnergy(next) - baseNew <= floor)
            {
                const real alpha = InternalEnergyCompression(state, increment, floor, baseOld, baseNew) * .99;
                for (auto &value : increment)
                    value *= alpha;
                for (int iteration = 0; iteration < 1000; ++iteration)
                {
                    next = Add(state, increment);
                    const real base = constraint == 0 ? 0 : next[0] * bounds.specificEnergyAtMinimumTemperature;
                    if (InternalEnergy(next) - base >= floor)
                        break;
                    for (auto &value : increment)
                        value *= .9;
                }
            }
        }
        return increment;
    }
}
