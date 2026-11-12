#include "Physics/Thermodynamics/TotalCondition.hpp"
#include <algorithm>
#include <cmath>
#include <new>

namespace DNDS::Physics::Thermodynamics
{
    NumericalResult<TotalConditionStaticState> TotalToStaticIdealMixture(
        Chemistry::ChemicalSource &source, const ReferenceScales &scales,
        real pressure, real temperature, real speedSquared,
        Chemistry::ConstSpeciesBufferView fractions) noexcept
    {
        if (!Detail::ValidTotalConditionInput(pressure, temperature, speedSquared))
            return NumericalStatus{NumericalError::InvalidState, 1, -1};
        try
        {
            if (!source.isIdealGas() || fractions.nSpecies != source.nSpecies())
                return NumericalStatus{NumericalError::InvalidState, 2, -1};
            const real t0 = temperature * scales.T0, p0 = pressure * scales.Pressure();
            const real h0 = source.mixtureEnthalpy(t0, fractions, p0);
            const real s0 = source.mixtureEntropy(t0, fractions, p0);
            const real minimum = std::min(t0 * .999, std::max(source.minTemperature(), 1.));
            const real hMinimum = source.mixtureEnthalpy(minimum, fractions, p0);
            const real maxKinetic = std::max(0., .95 * (h0 - hMinimum));
            const real v2 = std::min(speedSquared, 2 * maxKinetic / scales.SpecificEnergy());
            const real target = h0 - .5 * v2 * scales.SpecificEnergy();
            real low = minimum, high = t0, t = t0;
            bool converged = false;
            for (int iteration = 0; iteration < 60; ++iteration)
            {
                t = .5 * (low + high);
                const real h = source.mixtureEnthalpy(t, fractions, p0);
                if (std::abs(h - target) / std::max(std::abs(target), 1.) < 1e-10)
                {
                    converged = true;
                    break;
                }
                if (h < target)
                    low = t;
                else
                    high = t;
            }
            if (!converged)
                return NumericalStatus{NumericalError::InvalidState, 3, -1};
            const real gasConstant = source.mixtureR(fractions);
            const real entropy = source.mixtureEntropy(t, fractions, p0);
            const real p = p0 * std::exp((entropy - s0) / std::max(gasConstant, 1e-30));
            const real rho = p / (gasConstant * t) / scales.rho0;
            if (!Detail::ValidTotalConditionInput(p, t, v2) || !(std::isfinite(rho) && rho > 0))
                return NumericalStatus{NumericalError::InvalidState, 4, -1};
            return TotalConditionStaticState{rho, p / scales.Pressure(), t / scales.T0, v2};
        }
        catch (const std::bad_alloc &)
        {
            return NumericalStatus{NumericalError::AllocationFailure, 0, -1};
        }
        catch (...)
        {
            return NumericalStatus{NumericalError::UnexpectedFailure, 0, -1};
        }
    }
}
