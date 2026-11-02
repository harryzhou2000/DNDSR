#pragma once

#include "Cauchy/Physics/FlowState.hpp"
#include "DNDS/Core/Errors.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace DNDS::Cauchy
{
    /// Initial affine thermal bound. The energy reference may be negative.
    /// Future equation providers can supply composition/density-dependent
    /// MinimumInternalEnergyDensity(state), provided their admissible set is
    /// convex along the segments used by AdmissibleFraction.
    struct ThermalBounds
    {
        real densityFloor, specificEnergyAtMinimumTemperature, energyDensityFloor;
        template <size_t Components>
        DNDS_DEVICE_CALLABLE real MinimumInternalEnergyDensity(const std::array<real, Components> &state) const
        {
            return std::max(state[0] * specificEnergyAtMinimumTemperature, energyDensityFloor);
        }
    };
    template <size_t Components, class Bounds>
    DNDS_DEVICE_CALLABLE bool ThermallyAdmissible(const std::array<real, Components> &state, const Bounds &bounds)
    {
        static_assert(Components >= 5, "thermal flow profile requires rho,momentum3,total-energy");
        for (auto value : state)
            if (!std::isfinite(value))
                return false;
        if (!(state[0] >= bounds.densityFloor))
            return false;
        real kinetic = 0;
        for (int d = 0; d < 3; ++d)
            kinetic += .5 * state[1 + d] * state[1 + d] / state[0];
        const real minimum = bounds.MinimumInternalEnergyDensity(state);
        return std::isfinite(minimum) && state[4] - kinetic >= minimum;
    }
    template <size_t Components, class Bounds>
    DNDS_DEVICE_CALLABLE bool ThermallyInterior(const std::array<real, Components> &state, const Bounds &bounds)
    {
        if (!ThermallyAdmissible(state, bounds))
            return false;
        real kinetic = 0;
        for (int d = 0; d < 3; ++d)
            kinetic += .5 * state[1 + d] * state[1 + d] / state[0];
        const real minimum = bounds.MinimumInternalEnergyDensity(state);
        // Internal energy is a difference of much larger quantities at high
        // Mach. Theta-relative safety cannot protect a tiny beta from
        // cancellation when the stored polynomial is re-evaluated.
        constexpr real roundoff = 64 * std::numeric_limits<real>::epsilon();
        const real scale = std::max(std::max(std::abs(state[4]), kinetic), std::max(std::abs(minimum), real(1e-300)));
        return state[0] - bounds.densityFloor >= roundoff * std::max(state[0], bounds.densityFloor) &&
               state[4] - kinetic - minimum >= roundoff * scale;
    }
    /// Largest safe segment fraction by a bracketed convex-set search. Caller
    /// validates finite endpoints and an admissible base. No T inversion,
    /// pressure recovery, EOS equivalent gamma, or component clipping occurs.
    template <size_t Components, class Bounds>
    DNDS_DEVICE_CALLABLE real AdmissibleFraction(const std::array<real, Components> &base, const std::array<real, Components> &target, const Bounds &bounds)
    {
        if (ThermallyInterior(target, bounds))
            return 1;
        if (!ThermallyInterior(base, bounds))
            return 0; // The exact admissible mean remains usable; no positive slope is safe numerically.
        real low = 0, high = 1;
        for (int iteration = 0; iteration < 56; ++iteration)
        {
            const real middle = .5 * (low + high);
            std::array<real, Components> state;
            for (index j = 0; j < index(Components); ++j)
                state[j] = (1 - middle) * base[j] + middle * target[j];
            if (ThermallyInterior(state, bounds))
                low = middle;
            else
                high = middle;
        }
        return low * (1 - 1e-10);
    }
}
