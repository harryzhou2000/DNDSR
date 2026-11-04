#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>

namespace DNDS::Physics::Chemistry
{
    /// Euler's transported cell-mean species repair, independent of state
    /// ordering, storage layout, EOS and communication. Only the supplied Ns-1
    /// densities change, in the caller's code mass/volume units. Caller
    /// supplies finite values and positive density;
    /// validate the resulting thermodynamic state separately. This is not the
    /// temporary full-composition repair used by property evaluation.
    template <class Values>
    DNDS_DEVICE_CALLABLE bool RepairSpeciesDensities(real rho, Values values, index independent)
    {
        constexpr real floor = 1e-30, dependentFractionFloor = 1e-14;
        bool changed = false;
        for (index k = 0; k < independent; ++k)
        {
            const real value = std::max(values[k], floor);
            changed = changed || value != values[k];
            values[k] = value;
        }
        real sum = 0;
        for (index k = 0; k < independent; ++k)
            sum += values[k];
        const real maximum = rho * (1 - dependentFractionFloor);
        if (sum > maximum)
        {
            const real scale = maximum / sum;
            for (index k = 0; k < independent; ++k)
                values[k] *= scale;
            changed = true;
        }
        return changed;
    }
}
