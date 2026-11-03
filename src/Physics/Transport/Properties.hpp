#pragma once

#include <limits>

namespace DNDS::Physics::Transport
{
    /// Newtonian/Fourier molecular coefficients at one point. Species
    /// coefficients use separate caller-owned buffers; these two scalars do
    /// not prescribe a diagonal or matrix species-transport representation.
    struct Properties
    {
        double viscosity = std::numeric_limits<double>::quiet_NaN();
        double conductivity = std::numeric_limits<double>::quiet_NaN();
    };
}
