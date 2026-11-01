#pragma once

#include "DNDS/Core/Defines.hpp"
#include <array>

namespace DNDS::Cauchy
{
    /// Conservative compressible flow profile, including transverse momentum
    /// on planar geometry. Its energy origin is defined by the equation model.
    using FlowState = std::array<real, 5>;
    using Vector3 = std::array<real, 3>;
    /// Physical gradients: [rho,u,v,w,T], each with three directions.
    using FlowGradient = std::array<Vector3, 5>;
    struct PrimitiveState
    {
        real density, pressure, temperature, soundSpeed;
        Vector3 velocity;
    };
}
