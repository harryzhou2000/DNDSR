#pragma once

#include "Physics/RANS/Model.hpp"

namespace DNDS::Cauchy::CPU
{
    /// Euler no-slip RANS target. The caller selects viscosity at the prescribed
    /// wall temperature for an isothermal wall, otherwise at the cell mean.
    /// meanWallDistance is the arithmetic mean of cell quadrature samples.
    /// A physical face receives target; a Riemann exterior receives 2*target-U.
    /// SA/k targets are zero. Epsilon uses cell-mean k and reconstructed interior
    /// density, including for isothermal walls (not the prescribed wall density).
    std::array<real, 2> RANSWallTarget(const Physics::RANS::Model &model,
                                       const Physics::RANS::Point &mean, real interiorDensity);
}
