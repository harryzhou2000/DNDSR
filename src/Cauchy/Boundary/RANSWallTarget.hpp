#pragma once

#include "Physics/RANS/Point.hpp"

namespace DNDS::Cauchy
{
    /// Euler physical wall target, shared by CPU/device boundary producers.
    /// Viscosity is evaluated at prescribed wall T for an isothermal wall, or
    /// at the mean otherwise. Wall distance is the cell quadrature mean.
    /// SA/k targets are zero; omega is mu/d^2 times the legacy wall coefficient.
    /// RealizableKE retains its CPU formula using cell-mean k and reconstructed
    /// interior density. The caller validates positive rho/d and chooses whether
    /// to use this target directly or mirror it into an exterior flux state.
    DNDS_DEVICE_CALLABLE inline std::array<real, 2> PrepareRANSWallTarget(
        Physics::RANS::ModelKind kind, const Physics::RANS::Point &mean, real interiorDensity) noexcept
    {
        using Kind = Physics::RANS::ModelKind;
        std::array<real, 2> values{};
        if (kind == Kind::KOmegaWilcox || kind == Kind::KOmegaSST)
            values[1] = mean.molecularViscosity / sqr(mean.wallDistance) * Physics::RANS::kWallOmegaCoeff;
        else if (kind == Kind::RealizableKE)
            values[1] = 2 * mean.molecularViscosity / mean.state[0] *
                        (mean.state[5] / mean.state[0]) / sqr(mean.wallDistance) * interiorDensity;
        return values;
    }
}
