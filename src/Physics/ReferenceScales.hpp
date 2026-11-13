#pragma once

#include "DNDS/Core/Defines.hpp"
#include <array>
#include <utility>

namespace DNDS::Physics
{
    /// SI reference values. Solvers store nondimensional conservative fields;
    /// backends receive SI unless an API explicitly documents code-scaled data.
    /// Validate at setup; the arithmetic accessors add no hot-loop checks.
    struct ReferenceScales
    {
        real rho0 = 1, U0 = 1, L0 = 1, T0 = 1;
        void Validate(const std::string &prefix = "ReferenceScales") const
        {
            const std::array<std::pair<const char *, real>, 4> scales{{{"rho0", rho0}, {"U0", U0}, {"L0", L0}, {"T0", T0}}};
            for (const auto &[name, value] : scales)
                DNDS_check_throw_info(std::isfinite(value) && value > 0, prefix + "." + name + " must be finite and > 0");
        }
        real Pressure() const { return rho0 * U0 * U0; }
        real Time() const { return L0 / U0; }
        real SpecificEnergy() const { return U0 * U0; }
        real HeatCapacity() const { return U0 * U0 / T0; }
        real Viscosity() const { return rho0 * U0 * L0; }
        real Conductivity() const { return rho0 * U0 * U0 * U0 * L0 / T0; }
        real Diffusivity() const { return U0 * L0; }
        real SourceRate() const { return rho0 * U0 / L0; }
        real Momentum() const { return rho0 * U0; }
        real EnergyFlux() const { return rho0 * U0 * U0 * U0; }
    };
}
