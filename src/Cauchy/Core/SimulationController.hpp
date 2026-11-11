#pragma once

#include "Cauchy/Core/StepResult.hpp"
#include "DNDS/Core/Defines.hpp"

#include <optional>

namespace DNDS::Cauchy
{
    class Simulation;

    /// Sole owner of cross-domain accepted-step ordering. Simulation retains
    /// this controller; it borrows the facade's already-owned domains,
    /// placement, interface coordinator and clock. Domain-local numerical work
    /// remains behind DomainController transactions.
    class SimulationController
    {
        Simulation &_simulation;

    public:
        explicit SimulationController(Simulation &simulation)
            : _simulation(simulation) {}
        [[nodiscard]] StepResult Advance(
            real dt, std::optional<real> acceptedTime = {});
    };
}
