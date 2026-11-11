#pragma once

#include "Cauchy/Core/StepResult.hpp"
#include "DNDS/Core/Defines.hpp"

#include <memory>
#include <optional>

namespace DNDS::Cauchy
{
    struct SimulationControlAccess;

    /// Sole owner of cross-domain accepted-step ordering. Simulation retains
    /// this controller; it borrows the facade's already-owned domains,
    /// placement, interface coordinator and clock. Domain-local numerical work
    /// remains behind DomainController transactions.
    class SimulationController
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        explicit SimulationController(SimulationControlAccess access);
        ~SimulationController();
        SimulationController(const SimulationController &) = delete;
        SimulationController &operator=(const SimulationController &) = delete;
        [[nodiscard]] StepResult Advance(
            real dt, std::optional<real> acceptedTime = {});
    };
}
