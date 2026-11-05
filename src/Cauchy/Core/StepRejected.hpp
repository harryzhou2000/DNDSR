#pragma once

#include <stdexcept>

namespace DNDS::Cauchy
{
    /// A numerical rejection agreed by every rank in the domain. Throw only
    /// after the method's collectives have completed and its acceptance flag
    /// is global. Simulation promotes this agreement to the control communicator
    /// before the frontend may retry. Other exceptions are fatal to a distributed
    /// run: they must never be translated into timestep rejection/retry.
    class StepRejected : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };
}
