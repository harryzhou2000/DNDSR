#pragma once

#include "Cauchy/State/QueuedEquationOperations.hpp"
#include "Solver/TimeIntegration/TimeStepper.hpp"

namespace DNDS::Cauchy
{
    using EquationDITR = ODE::DITRMethod<EquationDofs, QueuedEquationOperations>;
    using EquationTimeStepper = ODE::StateTimeStepper<EquationDofs, QueuedEquationOperations>;
}
