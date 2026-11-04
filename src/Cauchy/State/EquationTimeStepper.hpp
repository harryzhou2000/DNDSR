#pragma once

#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Solver/TimeIntegration/TimeStepper.hpp"

namespace DNDS::Cauchy
{
    using EquationDITR = ODE::DITRMethod<EquationDofs, EquationDofOperations>;
    using EquationTimeStepper = ODE::StateTimeStepper<EquationDofs, EquationDofOperations>;
}
