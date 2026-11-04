#pragma once

#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Solver/StateTimeStepper.hpp"

namespace DNDS::Cauchy
{
    using EquationDITR = ODE::StateDITR<EquationDofs, EquationDofOperations>;
    using EquationTimeStepper = ODE::StateTimeStepper<EquationDofs, EquationDofOperations>;
}
