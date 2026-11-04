#pragma once

#include "Solver/FieldDITR.hpp"
#include "Solver/StateTimeStepper.hpp"

namespace DNDS::ODE
{
    using FieldTimeStepper = StateTimeStepper<Algebra::DofField<real>, FieldTimeOperations>;
}
