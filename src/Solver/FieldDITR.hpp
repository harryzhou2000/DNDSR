#pragma once

#include "Solver/FieldTimeOperations.hpp"
#include "Solver/StateDITR.hpp"

namespace DNDS::ODE
{
    using FieldDITR = StateDITR<Algebra::DofField<real>, FieldTimeOperations>;
}
