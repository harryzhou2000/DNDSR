#pragma once

#include "Solver/TimeIntegration/DITRMethod.hpp"

namespace DNDS::ODE
{
    /// Historical include/type spelling. The native nested nonlinear driver was
    /// removed; callers must use the public method's explicit stage protocol.
    template <class State, class Operations>
    using StateDITR = DITRMethod<State, Operations>;
}
