#pragma once

#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Solver/StateGMRES.hpp"

namespace DNDS::Cauchy
{
    /// Common left-GMRES mathematics on every logical component. Operators,
    /// preconditioners and products return synchronously and coordinate their
    /// own distributed failures. No hidden halo or payload repacking.
    using EquationGMRES = Linear::StateGMRES<EquationDofs, EquationDofOperations>;
}
