#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"

namespace DNDS::Cauchy
{
    RANSReconstructionView ResolveRANSReconstructionPolicy(const EquationSystem &equation)
    {
        return {equation.RANS().Kind(), equation.RANS().ComponentNumber()};
    }
}
