#include "Cauchy/Discretization/PerfectGasKernelPoint.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"

namespace DNDS::Cauchy
{
    PerfectGasSpatialParameters PerfectGasSpatialParameters::FromEquation(const EquationSystem &equation, int dimension)
    {
        DNDS_check_throw_info(!equation.HasMixture(), "perfect-gas CUDA spatial profile cannot use a mixture model");
        PerfectGasSpatialParameters result;
        result.gas = equation.Gas();
        result.dimension = dimension;
        result.viscous = equation.Viscous();
        if (equation.HasRANS())
            result.rans = equation.RANS().KernelParameters();
        result.Validate(5 + result.rans.ComponentNumber());
        return result;
    }
}
