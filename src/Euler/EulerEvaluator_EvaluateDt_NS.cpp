#include "EulerEvaluator_EvaluateDt.hxx"

namespace DNDS::Euler
{
    template void EulerEvaluator<NS>::EvaluateDt(
        std::vector<real> &dt,
        ArrayDOFV<nVars_Fixed> &u,
        real CFL, real &dtMinall, real MaxDt,
        bool UseLocaldt);
}