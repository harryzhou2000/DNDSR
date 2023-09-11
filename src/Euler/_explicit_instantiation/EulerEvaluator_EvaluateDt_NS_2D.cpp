#include "../EulerEvaluator_EvaluateDt.hxx"

namespace DNDS::Euler
{
    template void EulerEvaluator<NS_2D>::GetWallDist();

    template void EulerEvaluator<NS_2D>::EvaluateDt(
        std::vector<real> &dt,
        ArrayDOFV<nVars_Fixed> &u,
        real CFL, real &dtMinall, real MaxDt,
        bool UseLocaldt);
}