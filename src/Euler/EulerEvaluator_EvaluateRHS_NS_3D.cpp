#include "EulerEvaluator_EvaluateRHS.hxx"

namespace DNDS::Euler
{

    template void EulerEvaluator<NS_3D>::EvaluateRHS(
        ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec, real t);
}