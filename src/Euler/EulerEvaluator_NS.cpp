#include "EulerEvaluator.hxx"

namespace DNDS::Euler
{
    template void EulerEvaluator<NS>::LUSGSMatrixInit(
        std::vector<real> &dTau, real dt, real alphaDiag,
        ArrayDOFV<nVars_Fixed> &u, ArrayRECV<nVars_Fixed> &uRec,
        int jacobianCode,
        real t);

    template void EulerEvaluator<NS>::LUSGSMatrixVec(real alphaDiag, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);

    template void EulerEvaluator<NS>::UpdateLUSGSForward(real alphaDiag,
                                                         ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template void EulerEvaluator<NS>::UpdateLUSGSBackward(real alphaDiag,
                                                          ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template void EulerEvaluator<NS>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);

    template void EulerEvaluator<NS>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise);
}
