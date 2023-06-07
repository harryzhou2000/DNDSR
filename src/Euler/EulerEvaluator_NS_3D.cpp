#include "EulerEvaluator.hxx"

namespace DNDS::Euler
{
    template void EulerEvaluator<NS_3D>::LUSGSMatrixInit(
        std::vector<real> &dTau, real dt, real alphaDiag,
        ArrayDOFV<nVars_Fixed> &u, ArrayRECV<nVars_Fixed> &uRec,
        int jacobianCode,
        real t);

    template void EulerEvaluator<NS_3D>::LUSGSMatrixVec(real alphaDiag, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &AuInc);

    template void EulerEvaluator<NS_3D>::UpdateLUSGSForward(real alphaDiag,
                                                            ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template void EulerEvaluator<NS_3D>::UpdateLUSGSBackward(real alphaDiag,
                                                             ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &uInc, ArrayDOFV<nVars_Fixed> &uIncNew);

    template void EulerEvaluator<NS_3D>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);

    template void EulerEvaluator<NS_3D>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise);
}
