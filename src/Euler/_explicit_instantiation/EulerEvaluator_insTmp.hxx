#include "../EulerEvaluator.hxx"

#define DNDS_EulerEvaluator_INS(model)                                                                                                           \
    namespace DNDS::Euler                                                                                                                        \
    {                                                                                                                                            \
        template void EulerEvaluator<model>::LUSGSMatrixInit(                                                                                    \
            ArrayDOFV<nVars_Fixed> &JDiag,                                                                                                       \
            ArrayDOFV<nVars_Fixed> &JSource,                                                                                                     \
            std::vector<real> &dTau, real dt, real alphaDiag,                                                                                    \
            ArrayDOFV<nVars_Fixed> &u,                                                                                                           \
            ArrayRECV<nVars_Fixed> &uRec,                                                                                                        \
            int jacobianCode,                                                                                                                    \
            real t);                                                                                                                             \
                                                                                                                                                 \
        template void EulerEvaluator<model>::LUSGSMatrixVec(                                                                                     \
            real alphaDiag,                                                                                                                      \
            ArrayDOFV<nVars_Fixed> &u,                                                                                                           \
            ArrayDOFV<nVars_Fixed> &uInc,                                                                                                        \
            ArrayDOFV<nVars_Fixed> &JDiag,                                                                                                       \
            ArrayDOFV<nVars_Fixed> &AuInc);                                                                                                      \
                                                                                                                                                 \
        template void EulerEvaluator<model>::UpdateLUSGSForward(                                                                                 \
            real alphaDiag,                                                                                                                      \
            ArrayDOFV<nVars_Fixed> &rhs,                                                                                                         \
            ArrayDOFV<nVars_Fixed> &u,                                                                                                           \
            ArrayDOFV<nVars_Fixed> &uInc,                                                                                                        \
            ArrayDOFV<nVars_Fixed> &JDiag,                                                                                                       \
            ArrayDOFV<nVars_Fixed> &uIncNew);                                                                                                    \
                                                                                                                                                 \
        template void EulerEvaluator<model>::UpdateLUSGSBackward(                                                                                \
            real alphaDiag,                                                                                                                      \
            ArrayDOFV<nVars_Fixed> &rhs,                                                                                                         \
            ArrayDOFV<nVars_Fixed> &u,                                                                                                           \
            ArrayDOFV<nVars_Fixed> &uInc,                                                                                                        \
            ArrayDOFV<nVars_Fixed> &JDiag,                                                                                                       \
            ArrayDOFV<nVars_Fixed> &uIncNew);                                                                                                    \
                                                                                                                                                 \
        template void EulerEvaluator<model>::UpdateSGS(                                                                                          \
            real alphaDiag,                                                                                                                      \
            ArrayDOFV<nVars_Fixed> &rhs,                                                                                                         \
            ArrayDOFV<nVars_Fixed> &u,                                                                                                           \
            ArrayDOFV<nVars_Fixed> &uInc,                                                                                                        \
            ArrayDOFV<nVars_Fixed> &JDiag,                                                                                                       \
            bool forward, TU &sumInc);                                                                                                           \
        template void EulerEvaluator<model>::UpdateSGSWithRec(                                                                                    \
            real alphaDiag,                                                                                                                      \
            ArrayDOFV<nVars_Fixed> &rhs,                                                                                                         \
            ArrayDOFV<nVars_Fixed> &u,                                                                                                           \
            ArrayRECV<nVars_Fixed> &uRec,                                                                                                        \
            ArrayDOFV<nVars_Fixed> &uInc,                                                                                                        \
            ArrayRECV<nVars_Fixed> &uRecInc,                                                                                                     \
            ArrayDOFV<nVars_Fixed> &JDiag,                                                                                                       \
            bool forward, TU &sumInc);                                                                                                           \
                                                                                                                                                 \
        template void EulerEvaluator<model>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);                                                           \
                                                                                                                                                 \
        template void EulerEvaluator<model>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise); \
    }