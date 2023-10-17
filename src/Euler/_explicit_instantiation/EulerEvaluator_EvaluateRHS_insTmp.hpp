#include "../EulerEvaluator_EvaluateRHS.hxx"

#define DNDS_EulerEvaluator_EvaluateRHS_INS(model)        \
    namespace DNDS::Euler                                 \
    {                                                     \
        template void EulerEvaluator<model>::EvaluateRHS( \
            ArrayDOFV<nVars_Fixed> &rhs,                  \
            ArrayDOFV<nVars_Fixed> &JSource,              \
            ArrayDOFV<nVars_Fixed> &u,                    \
            ArrayRECV<nVars_Fixed> &uRec,                 \
            ArrayDOFV<1> &uRecBeta,                       \
            ArrayRECV<1> &cellRHSAlpha,                   \
            bool onlyOnHalfAlpha,                         \
            real t);                                      \
    }
