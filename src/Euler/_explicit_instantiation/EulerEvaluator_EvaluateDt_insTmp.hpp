#include "../EulerEvaluator_EvaluateDt.hxx"

#define DNDS_EulerEvaluator_EvaluateDt_INS(model)           \
    namespace DNDS::Euler                                   \
    {                                                       \
        template void EulerEvaluator<model>::GetWallDist(); \
        template void EulerEvaluator<model>::EvaluateDt(    \
            std::vector<real> &dt,                          \
            ArrayDOFV<nVars_Fixed> &u,                      \
            ArrayRECV<nVars_Fixed> &uRec,                   \
            real CFL, real &dtMinall, real MaxDt,           \
            bool UseLocaldt);                               \
    }
