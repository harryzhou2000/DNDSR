#include "../EulerSolver.hxx"

#define DNDS_EULERSOLVER_INS(model)                           \
    namespace DNDS::Euler                                     \
    {                                                         \
        template void EulerSolver<model>::RunImplicitEuler(); \
    }
