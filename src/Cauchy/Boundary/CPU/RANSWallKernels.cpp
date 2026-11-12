#include "Cauchy/Boundary/CPU/RANSWallKernels.hpp"
#include "Cauchy/Boundary/RANSWallTarget.hpp"

namespace DNDS::Cauchy::CPU
{
    std::array<real, 2> RANSWallTarget(const Physics::RANS::Model &model,
                                       const Physics::RANS::Point &mean, real interiorDensity)
    {
        return PrepareRANSWallTarget(model.Kind(), mean, interiorDensity);
    }
}
