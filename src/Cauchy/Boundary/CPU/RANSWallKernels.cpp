#include "Cauchy/Boundary/CPU/RANSWallKernels.hpp"

namespace DNDS::Cauchy::CPU
{
    std::array<real, 2> RANSWallTarget(const Physics::RANS::Model &model,
                                       const Physics::RANS::Point &mean, real interiorDensity)
    {
        using Kind = Physics::RANS::ModelKind;
        std::array<real, 2> values{};
        if (model.Kind() == Kind::KOmegaWilcox || model.Kind() == Kind::KOmegaSST)
            values[1] = mean.molecularViscosity / sqr(mean.wallDistance) * Physics::RANS::kWallOmegaCoeff;
        else if (model.Kind() == Kind::RealizableKE)
            values[1] = 2 * mean.molecularViscosity / mean.state[0] *
                        (mean.state[5] / mean.state[0]) / sqr(mean.wallDistance) * interiorDensity;
        return values;
    }
}
