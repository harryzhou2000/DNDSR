#pragma once

#include "Cauchy/Physics/FlowState.hpp"

#include <vector>

namespace DNDS::Cauchy
{
    /// Surface-integrated boundary flux and tangential/viscous force for one
    /// boundary zone. Uses Cauchy/Euler's negative outward RHS flux sign.
    struct BoundaryFluxIntegral
    {
        int zone = 0;
        real area = 0;
        std::vector<real> flux;
        Vector3 tangentialForce{};
    };
}
