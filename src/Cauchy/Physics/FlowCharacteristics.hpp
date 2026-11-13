#pragma once
#include "Cauchy/Physics/FlowState.hpp"

namespace DNDS::Cauchy
{
    /// Column-major conservative/characteristic maps for the flow-five block.
    /// Other equation components remain scalars. This record owns only numbers.
    struct FlowCharacteristics
    {
        std::array<real, 25> forward, inverse;
        Vector3 velocity{};
        real soundSpeed = 0;
        bool componentwiseFallback = false;
    };
}
