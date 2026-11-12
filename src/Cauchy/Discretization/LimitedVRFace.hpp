#pragma once
#include "DNDS/Core/Defines.hpp"
#include <array>
namespace DNDS::Cauchy
{
    /// Plain internal-face geometry used by the Cauchy physical LVR sensor.
    struct LimitedVRFace
    {
        index cell, other;
        std::array<real, 3> normal;
        std::array<real, 9> rotation;
    };
}
