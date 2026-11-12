#pragma once

#include "Cauchy/Physics/FlowState.hpp"

namespace DNDS::Cauchy
{
    /// One boundary quadrature incidence. Stored as a local Runtime array;
    /// condition indexes the CPU producer's resolved condition registry.
    struct BoundaryTraceGeometry
    {
        index cell = 0, face = 0, point = 0, condition = 0, interfaceRow = 0;
        int side = 0;
        Vector3 normal{}, location{};
    };
    static_assert(std::is_trivially_copyable_v<BoundaryTraceGeometry>);
}
