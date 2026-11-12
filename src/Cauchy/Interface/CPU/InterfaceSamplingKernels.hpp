#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include <array>

namespace DNDS::Cauchy::CPU
{
    /// Rigid placement of already selected reference-frame receiver queries.
    /// Row identity is stable until the coupling topology is rebuilt.
    struct InterfaceQueryRequest
    {
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<3>>
            referencePoints;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<3>>
            worldPoints;
        Runtime::ArrayView<index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            ids;
        std::array<real, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::array<real, 3> translation{};
        bool moving = false;
    };

    void BuildInterfaceQueries(const InterfaceQueryRequest &request);
}
