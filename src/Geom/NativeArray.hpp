#pragma once

#include "DNDS/Runtime/Array.hpp"

namespace DNDS::Geom
{
    /// Native topology storage uses Runtime allocation and row access directly.
    using Adjacency = Runtime::Array<index>;
    using AdjacencyPair = Runtime::ArrayPair<index>;
    template <size_t Width>
    using FixedAdjacency = Runtime::Array<index, Runtime::FixedRows<Width>>;
    template <size_t Width>
    using FixedAdjacencyPair = Runtime::ArrayPair<index, Runtime::FixedRows<Width>>;
}
