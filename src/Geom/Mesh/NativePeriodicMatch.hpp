#pragma once

#include "Geom/PeriodicTypes.hpp"
#include <algorithm>
#include <optional>
#include <vector>

namespace DNDS::Geom
{
    using NativePeriodicNodes = std::vector<std::pair<index, uint8_t>>;

    /// Exact node-image equality modulo one uniform frame change. Sorting
    /// after applying the mask is essential: XOR can permute repeated IDs'
    /// frame order. At most eight tiny subentity patterns are inspected.
    inline std::optional<uint8_t> NativePeriodicFrame(NativePeriodicNodes current, NativePeriodicNodes reference)
    {
        if (current.size() != reference.size() || current.empty())
            return {};
        for (const auto &entry : current)
            if (entry.second >= 8)
                return {};
        for (const auto &entry : reference)
            if (entry.second >= 8)
                return {};
        std::sort(reference.begin(), reference.end());
        for (uint8_t mask = 0; mask < 8; ++mask)
        {
            auto transformed = current;
            for (auto &[node, bits] : transformed)
                bits ^= mask;
            std::sort(transformed.begin(), transformed.end());
            if (transformed == reference)
                return mask;
        }
        return {};
    }
}
