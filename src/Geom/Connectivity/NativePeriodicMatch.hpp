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
        uint8_t previousMask = 0;
        for (uint8_t mask = 0; mask < 8; ++mask)
        {
            // Both inputs are already private values. Transform the previous
            // image in place rather than allocating one vector for every mask.
            for (auto &[node, bits] : current)
                bits ^= previousMask ^ mask;
            previousMask = mask;
            std::sort(current.begin(), current.end());
            if (current == reference)
                return mask;
        }
        return {};
    }
}
