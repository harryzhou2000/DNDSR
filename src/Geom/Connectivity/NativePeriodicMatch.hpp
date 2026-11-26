#pragma once

#include "Geom/Elements/Elements.hpp"
#include "Geom/Periodic/PeriodicTypes.hpp"
#include <algorithm>
#include <array>
#include <optional>
#include <vector>

namespace DNDS::Geom
{
    using NativePeriodicNodes = std::vector<std::pair<index, uint8_t>>;

    namespace Detail
    {
        struct CountingIntersectionIterator
        {
            using iterator_category = std::output_iterator_tag;
            using value_type = void;
            using difference_type = std::ptrdiff_t;
            using pointer = void;
            using reference = void;
            size_t *count;
            template <class T>
            CountingIntersectionIterator &operator=(const T &)
            {
                ++(*count);
                return *this;
            }
            CountingIntersectionIterator &operator*() { return *this; }
            CountingIntersectionIterator &operator++() { return *this; }
            CountingIntersectionIterator &operator++(int) { return *this; }
        };

        template <class ItA, class ItB>
        inline size_t CountSortedIntersection(ItA firstA, ItA lastA, ItB firstB, ItB lastB)
        {
            size_t shared = 0;
            std::set_intersection(firstA, lastA, firstB, lastB, CountingIntersectionIterator{&shared});
            return shared;
        }
    }

    /// Candidate uniform frame masks that map the minimum canonical node ID in
    /// pattern to frame 0. Used by canonical role-key construction.
    inline std::array<bool, 8> NativePeriodicFrameCandidates(const NativePeriodicNodes &pattern, bool periodic)
    {
        DNDS_check_throw_info(!pattern.empty(), "native topology identity requires node occurrences");
        std::array<bool, 8> candidates{};
        const auto firstNode = std::min_element(pattern.begin(), pattern.end())->first;
        for (const auto &[node, frame] : pattern)
        {
            DNDS_check_throw_info(frame < 8, "invalid native topology identity image");
            if (periodic && node == firstNode)
                candidates[frame] = true;
        }
        if (!periodic)
            candidates[0] = true;
        return candidates;
    }

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
        std::sort(current.begin(), current.end());
        std::sort(reference.begin(), reference.end());
        bool uniqueNodes = true;
        for (index iNode = 0; iNode < index(current.size()); ++iNode)
        {
            if (current[iNode].first != reference[iNode].first)
                return {};
            uniqueNodes = uniqueNodes && (iNode == 0 || current[iNode - 1].first != current[iNode].first);
        }
        if (uniqueNodes)
        {
            // With distinct IDs the correspondence fixes exactly one mask.
            // Most faces have this form, including faces touching a periodic
            // seam. Repeated IDs still need the full multiset search below.
            const uint8_t mask = current.front().second ^ reference.front().second;
            for (index iNode = 0; iNode < index(current.size()); ++iNode)
                if ((current[iNode].second ^ reference[iNode].second) != mask)
                    return {};
            return mask;
        }
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

    /// Check whether two cells share at least minimumShared vertices in a
    /// consistent periodic frame. Inputs must be sorted by (vertexId, frame)
    /// (or vertexId alone when frames are null). When XORing a candidate mask
    /// into otherFrames, the transformed pairs are re-sorted before counting
    /// the intersection so repeated canonical vertex IDs on 1-cell-thick
    /// periodic meshes match reliably.
    inline bool NativePeriodicFacialMatch(const index *current, const uint8_t *currentFrames, size_t currentSize,
                                          const index *other, const uint8_t *otherFrames, size_t otherSize,
                                          size_t minimumShared)
    {
        if (minimumShared == 0)
            return true;
        if (currentSize < minimumShared || otherSize < minimumShared)
            return false;
        const size_t rawShared = Detail::CountSortedIntersection(current, current + currentSize, other, other + otherSize);
        if (!currentFrames || !otherFrames)
            return rawShared >= minimumShared;
        if (rawShared < minimumShared)
            return false;
        constexpr size_t maxVertices = size_t(Elem::CellNumNodeMax);
        DNDS_check_throw_info(currentSize <= maxVertices && otherSize <= maxVertices,
                              "periodic facial match vertex count exceeds element maximum");
        std::array<std::pair<index, uint8_t>, maxVertices> currentPairs{};
        std::array<std::pair<index, uint8_t>, maxVertices> transformedOther{};
        for (size_t i = 0; i < currentSize; ++i)
            currentPairs[i] = {current[i], currentFrames[i]};
        uint8_t tried = 0;
        for (size_t i = 0; i < currentSize; ++i)
        {
            for (size_t j = 0; j < otherSize; ++j)
            {
                if (current[i] != other[j])
                    continue;
                const uint8_t mask = currentFrames[i] ^ otherFrames[j];
                if (mask >= 8 || (tried & (uint8_t(1) << mask)))
                    continue;
                tried |= uint8_t(1) << mask;
                for (size_t k = 0; k < otherSize; ++k)
                    transformedOther[k] = {other[k], uint8_t(otherFrames[k] ^ mask)};
                std::sort(transformedOther.begin(), transformedOther.begin() + ptrdiff_t(otherSize));
                if (Detail::CountSortedIntersection(currentPairs.begin(), currentPairs.begin() + ptrdiff_t(currentSize),
                                                    transformedOther.begin(), transformedOther.begin() + ptrdiff_t(otherSize)) >= minimumShared)
                    return true;
            }
        }
        return false;
    }
}
