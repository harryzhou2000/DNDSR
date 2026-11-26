#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/Errors.hpp"

#include <algorithm>
#include <iterator>
#include <vector>

namespace DNDS
{
    /// Return true iff @p values (subset or full list of indices in `[0, domain)`) contains any duplicate.
    /// Does not mutate the caller's container.
    template <class Range>
    bool PermutationHasDuplicates(const Range &values, index domain = -1)
    {
        const size_t n = size_t(std::distance(std::begin(values), std::end(values)));
        if (n <= 1)
            return false;
        if (domain >= 0 && size_t(domain) >= n)
        {
            std::vector<uint8_t> seen(size_t(domain), 0);
            for (const auto &raw : values)
            {
                const index v = index(raw);
                DNDS_check_throw_info(v >= 0 && v < domain, "permutation value out of range");
                if (seen[size_t(v)])
                    return true;
                seen[size_t(v)] = 1;
            }
            return false;
        }
        std::vector<index> sorted;
        sorted.reserve(n);
        for (const auto &raw : values)
            sorted.push_back(index(raw));
        std::sort(sorted.begin(), sorted.end());
        return std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
    }

    /// Validate that @p perm is a bijection of `[0, expectedSize)` (or `[0, perm.size())` if `expectedSize < 0`).
    template <class Range>
    void ValidatePermutation(const Range &perm, index expectedSize = -1, const char *message = "invalid permutation")
    {
        const index n = index(std::distance(std::begin(perm), std::end(perm)));
        if (expectedSize >= 0)
            DNDS_check_throw_info(n == expectedSize, message);
        std::vector<uint8_t> seen(size_t(n), 0);
        for (const auto &raw : perm)
        {
            const index v = index(raw);
            DNDS_check_throw_info(v >= 0 && v < n && !seen[size_t(v)], message);
            seen[size_t(v)] = 1;
        }
    }

    /// Validate @p perm as a bijection of `[0, n)` and return its inverse (`inverse[perm[i]] == i`).
    template <class Range>
    std::vector<index> InvertPermutationChecked(const Range &perm, const char *message = "invalid permutation")
    {
        const index n = index(std::distance(std::begin(perm), std::end(perm)));
        std::vector<index> inverse(size_t(n), -1);
        index i = 0;
        for (const auto &raw : perm)
        {
            const index target = index(raw);
            DNDS_check_throw_info(target >= 0 && target < n && inverse[size_t(target)] < 0, message);
            inverse[size_t(target)] = i++;
        }
        return inverse;
    }

    /// Concatenate disjoint index ranges (e.g., core + halo-dependent) into an order vector
    /// and verify it forms a bijection of `[0, n)`, returning `{order, position}` where `position[order[i]] == i`.
    template <class RangeA, class RangeB>
    std::pair<std::vector<index>, std::vector<index>> BuildPermutationFromRanges(
        index n, const RangeA &first, const RangeB &second, const char *message = "ranges do not form a permutation")
    {
        DNDS_check_throw_info(n >= 0, message);
        std::vector<index> order;
        order.reserve(size_t(n));
        for (const auto &v : first)
            order.push_back(index(v));
        for (const auto &v : second)
            order.push_back(index(v));
        DNDS_check_throw_info(index(order.size()) == n, message);
        auto position = InvertPermutationChecked(order, message);
        return {std::move(order), std::move(position)};
    }
}
