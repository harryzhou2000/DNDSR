#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/Errors.hpp"

#include <algorithm>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <vector>

namespace DNDS
{
    /// Sort a random-access container in-place and erase consecutive duplicates.
    template <class Container>
    void SortUnique(Container &container)
    {
        std::sort(container.begin(), container.end());
        container.erase(std::unique(container.begin(), container.end()), container.end());
    }

    /// Remove all occurrences of @p sentinel, then sort and erase duplicates.
    template <class Container, class T>
    void SortUniqueRemove(Container &container, const T &sentinel)
    {
        container.erase(std::remove(container.begin(), container.end(), sentinel), container.end());
        SortUnique(container);
    }

    /// Binary search in a sorted row; throws @p message if @p key is absent.
    template <class Range, class T>
    index CheckedLowerBoundRow(const Range &row, const T &key, const char *message = "key not found in sorted row")
    {
        const auto begin = std::begin(row);
        const auto end = std::end(row);
        const auto found = std::lower_bound(begin, end, key);
        DNDS_check_throw_info(found != end && *found == key, message);
        return index(std::distance(begin, found));
    }

    /// Return true iff all elements in @p range are pairwise distinct.
    template <class Range>
    bool AllUnique(const Range &range)
    {
        using Value = std::decay_t<decltype(*std::begin(range))>;
        std::vector<Value> sorted(std::begin(range), std::end(range));
        std::sort(sorted.begin(), sorted.end());
        return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
    }

    /// Return true iff @p range is a permutation of `[0, n)`.
    template <class Range>
    bool IsPermutation(const Range &range, index n)
    {
        if (n < 0)
            return false;
        const auto size = index(std::distance(std::begin(range), std::end(range)));
        if (size != n)
            return false;
        std::vector<uint8_t> seen(size_t(n), 0);
        for (const auto &raw : range)
        {
            const index v = index(raw);
            if (v < 0 || v >= n || seen[size_t(v)])
                return false;
            seen[size_t(v)] = 1;
        }
        return true;
    }
}
