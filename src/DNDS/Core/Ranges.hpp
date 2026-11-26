#pragma once

#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/Errors.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace DNDS
{
    /// Validate that a collection of half-open intervals `[begin, end)` (after sorting by begin)
    /// partitions `[0, totalWidth)` contiguously with no gaps, overlaps, or empty segments.
    /// Returns true when valid; throws if @p throwOnFailure is true.
    inline bool ValidateContiguousRanges(std::vector<std::pair<index, index>> intervals, index totalWidth,
                                         bool throwOnFailure = true,
                                         const char *message = "intervals do not form a contiguous cover")
    {
        const auto fail = [&]() -> bool
        {
            if (throwOnFailure)
                DNDS_check_throw_info(false, message);
            return false;
        };
        if (totalWidth < 0)
            return fail();
        if (totalWidth == 0)
            return intervals.empty() ? true : fail();
        if (intervals.empty())
            return fail();
        for (const auto &[b, e] : intervals)
            if (b < 0 || e <= b || e > totalWidth)
                return fail();
        std::sort(intervals.begin(), intervals.end());
        index cursor = 0;
        for (const auto &[b, e] : intervals)
        {
            if (b != cursor)
                return fail();
            cursor = e;
        }
        return cursor == totalWidth ? true : fail();
    }
}
