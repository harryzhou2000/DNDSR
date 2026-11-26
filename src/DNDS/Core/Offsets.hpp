#pragma once

#include "DNDS/Core/CheckedSize.hpp"
#include "DNDS/Core/Defines.hpp"
#include "DNDS/Core/Errors.hpp"

#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace DNDS
{
    /// Build checked exclusive prefix-sum offsets of length `counts.size() + 1`.
    /// Rejects negative counts and arithmetic overflow.
    template <class OutT = index, class CountRange>
    std::vector<OutT> ExclusiveOffsets(const CountRange &counts)
    {
        const size_t n = size_t(std::distance(std::begin(counts), std::end(counts)));
        std::vector<OutT> offsets(n + 1, OutT(0));
        OutT total = OutT(0);
        size_t i = 0;
        for (const auto &raw : counts)
        {
            if constexpr (std::is_signed_v<std::decay_t<decltype(raw)>>)
                DNDS_check_throw_info(raw >= 0, "negative count in ExclusiveOffsets");
            const OutT c = CheckedSize::Cast<OutT>(raw);
            total = CheckedSize::Add<OutT>(total, c);
            offsets[++i] = total;
        }
        return offsets;
    }

    /// Two-pass checked CSR flatten from an iterator range of rows with element filter/transform.
    template <class OffsetT = index, class ColT = index, class RowIt, class Filter>
    std::pair<std::vector<OffsetT>, std::vector<ColT>> BuildCSRFromRows(RowIt first, RowIt last, Filter &&accept)
    {
        const size_t nRows = size_t(std::distance(first, last));
        std::vector<OffsetT> offsets(nRows + 1, OffsetT(0));
        OffsetT total = OffsetT(0);
        size_t r = 0;
        for (auto it = first; it != last; ++it)
        {
            OffsetT count = OffsetT(0);
            for (const auto &entry : *it)
            {
                ColT mapped{};
                if (accept(entry, mapped))
                    count = CheckedSize::Add<OffsetT>(count, OffsetT(1));
            }
            total = CheckedSize::Add<OffsetT>(total, count);
            offsets[++r] = total;
        }
        std::vector<ColT> columns(static_cast<size_t>(total));
        r = 0;
        for (auto it = first; it != last; ++it)
        {
            OffsetT pos = offsets[r++];
            for (const auto &entry : *it)
            {
                ColT mapped{};
                if (accept(entry, mapped))
                    columns[size_t(pos++)] = mapped;
            }
        }
        return {std::move(offsets), std::move(columns)};
    }

    /// Two-pass checked CSR flatten from a range of rows with optional element filter/transform.
    template <class OffsetT = index, class ColT = index, class Rows, class Filter>
    std::pair<std::vector<OffsetT>, std::vector<ColT>> BuildCSRFromRows(const Rows &rows, Filter &&accept)
    {
        return BuildCSRFromRows<OffsetT, ColT>(std::begin(rows), std::end(rows), std::forward<Filter>(accept));
    }

    /// Two-pass checked CSR flatten copying all entries directly.
    template <class OffsetT = index, class ColT = index, class Rows>
    std::pair<std::vector<OffsetT>, std::vector<ColT>> BuildCSRFromRows(const Rows &rows)
    {
        return BuildCSRFromRows<OffsetT, ColT>(rows, [](const auto &entry, ColT &out)
                                               {
            out = static_cast<ColT>(entry);
            return true; });
    }
}
