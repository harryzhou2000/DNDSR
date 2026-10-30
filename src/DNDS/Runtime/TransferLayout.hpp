#pragma once

#include "Array.hpp"

namespace DNDS::Runtime
{
    /// Destination storage choice, independent of element type and access policy.
    /// Apply is local setup work: distributed callers must agree on failures
    /// before binding payloads. Logical lengths come from the row transfer plan;
    /// padding is neither transferred nor interpreted as values. Rank-local
    /// capacities/strides may differ if each accommodates its received rows.
    class TransferLayout
    {
        enum class Kind
        {
            Compact,
            Preserve,
            Table,
            Padded
        };
        Kind _kind = Kind::Compact;
        index _width = 0, _stride = 0;
        TransferLayout(Kind kind, index width, index stride) : _kind(kind), _width(width), _stride(stride) {}

    public:
        /// Compact CSR is the default, including uniform and empty destinations.
        TransferLayout() = default;
        static TransferLayout Compact() { return {}; }
        /// Preserve the local source's layout family, table width/stride or
        /// padded capacity. Reject incoming lengths that cannot fit that shape.
        static TransferLayout PreserveStorage() { return {Kind::Preserve, 0, 0}; }
        static TransferLayout Table(index width, index stride = 0) { return {Kind::Table, width, stride}; }
        static TransferLayout Padded(index capacity) { return {Kind::Padded, 0, capacity}; }

        RowLayout Apply(const RowLayout &source, const RowLayout &received) const
        {
            if (_kind == Kind::Compact || (_kind == Kind::Preserve && source.IsRagged()))
                return received;
            const bool table = _kind == Kind::Table || (_kind == Kind::Preserve && !source.IsVariable());
            const index width = _kind == Kind::Preserve ? index(source.Width()) : _width;
            const index stride = _kind == Kind::Preserve ? index(source.Stride()) : (_stride ? _stride : width);
            DNDS_check_throw_info(width >= 0 && stride >= 0 && (!table || stride >= width),
                                  "invalid transfer destination width or capacity");
            for (index iRow = 0; iRow < received.RowCount(); ++iRow)
                DNDS_check_throw_info(table ? received.RowSize(iRow) == width : received.RowSize(iRow) <= stride,
                                      "received row does not fit transfer destination layout");
            if (table)
                return RowLayout::Dense(received.RowCount(), size_t(width), size_t(stride));
            std::vector<size_t> lengths(size_t(received.RowCount()));
            for (index iRow = 0; iRow < received.RowCount(); ++iRow)
                lengths[size_t(iRow)] = received.RowSize(iRow);
            return RowLayout::Padded(lengths, size_t(stride));
        }
    };
}
