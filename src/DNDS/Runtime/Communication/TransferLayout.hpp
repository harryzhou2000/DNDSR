#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include <optional>

namespace DNDS::Runtime
{
    /// Destination storage choice, independent of element type and access policy.
    /// Apply is local setup work: distributed callers must agree on failures
    /// before binding payloads. Logical lengths come from the row transfer plan;
    /// padding is neither transferred nor interpreted as values. Rank-local
    /// capacities/strides may differ if each accommodates its received rows.
    /// Empty owners still declare the table width/padded capacity; an empty
    /// default Array is not a replacement for the typed storage prototype.
    class TransferLayout
    {
        enum class Kind
        {
            Compact,
            Preserve,
            Table,
            Padded,
            UniformFormat
        };
        Kind _kind = Kind::Preserve;
        index _width = 0, _stride = 0;
        RowLayout _prototype;
        TransferLayout(Kind kind, index width, index stride) : _kind(kind), _width(width), _stride(stride) {}

    public:
        /// Preserve the local source's family and stride/capacity by default.
        TransferLayout() = default;
        static TransferLayout Compact() { return {Kind::Compact, 0, 0}; }
        /// Preserve the local source's layout family, table width/stride or
        /// padded capacity. Reject incoming lengths that cannot fit that shape.
        static TransferLayout PreserveStorage() { return {Kind::Preserve, 0, 0}; }
        static TransferLayout Table(index width, index stride = 0) { return {Kind::Table, width, stride}; }
        static TransferLayout Padded(index capacity) { return {Kind::Padded, 0, capacity}; }

        static TransferLayout ColMajor(index width, ColMajorTableOptions options = {})
        {
            DNDS_check_throw_info(width >= 0, "negative transfer table width");
            TransferLayout policy{Kind::UniformFormat, width, 0};
            policy._prototype = RowLayout::ColMajor(0, size_t(width), options);
            return policy;
        }
        static TransferLayout TiledColMajor(index width, index tile)
        {
            DNDS_check_throw_info(width >= 0 && tile > 0, "invalid transfer tiled table shape");
            TransferLayout policy{Kind::UniformFormat, width, tile};
            policy._prototype = RowLayout::TiledColMajor(0, size_t(width), size_t(tile));
            return policy;
        }

        /// A fixed-width destination needs no row-length/CSR metadata. Sparse
        /// unrouted slots retain this declared width; their values are untouched.
        std::optional<RowLayout> TableLayout(const RowLayout &source, index rows) const
        {
            DNDS_check_throw_info(rows >= 0, "negative transfer destination row count");
            if (_kind == Kind::UniformFormat)
                return _prototype.WithRowCount(size_t(rows));
            if (_kind == Kind::Preserve && !source.IsVariable())
                return source.WithRowCount(size_t(rows));
            const bool table = _kind == Kind::Table;
            if (!table)
                return {};
            const index width = _kind == Kind::Preserve ? index(source.Width()) : _width;
            const index stride = _kind == Kind::Preserve ? index(source.Stride()) : (_stride ? _stride : width);
            DNDS_check_throw_info(rows >= 0 && width >= 0 && stride >= width,
                                  "invalid transfer destination width or capacity");
            return RowLayout::Dense(size_t(rows), size_t(width), size_t(stride));
        }
        /// Build directly from logical row sizes, never through a temporary CSR.
        RowLayout Apply(const RowLayout &source, const std::vector<size_t> &lengths) const
        {
            if (auto table = TableLayout(source, index(lengths.size())))
            {
                for (auto length : lengths)
                    DNDS_check_throw_info(length == size_t(table->Width()), "received row does not fit transfer destination layout");
                return *table;
            }
            if (_kind == Kind::Compact || (_kind == Kind::Preserve && source.IsRagged()))
                return RowLayout::Ragged(lengths);
            const index capacity = _kind == Kind::Preserve ? source.Stride() : _stride;
            DNDS_check_throw_info(capacity >= 0, "invalid transfer destination capacity");
            return RowLayout::Padded(lengths, size_t(capacity));
        }
        RowLayout Apply(const RowLayout &source, const RowLayout &received) const
        {
            if (auto table = TableLayout(source, received.RowCount()))
            {
                for (index iRow = 0; iRow < received.RowCount(); ++iRow)
                    DNDS_check_throw_info(received.RowSize(iRow) == table->Width(), "received row does not fit transfer destination layout");
                return *table;
            }
            if (received.IsRagged() && (_kind == Kind::Compact || (_kind == Kind::Preserve && source.IsRagged())))
                return received;
            std::vector<size_t> lengths(size_t(received.RowCount()));
            for (index iRow = 0; iRow < received.RowCount(); ++iRow)
                lengths[size_t(iRow)] = received.RowSize(iRow);
            return Apply(source, lengths);
        }
    };
}
