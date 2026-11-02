#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Algebra/EigenView.hpp"
#include <limits>
#include <memory>
#include <vector>

namespace DNDS::Algebra
{
    /// Borrowed CPU dense-block access. Retain the storage access object while
    /// using this view. Logical sparse rows contain consecutive dense blocks;
    /// each block uses the centralized column-major Eigen adapter.
    template <class Scalar>
    class BlockMatrixView
    {
        using ArrayView = Runtime::ArrayView<Scalar, Runtime::AccessBoundsChecks, Runtime::TableRows>;
        EigenRowMatrices<ArrayView> _blocks;
        const index *_offsets;
        index _rows;

    public:
        BlockMatrixView(ArrayView values, const index *offsets, index rows, index blockSize)
            : _blocks(values, blockSize, blockSize, Runtime::Device::Host()), _offsets(offsets), _rows(rows) {}
        index RowNumber() const { return _rows; }
        index BlockNumber(index iRow) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_assert(iRow >= 0 && iRow < _rows);
            return _offsets[iRow + 1] - _offsets[iRow];
        }
        auto operator()(index iRow, index iBlock) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_assert(iRow >= 0 && iRow < _rows && iBlock >= 0 && iBlock < BlockNumber(iRow));
            return _blocks[_offsets[iRow] + iBlock];
        }
    };

    /// Native numerical storage, independent of mesh adjacency and factorization.
    /// Shared copies retain both the Array allocation and immutable block-row
    /// offsets. All ownership/leases are acquired outside block arithmetic.
    class BlockMatrixStorage
    {
        struct Layout
        {
            std::vector<index> offsets;
            index blockSize;
        };
        std::shared_ptr<const Layout> _layout;
        Runtime::Array<real> _values;

    public:
        class ReadAccess
        {
            std::shared_ptr<const Layout> _layout;
            Runtime::Array<real>::ReadLease _lease;

        public:
            ReadAccess(std::shared_ptr<const Layout> layout, Runtime::Array<real>::ReadLease lease)
                : _layout(std::move(layout)), _lease(std::move(lease)) {}
            auto View() const
            {
                return BlockMatrixView<const real>(_lease.View<Runtime::TableRows>(), _layout->offsets.data(),
                                                   index(_layout->offsets.size()) - 1, _layout->blockSize);
            }
        };
        class WriteAccess
        {
            std::shared_ptr<const Layout> _layout;
            Runtime::Array<real>::WriteLease _lease;

        public:
            WriteAccess(std::shared_ptr<const Layout> layout, Runtime::Array<real>::WriteLease lease)
                : _layout(std::move(layout)), _lease(std::move(lease)) {}
            auto View() const
            {
                return BlockMatrixView<real>(_lease.View<Runtime::TableRows>(), _layout->offsets.data(),
                                             index(_layout->offsets.size()) - 1, _layout->blockSize);
            }
        };

        BlockMatrixStorage(const std::vector<index> &blocksPerRow, index blockSize)
        {
            DNDS_check_throw_info(blockSize > 0 && blockSize <= std::numeric_limits<rowsize>::max() / blockSize,
                                  "invalid dense block dimension");
            auto layout = std::make_shared<Layout>();
            layout->blockSize = blockSize;
            layout->offsets.reserve(blocksPerRow.size() + 1);
            layout->offsets.push_back(0);
            for (auto count : blocksPerRow)
            {
                DNDS_check_throw_info(count >= 0 && count <= std::numeric_limits<index>::max() - layout->offsets.back(),
                                      "invalid block row length or total block overflow");
                layout->offsets.push_back(layout->offsets.back() + count);
            }
            _values = Runtime::Array<real>(Runtime::RowLayout::Dense(layout->offsets.back(), blockSize * blockSize));
            _layout = std::move(layout);
        }
        index RowNumber() const { return index(_layout->offsets.size()) - 1; }
        index BlockSize() const { return _layout->blockSize; }
        const std::vector<index> &Offsets() const { return _layout->offsets; }
        const Runtime::Array<real> &Values() const { return _values; }
        ReadAccess Read() const
        {
            DNDS_check_throw_info(_values.GetDevice() == Runtime::Device::Host(), "CPU block matrix view requires host storage");
            return {_layout, _values.Read()};
        }
        WriteAccess Write()
        {
            DNDS_check_throw_info(_values.GetDevice() == Runtime::Device::Host(), "CPU block matrix view requires host storage");
            return {_layout, _values.Write()};
        }
    };
}
