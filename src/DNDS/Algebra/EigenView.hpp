#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include <Eigen/Core>

namespace DNDS::Algebra
{
    /// Borrowed per-row column-major dense matrices. Construct once in the
    /// operator prologue while retaining the Array access lease. Scalar rows may
    /// be strided (SoA/AoSoA); no packing or allocation occurs. CPU only.
    template <class ArrayView>
    class EigenRowMatrices
    {
        ArrayView _view;
        index _matrixRows, _matrixColumns;
        using Scalar = std::remove_const_t<typename ArrayView::value_type>;
        using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
        using QualifiedMatrix = std::conditional_t<std::is_const_v<typename ArrayView::value_type>, const Matrix, Matrix>;
        static constexpr int Inner = ArrayView::component_stride == Runtime::DynamicExtent ? Eigen::Dynamic : int(ArrayView::component_stride);
        using Stride = Eigen::Stride<Eigen::Dynamic, Inner>;

    public:
        EigenRowMatrices(ArrayView view, index rows, index columns, Runtime::Device device)
            : _view(view), _matrixRows(rows), _matrixColumns(columns)
        {
            DNDS_check_throw_info(device == Runtime::Device::Host() && rows >= 0 && columns >= 0,
                                  "Eigen row matrices require host placement and nonnegative dimensions");
            const auto extent = CheckedSize::Multiply(size_t(rows), size_t(columns));
            DNDS_check_throw_info(extent <= size_t(INT64_MAX), "local matrix extent exceeds signed index range");
            // This adapter is intended for uniform tables. Variable-row callers
            // construct an explicitly checked row adapter instead of rescanning
            // CSR lengths during every matrix access.
            DNDS_check_throw_info(!ArrayView::is_dynamic || (!view.offsets && !view.sizes),
                                  "Eigen row matrices require uniform logical rows");
            DNDS_check_throw_info(!view.RowCount() || view.Width() >= index(extent), "local matrix exceeds logical row width");
        }
        auto operator[](index iRow) const
        {
            const auto row = _view[iRow];
            const auto inner = row.ElementStride();
            return Eigen::Map<QualifiedMatrix, Eigen::Unaligned, Stride>(row.data(), _matrixRows, _matrixColumns,
                                                                         Stride(_matrixRows * inner, inner));
        }
        index RowNumber() const { return _view.RowCount(); }
    };

    template <class View>
    auto MakeEigenRowMatrices(View view, index rows, index columns, Runtime::Device device = Runtime::Device::Host())
    {
        return EigenRowMatrices<View>(view, rows, columns, device);
    }
}

namespace DNDS::Algebra
{
    template <class T, bool Checked, class Policy>
    class EigenPairRowMatrices
    {
        EigenRowMatrices<Runtime::ArrayView<const T, Checked, Policy>> _owned, _ghost;
        index _ownedRows;

    public:
        EigenPairRowMatrices(Runtime::PairReadView<T, Checked, Policy> view, index rows, index columns, Runtime::Device device)
            : _owned(view.owned, rows, columns, device), _ghost(view.ghost, rows, columns, device), _ownedRows(view.owned.RowCount()) {}
        auto operator[](index iRow) const { return iRow < _ownedRows ? _owned[iRow] : _ghost[iRow - _ownedRows]; }
    };
    template <class T, bool Checked, class Policy>
    auto MakeEigenRowMatrices(Runtime::PairReadView<T, Checked, Policy> view, index rows, index columns,
                              Runtime::Device device = Runtime::Device::Host())
    {
        return EigenPairRowMatrices<T, Checked, Policy>(view, rows, columns, device);
    }
}
