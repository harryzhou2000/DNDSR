#pragma once

#include "DNDS/Runtime/Array/Array.hpp"

namespace DNDS::Cauchy
{
    /// Plain sparse-row and logical-scan input. Every index refers to the same
    /// owned+ghost row space; no communicator, allocation or lease lives here.
    struct LinearPatternView
    {
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::CSRRows> columns;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> order, position, core, dependent;
        index ownedRows = 0;
    };

    /** @brief Canonical rank-local symbolic fields for matrix actions/scans.
     * @details Variable-length columns use a CSR Array. Logical order and
     * inverse positions are one-column Arrays. Setup vectors are discarded
     * after publication; numerical callers acquire one read scope per operation.
     */
    class LinearPatternStorage
    {
        Runtime::Array<index> _columns{Runtime::RowLayout::Ragged({})};
        Runtime::Array<index> _order{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _position{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _core{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<index> _dependent{Runtime::RowLayout::Dense(0, 1)};

    public:
        class ReadAccess
        {
            Runtime::Array<index>::ReadLease _columns, _order, _position, _core, _dependent;

        public:
            explicit ReadAccess(const LinearPatternStorage &storage)
                : _columns(storage._columns.Read()), _order(storage._order.Read()),
                  _position(storage._position.Read()), _core(storage._core.Read()),
                  _dependent(storage._dependent.Read()) {}
            LinearPatternView View() const
            {
                return {_columns.View<Runtime::CSRRows>(), _order.View<Runtime::FixedRows<1>>(),
                        _position.View<Runtime::FixedRows<1>>(), _core.View<Runtime::FixedRows<1>>(),
                        _dependent.View<Runtime::FixedRows<1>>(), _columns.Layout().RowCount()};
            }
        };
        LinearPatternStorage() = default;
        LinearPatternStorage(const std::vector<std::vector<index>> &columns,
                             const std::vector<index> &order, const std::vector<index> &position,
                             const std::vector<index> &core, const std::vector<index> &dependent);
        ReadAccess Read() const { return ReadAccess(*this); }
        /// Setup/diagnostic snapshot. Numerical operators use Read(), never a
        /// per-iteration host reconstruction of the symbolic pattern.
        std::vector<std::vector<index>> CopyColumns() const;
        std::vector<index> CopyOrder() const;
        std::vector<index> CopyCore() const;
        std::vector<index> CopyDependent() const;
    };
}
