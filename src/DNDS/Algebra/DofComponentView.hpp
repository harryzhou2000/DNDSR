#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include <vector>

namespace DNDS::Algebra
{
    /// One logical component in a physical parent. Keep the complete parent
    /// width/pitch: reducing it would corrupt tiled-column-major indexing.
    template <class T>
    struct DofComponentColumn
    {
        Runtime::ArrayView<T, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> owned, ghost;
        index component;
    };
    /// Borrowed logical vector rows across several physical arrays. No payload
    /// copy, reference count or lease lookup in scalar access. The descriptor
    /// table and all payload/shape scopes must outlive use. A device launch also
    /// needs that table in device-accessible memory; a host table is not enough.
    template <class T>
    struct DofComponentView
    {
        const DofComponentColumn<T> *columns = nullptr;
        index ownedRows = 0, ghostRows = 0, components = 0;
        DNDS_DEVICE_CALLABLE index RowCount() const { return ownedRows + ghostRows; }
        DNDS_DEVICE_CALLABLE index ComponentNumber() const { return components; }
        DNDS_DEVICE_CALLABLE DofComponentView OwnedPart() const { return {columns, ownedRows, 0, components}; }
        DNDS_DEVICE_CALLABLE DofComponentView GhostPart() const { return {columns, 0, ghostRows, components}; }
        DNDS_DEVICE_CALLABLE T &operator()(index iRow, index j) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_HD_assert(iRow >= 0 && iRow < RowCount() && j >= 0 && j < components);
            const auto &column = columns[j];
            return iRow < ownedRows ? column.owned(iRow, column.component) : column.ghost(iRow - ownedRows, column.component);
        }
    };
    /// Owns host descriptor metadata only. Payloads are borrowed from enclosing
    /// access scopes. Build once outside numerical loops, then freeze View().
    /// Moving/destroying this owner invalidates borrowed views; obtain a new
    /// view from its final location. No implicit upload, halo or synchronization.
    template <class T>
    class DofComponentLayout
    {
        std::vector<DofComponentColumn<T>> _columns;
        index _owned, _ghost;

    public:
        DofComponentLayout(std::vector<DofComponentColumn<T>> columns, index owned, index ghost)
            : _columns(std::move(columns)), _owned(owned), _ghost(ghost)
        {
            DNDS_check_throw_info(owned >= 0 && ghost >= 0 && owned <= INT64_MAX - ghost && !_columns.empty(), "invalid DOF component layout");
            for (const auto &column : _columns)
                DNDS_check_throw_info(column.owned.RowCount() == owned && column.ghost.RowCount() == ghost &&
                                          column.component >= 0 && column.component < column.owned.Width() &&
                                          (!ghost || column.component < column.ghost.Width()),
                                      "DOF component parent extent differs");
        }
        DofComponentView<T> View() const { return {_columns.data(), _owned, _ghost, index(_columns.size())}; }
    };
}
