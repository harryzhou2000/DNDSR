#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Runtime/Storage/Metadata.hpp"
#include <vector>
#include <optional>

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
        SpanView<const DofComponentColumn<T>> columns;
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
    /// Recognize a complete ordered AoS parent through logical columns once,
    /// outside numeric loops. No packing or ownership change: the caller must
    /// retain the original column table and payload leases. Split, permuted,
    /// subset and column-major layouts keep the generic component-view path.
    /// Select exactly one entity part; mixed owned/ghost extents are refused.
    template <class T>
    std::optional<Runtime::ArrayView<T, Runtime::AccessBoundsChecks, Runtime::TableRows>>
    TryTablePart(DofComponentView<T> view, bool ghost)
    {
        if (view.columns.empty() || view.components <= 0 || (ghost ? view.ownedRows : view.ghostRows))
            return std::nullopt;
        const auto first = ghost ? view.columns[0].ghost : view.columns[0].owned;
        if (first.kind != Runtime::RowLayoutKind::Table || first.width != view.components ||
            first.rows != (ghost ? view.ghostRows : view.ownedRows) || first.offsets || first.sizes)
            return std::nullopt;
        for (index j = 0; j < view.components; ++j)
        {
            const auto &column = view.columns[j];
            const auto parent = ghost ? column.ghost : column.owned;
            if (column.component != j || parent.data != first.data || parent.rows != first.rows ||
                parent.width != first.width || parent.stride != first.stride || parent.kind != first.kind ||
                parent.offsets || parent.sizes || parent.device != first.device)
                return std::nullopt;
        }
        return Runtime::ArrayView<T, Runtime::AccessBoundsChecks, Runtime::TableRows>{first.data, nullptr, first.rows,
                                                                                      first.width, first.stride, nullptr, first.kind, first.device};
    }
    /// Recognize only owned rows; pass OwnedPart() for paired descriptors.
    template <class T>
    auto TryOwnedTable(DofComponentView<T> view) { return TryTablePart(view, false); }
    /// Ghost-only counterpart; pass GhostPart(), retaining the enclosing
    /// read scope and its descriptor table through every stencil operation.
    template <class T>
    auto TryGhostTable(DofComponentView<T> view) { return TryTablePart(view, true); }
    /// Owns immutable backend metadata only. Payloads are borrowed from enclosing
    /// access scopes. Copies share descriptor storage; the enclosing operation
    /// retains both this owner and its payload scopes. No implicit upload,
    /// halo or synchronization. Device descriptors must contain device views.
    template <class T>
    class DofComponentLayout
    {
        Runtime::MetadataBuffer<DofComponentColumn<T>> _columns;
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
        DofComponentView<T> View() const { return {_columns.HostView(), _owned, _ghost, index(_columns.size())}; }
    };
}
