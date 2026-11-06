#pragma once

#include "Cauchy/State/EquationDofs.hpp"

namespace DNDS::Cauchy
{
    enum class EquationViewKind
    {
        Generic,
        SeparateFlow5,
        Combined,
        CombinedAoS,
        SeparateFlow5AoS
    };

    /// Borrowed whole-equation view. The layout owner and every original access
    /// scope must outlive use. Device execution also needs device-resident block
    /// descriptors/offsets. No ownership, schema lookup or lease operation occurs
    /// in row access; total width never determines physical equation semantics.
    /// CUDA plan: dispatch dedicated expected-layout/physics kernels with
    /// direct component arguments or statically composed inline segment views.
    /// Device-callability here is not a requirement to upload or interpret the
    /// generic CPU descriptor registry in production GPU kernels.
    template <class T, EquationViewKind Kind = EquationViewKind::Generic, size_t Width = Runtime::DynamicExtent>
    struct EquationBlockedView
    {
        using value_type = T;
        static constexpr EquationViewKind kind = Kind;
        static constexpr bool combined = Kind == EquationViewKind::Combined || Kind == EquationViewKind::CombinedAoS;
        static constexpr bool table = Kind == EquationViewKind::CombinedAoS;
        static constexpr size_t row_width = Width;
        static constexpr size_t flow_width = Kind == EquationViewKind::SeparateFlow5 || Kind == EquationViewKind::SeparateFlow5AoS || table ? 5 : Runtime::DynamicExtent;
        using ParentPolicy = std::conditional_t<table, Runtime::TableRows, Runtime::UniformTableRows>;
        using FlowPolicy = std::conditional_t<table || Kind == EquationViewKind::SeparateFlow5AoS, Runtime::TableRows, Runtime::UniformTableRows>;
        using WholeView = EquationBlockArrayView<T, Runtime::AccessBoundsChecks, Width, ParentPolicy>;
        using FlowView = EquationBlockArrayView<T, Runtime::AccessBoundsChecks, flow_width, FlowPolicy>;
        static constexpr bool has_single_parent = combined, has_static_layout = table;
        static constexpr size_t component_stride = combined ? WholeView::component_stride : Runtime::DynamicExtent;
        static constexpr size_t row_stride = combined ? WholeView::row_stride : Runtime::DynamicExtent;
        const EquationBlockArrayView<T> *blocks = nullptr;
        const index *offsets = nullptr;
        index blockNumber = 0, rows = 0, components = 0, rans = -1, species = -1, flowFirst = -1;
        WholeView whole;
        FlowView flow;
        DNDS_DEVICE_CALLABLE index RowCount() const { return rows; }
        DNDS_DEVICE_CALLABLE index ComponentNumber() const { return Width == Runtime::DynamicExtent ? components : index(Width); }
        DNDS_DEVICE_CALLABLE index RowSize(index) const { return ComponentNumber(); }
        DNDS_DEVICE_CALLABLE T &operator()(index iRow, index j) const
        {
            if constexpr (Runtime::AccessBoundsChecks)
                DNDS_HD_assert(iRow >= 0 && iRow < rows && j >= 0 && j < ComponentNumber());
            if constexpr (combined)
                return whole(iRow, j);
            else
            {
                if constexpr (flow_width == 5)
                    if (j >= flowFirst && j < flowFirst + 5)
                        return flow(iRow, j - flowFirst);
                index iBlock = 0;
                while (j >= offsets[iBlock + 1])
                    ++iBlock;
                return blocks[iBlock](iRow, j - offsets[iBlock]);
            }
        }
        struct Row;
        DNDS_DEVICE_CALLABLE Row operator[](index iRow) const;
    };

    template <class T, EquationViewKind Kind, size_t Width>
    struct EquationBlockedView<T, Kind, Width>::Row
    {
        using value_type = T;
        static constexpr EquationViewKind kind = Kind;
        static constexpr bool has_single_parent = combined, has_static_layout = table;
        static constexpr size_t row_width = Width, component_stride = EquationBlockedView::component_stride;
        EquationBlockedView view;
        index iRow;
        DNDS_DEVICE_CALLABLE index size() const { return view.ComponentNumber(); }
        DNDS_DEVICE_CALLABLE T &operator[](index j) const { return view(iRow, j); }
        /// A split logical row has no single raw pointer. Request Flow(),
        /// RANS(), Species() or Block() and retain that segment's stride.
        /// A combined raw row still needs ElementStride(), including SoA and
        /// tiled layouts; data()+j is valid only when that stride is one.
        template <bool C = combined, std::enable_if_t<C && combined, int> = 0>
        DNDS_DEVICE_CALLABLE T *data() const { return view.whole[iRow].data(); }
        template <bool C = combined, std::enable_if_t<C && combined, int> = 0>
        DNDS_DEVICE_CALLABLE index ElementStride() const { return view.whole[iRow].ElementStride(); }
        DNDS_DEVICE_CALLABLE auto Flow() const
        {
            if constexpr (flow_width == Runtime::DynamicExtent)
                if (view.flowFirst < 0)
                    return typename decltype(view.flow)::RowType{nullptr, 0};
            return view.flow[iRow];
        }
        DNDS_DEVICE_CALLABLE auto Block(index iBlock) const
        {
            return iBlock < 0 ? Runtime::RowSpan<T>{nullptr, 0} : view.blocks[iBlock][iRow];
        }
        DNDS_DEVICE_CALLABLE auto RANS() const { return Block(view.rans); }
        DNDS_DEVICE_CALLABLE auto Species() const { return Block(view.species); }
    };
    template <class T, EquationViewKind Kind, size_t Width>
    DNDS_DEVICE_CALLABLE typename EquationBlockedView<T, Kind, Width>::Row
    EquationBlockedView<T, Kind, Width>::operator[](index iRow) const { return {*this, iRow}; }

    /// Host metadata owner, prepared once from an already acquired scope. This
    /// owns descriptors only, never payloads or leases. Moving it invalidates
    /// existing borrowed views. Select owned and ghost parts independently.
    template <class T>
    class EquationViewLayout
    {
        std::vector<EquationBlockArrayView<T>> _blocks;
        std::vector<index> _offsets{0};
        index _flow = -1, _rans = -1, _species = -1;
        bool _combined = false;

    public:
        template <class Scope>
        EquationViewLayout(const EquationDofs &state, const Scope &scope, bool ghost = false)
        {
            _combined = state.PhysicalFieldNumber() == 1;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
            {
                const auto &name = state.Schema().blocks[iBlock].name;
                if (name == "flow")
                    _flow = iBlock;
                if (name == "rans")
                    _rans = iBlock;
                if (name == "species")
                    _species = iBlock;
                if constexpr (std::is_const_v<T>)
                {
                    const auto part = scope.Block(iBlock);
                    _blocks.push_back(ghost ? part.ghost : part.owned);
                }
                else
                {
                    DNDS_check_throw_info(!ghost, "write equation view has no ghost part");
                    _blocks.push_back(scope.Block(iBlock));
                }
                const auto location = state.LocateBlock(iBlock);
                _combined = _combined && location.firstComponent == _offsets.back();
                _offsets.push_back(_offsets.back() + location.componentNumber);
            }
        }
        bool Combined() const { return _combined; }
        bool Flow5() const { return _flow >= 0 && _blocks[_flow].ComponentNumber() == 5; }
        bool FlowTable() const { return Flow5() && _blocks[_flow].HasTableLayout(); }
        bool Table() const { return _combined && _blocks[0].HasTableLayout(); }
        index ComponentNumber() const { return _offsets.back(); }
        template <EquationViewKind Kind = EquationViewKind::Generic, size_t Width = Runtime::DynamicExtent>
        auto View() const
        {
            using V = EquationBlockedView<T, Kind, Width>;
            DNDS_check_throw_info(!V::combined || _combined, "combined view requires ordered single-parent storage");
            DNDS_check_throw_info(!V::table || Table(), "combined AoS view requires table storage");
            DNDS_check_throw_info(V::flow_width != 5 || Flow5(), "static flow view requires flow5 block");
            DNDS_check_throw_info(!(V::table || Kind == EquationViewKind::SeparateFlow5AoS) || FlowTable(), "static AoS flow requires table storage");
            DNDS_check_throw_info(Width == Runtime::DynamicExtent || index(Width) == ComponentNumber(), "whole-equation static width differs");
            V view;
            view.blocks = _blocks.data();
            view.offsets = _offsets.data();
            view.blockNumber = index(_blocks.size());
            view.rows = _blocks[0].RowCount();
            view.components = ComponentNumber();
            view.rans = _rans;
            view.species = _species;
            const auto parent = _blocks[0].parent;
            view.whole = {{parent.data, parent.offsets, parent.rows, parent.width, parent.stride, parent.sizes, parent.kind}, 0, ComponentNumber()};
            if (_flow >= 0)
            {
                const auto block = _blocks[_flow];
                view.flow = {{block.parent.data, block.parent.offsets, block.parent.rows, block.parent.width, block.parent.stride, block.parent.sizes, block.parent.kind}, block.first, block.components};
                view.flowFirst = _offsets[_flow];
            }
            return view;
        }
    };

    /// Explicit operator-selected dispatch, never an automatic kernel policy.
    /// Each operator chooses supported families in its own preparation code.
    /// Joint construction prevents input/output specialization products.
    /// CPU placement, schema compatibility, row spaces and nonaliasing remain
    /// operator-entry responsibilities. Callback return types must agree.
    template <EquationViewKind Kind = EquationViewKind::Generic, size_t Width = Runtime::DynamicExtent,
              class Kernel, class T, class... Other>
    decltype(auto) DispatchCPUEquationViews(Kernel &&kernel, const EquationViewLayout<T> &first, const EquationViewLayout<Other> &...other)
    {
        DNDS_check_throw_info(((first.ComponentNumber() == other.ComponentNumber()) && ...), "equation view widths differ");
        return std::forward<Kernel>(kernel)(first.template View<Kind, Width>(), other.template View<Kind, Width>()...);
    }
}
