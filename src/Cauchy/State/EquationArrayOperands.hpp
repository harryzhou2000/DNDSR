#pragma once

#include "Cauchy/State/EquationViews.hpp"
#include "DNDS/Runtime/Execution/EagerArrayBatch.hpp"

namespace DNDS::Cauchy::EquationArrayOperands
{
    namespace R = Runtime;
    using Read = R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows>;
    using Write = R::ArrayView<real, R::AccessBoundsChecks, R::UniformTableRows>;
    using Inputs = std::vector<R::ArrayOperand<real, false, R::UniformTableRows>>;
    using Outputs = std::vector<R::ArrayOperand<real, true, R::UniformTableRows>>;
    /// Host component-scale binding and a shallow schema/allocation handle.
    /// No field payload or device pointer table is copied. Batch operands own
    /// the access contract; the alias only supplies immutable layout metadata.
    struct Binding
    {
        EquationDofs state;
        struct Column
        {
            index slot, component;
        };
        R::MetadataBuffer<Column> columns;
        index rows = 0, firstSlot = 0;
        struct ReadBlocks
        {
            const Binding &binding;
            SpanView<const Read> arrays;
            EquationBlockReadView Block(index iBlock) const
            {
                const auto location = binding.state.LocateBlock(iBlock);
                return {{arrays[binding.firstSlot + location.physicalField], location.firstComponent, location.componentNumber},
                        {{}, location.firstComponent, location.componentNumber}};
            }
        };
        struct WriteBlocks
        {
            const Binding &binding;
            SpanView<const Write> arrays;
            EquationBlockArrayView<real> Block(index iBlock) const
            {
                const auto location = binding.state.LocateBlock(iBlock);
                return {arrays[binding.firstSlot + location.physicalField], location.firstComponent, location.componentNumber};
            }
        };
        auto Layout(SpanView<const Read> arrays) const { return EquationViewLayout<const real>(state, ReadBlocks{*this, arrays}); }
        auto Layout(SpanView<const Write> arrays) const { return EquationViewLayout<real>(state, WriteBlocks{*this, arrays}); }
        template <class T>
        auto Components(SpanView<const R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows>> arrays) const
        {
            std::vector<Algebra::DofComponentColumn<T>> views;
            views.reserve(columns.size());
            for (const auto column : columns.HostView())
                views.push_back({arrays[column.slot], {}, column.component});
            return Algebra::DofComponentLayout<T>(std::move(views), rows, 0);
        }
    };
    inline Binding Describe(const EquationDofs &state, index slot)
    {
        std::vector<Binding::Column> columns;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            const auto location = state.LocateBlock(iBlock);
            for (index j = 0; j < location.componentNumber; ++j)
                columns.push_back({slot + location.physicalField, location.firstComponent + j});
        }
        return {state.NumericAlias(), R::MetadataBuffer<Binding::Column>(std::move(columns)), index(state.Space().OwnedRows()), slot};
    }
    inline Binding AddInput(const EquationDofs &state, Inputs &inputs)
    {
        const auto binding = Describe(state, index(inputs.size()));
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            inputs.push_back(R::ReadAs<R::UniformTableRows>(state.PhysicalField(iField).Owned()));
        return binding;
    }
    inline Binding AddOutput(EquationDofs &state, Outputs &outputs, bool readWrite = false)
    {
        const auto binding = Describe(state, index(outputs.size()));
        for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        {
            auto &array = state.PhysicalField(iField).Owned();
            outputs.push_back(readWrite ? R::ReadWriteAs<R::UniformTableRows>(array) : R::WriteAs<R::UniformTableRows>(array));
        }
        return binding;
    }
    template <class T>
    auto Table(R::ArrayView<T, R::AccessBoundsChecks, R::UniformTableRows> view)
    {
        DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table, "nonlinear CUDA field requires AoS");
        return R::ArrayView<T, R::AccessBoundsChecks, R::TableRows>{view.data, nullptr, view.rows,
                                                                    view.width, view.stride, nullptr, view.kind, view.device};
    }
}
