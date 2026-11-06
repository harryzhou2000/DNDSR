#pragma once

#include "Cauchy/Physics/EquationSchema.hpp"
#include "DNDS/Algebra/DofBundle.hpp"
#include "DNDS/Algebra/DofComponentView.hpp"
#include <array>
#include <utility>

namespace DNDS::Cauchy
{
    /// One logical equation block inside a physical field. Width and logical
    /// ordering come from EquationSchema, not from the physical storage name.
    struct EquationDofBinding
    {
        std::string block, field;
        index first = 0;
    };

    /// Host dispatch metadata, resolved without guessing allocation grouping
    /// from pointers. The physical field retains its complete layout/pitch.
    struct EquationBlockLocation
    {
        index physicalField, firstComponent, componentNumber;
    };

    /// Borrowed component range. Preserve the complete parent layout: changing
    /// its width would change the tile pitch of an AoSoA array incorrectly.
    /// Resolve segments outside kernels; these plain views retain no owner.
    template <class T, bool Checked = Runtime::AccessBoundsChecks, size_t Width = Runtime::DynamicExtent,
              class ParentPolicy = Runtime::UniformTableRows>
    struct EquationBlockArrayView
    {
        using ParentView = Runtime::ArrayView<T, Checked, ParentPolicy>;
        using value_type = T;
        static constexpr size_t row_width = Width, component_stride = ParentView::component_stride;
        static constexpr size_t row_stride = ParentView::row_stride;
        static constexpr bool is_dynamic = ParentView::is_dynamic;
        /// layout_kind is a compile-time guarantee only when !is_dynamic.
        static constexpr Runtime::RowLayoutKind layout_kind = ParentView::layout_kind;
        using RowType = Runtime::RowSpan<T, Checked, Width, Runtime::DynamicExtent, component_stride>;
        ParentView parent;
        index first = 0, components = 0;
        DNDS_DEVICE_CALLABLE index RowCount() const { return parent.RowCount(); }
        DNDS_DEVICE_CALLABLE index ComponentNumber() const
        {
            if constexpr (Width != Runtime::DynamicExtent)
                return index(Width);
            else
                return components;
        }
        DNDS_DEVICE_CALLABLE index RowSize(index iRow) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(iRow >= 0 && iRow < RowCount());
            return ComponentNumber();
        }
        DNDS_DEVICE_CALLABLE T &operator()(index iRow, index iComponent) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(iComponent >= 0 && iComponent < ComponentNumber());
            return parent(iRow, first + iComponent);
        }
        DNDS_DEVICE_CALLABLE RowType operator[](index iRow) const
        {
            const auto row = parent[iRow];
            return {row.data() + first * row.ElementStride(), ComponentNumber(), row.ElementStride()};
        }
        template <size_t FixedWidth>
        EquationBlockArrayView<T, Checked, FixedWidth, ParentPolicy> Fixed() const
        {
            DNDS_check_throw_info(ComponentNumber() == index(FixedWidth), "equation block static width differs");
            return {parent, first, components};
        }
        /// Prologue query/projection only; neither acquires access nor changes
        /// storage. The enclosing scope must retain the parent through use.
        bool HasTableLayout() const
        {
            return parent.kind == Runtime::RowLayoutKind::Table && !parent.offsets && !parent.sizes;
        }
        EquationBlockArrayView<T, Checked, Width, Runtime::TableRows> AsTable() const
        {
            DNDS_check_throw_info(HasTableLayout(), "equation block needs AoS table storage");
            // Keep the complete parent width/pitch, including component ranges
            // within wider coupled rows. An absent ghost has no parent rows.
            return {{parent.data, nullptr, parent.rows, parent.width, parent.stride, nullptr, parent.kind}, first, components};
        }
    };

    template <size_t Width = Runtime::DynamicExtent, class ParentPolicy = Runtime::UniformTableRows>
    struct EquationBlockPairReadView
    {
        EquationBlockArrayView<const real, Runtime::AccessBoundsChecks, Width, ParentPolicy> owned, ghost;
        static constexpr size_t row_width = Width, component_stride = decltype(owned)::component_stride;
        DNDS_DEVICE_CALLABLE index RowCount() const { return owned.RowCount() + ghost.RowCount(); }
        DNDS_DEVICE_CALLABLE index ComponentNumber() const { return owned.ComponentNumber(); }
        DNDS_DEVICE_CALLABLE const real &operator()(index iRow, index iComponent) const
        {
            return iRow < owned.RowCount() ? owned(iRow, iComponent) : ghost(iRow - owned.RowCount(), iComponent);
        }
        DNDS_DEVICE_CALLABLE auto operator[](index iRow) const
        {
            return iRow < owned.RowCount() ? owned[iRow] : ghost[iRow - owned.RowCount()];
        }
        template <size_t FixedWidth>
        EquationBlockPairReadView<FixedWidth, ParentPolicy> Fixed() const
        {
            return {owned.template Fixed<FixedWidth>(), ghost.template Fixed<FixedWidth>()};
        }
        bool HasTableLayout() const { return owned.HasTableLayout() && ghost.HasTableLayout(); }
        EquationBlockPairReadView<Width, Runtime::TableRows> AsTable() const { return {owned.AsTable(), ghost.AsTable()}; }
    };
    using EquationBlockReadView = EquationBlockPairReadView<>;

    /// Semantic state binding over native fields, with no payload repacking.
    /// Copies share allocations and immutable binding metadata. Clone copies
    /// owned values; callers refresh ghosts afterwards. Replacing allocations,
    /// row shapes or row order requires an explicit new binding.
    ///
    /// This is not a physical-model implementation. Domain/FV provide explicit
    /// perfect-gas and ideal-mixture CPU profiles. Other schemas still need a
    /// matching physical operator; extra fields do not imply model support.
    class EquationDofs
    {
        struct Definition;
        struct ArrayPin
        {
            Runtime::ArrayStorage::Shared storage;
            uint64_t identity = 0, shape = 0;
        };
        std::shared_ptr<const Definition> _definition;
        Algebra::DofBundle _fields;
        std::vector<std::array<ArrayPin, 2>> _pins;
        void PinFields();
        EquationDofs(std::shared_ptr<const Definition> definition, Algebra::DofBundle fields);

    public:
        /// Every logical block is mapped exactly once. Physical components must
        /// be covered without overlap or gaps. Fields use one row space/device.
        EquationDofs(EquationSchema schema, Algebra::DofBundle fields, std::vector<EquationDofBinding> bindings);
        static EquationDofs Coupled(EquationSchema schema, Algebra::DofField<real> field, std::string name = "flow");
        static EquationDofs Separate(EquationSchema schema, Algebra::DofBundle fields);
        const EquationSchema &Schema() const;
        index ComponentNumber() const;
        index BlockNumber() const;
        const Algebra::DofSpace &Space() const { return _fields.Space(); }
        const Algebra::DofBundle &Fields() const { return _fields; }
        index PhysicalFieldNumber() const { return _fields.BlockNumber(); }
        /// Validate once at operator entry, then inspect PhysicalField(...)
        /// for parent layout/device and acquire its normal access scope. This
        /// metadata query grants no pointer access or asynchronous permission.
        EquationBlockLocation LocateBlock(index iBlock) const;
        EquationBlockLocation LocateBlock(const std::string &block) const;
        /// Communication/IO adapters access a physical field explicitly. Value
        /// changes are allowed; replacement handles invalidate this binding.
        Algebra::DofField<real> &PhysicalField(index iField);
        const Algebra::DofField<real> &PhysicalField(index iField) const;
        Runtime::Device Device() const;
        void Validate() const;
        /// Exact semantic comparison outside kernels; component count alone
        /// cannot distinguish flow4+one scalar from compressible flow5.
        void RequireSchema(const EquationSchema &schema) const;
        void RequireCompatible(const EquationDofs &other) const;
        /// Current physical FV/nonlinear/linear adapter: one array in logical order.
        /// Refuses split/permuted storage rather than silently flattening it.
        Algebra::DofField<real> &CoupledField();
        const Algebra::DofField<real> &CoupledField() const;
        /// Bind a new stage field to the same coupled descriptor. This shares
        /// metadata, not payload, and never copies or flattens its values.
        EquationDofs RebindCoupled(Algebra::DofField<real> field) const;
        /// Bind stage/workspace fields to the same immutable logical map.
        /// Names/order, widths, device and DofSpace must match; per-field physical
        /// layouts may differ. Shares metadata/handles, never flattens/copies data.
        EquationDofs RebindFields(Algebra::DofBundle fields) const;
        EquationDofs Clone() const;
        /// Shallow payload/schema alias retaining numeric row identity only.
        /// Use for worker-owned staged scopes; the controller must retain its
        /// separate communicator/transfer owners. No values or layouts change.
        EquationDofs NumericAlias() const;

        class ReadScope
        {
            friend class EquationDofs;
            std::shared_ptr<const Definition> _definition;
            std::vector<Runtime::Array<real>::ReadLease> _owned, _ghost;
            ReadScope(const EquationDofs &state, bool ghosts, Runtime::ExecutionContext context);

        public:
            ReadScope(ReadScope &&) noexcept = default;
            ReadScope &operator=(ReadScope &&) noexcept = default;
            ReadScope(const ReadScope &) = delete;
            ReadScope &operator=(const ReadScope &) = delete;
            EquationBlockReadView Block(index iBlock) const;
            EquationBlockReadView Block(const std::string &name) const;
            /// Check a known logical block width once, outside point loops.
            /// This fixes access shape only; parent layout/grouping is unchanged.
            template <size_t Width>
            auto Block(index iBlock) const { return Block(iBlock).template Fixed<Width>(); }
            template <size_t Width>
            auto Block(const std::string &name) const { return Block(name).template Fixed<Width>(); }
            /// Resolve logical component metadata once for coupled dense-row
            /// kernels. Retain this scope and the returned host layout owner.
            /// This is not a device upload or a payload packing operation.
            Algebra::DofComponentLayout<const real> ComponentLayout() const;
            /// Attach one enclosing execution-prefix completion to all leases;
            /// caller records the correct stream after their last use. CPU
            /// workers instead retain this scope until execution returns.
            void Complete(const Runtime::Completion &completion);
        };
        class WriteScope
        {
            friend class EquationDofs;
            std::shared_ptr<const Definition> _definition;
            std::vector<Runtime::Array<real>::WriteLease> _owned;
            WriteScope(EquationDofs &state, Runtime::ExecutionContext context);

        public:
            WriteScope(WriteScope &&) noexcept = default;
            WriteScope &operator=(WriteScope &&) noexcept = default;
            WriteScope(const WriteScope &) = delete;
            WriteScope &operator=(const WriteScope &) = delete;
            EquationBlockArrayView<real> Block(index iBlock) const;
            EquationBlockArrayView<real> Block(const std::string &name) const;
            template <size_t Width>
            auto Block(index iBlock) const { return Block(iBlock).template Fixed<Width>(); }
            template <size_t Width>
            auto Block(const std::string &name) const { return Block(name).template Fixed<Width>(); }
            Algebra::DofComponentLayout<real> ComponentLayout() const;
            void Complete(const Runtime::Completion &completion);
            /// Only before any values have changed, as for native Array leases.
            void CancelUnmodified();
            /// Mark partially written output unusable on an abandoned staged
            /// operator. Every physical output lease is retired consistently.
            void Fail();
        };
        /// Pointwise algebra/reductions stay unsplit and owned-only. Include
        /// ghosts only for a stencil after its halo has become ready. Scopes
        /// retain allocations/metadata; copied plain views do not.
        ReadScope ReadOwned(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const;
        ReadScope ReadWithGhost(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host()) const;
        /// One writer per physical owned field, including when several logical
        /// segments share it. Reservations fail before any local output changes.
        WriteScope WriteOwned(Runtime::ExecutionContext context = Runtime::ExecutionContext::Host());
    };
}
