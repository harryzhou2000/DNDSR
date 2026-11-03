#pragma once

#include "Cauchy/Physics/EquationSchema.hpp"
#include "DNDS/Algebra/DofBundle.hpp"
#include <array>

namespace DNDS::Cauchy
{
    /// One logical equation block inside a physical field. Width and logical
    /// ordering come from EquationSchema, not from the physical storage name.
    struct EquationDofBinding
    {
        std::string block, field;
        index first = 0;
    };

    /// Borrowed component range. Preserve the complete parent layout: changing
    /// its width would change the tile pitch of an AoSoA array incorrectly.
    /// Resolve segments outside kernels; these plain views retain no owner.
    template <class T, bool Checked = Runtime::AccessBoundsChecks, size_t Width = Runtime::DynamicExtent>
    struct EquationBlockArrayView
    {
        Runtime::ArrayView<T, Checked, Runtime::UniformTableRows> parent;
        index first = 0, components = 0;
        DNDS_DEVICE_CALLABLE index RowCount() const { return parent.RowCount(); }
        DNDS_DEVICE_CALLABLE index ComponentNumber() const
        {
            if constexpr (Width != Runtime::DynamicExtent)
                return index(Width);
            else
                return components;
        }
        DNDS_DEVICE_CALLABLE T &operator()(index iRow, index iComponent) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(iComponent >= 0 && iComponent < ComponentNumber());
            return parent(iRow, first + iComponent);
        }
        DNDS_DEVICE_CALLABLE Runtime::RowSpan<T, Checked, Width> operator[](index iRow) const
        {
            const auto row = parent[iRow];
            return {row.data() + first * row.ElementStride(), ComponentNumber(), row.ElementStride()};
        }
        template <size_t FixedWidth>
        EquationBlockArrayView<T, Checked, FixedWidth> Fixed() const
        {
            DNDS_check_throw_info(ComponentNumber() == index(FixedWidth), "equation block static width differs");
            return {parent, first, components};
        }
    };

    struct EquationBlockReadView
    {
        EquationBlockArrayView<const real> owned, ghost;
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
    };

    /// Semantic state binding over native fields, with no payload repacking.
    /// Copies share allocations and immutable binding metadata. Clone copies
    /// owned values; callers refresh ghosts afterwards. Replacing allocations,
    /// row shapes or row order requires an explicit new binding.
    ///
    /// This is not a physical-model implementation. Current production dispatch
    /// remains compressible flow5; mock and future operators can use other schemas.
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
        /// Initial stage/linear adapter: one physical array in logical order.
        /// Refuses split/permuted storage rather than silently flattening it.
        Algebra::DofField<real> &CoupledField();
        const Algebra::DofField<real> &CoupledField() const;
        /// Bind a new stage field to the same coupled descriptor. This shares
        /// metadata, not payload, and never copies or flattens its values.
        EquationDofs RebindCoupled(Algebra::DofField<real> field) const;
        EquationDofs Clone() const;

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
            void Complete(const Runtime::Completion &completion);
            /// Only before any values have changed, as for native Array leases.
            void CancelUnmodified();
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
