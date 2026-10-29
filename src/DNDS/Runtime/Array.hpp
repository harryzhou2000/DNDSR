#pragma once

#include "Buffer.hpp"
#include "Metadata.hpp"
#include <algorithm>
#include <vector>
#include <limits>
#include <mutex>

namespace DNDS::Runtime
{
    enum class RowLayoutKind
    {
        Table,
        Padded,
        CSR
    };

    /// Immutable row shape in element units. Padding belongs to storage, not a row.
    class RowLayout
    {
        size_t _rows = 0, _width = 0, _stride = 0;
        RowLayoutKind _kind = RowLayoutKind::Table;
        MetadataBuffer<size_t> _offsets;
        MetadataBuffer<size_t> _sizes;

    public:
        RowLayout() = default;
        static RowLayout Dense(size_t rows, size_t width, size_t stride = 0)
        {
            DNDS_check_throw_info(rows <= size_t(INT64_MAX) && width <= size_t(INT64_MAX) && stride <= size_t(INT64_MAX),
                                  "row shape exceeds signed index range");
            RowLayout layout;
            layout._rows = rows;
            layout._width = width;
            layout._stride = stride ? stride : width;
            DNDS_check_throw_info(layout._stride >= width, "row stride is smaller than width");
            DNDS_check_throw_info(CheckedSize::Multiply(rows, layout._stride) <= size_t(INT64_MAX),
                                  "row offsets exceed signed index range");
            return layout;
        }
        static RowLayout Padded(const std::vector<size_t> &lengths, size_t capacity)
        {
            DNDS_check_throw_info(lengths.size() <= size_t(INT64_MAX) && capacity <= size_t(INT64_MAX),
                                  "row shape exceeds signed index range");
            RowLayout layout;
            layout._kind = RowLayoutKind::Padded;
            layout._rows = lengths.size();
            layout._stride = capacity;
            DNDS_check_throw_info(CheckedSize::Multiply(layout._rows, capacity) <= size_t(INT64_MAX),
                                  "row offsets exceed signed index range");
            for (auto n : lengths)
                DNDS_check_throw_info(n <= capacity, "row size exceeds padded capacity");
            layout._sizes = MetadataBuffer<size_t>(lengths);
            return layout;
        }
        static RowLayout Ragged(const std::vector<size_t> &lengths)
        {
            DNDS_check_throw_info(lengths.size() <= size_t(INT64_MAX), "row count exceeds signed index range");
            RowLayout layout;
            layout._kind = RowLayoutKind::CSR;
            layout._rows = lengths.size();
            std::vector<size_t> offsets{0};
            offsets.reserve(CheckedSize::Add(lengths.size(), size_t(1)));
            for (auto n : lengths)
            {
                DNDS_check_throw_info(n <= size_t(INT64_MAX) - offsets.back(), "row offsets exceed signed index range");
                offsets.push_back(offsets.back() + n);
            }
            layout._offsets = MetadataBuffer<size_t>(offsets);
            return layout;
        }
        index RowCount() const { return index(_rows); }
        index Rows() const { return RowCount(); }
        RowLayoutKind Kind() const { return _kind; }
        bool IsRagged() const { return _kind == RowLayoutKind::CSR; }
        bool IsPadded() const { return _kind == RowLayoutKind::Padded; }
        bool IsVariable() const { return _kind != RowLayoutKind::Table; }
        index Width() const { return index(_width); }
        index Stride() const { return index(_stride); }
        size_t Elements() const { return IsRagged() ? _offsets.back() : _rows * _stride; }
        const MetadataBuffer<size_t> &Offsets() const { return _offsets; }
        const MetadataBuffer<size_t> &Sizes() const { return _sizes; }
        index RowSize(index row) const
        {
            DNDS_check_throw_info(row >= 0 && size_t(row) < _rows, "row outside layout");
            return IsRagged() ? _offsets[row + 1] - _offsets[row] : IsPadded() ? _sizes[row]
                                                                               : _width;
        }
        index Offset(index row) const
        {
            DNDS_check_throw_info(row >= 0 && size_t(row) < _rows, "row outside layout");
            return IsRagged() ? _offsets[row] : row * _stride;
        }
        bool operator==(const RowLayout &other) const
        {
            return _kind == other._kind && _rows == other._rows && _width == other._width && _stride == other._stride &&
                   _offsets == other._offsets && _sizes == other._sizes;
        }
    };

    inline constexpr size_t DynamicExtent = std::numeric_limits<size_t>::max();

    /// Layout-specific access with runtime widths/capacities. These policies
    /// remove layout dispatch from kernels without fixing the row extent.
    struct TableRows
    {
        static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::Table;
        static constexpr bool is_dynamic = false, is_static_max = false;
        static RowLayout Default() { return RowLayout::Dense(0, 0); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind, "table row layout mismatch");
        }
    };

    struct PaddedRows
    {
        static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::Padded;
        static constexpr bool is_dynamic = false, is_static_max = false;
        static RowLayout Default() { return RowLayout::Padded({}, 0); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind, "padded row layout mismatch");
        }
    };

    struct CSRRows
    {
        static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
        static constexpr RowLayoutKind layout_kind = RowLayoutKind::CSR;
        static constexpr bool is_dynamic = false, is_static_max = false;
        static RowLayout Default() { return RowLayout::Ragged({}); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.Kind() == layout_kind, "CSR row layout mismatch");
        }
    };

    /// Fixed logical width, with either a runtime stride or a static physical
    /// stride. The shape remains shared with dynamic language and MPI adapters.
    template <size_t Width, size_t Stride = DynamicExtent>
    struct FixedRows : TableRows
    {
        static_assert(Width != DynamicExtent && (Stride == DynamicExtent || Stride >= Width));
        static constexpr size_t row_width = Width, row_capacity = Stride, row_stride = Stride;
        static constexpr bool is_static_max = false;
        static RowLayout Default() { return RowLayout::Dense(0, Width, Stride == DynamicExtent ? Width : Stride); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(!layout.IsVariable() && layout.Width() == Width, "fixed row width mismatch");
            DNDS_check_throw_info(Stride == DynamicExtent || layout.Stride() == Stride, "static row stride mismatch");
        }
    };

    /// Legacy static-max equivalent: variable logical lengths in fixed-capacity
    /// rows. Capacity is a compile-time bound and physical stride, never a width.
    template <size_t Capacity>
    struct StaticMaxRows : PaddedRows
    {
        static_assert(Capacity != DynamicExtent);
        static constexpr size_t row_width = DynamicExtent, row_capacity = Capacity, row_stride = Capacity;
        static constexpr bool is_static_max = true;
        static RowLayout Default() { return RowLayout::Padded({}, Capacity); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(layout.IsPadded() && layout.Stride() == Capacity, "static row capacity mismatch");
        }
    };

    namespace Detail
    {
        template <class Policy>
        struct RowPolicyTraits : Policy
        {
        };
        template <>
        struct RowPolicyTraits<RowLayout>
        {
            static constexpr size_t row_width = DynamicExtent, row_capacity = DynamicExtent, row_stride = DynamicExtent;
            static constexpr RowLayoutKind layout_kind = RowLayoutKind::Table;
            static constexpr bool is_dynamic = true, is_static_max = false;
        };
    }

    /// Borrowed logical row, shared by native and compatibility views. A span
    /// retains neither values nor metadata; its outer lease must remain active.
    template <class T, bool Checked = false, size_t Width = DynamicExtent, size_t Capacity = DynamicExtent>
    class RowSpan
    {
        T *_data = nullptr;
        index _size = 0;

    public:
        static constexpr size_t row_width = Width, row_capacity = Capacity;
        DNDS_DEVICE_CALLABLE RowSpan() = default;
        DNDS_DEVICE_CALLABLE RowSpan(T *data, index size) : _data(data), _size(size) {}
        DNDS_DEVICE_CALLABLE T &operator[](index column) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(column >= 0 && column < this->size());
            return _data[column];
        }
        DNDS_DEVICE_CALLABLE T *data() const { return _data; }
        DNDS_DEVICE_CALLABLE index size() const
        {
            if constexpr (Width != DynamicExtent)
                return Width;
            else
                return _size;
        }
        DNDS_DEVICE_CALLABLE bool empty() const { return this->size() == 0; }
        DNDS_DEVICE_CALLABLE T *begin() const { return _data; }
        DNDS_DEVICE_CALLABLE T *end() const { return this->size() ? _data + this->size() : _data; }
        DNDS_DEVICE_CALLABLE const T *cbegin() const { return _data; }
        DNDS_DEVICE_CALLABLE const T *cend() const { return this->size() ? _data + this->size() : _data; }
        template <size_t W = Width, size_t C = Capacity, std::enable_if_t<W != DynamicExtent || C != DynamicExtent, int> = 0>
        DNDS_DEVICE_CALLABLE operator RowSpan<T, Checked>() const { return {_data, this->size()}; }
    };

    /// Plain kernel view: no ownership, placement dispatch or tracking per element.
    /// Checked=true is a separate development policy, independent of NDEBUG.
    /// @pre Caller retains the lease/operation through actual use, validates row
    /// and component indices (including stencil indirection), and uses the correct
    /// device. An owned-only core stencil must be reclassified after mesh changes.
    /// Copying this view does not retain values or offset/size metadata.
    template <class T, bool Checked = false, class LayoutPolicy = RowLayout>
    struct ArrayView
    {
        using Policy = Detail::RowPolicyTraits<LayoutPolicy>;
        static constexpr size_t row_width = Policy::row_width, row_capacity = Policy::row_capacity, row_stride = Policy::row_stride;
        static constexpr RowLayoutKind layout_kind = Policy::layout_kind;
        static constexpr bool is_dynamic = Policy::is_dynamic;
        using RowType = RowSpan<T, Checked, row_width, row_capacity>;
        T *data = nullptr;
        const size_t *offsets = nullptr;
        index rows = 0, width = 0, stride = 0;
        const size_t *sizes = nullptr;

        DNDS_DEVICE_CALLABLE index RowCount() const { return rows; }
        DNDS_DEVICE_CALLABLE index Rows() const { return RowCount(); }
        DNDS_DEVICE_CALLABLE index RowSize(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < rows);
            if constexpr (row_width != DynamicExtent)
                return row_width;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::Table)
                return width;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::Padded)
                return sizes ? sizes[row] : width;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::CSR)
                return offsets ? offsets[row + 1] - offsets[row] : width;
            else
                return offsets ? offsets[row + 1] - offsets[row] : sizes ? sizes[row]
                                                                         : width;
        }
        DNDS_DEVICE_CALLABLE T *Row(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < rows);
            index offset;
            if constexpr (row_stride != DynamicExtent)
                offset = row * row_stride;
            else if constexpr (!is_dynamic && layout_kind != RowLayoutKind::CSR)
                offset = row * stride;
            else if constexpr (!is_dynamic && layout_kind == RowLayoutKind::CSR)
                offset = offsets ? offsets[row] : row * stride;
            else
                offset = offsets ? offsets[row] : row * stride;
            return data ? data + offset : nullptr;
        }
        DNDS_DEVICE_CALLABLE T &operator()(index row, index component) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < rows && component >= 0 && component < RowSize(row));
            return Row(row)[component];
        }
        DNDS_DEVICE_CALLABLE RowType operator[](index row) const { return {Row(row), RowSize(row)}; }
        DNDS_DEVICE_CALLABLE ArrayView<T, Checked> DynamicView() const { return {data, offsets, rows, width, stride, sizes}; }
        template <class P = LayoutPolicy, std::enable_if_t<!std::is_same_v<P, RowLayout>, int> = 0>
        DNDS_DEVICE_CALLABLE operator ArrayView<T, Checked>() const { return DynamicView(); }
    };

    namespace Detail
    {
        /// Construct a typed view with a one-time projection check. Table
        /// storage may be exposed as padded/uniform CSR; variable storage may
        /// be exposed as a table/fixed view only when every row is uniform and
        /// physically contiguous. Metadata and values remain borrowed.
        template <class T, bool Checked, class Policy>
        ArrayView<T, Checked, Policy> ProjectView(ArrayView<T, Checked> source, const RowLayout &shape)
        {
            if constexpr (std::is_same_v<Policy, RowLayout>)
                return source;
            using Traits = RowPolicyTraits<Policy>;
            const bool sourceTable = !source.offsets && !source.sizes;
            const bool sourcePadded = source.sizes;
            const bool sourceCSR = source.offsets;
            const bool keepPadded = Traits::layout_kind == RowLayoutKind::Padded && sourcePadded;
            const bool keepCSR = Traits::layout_kind == RowLayoutKind::CSR && sourceCSR;
            index width = source.width, stride = source.stride;
            const size_t *offsets = nullptr, *sizes = nullptr;
            if (keepPadded)
                sizes = source.sizes;
            else if (keepCSR)
                offsets = source.offsets;
            else if (!sourceTable)
            {
                // Projection runs on the submitting host. The view may contain
                // CUDA metadata pointers; only the retained host shape is read.
                width = shape.RowCount() ? shape.RowSize(0) : 0;
                for (index row = 0; row < source.rows; ++row)
                    DNDS_check_throw_info(shape.RowSize(row) == width &&
                                              shape.Offset(row) == row * width,
                                          "row view projection requires uniform contiguous rows");
                stride = width;
            }
            if constexpr (Traits::row_width != DynamicExtent)
                DNDS_check_throw_info(width == Traits::row_width, "fixed row width view mismatch");
            if constexpr (Traits::row_stride != DynamicExtent)
                DNDS_check_throw_info(stride == Traits::row_stride, "fixed row stride view mismatch");
            return {source.data, offsets, source.rows, width, stride, sizes};
        }
    }

    /// Layout-generic debugging/setup view over table, padded or CSR storage.
    template <class T, bool Checked = false>
    using GenericRowView = ArrayView<T, Checked, RowLayout>;

    namespace Detail
    {
        /// Published row metadata is retained on host and mirrored for device views.
        struct ArrayLayout
        {
        private:
            Buffer deviceOffsets;
            Buffer deviceSizes;

        public:
            RowLayout rows;
            uint64_t DeviceOffsetIdentity() const { return deviceOffsets.Identity(); }
            uint64_t DeviceSizeIdentity() const { return deviceSizes.Identity(); }
            template <class F>
            void VisitAllocations(F &&visit) const
            {
                rows.Offsets().VisitAllocations(visit);
                rows.Sizes().VisitAllocations(visit);
                for (const auto *buffer : {&deviceOffsets, &deviceSizes})
                    if (buffer->Identity())
                        visit(buffer->Identity(), buffer->AllocationHandle().Bytes(), buffer->GetDevice());
            }
            bool CanReleaseOrdered(ExecutionContext context) const
            {
                return deviceOffsets.CanReleaseOrdered(context) && deviceSizes.CanReleaseOrdered(context);
            }
            static void ReleaseHandle(std::shared_ptr<const ArrayLayout> &layout, ExecutionContext context)
            {
                if (!layout)
                    return;
                DNDS_check_throw_info(layout->CanReleaseOrdered(context), "metadata provider cannot release on this stream");
                auto offsets = layout->deviceOffsets, sizes = layout->deviceSizes;
                layout.reset();
                offsets.ReleaseOn(context);
                sizes.ReleaseOn(context);
            }
            explicit ArrayLayout(RowLayout shape, Device device) : rows(std::move(shape))
            {
                auto upload = [device](const MetadataBuffer<size_t> &metadata, Buffer &destination)
                {
                    auto host = metadata.Read();
                    destination = Buffer::Allocate(host.Bytes(), device);
                    auto target = destination.Write({device, nullptr});
                    CopyBytes(host.View<size_t>().data, Device::Host(), target.View<size_t>().data,
                              device, host.Bytes(), {device, nullptr})
                        .Wait();
                };
                if (rows.IsRagged() && device.space != MemorySpace::Host)
                    upload(rows.Offsets(), deviceOffsets);
                if (rows.IsPadded() && device.space != MemorySpace::Host)
                    upload(rows.Sizes(), deviceSizes);
            }
            const size_t *Offsets(Device device) const
            {
                if (!rows.IsRagged())
                    return nullptr;
                return device.space == MemorySpace::Host ? rows.Offsets().data() : static_cast<const size_t *>(deviceOffsets.LegacyPointer());
            }
            const size_t *Sizes(Device device) const
            {
                if (!rows.IsPadded())
                    return nullptr;
                return device.space == MemorySpace::Host ? rows.Sizes().data() : static_cast<const size_t *>(deviceSizes.LegacyPointer());
            }
        };
    }

    /// One immutable logical layout and its allocation, shared by typed and
    /// language bindings. Row metadata stays in element units for every handle.
    /// Buffer owns values/access state; ArrayLayout owns CPU/CUDA row offsets.
    class ExclusiveRows;
    class ArrayStorage
    {
        friend class ExclusiveRows;
        mutable Buffer _values;
        mutable std::shared_ptr<const Detail::ArrayLayout> _layout;
        size_t _elementBytes;
        mutable std::mutex _shapeMutex;
        mutable bool _editing = false, _shapeValid = true;
        mutable uint64_t _shapeVersion = 0;
        void CheckPublished() const
        {
            DNDS_check_throw_info(!_editing && _shapeValid, "Array shape is unpublished or failed");
        }

        ArrayStorage(RowLayout layout, Buffer values, size_t elementBytes)
            : _values(std::move(values)),
              _layout(std::make_shared<const Detail::ArrayLayout>(std::move(layout), _values.GetDevice())),
              _elementBytes(elementBytes) {}
        ArrayStorage(std::shared_ptr<const Detail::ArrayLayout> layout, Buffer values, size_t elementBytes)
            : _values(std::move(values)), _layout(std::move(layout)), _elementBytes(elementBytes) {}

    public:
        using Shared = std::shared_ptr<const ArrayStorage>;
        bool CanReleaseOrdered(ExecutionContext context) const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            return _values.CanReleaseOrdered(context) && _layout->CanReleaseOrdered(context);
        }
        static void ReleaseHandle(Shared &storage, ExecutionContext context)
        {
            if (!storage)
                return;
            Buffer values;
            std::shared_ptr<const Detail::ArrayLayout> layout;
            {
                std::lock_guard<std::mutex> lock(storage->_shapeMutex);
                DNDS_check_throw_info(storage->_values.CanReleaseOrdered(context) && storage->_layout->CanReleaseOrdered(context),
                                      "Array provider cannot release on this stream");
                values = storage->_values;
                layout = storage->_layout;
            }
            storage.reset();
            values.ReleaseOn(context);
            Detail::ArrayLayout::ReleaseHandle(layout, context);
        }
        static Shared Share(const Shared &source)
        {
            if (!source)
                return {};
            std::lock_guard<std::mutex> lock(source->_shapeMutex);
            source->CheckPublished();
            return source;
        }
        uint64_t ShapeVersion() const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _shapeVersion;
        }
        bool IsValid() const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            return !_editing && _shapeValid && _values.IsValid();
        }
        static Shared FromBuffer(RowLayout layout, Buffer values, size_t elementBytes)
        {
            DNDS_check_throw_info(elementBytes && values.Identity() &&
                                      values.Bytes() == CheckedSize::Multiply(layout.Elements(), elementBytes),
                                  "Array storage size/layout mismatch");
            return Shared(new ArrayStorage(std::move(layout), std::move(values), elementBytes));
        }
        static Shared Allocate(RowLayout layout, size_t elementBytes, Device device = Device::Host())
        {
            auto values = Buffer::Allocate(CheckedSize::Multiply(layout.Elements(), elementBytes), device);
            return FromBuffer(std::move(layout), std::move(values), elementBytes);
        }
        /// Independent values with exactly the same retained metadata, including
        /// CUDA offsets. Useful for SoA field bundles on a common row shape.
        Shared WithValues(Buffer values, size_t elementBytes) const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            DNDS_check_throw_info(values.Identity() && values.GetDevice() == _values.GetDevice() && elementBytes &&
                                      values.Bytes() == CheckedSize::Multiply(_layout->rows.Elements(), elementBytes),
                                  "shared metadata value size/device mismatch");
            return Shared(new ArrayStorage(_layout, std::move(values), elementBytes));
        }
        RowLayout Layout() const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _layout->rows;
        }
        index RowCount() const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _layout->rows.RowCount();
        }
        index Rows() const { return RowCount(); }
        index RowSize(index row) const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _layout->rows.RowSize(row);
        }
        std::shared_ptr<const Detail::ArrayLayout> SharedLayout() const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _layout;
        }
        Buffer Values() const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _values;
        }
        size_t ElementBytes() const { return _elementBytes; }
        /// Visit retained value/host metadata/device metadata allocations.
        /// Capacity covers the whole allocation, including imported subspans;
        /// callers deduplicate identities shared by multiple handles/layouts.
        /// This neither accesses values nor synchronizes device work.
        template <class F>
        void VisitAllocations(F &&visit) const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            if (_values.Identity())
                visit(_values.Identity(), _values.AllocationHandle().Bytes(), _values.GetDevice());
            _layout->VisitAllocations(visit);
        }
        ReadAccess Read(ExecutionContext context = ExecutionContext::Host()) const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _values.Read(context);
        }
        WriteAccess Write(ExecutionContext context = ExecutionContext::Host()) const
        {
            std::lock_guard<std::mutex> lock(_shapeMutex);
            CheckPublished();
            return _values.Write(context);
        }
        CopyOperation CopyFrom(const ArrayStorage &source, ExecutionContext context = ExecutionContext::Host()) const
        {
            auto destination = Values();
            auto input = source.Values();
            DNDS_check_throw_info(_elementBytes == source._elementBytes && Layout() == source.Layout(),
                                  "Array copy requires matching element size and layout");
            return Copy(input, destination, context);
        }
    };

    template <class T, bool Writable>
    class ArrayAccess
    {
        template <class, bool>
        friend class ArrayAccess;
        using Lease = std::conditional_t<Writable, WriteAccess, ReadAccess>;
        using Value = std::conditional_t<Writable, T, const T>;
        std::shared_ptr<const Detail::ArrayLayout> _layout;
        Lease _lease;

    public:
        ArrayAccess(Lease lease, std::shared_ptr<const Detail::ArrayLayout> layout)
            : _layout(std::move(layout)), _lease(std::move(lease))
        {
        }
        ArrayAccess(const ArrayAccess &) = delete;
        ArrayAccess &operator=(const ArrayAccess &) = delete;
        ArrayAccess(ArrayAccess &&) noexcept = default;
        ArrayAccess &operator=(ArrayAccess &&other) noexcept
        {
            if (this != &other)
            {
                _lease = std::move(other._lease); // Drains old work before dropping its layout.
                _layout = std::move(other._layout);
            }
            return *this;
        }
        bool Active() const { return _lease.Active(); }
        Device GetDevice() const { return _lease.GetDevice(); }
        const RowLayout &Layout() const { return _layout->rows; }
        template <bool Checked = false>
        ArrayView<Value, Checked> View() const
        {
            auto flat = _lease.template View<T>();
            const auto &r = _layout->rows;
            return {flat.data, _layout->Offsets(_lease.GetDevice()), r.RowCount(), r.Width(), r.Stride(), _layout->Sizes(_lease.GetDevice())};
        }
        template <class LayoutPolicy, bool Checked = false>
        ArrayView<Value, Checked, LayoutPolicy> View() const
        {
            return Detail::ProjectView<Value, Checked, LayoutPolicy>(View<Checked>(), _layout->rows);
        }
        template <bool Checked = false>
        ArrayView<Value, Checked> DynamicView() const { return View<Checked>().DynamicView(); }
        /// Checked host access for frontends, never used inside native kernels.
        Value &At(index row, index component) const
        {
            DNDS_check_throw_info(GetDevice() == Device::Host(), "host indexing requires a host Array");
            DNDS_check_throw_info(row >= 0 && component >= 0 && row < Layout().RowCount() &&
                                      component < Layout().RowSize(row),
                                  "Array index outside row");
            return View()(row, component);
        }
        void Complete(Completion completion) { _lease.Complete(std::move(completion)); }
        void Wait() { _lease.Wait(); }
        void Fail() { _lease.Fail(); }
    };

    /// Native shallow handle. Layout replacement retains existing aliases/leases.
    /// Legacy DNDS::Array remains a separate deep-copy facade over shared facilities.
    template <class T>
    class ArrayHandle
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        static_assert(alignof(T) <= 64, "native Array supports alignment up to 64 bytes");
        ArrayStorage::Shared _storage;
        explicit ArrayHandle(ArrayStorage::Shared storage) : _storage(std::move(storage)) {}

    public:
        using value_type = T;
        using ReadLease = ArrayAccess<T, false>;
        using WriteLease = ArrayAccess<T, true>;
        ArrayHandle(const ArrayHandle &other) : _storage(ArrayStorage::Share(other._storage)) {}
        ArrayHandle &operator=(const ArrayHandle &other)
        {
            if (this != &other)
                _storage = ArrayStorage::Share(other._storage);
            return *this;
        }
        ArrayHandle(ArrayHandle &&) noexcept = default;
        ArrayHandle &operator=(ArrayHandle &&) noexcept = default;
        ArrayHandle() { Reset(RowLayout{}); }
        explicit ArrayHandle(RowLayout layout, Device device = Device::Host()) { Reset(std::move(layout), device); }
        /// Reinterpret owned storage with an explicit row/type layout. This
        /// retains the original allocation's controller; it never wraps raw pointers.
        static ArrayHandle FromStorage(RowLayout layout, Buffer storage)
        {
            return FromSharedStorage(ArrayStorage::FromBuffer(std::move(layout), std::move(storage), sizeof(T)));
        }
        /// Shares values, logical layout and device offsets without allocating or
        /// copying. Language adapters additionally validate the semantic dtype.
        /// @pre T describes the actual stored values and their ABI. Size and
        /// alignment checks alone cannot establish semantic type compatibility
        /// for this deliberately type-erased native storage boundary.
        static ArrayHandle FromSharedStorage(ArrayStorage::Shared storage)
        {
            DNDS_check_throw_info(storage && storage->ElementBytes() == sizeof(T), "Array element size mismatch");
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(storage->Values().LegacyPointer()) % alignof(T) == 0,
                                  "Array storage alignment mismatch");
            return ArrayHandle(std::move(storage));
        }
        void Reset(RowLayout layout, Device device = Device::Host())
        {
            _storage = ArrayStorage::Allocate(std::move(layout), sizeof(T), device);
        }
        ArrayStorage::Shared SharedStorage() const { return ArrayStorage::Share(_storage); }
        ExclusiveRows EditRows(); // Defined by RowBuilder.hpp; exclusive CPU setup.
        uint64_t ShapeVersion() const { return _storage->ShapeVersion(); }
        RowLayout Layout() const { return _storage->Layout(); }
        index RowCount() const { return _storage->RowCount(); }
        index Rows() const { return RowCount(); }
        index RowSize(index row) const { return _storage->RowSize(row); }
        Device GetDevice() const { return Storage().GetDevice(); }
        uint64_t Identity() const { return Storage().Identity(); }
        uint64_t Version() const { return Storage().Version(); }
        bool IsValid() const { return _storage->IsValid(); }
        Buffer Storage() const { return _storage->Values(); }
        bool CanReleaseOrdered(ExecutionContext context) const { return !_storage || _storage->CanReleaseOrdered(context); }
        void ReleaseOn(ExecutionContext context) { ArrayStorage::ReleaseHandle(_storage, context); }
        ReadLease Read(ExecutionContext context = ExecutionContext::Host()) const { return {_storage->Read(context), _storage->SharedLayout()}; }
        WriteLease Write(ExecutionContext context = ExecutionContext::Host()) { return {_storage->Write(context), _storage->SharedLayout()}; }
        CopyOperation CopyFrom(const ArrayHandle &other, ExecutionContext context = ExecutionContext::Host())
        {
            return _storage->CopyFrom(*other._storage, context);
        }
        /// Synchronous explicit snapshot/placement conversion. CopyFrom is async.
        ArrayHandle To(Device device, ExecutionContext context) const
        {
            ArrayHandle result(Layout(), device);
            result.CopyFrom(*this, context).Wait();
            return result;
        }
        ArrayHandle To(Device device) const { return To(device, {device == Device::Host() ? GetDevice() : device, nullptr}); }
        ArrayHandle Clone(ExecutionContext context) const { return To(GetDevice(), context); }
        ArrayHandle Clone() const { return To(GetDevice()); }
    };

    /// Public storage handle. Layout policy belongs to an acquired view.
    /// The optional second alias argument is a migration spelling only.
    template <class T, class LayoutPolicy = RowLayout>
    using Array = ArrayHandle<T>;

    /// Logical combined indexing without combining storage or access controllers.
    template <class T, bool Checked = false, class LayoutPolicy = RowLayout>
    struct PairReadView
    {
        static constexpr size_t row_width = Detail::RowPolicyTraits<LayoutPolicy>::row_width;
        static constexpr size_t row_capacity = Detail::RowPolicyTraits<LayoutPolicy>::row_capacity;
        using RowType = RowSpan<const T, Checked, row_width, row_capacity>;
        ArrayView<const T, Checked, LayoutPolicy> owned, ghost;
        DNDS_DEVICE_CALLABLE index RowCount() const { return owned.RowCount() + ghost.RowCount(); }
        DNDS_DEVICE_CALLABLE index Rows() const { return RowCount(); }
        DNDS_DEVICE_CALLABLE const T *Row(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < RowCount());
            return row < owned.RowCount() ? owned.Row(row) : ghost.Row(row - owned.RowCount());
        }
        DNDS_DEVICE_CALLABLE index RowSize(index row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < RowCount());
            return row < owned.RowCount() ? owned.RowSize(row) : ghost.RowSize(row - owned.RowCount());
        }
        DNDS_DEVICE_CALLABLE const T &operator()(index row, index column) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row >= 0 && row < RowCount() && column >= 0 && column < RowSize(row));
            return Row(row)[column];
        }
        DNDS_DEVICE_CALLABLE RowType operator[](index row) const { return {Row(row), RowSize(row)}; }
        DNDS_DEVICE_CALLABLE PairReadView<T, Checked> DynamicView() const { return {owned.DynamicView(), ghost.DynamicView()}; }
        template <class P = LayoutPolicy, std::enable_if_t<!std::is_same_v<P, RowLayout>, int> = 0>
        DNDS_DEVICE_CALLABLE operator PairReadView<T, Checked>() const { return DynamicView(); }
    };

    template <class T>
    struct ArrayPairHandle
    {
        Array<T> owned, ghost;
        /// Core kernels receive owned.Read().View(), never a full pair view.
        struct ReadLease
        {
            typename Array<T>::ReadLease owned, ghost;
            template <bool Checked = false>
            PairReadView<T, Checked> View() const { return {owned.template View<Checked>(), ghost.template View<Checked>()}; }
            template <class Policy, bool Checked = false>
            PairReadView<T, Checked, Policy> View() const { return {owned.template View<Policy, Checked>(), ghost.template View<Policy, Checked>()}; }
            template <bool Checked = false>
            PairReadView<T, Checked> DynamicView() const { return View<Checked>(); }
            void Complete(Completion done)
            {
                owned.Complete(done);
                ghost.Complete(done);
            }
            void Wait()
            {
                owned.Wait();
                ghost.Wait();
            }
        };
        typename Array<T>::ReadLease ReadOwned(ExecutionContext context = ExecutionContext::Host()) const { return owned.Read(context); }
        /// Acquire only after ghost readiness. This never combines the writers.
        ReadLease ReadCombined(ExecutionContext context = ExecutionContext::Host()) const { return {owned.Read(context), ghost.Read(context)}; }
        ReadLease Read(ExecutionContext context = ExecutionContext::Host()) const { return ReadCombined(context); }
    };

    template <class T, class LayoutPolicy = RowLayout>
    using ArrayPair = ArrayPairHandle<T>;
}
