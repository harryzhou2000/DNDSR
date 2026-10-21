#pragma once

#include "Buffer.hpp"
#include <algorithm>
#include <vector>
#include <limits>

namespace DNDS::Runtime
{
    /// Immutable row shape in element units. Padding belongs to storage, not a row.
    class RowLayout
    {
        size_t _rows = 0, _width = 0, _stride = 0;
        std::vector<size_t> _offsets;

    public:
        RowLayout() = default;
        static RowLayout Dense(size_t rows, size_t width, size_t stride = 0)
        {
            RowLayout layout;
            layout._rows = rows;
            layout._width = width;
            layout._stride = stride ? stride : width;
            DNDS_check_throw_info(layout._stride >= width, "row stride is smaller than width");
            CheckedSize::Multiply(rows, layout._stride);
            return layout;
        }
        static RowLayout Ragged(const std::vector<size_t> &lengths)
        {
            RowLayout layout;
            layout._rows = lengths.size();
            layout._offsets.push_back(0);
            for (auto n : lengths)
            {
                DNDS_check_throw_info(n <= SIZE_MAX - layout._offsets.back(), "row offsets overflow");
                layout._offsets.push_back(layout._offsets.back() + n);
            }
            return layout;
        }
        size_t Rows() const { return _rows; }
        bool IsRagged() const { return !_offsets.empty(); }
        size_t Width() const { return _width; }
        size_t Stride() const { return _stride; }
        size_t Elements() const { return IsRagged() ? _offsets.back() : _rows * _stride; }
        const std::vector<size_t> &Offsets() const { return _offsets; }
        size_t RowSize(size_t row) const
        {
            DNDS_check_throw_info(row < _rows, "row outside layout");
            return IsRagged() ? _offsets[row + 1] - _offsets[row] : _width;
        }
        size_t Offset(size_t row) const
        {
            DNDS_check_throw_info(row < _rows, "row outside layout");
            return IsRagged() ? _offsets[row] : row * _stride;
        }
        bool operator==(const RowLayout &other) const
        {
            return _rows == other._rows && _width == other._width && _stride == other._stride && _offsets == other._offsets;
        }
    };

    /// Optional compile-time row width; the runtime shape is still shared by MPI/I/O.
    template <size_t Width>
    struct FixedRows
    {
        static RowLayout Default() { return RowLayout::Dense(0, Width); }
        static void Validate(const RowLayout &layout)
        {
            DNDS_check_throw_info(!layout.IsRagged() && layout.Width() == Width, "fixed row width mismatch");
        }
    };

    /// Plain kernel view: no ownership, placement dispatch or tracking per element.
    /// Checked=true is a separate development policy, independent of NDEBUG.
    template <class T, bool Checked = false>
    struct ArrayView
    {
        T *data = nullptr;
        const size_t *offsets = nullptr;
        size_t rows = 0, width = 0, stride = 0;

        DNDS_DEVICE_CALLABLE size_t Rows() const { return rows; }
        DNDS_DEVICE_CALLABLE size_t RowSize(size_t row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row < rows);
            return offsets ? offsets[row + 1] - offsets[row] : width;
        }
        DNDS_DEVICE_CALLABLE T *Row(size_t row) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row < rows);
            const auto offset = offsets ? offsets[row] : row * stride;
            return data ? data + offset : nullptr;
        }
        DNDS_DEVICE_CALLABLE T &operator()(size_t row, size_t component) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(row < rows && component < RowSize(row));
            return Row(row)[component];
        }
    };

    namespace Detail
    {
        /// CPU shape is retained for setup; only immutable CSR offsets are mirrored.
        struct ArrayLayout
        {
            RowLayout rows;
            Buffer deviceOffsets;
            explicit ArrayLayout(RowLayout shape, Device device) : rows(std::move(shape))
            {
                if (rows.IsRagged() && device.space != MemorySpace::Host)
                {
                    auto host = Buffer::Allocate(CheckedSize::Multiply(rows.Offsets().size(), sizeof(size_t)));
                    {
                        auto edit = host.Write();
                        std::copy(rows.Offsets().begin(), rows.Offsets().end(), edit.View<size_t>().data);
                    }
                    deviceOffsets = Buffer::Allocate(host.Bytes(), device);
                    Copy(host, deviceOffsets, {device, nullptr}).Wait();
                }
            }
            const size_t *Offsets(Device device) const
            {
                if (!rows.IsRagged())
                    return nullptr;
                return device.space == MemorySpace::Host ? rows.Offsets().data() : static_cast<const size_t *>(deviceOffsets.LegacyPointer());
            }
        };
    }

    /// One immutable logical layout and its allocation, shared by typed and
    /// language bindings. Row metadata stays in element units for every handle.
    /// Buffer owns values/access state; ArrayLayout owns CPU/CUDA row offsets.
    class ArrayStorage
    {
        Buffer _values;
        std::shared_ptr<const Detail::ArrayLayout> _layout;
        size_t _elementBytes;

        ArrayStorage(RowLayout layout, Buffer values, size_t elementBytes)
            : _values(std::move(values)),
              _layout(std::make_shared<const Detail::ArrayLayout>(std::move(layout), _values.GetDevice())),
              _elementBytes(elementBytes) {}

    public:
        using Shared = std::shared_ptr<const ArrayStorage>;
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
        const RowLayout &Layout() const { return _layout->rows; }
        const std::shared_ptr<const Detail::ArrayLayout> &SharedLayout() const { return _layout; }
        const Buffer &Values() const { return _values; }
        size_t ElementBytes() const { return _elementBytes; }
        ReadAccess Read(ExecutionContext context = ExecutionContext::Host()) const { return _values.Read(context); }
        WriteAccess Write(ExecutionContext context = ExecutionContext::Host()) const
        {
            auto handle = _values;
            return handle.Write(context);
        }
        CopyOperation CopyFrom(const ArrayStorage &source, ExecutionContext context = ExecutionContext::Host()) const
        {
            DNDS_check_throw_info(_elementBytes == source._elementBytes && Layout() == source.Layout(),
                                  "Array copy requires matching element size and layout");
            auto destination = _values;
            return Copy(source._values, destination, context);
        }
    };

    template <class T, bool Writable>
    class ArrayAccess
    {
        using Lease = std::conditional_t<Writable, WriteAccess, ReadAccess>;
        using Value = std::conditional_t<Writable, T, const T>;
        std::shared_ptr<const Detail::ArrayLayout> _layout;
        Lease _lease;

    public:
        ArrayAccess(Lease lease, std::shared_ptr<const Detail::ArrayLayout> layout)
            : _layout(std::move(layout)), _lease(std::move(lease)) {}
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
            return {flat.data, _layout->Offsets(_lease.GetDevice()), r.Rows(), r.Width(), r.Stride()};
        }
        /// Checked host access for frontends, never used inside native kernels.
        Value &At(size_t row, size_t component) const
        {
            DNDS_check_throw_info(GetDevice() == Device::Host(), "host indexing requires a host Array");
            DNDS_check_throw_info(row < Layout().Rows() && component < Layout().RowSize(row), "Array index outside row");
            return View()(row, component);
        }
        void Complete(Completion completion) { _lease.Complete(std::move(completion)); }
        void Wait() { _lease.Wait(); }
        void Fail() { _lease.Fail(); }
    };

    /// Native shallow handle. Layout replacement retains existing aliases/leases.
    /// Legacy DNDS::Array remains a separate deep-copy facade over shared facilities.
    template <class T, class LayoutPolicy = RowLayout>
    class Array
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        static_assert(alignof(T) <= 64, "native Array supports alignment up to 64 bytes");
        ArrayStorage::Shared _storage;
        explicit Array(ArrayStorage::Shared storage) : _storage(std::move(storage)) {}

    public:
        using value_type = T;
        using ReadLease = ArrayAccess<T, false>;
        using WriteLease = ArrayAccess<T, true>;
        Array()
        {
            if constexpr (std::is_same_v<LayoutPolicy, RowLayout>)
                Reset(RowLayout{});
            else
                Reset(LayoutPolicy::Default());
        }
        explicit Array(RowLayout layout, Device device = Device::Host()) { Reset(std::move(layout), device); }
        /// Reinterpret owned storage with an explicit row/type layout. This
        /// retains the original allocation's controller; it never wraps raw pointers.
        static Array FromStorage(RowLayout layout, Buffer storage)
        {
            return FromSharedStorage(ArrayStorage::FromBuffer(std::move(layout), std::move(storage), sizeof(T)));
        }
        /// Shares values, logical layout and device offsets without allocating or
        /// copying. Language adapters additionally validate the semantic dtype.
        static Array FromSharedStorage(ArrayStorage::Shared storage)
        {
            DNDS_check_throw_info(storage && storage->ElementBytes() == sizeof(T), "Array element size mismatch");
            if constexpr (!std::is_same_v<LayoutPolicy, RowLayout>)
                LayoutPolicy::Validate(storage->Layout());
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(storage->Values().LegacyPointer()) % alignof(T) == 0,
                                  "Array storage alignment mismatch");
            return Array(std::move(storage));
        }
        void Reset(RowLayout layout, Device device = Device::Host())
        {
            if constexpr (!std::is_same_v<LayoutPolicy, RowLayout>)
                LayoutPolicy::Validate(layout);
            _storage = ArrayStorage::Allocate(std::move(layout), sizeof(T), device);
        }
        const ArrayStorage::Shared &SharedStorage() const { return _storage; }
        const RowLayout &Layout() const { return _storage->Layout(); }
        size_t Rows() const { return Layout().Rows(); }
        size_t RowSize(size_t row) const { return Layout().RowSize(row); }
        Device GetDevice() const { return Storage().GetDevice(); }
        uint64_t Identity() const { return Storage().Identity(); }
        uint64_t Version() const { return Storage().Version(); }
        bool IsValid() const { return Storage().IsValid(); }
        const Buffer &Storage() const { return _storage->Values(); }
        ReadLease Read(ExecutionContext context = ExecutionContext::Host()) const { return {_storage->Read(context), _storage->SharedLayout()}; }
        WriteLease Write(ExecutionContext context = ExecutionContext::Host()) { return {_storage->Write(context), _storage->SharedLayout()}; }
        CopyOperation CopyFrom(const Array &other, ExecutionContext context = ExecutionContext::Host())
        {
            return _storage->CopyFrom(*other._storage, context);
        }
        /// Synchronous explicit snapshot/placement conversion. CopyFrom is async.
        Array To(Device device, ExecutionContext context) const
        {
            Array result(Layout(), device);
            result.CopyFrom(*this, context).Wait();
            return result;
        }
        Array To(Device device) const { return To(device, {device == Device::Host() ? GetDevice() : device, nullptr}); }
        Array Clone(ExecutionContext context) const { return To(GetDevice(), context); }
        Array Clone() const { return To(GetDevice()); }
    };

    /// Logical combined indexing without combining storage or access controllers.
    template <class T, bool Checked = false>
    struct PairReadView
    {
        ArrayView<const T, Checked> owned, ghost;
        DNDS_DEVICE_CALLABLE const T *Row(size_t row) const
        {
            return row < owned.Rows() ? owned.Row(row) : ghost.Row(row - owned.Rows());
        }
        DNDS_DEVICE_CALLABLE size_t RowSize(size_t row) const
        {
            return row < owned.Rows() ? owned.RowSize(row) : ghost.RowSize(row - owned.Rows());
        }
        DNDS_DEVICE_CALLABLE const T &operator()(size_t row, size_t column) const
        {
            if constexpr (Checked)
                DNDS_HD_assert(column < RowSize(row));
            return Row(row)[column];
        }
    };

    template <class T, class LayoutPolicy = RowLayout>
    struct ArrayPair
    {
        Array<T, LayoutPolicy> owned, ghost;
        /// Core kernels receive owned.Read().View(), never a full pair view.
        struct ReadLease
        {
            typename Array<T, LayoutPolicy>::ReadLease owned, ghost;
            template <bool Checked = false>
            PairReadView<T, Checked> View() const { return {owned.template View<Checked>(), ghost.template View<Checked>()}; }
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
        ReadLease Read(ExecutionContext context = ExecutionContext::Host()) const { return {owned.Read(context), ghost.Read(context)}; }
    };
}
