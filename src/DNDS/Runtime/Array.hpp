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
        Buffer _values;
        std::shared_ptr<const Detail::ArrayLayout> _layout;

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
            if constexpr (!std::is_same_v<LayoutPolicy, RowLayout>)
                LayoutPolicy::Validate(layout);
            DNDS_check_throw_info(storage.Identity() && storage.Bytes() == CheckedSize::Multiply(layout.Elements(), sizeof(T)),
                                  "Array storage size/layout mismatch");
            DNDS_check_throw_info(reinterpret_cast<uintptr_t>(storage.LegacyPointer()) % alignof(T) == 0, "Array storage alignment mismatch");
            Array result;
            result._layout = std::make_shared<const Detail::ArrayLayout>(std::move(layout), storage.GetDevice());
            result._values = std::move(storage);
            return result;
        }
        void Reset(RowLayout layout, Device device = Device::Host())
        {
            if constexpr (!std::is_same_v<LayoutPolicy, RowLayout>)
                LayoutPolicy::Validate(layout);
            auto shape = std::make_shared<const Detail::ArrayLayout>(std::move(layout), device);
            auto values = Buffer::Allocate(CheckedSize::Multiply(shape->rows.Elements(), sizeof(T)), device);
            _layout = std::move(shape);
            _values = std::move(values);
        }
        const RowLayout &Layout() const { return _layout->rows; }
        size_t Rows() const { return Layout().Rows(); }
        size_t RowSize(size_t row) const { return Layout().RowSize(row); }
        Device GetDevice() const { return _values.GetDevice(); }
        uint64_t Identity() const { return _values.Identity(); }
        uint64_t Version() const { return _values.Version(); }
        bool IsValid() const { return _values.IsValid(); }
        const Buffer &Storage() const { return _values; }
        ReadLease Read(ExecutionContext context = ExecutionContext::Host()) const { return {_values.Read(context), _layout}; }
        WriteLease Write(ExecutionContext context = ExecutionContext::Host()) { return {_values.Write(context), _layout}; }
        CopyOperation CopyFrom(const Array &other, ExecutionContext context = ExecutionContext::Host())
        {
            DNDS_check_throw_info(Layout() == other.Layout(), "Array copy requires matching layouts");
            return Copy(other._values, _values, context);
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
