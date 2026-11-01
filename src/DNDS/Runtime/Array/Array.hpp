#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"
#include <mutex>

namespace DNDS::Runtime
{
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
        static Shared Allocate(RowLayout layout, size_t elementBytes, Device device = Device::Host(), AllocationOptions options = {})
        {
            auto values = Buffer::Allocate(CheckedSize::Multiply(layout.Elements(), elementBytes), device, options);
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
        const RowLayout &Layout() const
        {
            DNDS_check_throw_info(_layout, "Array access was moved");
            return _layout->rows;
        }
        template <bool Checked = AccessBoundsChecks>
        ArrayView<Value, Checked> View() const
        {
            auto flat = _lease.template View<T>();
            const auto &r = _layout->rows;
            return {flat.data, _layout->Offsets(_lease.GetDevice()), r.RowCount(), r.Width(), r.Stride(), _layout->Sizes(_lease.GetDevice()), r.Kind()};
        }
        template <class LayoutPolicy, bool Checked = AccessBoundsChecks>
        ArrayView<Value, Checked, LayoutPolicy> View() const
        {
            return Detail::ProjectView<Value, Checked, LayoutPolicy>(View<Checked>(), _layout->rows);
        }
        template <bool Checked = AccessBoundsChecks>
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
    /// A moved/released handle is invalid: IsValid returns false, SharedStorage
    /// returns null, and storage-dependent operations throw. Reset or assignment
    /// restores it; existing retained access scopes keep their own allocation.
    template <class T>
    class ArrayHandle
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        ArrayStorage::Shared _storage;
        explicit ArrayHandle(ArrayStorage::Shared storage) : _storage(std::move(storage)) {}
        const ArrayStorage &RequireStorage() const
        {
            DNDS_check_throw_info(_storage, "Array handle was moved or released");
            return *_storage;
        }

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
        explicit ArrayHandle(RowLayout layout, Device device = Device::Host(), AllocationOptions options = {}) { Reset(std::move(layout), device, options); }
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
        void Reset(RowLayout layout, Device device = Device::Host(), AllocationOptions options = {})
        {
            DNDS_check_throw_info(!options.alignmentBytes || !(options.alignmentBytes & (options.alignmentBytes - 1)),
                                  "allocation alignment must be a power of two");
            options.alignmentBytes = std::max(options.alignmentBytes, size_t(alignof(T)));
            _storage = ArrayStorage::Allocate(std::move(layout), sizeof(T), device, options);
        }
        ArrayStorage::Shared SharedStorage() const { return ArrayStorage::Share(_storage); }
        ExclusiveRows EditRows(); // Defined by RowBuilder.hpp; exclusive CPU setup.
        uint64_t ShapeVersion() const { return RequireStorage().ShapeVersion(); }
        RowLayout Layout() const { return RequireStorage().Layout(); }
        index RowCount() const { return RequireStorage().RowCount(); }
        index Rows() const { return RowCount(); }
        index RowSize(index row) const { return RequireStorage().RowSize(row); }
        Device GetDevice() const { return Storage().GetDevice(); }
        uint64_t Identity() const { return Storage().Identity(); }
        uint64_t Version() const { return Storage().Version(); }
        bool IsValid() const { return _storage && _storage->IsValid(); }
        Buffer Storage() const { return RequireStorage().Values(); }
        bool CanReleaseOrdered(ExecutionContext context) const { return !_storage || _storage->CanReleaseOrdered(context); }
        void ReleaseOn(ExecutionContext context) { ArrayStorage::ReleaseHandle(_storage, context); }
        ReadLease Read(ExecutionContext context = ExecutionContext::Host()) const
        {
            const auto &storage = RequireStorage();
            return {storage.Read(context), storage.SharedLayout()};
        }
        WriteLease Write(ExecutionContext context = ExecutionContext::Host())
        {
            const auto &storage = RequireStorage();
            return {storage.Write(context), storage.SharedLayout()};
        }
        CopyOperation CopyFrom(const ArrayHandle &other, ExecutionContext context = ExecutionContext::Host())
        {
            return RequireStorage().CopyFrom(other.RequireStorage(), context);
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

    template <class T>
    struct ArrayPairHandle
    {
        Array<T> owned, ghost;
        /// Core kernels receive owned.Read().View(), never a full pair view.
        struct ReadLease
        {
            typename Array<T>::ReadLease owned, ghost;
            template <bool Checked = AccessBoundsChecks>
            PairReadView<T, Checked> View() const { return {owned.template View<Checked>(), ghost.template View<Checked>()}; }
            template <class Policy, bool Checked = AccessBoundsChecks>
            PairReadView<T, Checked, Policy> View() const { return {owned.template View<Policy, Checked>(), ghost.template View<Policy, Checked>()}; }
            template <bool Checked = AccessBoundsChecks>
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
