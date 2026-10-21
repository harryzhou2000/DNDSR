#include "Buffer.hpp"
#include "AllocationResource.hpp"
#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <utility>
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif

namespace DNDS::Runtime
{
    namespace
    {
        std::atomic<uint64_t> nextIdentity{1};

        void ValidateDevice(Device device)
        {
            DNDS_check_throw_info((device.space == MemorySpace::Host && device.ordinal == -1) ||
                                      (device.space == MemorySpace::CUDA && device.ordinal >= 0),
                                  "runtime: invalid device descriptor");
#ifndef DNDS_USE_CUDA
            DNDS_check_throw_info(device.space == MemorySpace::Host, "runtime: CUDA is not enabled in this build");
#endif
        }

#ifdef DNDS_USE_CUDA
        void CheckCUDA(cudaError_t status)
        {
            DNDS_check_throw_info(status == cudaSuccess, cudaGetErrorString(status));
        }

        class DeviceScope
        {
            int _previous = -1;

        public:
            explicit DeviceScope(int ordinal)
            {
                CheckCUDA(cudaGetDevice(&_previous));
                CheckCUDA(cudaSetDevice(ordinal));
            }
            ~DeviceScope()
            {
                if (cudaSetDevice(_previous) != cudaSuccess)
                    std::terminate();
            }
        };

        void ValidateStream(ExecutionContext context)
        {
            DeviceScope scope(context.device.ordinal);
#    if CUDART_VERSION >= 13000
            int ordinal = -1;
            CheckCUDA(cudaStreamGetDevice(static_cast<cudaStream_t>(context.stream), &ordinal));
            DNDS_check_throw_info(ordinal == context.device.ordinal, "runtime: stream belongs to another CUDA device");
#    else
            // Older CUDA runtimes lack this query. The caller must supply a
            // stream from the declared device; event recording validates usage.
            unsigned flags = 0;
            CheckCUDA(cudaStreamGetFlags(static_cast<cudaStream_t>(context.stream), &flags));
#    endif
        }
#endif

        void Fence(ExecutionContext context)
        {
            ValidateDevice(context.device);
            if (context.device.space == MemorySpace::Host)
                DNDS_check_throw_info(!context.stream, "host context cannot contain a CUDA stream");
#ifdef DNDS_USE_CUDA
            else
            {
                DeviceScope device(context.device.ordinal);
                CheckCUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(context.stream)));
            }
#endif
        }

        class SharedResource final : public AllocationResource
        {
            std::shared_ptr<Allocator> _allocator;
            std::shared_ptr<void> _owner;
            size_t _bytes;
            Device _device;

        public:
            SharedResource(std::shared_ptr<void> owner, size_t bytes, Device device, std::shared_ptr<Allocator> allocator = {})
                : _allocator(std::move(allocator)), _owner(std::move(owner)), _bytes(bytes), _device(device) {}
            void *Data() const noexcept override { return _owner.get(); }
            size_t Bytes() const noexcept override { return _bytes; }
            Device GetDevice() const noexcept override { return _device; }
            void Release(const ExecutionContext *) noexcept override
            {
                _owner.reset();
                _allocator.reset();
            }
        };

        class NativeResource final : public AllocationResource
        {
            void *_pointer = nullptr;
            size_t _bytes;
            Device _device;
            bool _orderedCUDA = false;
            bool _asyncAllocation = false, _unpublished = false;
            ExecutionContext _construction;

        public:
            NativeResource(size_t bytes, Device device, const ExecutionContext *async = nullptr)
                : _bytes(bytes), _device(device), _asyncAllocation(async != nullptr), _unpublished(async != nullptr),
                  _construction(async ? *async : ExecutionContext{})
            {
                if (!bytes)
                    return;
                if (device == Device::Host())
                {
                    _pointer = ::operator new(bytes, std::align_val_t(64));
                    std::memset(_pointer, 0, bytes);
                }
#ifdef DNDS_USE_CUDA
                else
                {
                    DeviceScope scope(device.ordinal);
                    try
                    {
#    if CUDART_VERSION >= 11020
                        int pools = 0;
                        CheckCUDA(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device.ordinal));
                        _orderedCUDA = bool(pools);
#    endif
                        if (async)
                        {
#    if CUDART_VERSION >= 11020
                            DNDS_check_throw_info(_orderedCUDA, "device does not support stream ordered allocation");
                            auto stream = static_cast<cudaStream_t>(async->stream);
                            CheckCUDA(cudaMallocAsync(&_pointer, bytes, stream));
                            CheckCUDA(cudaMemsetAsync(_pointer, 0, bytes, stream));
#    else
                            DNDS_check_throw_info(false, "CUDA runtime lacks stream ordered allocation");
#    endif
                        }
                        else
                        {
                            CheckCUDA(cudaMalloc(&_pointer, bytes));
                            CheckCUDA(cudaMemset(_pointer, 0, bytes));
                            CheckCUDA(cudaStreamSynchronize(nullptr));
                        }
                    }
                    catch (...)
                    {
                        Release(async);
                        throw;
                    }
                }
#endif
            }
            ~NativeResource() override { Release(_unpublished ? &_construction : nullptr); }
            void Published() { _unpublished = false; }
            void *Data() const noexcept override { return _pointer; }
            size_t Bytes() const noexcept override { return _bytes; }
            Device GetDevice() const noexcept override { return _device; }
            bool CanReleaseOrdered(ExecutionContext context) const noexcept override
            {
                return context.device == _device && (_device != Device::Host() || !context.stream) &&
                       (_device == Device::Host() || _orderedCUDA || !_bytes);
            }
            void Release(const ExecutionContext *context) noexcept override
            {
                auto *pointer = std::exchange(_pointer, nullptr);
                if (!pointer)
                    return;
                if (_device == Device::Host())
                    ::operator delete(pointer, std::align_val_t(64));
#ifdef DNDS_USE_CUDA
                else
                {
                    DeviceScope device(_device.ordinal);
                    cudaError_t status;
#    if CUDART_VERSION >= 11020
                    if (context)
                        status = cudaFreeAsync(pointer, static_cast<cudaStream_t>(context->stream));
                    else
#    endif
                    {
                        // cudaFree does not synchronize a cudaMallocAsync
                        // allocation. Unplaced RAII uses the conservative path.
                        if (_asyncAllocation && cudaDeviceSynchronize() != cudaSuccess)
                            std::terminate();
                        status = cudaFree(pointer);
                    }
                    if (status != cudaSuccess)
                        std::terminate();
                }
#endif
            }
        };
    }

    struct Completion::State
    {
        Device device;
        void *stream = nullptr;
#ifdef DNDS_USE_CUDA
        cudaEvent_t event = nullptr;
        ~State()
        {
            if (event)
            {
                DeviceScope scope(device.ordinal);
                if (cudaEventSynchronize(event) != cudaSuccess || cudaEventDestroy(event) != cudaSuccess)
                    std::terminate();
            }
        }
#endif
    };

    Completion Completion::Record(ExecutionContext context)
    {
        ValidateDevice(context.device);
        if (context.device.space == MemorySpace::Host)
        {
            DNDS_check_throw_info(!context.stream, "host context cannot contain a CUDA stream");
            return {};
        }
        auto state = std::make_shared<State>();
        state->device = context.device;
        state->stream = context.stream;
#ifdef DNDS_USE_CUDA
        ValidateStream(context);
        DeviceScope device(context.device.ordinal);
        CheckCUDA(cudaEventCreateWithFlags(&state->event, cudaEventDisableTiming));
        CheckCUDA(cudaEventRecord(state->event, static_cast<cudaStream_t>(context.stream)));
#endif
        return Completion(std::move(state));
    }

    void Completion::Wait() const
    {
#ifdef DNDS_USE_CUDA
        if (_state)
        {
            DeviceScope device(_state->device.ordinal);
            CheckCUDA(cudaEventSynchronize(_state->event));
        }
#endif
    }

    bool Completion::Ready() const
    {
#ifdef DNDS_USE_CUDA
        if (_state)
        {
            DeviceScope device(_state->device.ordinal);
            auto status = cudaEventQuery(_state->event);
            if (status == cudaErrorNotReady)
                return false;
            CheckCUDA(status);
        }
#endif
        return true;
    }

    Device Completion::GetDevice() const { return _state ? _state->device : Device::Host(); }
    bool Completion::Matches(ExecutionContext context) const
    {
        return GetDevice() == context.device && (_state ? _state->stream : nullptr) == context.stream;
    }

    struct BufferState
    {
        Detail::AllocationReference allocation;
        size_t bytes = 0;
        Device device;
        uint64_t identity = nextIdentity.fetch_add(1);
        uint64_t version = 0;
        size_t readers = 0;
        size_t pendingReaders = 0, pendingWriters = 0;
        bool writer = false;
        bool failed = false;
        bool writable = true;
        mutable std::mutex mutex;
    };

    namespace
    {
        std::shared_ptr<BufferState> MakeState(std::unique_ptr<AllocationResource> resource)
        {
            DNDS_check_throw_info(resource, "runtime: null allocation resource");
            ValidateDevice(resource->GetDevice());
            DNDS_check_throw_info(!resource->Bytes() || resource->Data(), "runtime: allocation resource has null storage");
            DNDS_check_throw_info(resource->Bytes() <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(resource->Data()),
                                  "runtime: allocation address range overflow");
            auto state = std::make_shared<BufferState>();
            state->bytes = resource->Bytes();
            state->device = resource->GetDevice();
            state->allocation = Detail::AllocationReference(std::move(resource));
            return state;
        }
    }

    Allocation Allocation::Import(std::shared_ptr<void> owner, size_t bytes, Device device, bool writable)
    {
        ValidateDevice(device);
        DNDS_check_throw_info(owner.use_count() && (!bytes || owner.get()), "runtime: imported allocation requires shared ownership");
        DNDS_check_throw_info(bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(owner.get()), "runtime: imported address range overflow");
        auto state = MakeState(std::make_unique<SharedResource>(std::move(owner), bytes, device));
        state->writable = writable;
        return Allocation(std::move(state));
    }

    size_t Allocation::Bytes() const { return _state ? _state->bytes : 0; }
    Device Allocation::GetDevice() const { return _state ? _state->device : Device::Host(); }
    Buffer::Buffer(std::shared_ptr<BufferState> state) : _state(std::move(state)), _bytes(_state->bytes) {}

    Buffer Buffer::FromAllocation(const Allocation &allocation, size_t offset, size_t bytes)
    {
        DNDS_check_throw_info(allocation._state && offset <= allocation.Bytes() && bytes <= allocation.Bytes() - offset,
                              "runtime: buffer range outside allocation");
        Buffer result(allocation._state);
        result._offset = offset;
        result._bytes = bytes;
        return result;
    }

    Buffer Buffer::Subspan(size_t offset, size_t bytes) const
    {
        DNDS_check_throw_info(offset <= _bytes && bytes <= _bytes - offset, "runtime: subspan outside buffer");
        return FromAllocation(AllocationHandle(), _offset + offset, bytes);
    }

    Buffer Buffer::Allocate(size_t bytes, Device device)
    {
        ValidateDevice(device);
        return FromResource(std::make_unique<NativeResource>(bytes, device));
    }

    Buffer Buffer::Allocate(size_t bytes, Device device, std::shared_ptr<Allocator> allocator)
    {
        ValidateDevice(device);
        DNDS_check_throw_info(allocator, "runtime: null allocator policy");
        auto owner = allocator->Allocate(bytes, device);
        return FromResource(std::make_unique<SharedResource>(std::move(owner), bytes, device, std::move(allocator)));
    }
    Buffer Buffer::FromResource(std::unique_ptr<AllocationResource> resource) { return Buffer(MakeState(std::move(resource))); }
    Buffer Buffer::AllocateCUDAAsync(size_t bytes, ExecutionContext context)
    {
        ValidateDevice(context.device);
        DNDS_check_throw_info(context.device.space == MemorySpace::CUDA, "async device allocation requires CUDA");
#ifdef DNDS_USE_CUDA
        ValidateStream(context);
#endif
        auto resource = std::make_unique<NativeResource>(bytes, context.device, &context);
        auto *native = resource.get();
        auto result = FromResource(std::move(resource));
        native->Published();
        return result;
    }

    size_t Buffer::Bytes() const { return _state ? _bytes : 0; }
    Device Buffer::GetDevice() const { return _state ? _state->device : Device::Host(); }
    uint64_t Buffer::Identity() const { return _state ? _state->identity : 0; }
    uint64_t Buffer::Version() const
    {
        if (!_state)
            return 0;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return _state->version;
    }
    bool Buffer::IsIdle() const
    {
        if (!_state)
            return true;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return !_state->writer && !_state->readers && !_state->pendingReaders && !_state->pendingWriters;
    }
    bool Buffer::IsUnique() const { return _state && _state.use_count() == 1; }
    void *Buffer::LegacyPointer() const
    {
        auto *base = _state ? static_cast<std::byte *>(_state->allocation.Data()) : nullptr;
        return base ? base + _offset : nullptr;
    }
    bool Buffer::IsValid() const
    {
        if (!_state)
            return false;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return !_state->failed;
    }
    bool Buffer::CanReleaseOrdered(ExecutionContext context) const
    {
        return !_state || _state->allocation.CanReleaseOrdered(context);
    }
    void Buffer::ReleaseOn(ExecutionContext context)
    {
        if (!_state)
            return;
        ValidateDevice(context.device);
        DNDS_check_throw_info(context.device != Device::Host() || !context.stream, "invalid host release context");
#ifdef DNDS_USE_CUDA
        if (context.device.space == MemorySpace::CUDA)
            ValidateStream(context);
#endif
        DNDS_check_throw_info(CanReleaseOrdered(context), "allocation provider cannot release on this stream");
        // The extra payload reference carries the explicit context through
        // controller destruction. It never tests use_count or force-frees aliases.
        auto payload = _state->allocation;
        _state.reset();
        _offset = _bytes = 0;
        payload.ReleaseOn(context);
    }
    ReadAccess Buffer::Read(ExecutionContext context) const { return ReadAccess(Access(_state, _offset, _bytes, context, false)); }
    WriteAccess Buffer::Write(ExecutionContext context) { return WriteAccess(Access(_state, _offset, _bytes, context, true)); }
    ScheduledAccess Buffer::Reserve(AccessMode mode, Device device) const
    {
        return ScheduledAccess(_state, _offset, _bytes, device, mode);
    }

    ScheduledAccess::ScheduledAccess(std::shared_ptr<BufferState> state, size_t offset, size_t bytes, Device device, AccessMode mode)
        : _state(std::move(state)), _offset(offset), _bytes(bytes), _write(mode != AccessMode::Read)
    {
        DNDS_check_throw_info(mode == AccessMode::Read || mode == AccessMode::Write || mode == AccessMode::ReadWrite,
                              "runtime: invalid scheduled access mode");
        DNDS_check_throw_info(_state && _state->device == device, "runtime: scheduled access placement mismatch");
        std::lock_guard<std::mutex> lock(_state->mutex);
        DNDS_check_throw_info(!_state->failed, "runtime: failed allocation must be replaced");
        DNDS_check_throw_info(!_write || _state->writable, "runtime: imported allocation is read-only");
        DNDS_check_throw_info(!_state->writer && (!_write || !_state->readers), "runtime: conflicting immediate buffer access");
        auto &count = _write ? _state->pendingWriters : _state->pendingReaders;
        DNDS_check_throw_info(count < SIZE_MAX, "runtime: scheduled access count overflow");
        ++count;
    }
    ScheduledAccess &ScheduledAccess::operator=(ScheduledAccess &&other) noexcept
    {
        if (this != &other)
        {
            Cancel();
            _state = std::move(other._state);
            _issued = std::move(other._issued);
            _offset = other._offset;
            _bytes = other._bytes;
            _write = other._write;
        }
        return *this;
    }
    ScheduledAccess::~ScheduledAccess() { Cancel(); }
    void ScheduledAccess::Validate() const
    {
        auto state = _state ? _state : _issued.lock();
        DNDS_check_throw_info(state, "runtime: inactive scheduled access");
        std::lock_guard<std::mutex> lock(state->mutex);
        DNDS_check_throw_info(!state->failed, "runtime: scheduled operand has failed");
    }
    void *ScheduledAccess::Pointer(size_t alignment, size_t elementBytes) const
    {
        DNDS_check_throw_info(_state, "runtime: scheduled view requested after handoff");
        Validate();
        auto *base = static_cast<std::byte *>(_state->allocation.Data());
        auto *pointer = base ? base + _offset : nullptr;
        DNDS_check_throw_info(elementBytes && alignment && _bytes % elementBytes == 0 &&
                                  reinterpret_cast<uintptr_t>(pointer) % alignment == 0,
                              "runtime: incompatible scheduled view size/alignment");
        return pointer;
    }
    void ScheduledAccess::Handoff(ExecutionContext context)
    {
        DNDS_check_throw_info(_state && _state->device.space == MemorySpace::CUDA &&
                                  _state->allocation.CanReleaseOrdered(context),
                              "runtime: unsupported scheduled handoff");
        auto payload = _state->allocation;
        _issued = _state;
        _state.reset();
        payload.ReleaseOn(context);
    }
    void ScheduledAccess::Complete(bool success) noexcept
    {
        auto state = _state ? std::move(_state) : _issued.lock();
        _issued.reset();
        if (!state)
            return;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (_write)
        {
            --state->pendingWriters;
            if (success && !state->failed)
                ++state->version;
            else
                state->failed = true;
        }
        else
            --state->pendingReaders;
    }
    void ScheduledAccess::Cancel() noexcept
    {
        auto state = _state ? std::move(_state) : _issued.lock();
        _issued.reset();
        if (!state)
            return;
        std::lock_guard<std::mutex> lock(state->mutex);
        --(_write ? state->pendingWriters : state->pendingReaders);
    }

    Access::Access(std::shared_ptr<BufferState> state, size_t offset, size_t bytes, ExecutionContext context, bool write)
        : _state(std::move(state)), _context(context), _write(write), _offset(offset), _bytes(bytes), _uncaughtOnAcquire(std::uncaught_exceptions())
    {
        ValidateDevice(context.device);
        DNDS_check_throw_info(context.device.space != MemorySpace::Host || !context.stream,
                              "host context cannot contain a CUDA stream");
        DNDS_check_throw_info(_state, "runtime: access to an unallocated buffer");
        DNDS_check_throw_info(_state->device == context.device,
                              "runtime: placement mismatch; copy explicitly before accessing");
#ifdef DNDS_USE_CUDA
        if (context.device.space == MemorySpace::CUDA)
            ValidateStream(context);
#endif
        std::lock_guard<std::mutex> lock(_state->mutex);
        DNDS_check_throw_info(!_state->failed, "runtime: failed allocation must be replaced");
        DNDS_check_throw_info(!write || _state->writable, "runtime: imported allocation is read-only");
        DNDS_check_throw_info(!_state->writer && (!write || !_state->readers), "runtime: conflicting buffer access");
        DNDS_check_throw_info(!_state->pendingWriters && (!write || !_state->pendingReaders),
                              "runtime: pending scheduled access; observe stream completion first");
        if (write)
            _state->writer = true;
        else
            ++_state->readers;
    }

    Access::Access(Access &&other) noexcept
        : _state(std::move(other._state)), _context(other._context), _completion(std::move(other._completion)),
          _hasCompletion(other._hasCompletion), _write(other._write), _offset(other._offset), _bytes(other._bytes), _uncaughtOnAcquire(other._uncaughtOnAcquire) {}

    Access &Access::operator=(Access &&other) noexcept
    {
        if (this != &other)
        {
            try
            {
                Wait();
            }
            catch (...)
            {
                std::terminate();
            }
            _state = std::move(other._state);
            _context = other._context;
            _completion = std::move(other._completion);
            _hasCompletion = other._hasCompletion;
            _write = other._write;
            _offset = other._offset;
            _bytes = other._bytes;
            _uncaughtOnAcquire = other._uncaughtOnAcquire;
        }
        return *this;
    }

    Access::~Access()
    {
        try
        {
            if (_write && std::uncaught_exceptions() > _uncaughtOnAcquire)
                Fail();
            else
                Wait();
        }
        catch (...)
        {
            std::terminate();
        }
    }

    void Access::Complete(Completion completion)
    {
        DNDS_check_throw_info(_state && !_hasCompletion, "runtime: access already submitted or released");
        DNDS_check_throw_info(completion.Matches(_context), "runtime: completion device/stream mismatch");
        _completion = std::move(completion);
        _hasCompletion = true;
    }

    void Access::Wait()
    {
        if (!_state)
            return;
        std::exception_ptr error;
        try
        {
            if (_hasCompletion)
                _completion.Wait();
            else
                Fence(_context);
        }
        catch (...)
        {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            if (_write)
            {
                ++_state->version;
                _state->failed |= bool(error);
                _state->writer = false;
            }
            else
                --_state->readers;
        }
        _state.reset();
        if (error)
            std::rethrow_exception(error);
    }

    void Access::CancelUnmodified()
    {
        if (!_state)
            return;
        DNDS_check_throw_info(!_hasCompletion, "cannot cancel submitted access");
        auto state = std::move(_state);
        std::lock_guard<std::mutex> lock(state->mutex);
        if (_write)
            state->writer = false;
        else
            --state->readers;
    }

    void Access::Fail()
    {
        if (!_state)
            return;
        if (_write)
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            _state->failed = true;
        }
        Wait();
    }

    size_t Access::Bytes() const
    {
        DNDS_check_throw(_state);
        return _bytes;
    }
    Device Access::GetDevice() const
    {
        DNDS_check_throw(_state);
        return _state->device;
    }
    void *Access::Pointer(size_t alignment, size_t elementSize) const
    {
        DNDS_check_throw_info(_state && !_hasCompletion, "runtime: view requested after submit/release");
        auto *base = static_cast<std::byte *>(_state->allocation.Data());
        auto *p = base ? base + _offset : nullptr;
        DNDS_check_throw_info(_bytes % elementSize == 0 && reinterpret_cast<uintptr_t>(p) % alignment == 0,
                              "runtime: incompatible typed view size/alignment");
        return p;
    }

    void Detail::EnqueueCopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
                                  size_t bytes, ExecutionContext context)
    {
        ValidateDevice(context.device);
        const bool gpu = sourceDevice != Device::Host() || destinationDevice != Device::Host();
        DNDS_check_throw_info(gpu == (context.device.space == MemorySpace::CUDA), "copy execution device mismatch");
        DNDS_check_throw_info((sourceDevice == Device::Host() || sourceDevice == context.device) &&
                                  (destinationDevice == Device::Host() || destinationDevice == context.device),
                              "copy device ordinal mismatch");
        if (!gpu)
        {
            DNDS_check_throw_info(!context.stream, "host copy cannot use a device stream");
            if (bytes)
                std::memcpy(destination, source, bytes);
        }
#ifdef DNDS_USE_CUDA
        else
        {
            DeviceScope device(context.device.ordinal);
            ValidateStream(context);
            if (bytes)
                CheckCUDA(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDefault, static_cast<cudaStream_t>(context.stream)));
        }
#endif
    }

    Completion Detail::CopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
                                 size_t bytes, ExecutionContext context)
    {
        EnqueueCopyBytes(source, sourceDevice, destination, destinationDevice, bytes, context);
        return Completion::Record(context);
    }

    CopyOperation Copy(const Buffer &source, Buffer &destination, ExecutionContext context)
    {
        DNDS_check_throw_info(source.Identity() && destination.Identity() && source.Identity() != destination.Identity(),
                              "runtime: copy requires distinct allocated buffers");
        DNDS_check_throw_info(source.Bytes() == destination.Bytes(), "runtime: copy size mismatch");
        ValidateDevice(context.device);
        auto sourceContext = source.GetDevice().space == MemorySpace::Host ? ExecutionContext::Host() : context;
        DNDS_check_throw_info(context.device.space != MemorySpace::Host || !context.stream,
                              "host context cannot contain a CUDA stream");
        auto destinationContext = destination.GetDevice().space == MemorySpace::Host ? ExecutionContext::Host() : context;
        const bool onDevice = source.GetDevice().space == MemorySpace::CUDA || destination.GetDevice().space == MemorySpace::CUDA;
        DNDS_check_throw_info(onDevice == (context.device.space == MemorySpace::CUDA), "runtime: copy execution device mismatch");
        CopyOperation operation;
        operation._source = source.Read(sourceContext);
        operation._destination = destination.Write(destinationContext);
        auto src = operation._source.View<std::byte>();
        auto dst = operation._destination.View<std::byte>();
        if (!onDevice)
        {
            if (source.Bytes())
                std::memcpy(dst.data, src.data, source.Bytes());
            operation.Wait();
        }
#ifdef DNDS_USE_CUDA
        else
        {
            DeviceScope device(context.device.ordinal);
            operation._source._context = context;
            operation._destination._context = context;
            if (source.Bytes())
                CheckCUDA(cudaMemcpyAsync(dst.data, src.data, source.Bytes(), cudaMemcpyDefault,
                                          static_cast<cudaStream_t>(context.stream)));
            // Host staging leases also have to wait on the CUDA copy. Attach
            // the copy context before recording its event (no host-side access).
            auto done = Completion::Record(context);
            operation._source.Complete(done);
            operation._destination.Complete(done);
        }
#endif
        return operation;
    }

    void CopyOperation::Wait()
    {
        _source.Wait();
        _destination.Wait();
    }
}
