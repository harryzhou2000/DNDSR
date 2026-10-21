#include "Buffer.hpp"
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
        std::shared_ptr<Allocator> allocator;
        std::shared_ptr<void> allocation;
        size_t bytes = 0;
        Device device;
        uint64_t identity = nextIdentity.fetch_add(1);
        uint64_t version = 0;
        size_t readers = 0;
        bool writer = false;
        bool failed = false;
        mutable std::mutex mutex;
    };

    Buffer Buffer::Allocate(size_t bytes, Device device)
    {
        ValidateDevice(device);
        auto state = std::make_shared<BufferState>();
        state->bytes = bytes;
        state->device = device;
        if (bytes && device.space == MemorySpace::Host)
        {
            void *p = ::operator new(bytes, std::align_val_t(64));
            state->allocation = std::shared_ptr<void>(p, [](void *q)
                                                      { ::operator delete(q, std::align_val_t(64)); });
            std::memset(p, 0, bytes);
        }
#ifdef DNDS_USE_CUDA
        else if (bytes)
        {
            DeviceScope scope(device.ordinal);
            void *p = nullptr;
            CheckCUDA(cudaMalloc(&p, bytes));
            state->allocation = std::shared_ptr<void>(p, [device](void *q)
                                                      {
                                                        DeviceScope scope(device.ordinal);
                                                        if (cudaFree(q) != cudaSuccess)
                                                            std::terminate(); });
            CheckCUDA(cudaMemset(p, 0, bytes));
            CheckCUDA(cudaStreamSynchronize(nullptr));
        }
#endif
        return Buffer(std::move(state));
    }

    Buffer Buffer::Allocate(size_t bytes, Device device, std::shared_ptr<Allocator> allocator)
    {
        ValidateDevice(device);
        DNDS_check_throw_info(allocator, "runtime: null allocator policy");
        auto state = std::make_shared<BufferState>();
        state->allocator = std::move(allocator);
        state->bytes = bytes;
        state->device = device;
        state->allocation = state->allocator->Allocate(bytes, device);
        DNDS_check_throw_info(!bytes || state->allocation, "runtime: allocator returned null storage");
        return Buffer(std::move(state));
    }

    size_t Buffer::Bytes() const { return _state ? _state->bytes : 0; }
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
        return !_state->writer && !_state->readers;
    }
    void *Buffer::LegacyPointer() const { return _state ? _state->allocation.get() : nullptr; }
    bool Buffer::IsValid() const
    {
        if (!_state)
            return false;
        std::lock_guard<std::mutex> lock(_state->mutex);
        return !_state->failed;
    }
    ReadAccess Buffer::Read(ExecutionContext context) const { return ReadAccess(Access(_state, context, false)); }
    WriteAccess Buffer::Write(ExecutionContext context) { return WriteAccess(Access(_state, context, true)); }

    Access::Access(std::shared_ptr<BufferState> state, ExecutionContext context, bool write)
        : _state(std::move(state)), _context(context), _write(write), _uncaughtOnAcquire(std::uncaught_exceptions())
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
        DNDS_check_throw_info(!_state->writer && (!write || !_state->readers), "runtime: conflicting buffer access");
        if (write)
            _state->writer = true;
        else
            ++_state->readers;
    }

    Access::Access(Access &&other) noexcept
        : _state(std::move(other._state)), _context(other._context), _completion(std::move(other._completion)),
          _hasCompletion(other._hasCompletion), _write(other._write), _uncaughtOnAcquire(other._uncaughtOnAcquire) {}

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
        return _state->bytes;
    }
    Device Access::GetDevice() const
    {
        DNDS_check_throw(_state);
        return _state->device;
    }
    void *Access::Pointer(size_t alignment, size_t elementSize) const
    {
        DNDS_check_throw_info(_state && !_hasCompletion, "runtime: view requested after submit/release");
        auto *p = _state->allocation.get();
        DNDS_check_throw_info(_state->bytes % elementSize == 0 && reinterpret_cast<uintptr_t>(p) % alignment == 0,
                              "runtime: incompatible typed view size/alignment");
        return p;
    }

    Completion Detail::CopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
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
