#pragma once

#include "DNDS/Errors.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/CheckedSize.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace DNDS::Runtime
{
    enum class MemorySpace
    {
        Host,
        CUDA
    };

    /// Explicit placement; CUDA remains a valid descriptor in CPU-only builds,
    /// where attempting to allocate it reports an unsupported capability.
    struct Device
    {
        MemorySpace space = MemorySpace::Host;
        int ordinal = -1;
        static Device Host() { return {}; }
        static Device CUDA(int ordinal) { return {MemorySpace::CUDA, ordinal}; }
        bool operator==(Device r) const { return space == r.space && ordinal == r.ordinal; }
        bool operator!=(Device r) const { return !(*this == r); }
    };

    /// Borrowed execution stream. The caller keeps the stream alive until all
    /// leases using it have completed. No implicit host/device movement.
    struct ExecutionContext
    {
        Device device;
        void *stream = nullptr;
        static ExecutionContext Host() { return {}; }
        static ExecutionContext CUDA(int ordinal, void *stream = nullptr) { return {Device::CUDA(ordinal), stream}; }
    };

    class Completion
    {
        struct State;
        std::shared_ptr<State> _state;
        explicit Completion(std::shared_ptr<State> state) : _state(std::move(state)) {}

    public:
        Completion() = default;
        static Completion Record(ExecutionContext context);
        void Wait() const;
        bool Ready() const;
        Device GetDevice() const;
        bool Matches(ExecutionContext context) const;
    };

    namespace Detail
    {
        /// Backend primitive for operation implementations holding all required
        /// leases. It does not acquire/release access or publish generations.
        Completion CopyBytes(const void *source, Device sourceDevice, void *destination, Device destinationDevice,
                             size_t bytes, ExecutionContext context);
    }

    template <class T>
    struct Span
    {
        T *data = nullptr;
        size_t size = 0;
        MemorySpace space = MemorySpace::Host;

        DNDS_DEVICE_CALLABLE T &operator[](size_t i) const
        {
#ifdef __CUDA_ARCH__
            DNDS_HD_assert(space == MemorySpace::CUDA && i < size);
#else
            DNDS_check_throw_info(space == MemorySpace::Host && i < size, "runtime span: invalid host access or index");
#endif
            return data[i];
        }
    };

    struct BufferState;
    class Buffer;
    class CopyOperation;

    /// Optional allocator policy for pools/pinned memory. Returned ownership
    /// must free on the named device and remain valid until its last lease ends.
    /// Implementations return zero-initialized storage before publication.
    class Allocator
    {
    public:
        virtual ~Allocator() = default;
        virtual std::shared_ptr<void> Allocate(size_t bytes, Device device) = 0;
    };

    /// Move-only access owns its allocation until completion. Complete records
    /// a launch event; Wait releases access. Destruction waits as a safe fallback.
    /// A CUDA lease without Complete fences its stream before releasing storage.
    class Access
    {
        friend class Buffer;
        friend CopyOperation Copy(const Buffer &, Buffer &, ExecutionContext);
        std::shared_ptr<BufferState> _state;
        ExecutionContext _context;
        Completion _completion;
        bool _hasCompletion = false;
        bool _write = false;
        int _uncaughtOnAcquire = 0;
        Access(std::shared_ptr<BufferState> state, ExecutionContext context, bool write);

    protected:
        void *Pointer(size_t alignment, size_t elementSize) const;

    public:
        Access() = default;
        Access(const Access &) = delete;
        Access &operator=(const Access &) = delete;
        Access(Access &&other) noexcept;
        Access &operator=(Access &&other) noexcept;
        ~Access();
        void Complete(Completion completion);
        void Wait();
        /// Discard an unissued access after collective preflight failed. The
        /// caller guarantees no read/write work has used this lease.
        void CancelUnmodified();
        /// Drain submitted work, then make a potentially partial write unreadable.
        /// Recovery requires replacing the allocation, not accepting stale values.
        void Fail();
        bool Active() const { return bool(_state); }
        size_t Bytes() const;
        Device GetDevice() const;
    };

    class ReadAccess : public Access
    {
        friend class Buffer;
        explicit ReadAccess(Access access) : Access(std::move(access)) {}

    public:
        ReadAccess() = default;
        template <class T>
        Span<const T> View() const
        {
            static_assert(std::is_trivially_copyable_v<T>);
            auto *p = static_cast<const T *>(Pointer(alignof(T), sizeof(T)));
            return {p, Bytes() / sizeof(T), GetDevice().space};
        }
    };

    class WriteAccess : public Access
    {
        friend class Buffer;
        explicit WriteAccess(Access access) : Access(std::move(access)) {}

    public:
        WriteAccess() = default;
        template <class T>
        Span<T> View() const
        {
            static_assert(std::is_trivially_copyable_v<T>);
            auto *p = static_cast<T *>(Pointer(alignof(T), sizeof(T)));
            return {p, Bytes() / sizeof(T), GetDevice().space};
        }
    };

    /// A shallow, shared handle to exactly one allocation and access controller.
    /// Allocate creates independent storage; Copy creates independent values.
    /// Layout, mesh association and distributed routing belong in other objects.
    class Buffer
    {
        std::shared_ptr<BufferState> _state;
        explicit Buffer(std::shared_ptr<BufferState> state) : _state(std::move(state)) {}

    public:
        Buffer() = default;
        static Buffer Allocate(size_t bytes, Device device = Device::Host());
        static Buffer Allocate(size_t bytes, Device device, std::shared_ptr<Allocator> allocator);
        size_t Bytes() const;
        Device GetDevice() const;
        uint64_t Identity() const;
        uint64_t Version() const;
        bool IsIdle() const;
        bool IsValid() const;
        ReadAccess Read(ExecutionContext context = ExecutionContext::Host()) const;
        WriteAccess Write(ExecutionContext context = ExecutionContext::Host());

        /// Compatibility boundary only. Raw pointers cannot enforce future
        /// writes, synchronization or lifetimes. Native kernels use access leases.
        void *LegacyPointer() const;
    };

    class CopyOperation
    {
        friend CopyOperation Copy(const Buffer &, Buffer &, ExecutionContext);
        ReadAccess _source;
        WriteAccess _destination;

    public:
        CopyOperation() = default;
        CopyOperation(const CopyOperation &) = delete;
        CopyOperation &operator=(const CopyOperation &) = delete;
        CopyOperation(CopyOperation &&) noexcept = default;
        CopyOperation &operator=(CopyOperation &&) noexcept = default;
        void Wait();
    };

    /// Explicit equal-size copy. CUDA copies require a context matching every
    /// CUDA allocation involved; cross-device transfers are rejected for now.
    CopyOperation Copy(const Buffer &source, Buffer &destination,
                       ExecutionContext context = ExecutionContext::Host());
}
