#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"
#include "DNDS/Runtime/Execution/AccessTracking.hpp"
#include <functional>
#include <vector>
#include <optional>

namespace DNDS::Runtime
{
    enum class CommandKind
    {
        Compute,
        Cleanup,
        OrderedRelease
    };
    namespace Detail
    {
        template <class T, class = void>
        struct HasOrderedRelease : std::false_type
        {
        };
        template <class T>
        struct HasOrderedRelease<T, std::void_t<decltype(std::declval<const T &>().CanReleaseOrdered(ExecutionContext{})),
                                                decltype(std::declval<T &>().ReleaseOn(ExecutionContext{}))>> : std::true_type
        {
        };
        template <class T, class = void>
        struct HasReleaseBuffers : std::false_type
        {
        };
        template <class T>
        struct HasReleaseBuffers<T, std::void_t<decltype(std::declval<const T &>().ReleaseBuffers())>> : std::true_type
        {
        };
    }
    /// Backend-owned completion for an explicit stream prefix. Ordinary kernel
    /// submission does not create or return one of these objects.
    class StreamEvent
    {
    public:
        class State
        {
        public:
            virtual ~State() = default;
            virtual bool Ready() const = 0;
            virtual void Wait() const = 0;
        };

    private:
        std::shared_ptr<const State> _state;
        std::shared_ptr<const AccessOrder> _order;
        friend class ExecutionStream;
        friend class TransferBinding;
        friend class MPIProgress;

    public:
        StreamEvent() = default;
        explicit StreamEvent(std::shared_ptr<const State> state) : _state(std::move(state)) {}
        bool Ready() const { return !_state || _state->Ready(); }
        void WaitHost() const
        {
            if (_state)
                _state->Wait();
        }
        const std::shared_ptr<const State> &Provider() const { return _state; }
    };

    /// One ordered lane. Launchers own queueing, dependencies and completion;
    /// the engine uses this contract for inline/worker CPU and native devices.
    class StreamLauncher
    {
    public:
        using Work = std::function<void(ExecutionContext)>;
        using Retirement = std::function<void(std::exception_ptr)>;
        virtual ~StreamLauncher() = default;
        virtual ExecutionContext Context() const = 0;
        /// Rejection throws before accepting ownership; an accepted command's
        /// execution errors are reported by its completion/drain path instead.
        /// Retirement runs once after actual completion (or accepted work is
        /// skipped after failure), before the stream prefix becomes complete.
        /// It must not throw. Rejected work is not retired by the launcher.
        virtual void Submit(Work work, CommandKind kind, Retirement retire = {}) = 0;
        virtual void WaitFor(StreamEvent event) = 0;
        virtual StreamEvent Record() = 0;
        virtual void Drain() = 0;
        virtual Buffer Allocate(size_t)
        {
            DNDS_check_throw_info(false, "stream provider has no allocation capability");
            return {};
        }
        /// Provider extension for requested alignment. Existing custom launchers
        /// keep their allocation hook; this fallback fails if it cannot honor
        /// the requested address alignment. Native providers override it.
        virtual Buffer Allocate(size_t bytes, AllocationOptions options)
        {
            DNDS_check_throw_info(!options.alignmentBytes || !(options.alignmentBytes & (options.alignmentBytes - 1)),
                                  "allocation alignment must be a power of two");
            auto result = Allocate(bytes);
            DNDS_check_throw_info(!options.alignmentBytes ||
                                      reinterpret_cast<uintptr_t>(result.LegacyPointer()) % options.alignmentBytes == 0,
                                  "stream allocator cannot satisfy requested alignment");
            return result;
        }
        /// Native asynchronous initialization must use checked reservations when
        /// called by a checked engine. Ready CPU providers use the default.
        virtual Buffer AllocateChecked(size_t bytes, AllocationOptions options) { return Allocate(bytes, options); }
    };

    class ExecutionStream
    {
        friend class EagerEngine;
        friend class TransferBinding;
        std::shared_ptr<StreamLauncher> _launcher;
        std::shared_ptr<const int> _owner;
        std::shared_ptr<AccessTrackingContext> _tracking;
        uint64_t _lane = 0;
        std::shared_ptr<void> _laneLifetime;
        ExecutionStream(std::shared_ptr<StreamLauncher> launcher, std::shared_ptr<const int> owner,
                        std::shared_ptr<AccessTrackingContext> tracking = {}, uint64_t lane = 0)
            : _launcher(std::move(launcher)), _owner(std::move(owner)), _tracking(std::move(tracking)), _lane(lane)
        {
            if (_tracking)
                _laneLifetime = std::shared_ptr<void>(_tracking.get(), [tracking = _tracking, lane](void *)
                                                      { tracking->RetireLane(lane); });
        }

    public:
        ExecutionStream() = default;
        ExecutionContext Context() const;
        /// Caller supplies every cross-stream dependency; waiting is not inferred
        /// from Array aliases. CUDA rejects an unfinished portable MPI/CPU event.
        void WaitFor(StreamEvent event) const;
        /// Capture only the already-submitted prefix. Later work is not covered;
        /// caller must record again before exporting that work's readiness.
        StreamEvent Record() const;
    };

    /// Fully eager execution. The caller supplies dependency/release ordering.
    /// HostThreads=0 runs ready CPU work inline; >0 shares a bounded worker pool
    /// between all CPU lanes. Waiting lanes never consume blocked workers.
    /// @pre Caller orders all conflicting scheduled accesses and joins every
    /// independent use before a final release. An optional shared tracking
    /// context checks declared accesses at submission. Unchecked eager mode
    /// leaves this proof with the caller; omitted edges can corrupt values.
    /// @pre CPU work must not block waiting for work that needs the same bounded
    /// pool; express dependencies with WaitFor. Synchronize explicitly to observe
    /// failures. Concurrent mutation of one public Array handle is not supported.
    /// See docs/architecture/RuntimeCallerContract.md for the complete contract.
    class EagerEngine
    {
        struct State;
        std::unique_ptr<State> _state;

    public:
        explicit EagerEngine(size_t hostThreads = 0, std::shared_ptr<AccessTrackingContext> tracking = {});
        EagerEngine(const EagerEngine &) = delete;
        EagerEngine &operator=(const EagerEngine &) = delete;
        ~EagerEngine();
        ExecutionStream CreateCPUStream();
        ExecutionStream CreateStream(std::shared_ptr<StreamLauncher> launcher);
        bool ChecksAccess() const;
        /// CPU allocation is ready immediately; CUDA allocation/zeroing follows
        /// this stream. Record/Wait exports readiness when another stream needs it.
        Buffer AllocateBuffer(const ExecutionStream &stream, size_t bytes, AllocationOptions options = {});
        void Submit(const ExecutionStream &stream, StreamLauncher::Work work);
        /// Backend/operator integration; no per-kernel public completion token.
        void SubmitOperation(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                             std::vector<ScheduledBufferAccess> accesses = {});
        /// Ordered cleanup still runs after a preceding computation failed.
        void SubmitCleanup(const ExecutionStream &stream, StreamLauncher::Work work, std::vector<ScheduledBufferAccess> accesses = {});
        void SubmitOrderedRelease(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                                  std::vector<ScheduledBufferAccess> accesses = {});
        /// Consume one handle, not all aliases. Caller must join every independent
        /// use onto stream before any potentially final release. Different alias
        /// release streams are legal only when the actual final decrement follows
        /// all uses; reference counting alone does not establish that ordering.
        template <class T>
        void ReleaseOn(T &&handle, const ExecutionStream &stream)
        {
            static_assert(!std::is_lvalue_reference_v<T>, "ReleaseOn consumes a moved handle");
            static_assert(std::is_nothrow_move_assignable_v<std::decay_t<T>>, "release handles need noexcept move assignment");
            DNDS_check_throw_info(stream._launcher, "invalid release stream");
            std::vector<ScheduledBufferAccess> accesses;
            if (ChecksAccess())
            {
                if constexpr (Detail::HasReleaseBuffers<std::decay_t<T>>::value)
                {
                    for (const auto &buffer : handle.ReleaseBuffers())
                        if (buffer.Identity())
                            accesses.push_back({buffer, AccessMode::Write});
                }
                else
                    DNDS_check_throw_info(false, "checked release needs explicit ReleaseBuffers allocation declarations");
            }
            bool ordered = false;
            if constexpr (Detail::HasOrderedRelease<std::decay_t<T>>::value)
                ordered = handle.CanReleaseOrdered(stream.Context());
            auto retained = std::make_shared<std::optional<std::decay_t<T>>>(std::in_place, std::move(handle));
            try
            {
                if constexpr (Detail::HasOrderedRelease<std::decay_t<T>>::value)
                    if (ordered)
                    {
                        SubmitOrderedRelease(stream, [retained](ExecutionContext context)
                                             { retained->value().ReleaseOn(context); retained->reset(); }, [retained](std::exception_ptr) noexcept
                                             { retained->reset(); }, std::move(accesses));
                        return;
                    }
                SubmitCleanup(stream, [retained](ExecutionContext)
                              { retained->reset(); }, std::move(accesses));
            }
            catch (...)
            {
                handle = std::move(retained->value()); // Rejected: return ownership.
                throw;
            }
        }
        /// Drain all streams even if one fails, then rethrow the first failure.
        void Synchronize();
    };
}
