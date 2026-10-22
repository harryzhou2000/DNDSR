#pragma once

#include "Buffer.hpp"
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
    };

    class ExecutionStream
    {
        friend class EagerEngine;
        std::shared_ptr<StreamLauncher> _launcher;
        std::shared_ptr<const int> _owner;
        ExecutionStream(std::shared_ptr<StreamLauncher> launcher, std::shared_ptr<const int> owner)
            : _launcher(std::move(launcher)), _owner(std::move(owner)) {}

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
    /// independent use before a final release. No eager schedule proof exists:
    /// omitted edges can silently produce stale values or unsafe memory access.
    /// @pre CPU work must not block waiting for work that needs the same bounded
    /// pool; express dependencies with WaitFor. Synchronize explicitly to observe
    /// failures. Concurrent mutation of one public Array handle is not supported.
    /// See docs/architecture/RuntimeCallerContract.md for the complete contract.
    class EagerEngine
    {
        struct State;
        std::unique_ptr<State> _state;

    public:
        explicit EagerEngine(size_t hostThreads = 0);
        EagerEngine(const EagerEngine &) = delete;
        EagerEngine &operator=(const EagerEngine &) = delete;
        ~EagerEngine();
        ExecutionStream CreateCPUStream();
        ExecutionStream CreateStream(std::shared_ptr<StreamLauncher> launcher);
        /// CPU allocation is ready immediately; CUDA allocation/zeroing follows
        /// this stream. Record/Wait exports readiness when another stream needs it.
        Buffer AllocateBuffer(const ExecutionStream &stream, size_t bytes);
        void Submit(const ExecutionStream &stream, StreamLauncher::Work work);
        /// Backend/operator integration; no per-kernel public completion token.
        void SubmitOperation(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire);
        /// Ordered cleanup still runs after a preceding computation failed.
        void SubmitCleanup(const ExecutionStream &stream, StreamLauncher::Work work);
        void SubmitOrderedRelease(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire);
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
                                             { retained->reset(); });
                        return;
                    }
                SubmitCleanup(stream, [retained](ExecutionContext)
                              { retained->reset(); });
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
