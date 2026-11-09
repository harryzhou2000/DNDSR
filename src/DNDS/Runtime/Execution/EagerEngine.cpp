#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/MPI/Failure.hpp"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>

namespace DNDS::Runtime
{
    namespace
    {
        struct CPUCommand
        {
            StreamLauncher::Work work;
            StreamEvent dependency;
            bool wait = false, cleanup = false;
            StreamLauncher::Retirement retire;
        };
        struct CPULane
        {
            std::deque<CPUCommand> commands;
            uint64_t submitted = 0, completed = 0, failedAt = 0;
            std::exception_ptr error;
            bool running = false;
        };

        class CPUExecutor
        {
            std::mutex _mutex;
            std::condition_variable _changed;
            std::vector<std::weak_ptr<CPULane>> _lanes;
            std::vector<std::thread> _workers;
            size_t _next = 0;
            bool _stopping = false;
            const bool _solverFailures = MPI::SolverFailureScopeActive();

            bool Step()
            {
                std::shared_ptr<CPULane> lane;
                CPUCommand command;
                std::exception_ptr priorError;
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    const auto count = _lanes.size();
                    for (size_t i = 0; i < count; ++i)
                    {
                        const auto candidate = (_next + i) % count;
                        auto current = _lanes[candidate].lock();
                        if (current && !current->running && !current->commands.empty())
                        {
                            lane = std::move(current);
                            lane->running = true;
                            command = std::move(lane->commands.front());
                            lane->commands.pop_front();
                            priorError = lane->error;
                            _next = (candidate + 1) % count;
                            break;
                        }
                    }
                }
                if (!lane)
                    return false;
                std::exception_ptr error;
                bool ready = true;
                try
                {
                    // Invoke backend/event/user code with no executor lock held.
                    // An unresolved dependency is put back; no worker blocks.
                    if (command.wait)
                        ready = command.dependency.Ready();
                    if (ready && !command.wait && (!priorError || command.cleanup))
                        command.work(ExecutionContext::Host());
                }
                catch (...)
                {
                    if (_solverFailures)
                    {
                        MPI::AbortIfSolverFailure("solver CPU command failed");
                        MPI::AbortCaught(MPI_COMM_NULL, "solver CPU worker failed");
                    }
                    error = std::current_exception();
                }
                if (ready && command.retire)
                    command.retire(error ? error : priorError);
                // Destroy captures before advancing completion: owning CPU
                // resources and cleanup belong to this ordered command.
                if (ready)
                    command = {};
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    if (ready)
                    {
                        ++lane->completed;
                        if (error && !lane->error)
                        {
                            lane->error = error;
                            lane->failedAt = lane->completed;
                        }
                    }
                    else
                        lane->commands.push_front(std::move(command));
                    lane->running = false;
                }
                _changed.notify_all();
                return ready;
            }
            void Worker()
            {
                for (;;)
                {
                    if (Step())
                        continue;
                    std::unique_lock<std::mutex> lock(_mutex);
                    if (_stopping)
                        return;
                    // External completions may not signal our condition variable.
                    _changed.wait_for(lock, std::chrono::milliseconds(1));
                }
            }

        public:
            explicit CPUExecutor(size_t threads)
            {
                try
                {
                    for (size_t i = 0; i < threads; ++i)
                        _workers.emplace_back([this]
                                              { Worker(); });
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("solver CPU worker creation failed");
                    {
                        std::lock_guard<std::mutex> lock(_mutex);
                        _stopping = true;
                    }
                    _changed.notify_all();
                    for (auto &worker : _workers)
                        worker.join();
                    throw;
                }
            }
            ~CPUExecutor()
            {
                if (std::uncaught_exceptions())
                    MPI::AbortIfSolverFailure("solver unwinding before CPU worker join");
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _stopping = true;
                }
                _changed.notify_all();
                for (auto &worker : _workers)
                    worker.join();
            }
            std::shared_ptr<CPULane> NewLane()
            {
                auto lane = std::make_shared<CPULane>();
                std::lock_guard<std::mutex> lock(_mutex);
                // Reuse expired slots instead of retaining all historical lanes.
                for (auto &slot : _lanes)
                    if (slot.expired())
                    {
                        slot = lane;
                        return lane;
                    }
                _lanes.emplace_back(lane);
                return lane;
            }
            void Submit(const std::shared_ptr<CPULane> &lane, CPUCommand command)
            {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    DNDS_check_throw_info(!_stopping && lane->submitted < UINT64_MAX, "CPU execution lane unavailable");
                    lane->commands.push_back(std::move(command));
                    ++lane->submitted;
                }
                _changed.notify_all();
                if (_workers.empty())
                    while (Step())
                    {
                    }
            }
            uint64_t Tail(const std::shared_ptr<CPULane> &lane)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return lane->submitted;
            }
            bool Ready(const std::shared_ptr<CPULane> &lane, uint64_t prefix)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (lane->completed < prefix)
                    return false;
                if (lane->error && lane->failedAt <= prefix)
                    std::rethrow_exception(lane->error);
                return true;
            }
            bool Poll(const std::shared_ptr<CPULane> &lane, uint64_t prefix)
            {
                if (Ready(lane, prefix))
                    return true;
                if (_workers.empty())
                    Step(); // Allows an imported inline-executor dependency to progress.
                return Ready(lane, prefix);
            }
            void Wait(const std::shared_ptr<CPULane> &lane, uint64_t prefix)
            {
                while (!Ready(lane, prefix))
                {
                    if (_workers.empty() && Step())
                        continue;
                    std::unique_lock<std::mutex> lock(_mutex);
                    _changed.wait_for(lock, std::chrono::milliseconds(1));
                }
            }
        };

        class CPUEvent : public StreamEvent::State
        {
            std::shared_ptr<CPUExecutor> _executor;
            std::shared_ptr<CPULane> _lane;
            uint64_t _prefix;

        public:
            CPUEvent(std::shared_ptr<CPUExecutor> executor, std::shared_ptr<CPULane> lane, uint64_t prefix)
                : _executor(std::move(executor)), _lane(std::move(lane)), _prefix(prefix) {}
            bool Ready() const override { return _executor->Poll(_lane, _prefix); }
            void Wait() const override { _executor->Wait(_lane, _prefix); }
        };

        class CPULauncher : public StreamLauncher
        {
            std::shared_ptr<CPUExecutor> _executor;
            std::shared_ptr<CPULane> _lane;

        public:
            explicit CPULauncher(std::shared_ptr<CPUExecutor> executor)
                : _executor(std::move(executor)), _lane(_executor->NewLane()) {}
            ExecutionContext Context() const override { return ExecutionContext::Host(); }
            void Submit(Work work, CommandKind kind, Retirement retire) override
            {
                DNDS_check_throw_info(work, "empty CPU command");
                _executor->Submit(_lane, {std::move(work), {}, false, kind != CommandKind::Compute, std::move(retire)});
            }
            void WaitFor(StreamEvent event) override { _executor->Submit(_lane, {{}, std::move(event), true, false}); }
            StreamEvent Record() override
            {
                return StreamEvent(std::make_shared<CPUEvent>(_executor, _lane, _executor->Tail(_lane)));
            }
            void Drain() override { _executor->Wait(_lane, _executor->Tail(_lane)); }
            Buffer Allocate(size_t bytes) override { return Buffer::Allocate(bytes); }
            Buffer Allocate(size_t bytes, AllocationOptions options) override { return Buffer::Allocate(bytes, Device::Host(), options); }
        };
    }

    struct EagerEngine::State
    {
        std::shared_ptr<CPUExecutor> host;
        std::shared_ptr<const int> identity = std::make_shared<const int>(0);
        std::vector<std::shared_ptr<StreamLauncher>> streams;
        std::shared_ptr<AccessTrackingContext> tracking;
        std::mutex mutex;
        explicit State(size_t threads) : host(std::make_shared<CPUExecutor>(threads)) {}
    };
    EagerEngine::EagerEngine(size_t threads, std::shared_ptr<AccessTrackingContext> tracking) : _state(std::make_unique<State>(threads))
    {
        _state->tracking = std::move(tracking);
    }
    EagerEngine::~EagerEngine()
    {
        if (std::uncaught_exceptions())
            MPI::AbortIfSolverFailure("solver unwinding before stream shutdown");
        try
        {
            Synchronize();
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("solver stream shutdown failed");
        } // Explicit Synchronize reports errors; shutdown still drains.
    }
    ExecutionStream EagerEngine::CreateCPUStream() { return CreateStream(std::make_shared<CPULauncher>(_state->host)); }
    bool EagerEngine::ChecksAccess() const { return bool(_state->tracking); }
    ExecutionStream EagerEngine::CreateStream(std::shared_ptr<StreamLauncher> launcher)
    {
        DNDS_check_throw_info(launcher, "stream needs a launcher");
        std::lock_guard<std::mutex> lock(_state->mutex);
        _state->streams.push_back(launcher);
        return ExecutionStream(std::move(launcher), _state->identity, _state->tracking,
                               _state->tracking ? _state->tracking->NewLane() : 0);
    }
    void EagerEngine::Submit(const ExecutionStream &stream, StreamLauncher::Work work)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work, "invalid eager submission or foreign stream");
        SubmitOperation(stream, std::move(work), [](std::exception_ptr) {});
    }
    Buffer EagerEngine::AllocateBuffer(const ExecutionStream &stream, size_t bytes, AllocationOptions options)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity, "invalid allocation stream");
        if (!_state->tracking)
            return stream._launcher->Allocate(bytes, options);
        auto buffer = stream._launcher->AllocateChecked(bytes, options);
        // Allocation/zero initialization is itself a producer, even when native
        // allocation returned a host handle before its device work completed.
        SubmitOperation(stream, [](ExecutionContext) {}, [](std::exception_ptr) {}, {{buffer, AccessMode::Write}});
        return buffer;
    }
    void EagerEngine::SubmitOperation(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                                      std::vector<ScheduledBufferAccess> accesses)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work && retire,
                              "invalid eager operation or foreign stream");
        if (!_state->tracking)
            stream._launcher->Submit(std::move(work), CommandKind::Compute, std::move(retire));
        else
            _state->tracking->Submit(stream._lane, accesses, [&](AccessTrackingContext::Submission submission)
                                     { stream._launcher->Submit(std::move(work), CommandKind::Compute,
                                                                [submission, retire = std::move(retire)](std::exception_ptr error) noexcept
                                                                {
                                                                    retire(error);
                                                                    submission.Complete();
                                                                }); });
    }
    void EagerEngine::SubmitCleanup(const ExecutionStream &stream, StreamLauncher::Work work, std::vector<ScheduledBufferAccess> accesses)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work, "invalid eager cleanup or foreign stream");
        if (!_state->tracking)
            stream._launcher->Submit(std::move(work), CommandKind::Cleanup);
        else
            _state->tracking->Submit(stream._lane, accesses, [&](AccessTrackingContext::Submission submission)
                                     { stream._launcher->Submit(std::move(work), CommandKind::Cleanup,
                                                                [submission](std::exception_ptr) noexcept
                                                                { submission.Complete(); }); });
    }
    void EagerEngine::SubmitOrderedRelease(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire,
                                           std::vector<ScheduledBufferAccess> accesses)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work && retire,
                              "invalid eager release or foreign stream");
        if (!_state->tracking)
            stream._launcher->Submit(std::move(work), CommandKind::OrderedRelease, std::move(retire));
        else
            _state->tracking->Submit(stream._lane, accesses, [&](AccessTrackingContext::Submission submission)
                                     { stream._launcher->Submit(std::move(work), CommandKind::OrderedRelease,
                                                                [submission, retire = std::move(retire)](std::exception_ptr error) noexcept
                                                                { retire(error); submission.Complete(); }); });
    }
    void EagerEngine::Synchronize()
    {
        std::vector<std::shared_ptr<StreamLauncher>> streams;
        {
            std::lock_guard<std::mutex> lock(_state->mutex);
            streams = _state->streams;
        }
        std::exception_ptr first;
        for (const auto &stream : streams)
            try
            {
                stream->Drain();
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("solver stream synchronization failed");
                if (!first)
                    first = std::current_exception();
            }
        if (first)
            std::rethrow_exception(first);
    }
    ExecutionContext ExecutionStream::Context() const
    {
        DNDS_check_throw_info(_launcher, "empty execution stream");
        return _launcher->Context();
    }
    void ExecutionStream::WaitFor(StreamEvent event) const
    {
        DNDS_check_throw_info(_launcher, "empty execution stream");
        if (_tracking)
        {
            const auto order = event._order;
            _tracking->WaitFor(_lane, order, [&]
                               { _launcher->WaitFor(std::move(event)); });
        }
        else
            _launcher->WaitFor(std::move(event));
    }
    StreamEvent ExecutionStream::Record() const
    {
        DNDS_check_throw_info(_launcher, "empty execution stream");
        if (!_tracking)
            return _launcher->Record();
        StreamEvent event;
        const auto order = _tracking->Record(_lane, [&]
                                             { event = _launcher->Record(); });
        event._order = order;
        return event;
    }
}
