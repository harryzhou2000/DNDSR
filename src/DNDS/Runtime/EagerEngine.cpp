#include "EagerEngine.hpp"
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
        };
    }

    struct EagerEngine::State
    {
        std::shared_ptr<CPUExecutor> host;
        std::shared_ptr<const int> identity = std::make_shared<const int>(0);
        std::vector<std::shared_ptr<StreamLauncher>> streams;
        std::mutex mutex;
        explicit State(size_t threads) : host(std::make_shared<CPUExecutor>(threads)) {}
    };
    EagerEngine::EagerEngine(size_t threads) : _state(std::make_unique<State>(threads)) {}
    EagerEngine::~EagerEngine()
    {
        try
        {
            Synchronize();
        }
        catch (...)
        {
        } // Explicit Synchronize reports errors; shutdown still drains.
    }
    ExecutionStream EagerEngine::CreateCPUStream() { return CreateStream(std::make_shared<CPULauncher>(_state->host)); }
    ExecutionStream EagerEngine::CreateStream(std::shared_ptr<StreamLauncher> launcher)
    {
        DNDS_check_throw_info(launcher, "stream needs a launcher");
        std::lock_guard<std::mutex> lock(_state->mutex);
        _state->streams.push_back(launcher);
        return ExecutionStream(std::move(launcher), _state->identity);
    }
    void EagerEngine::Submit(const ExecutionStream &stream, StreamLauncher::Work work)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work, "invalid eager submission or foreign stream");
        stream._launcher->Submit(std::move(work), CommandKind::Compute);
    }
    Buffer EagerEngine::AllocateBuffer(const ExecutionStream &stream, size_t bytes)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity, "invalid allocation stream");
        return stream._launcher->Allocate(bytes);
    }
    void EagerEngine::SubmitOperation(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work && retire,
                              "invalid eager operation or foreign stream");
        stream._launcher->Submit(std::move(work), CommandKind::Compute, std::move(retire));
    }
    void EagerEngine::SubmitCleanup(const ExecutionStream &stream, StreamLauncher::Work work)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work, "invalid eager cleanup or foreign stream");
        stream._launcher->Submit(std::move(work), CommandKind::Cleanup);
    }
    void EagerEngine::SubmitOrderedRelease(const ExecutionStream &stream, StreamLauncher::Work work, StreamLauncher::Retirement retire)
    {
        DNDS_check_throw_info(stream._launcher && stream._owner == _state->identity && work && retire,
                              "invalid eager release or foreign stream");
        stream._launcher->Submit(std::move(work), CommandKind::OrderedRelease, std::move(retire));
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
        _launcher->WaitFor(std::move(event));
    }
    StreamEvent ExecutionStream::Record() const
    {
        DNDS_check_throw_info(_launcher, "empty execution stream");
        return _launcher->Record();
    }
}
