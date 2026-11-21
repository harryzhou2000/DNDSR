#include "DNDS/Runtime/Execution/SYCLLauncher.hpp"
#include "DNDS/Runtime/Execution/SYCLQueue.hpp"
#include <deque>

namespace DNDS::Runtime
{
    namespace
    {
        struct Command
        {
            StreamLauncher::Work work;
            StreamLauncher::Retirement retire;
            CommandKind kind;
            std::exception_ptr error;
        };

        // Run host retirements on observation. A SYCL host_task is unsuitable
        // here because dropping the last Array may itself call into SYCL.
        struct Batch
        {
            std::vector<Command> commands;
            std::vector<StreamEvent> dependencies;

            void Finish(const ExecutionContext &context, std::exception_ptr &error)
            {
                for (const auto &event : dependencies)
                    try
                    {
                        event.WaitHost();
                    }
                    catch (...)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                dependencies.clear();
                for (auto &command : commands)
                {
                    if (!error)
                        error = command.error;
                    if (command.kind == CommandKind::Cleanup)
                        try
                        {
                            command.work(context);
                        }
                        catch (...)
                        {
                            if (!error)
                                error = std::current_exception();
                        }
                    if (command.retire)
                        command.retire(error);
                }
                commands.clear();
            }
        };

        class Prefix final : public StreamEvent::State
        {
            mutable std::once_flag _once;
            mutable Batch _batch;
            mutable std::exception_ptr _error;
            mutable ExecutionContext _context;

        public:
            const Device device;
            const std::weak_ptr<StreamResource> origin;
            const sycl::event native;

            Prefix(ExecutionContext context, Batch &batch)
                : _context(context), device(context.device), origin(context.stream),
                  native(SYCLQueue(context).ext_oneapi_submit_barrier())
            {
                std::swap(_batch, batch);
            }
            unsigned AutonomousWaitPriority() const noexcept override { return 2; }
            void Wait() const override
            {
                std::call_once(_once, [&]
                               {
                    try { auto event = native; event.wait_and_throw(); }
                    catch (...) { _error = std::current_exception(); }
                    _batch.Finish(_context, _error);
                    _context.stream.reset(); });
                NotifyCompletion();
                if (_error)
                    std::rethrow_exception(_error);
            }
            bool Ready() const override
            {
                if (native.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete)
                    return false;
                Wait();
                return true;
            }
        };

        class Launcher final : public StreamLauncher
        {
            ExecutionContext _context;
            Batch _batch;
            std::exception_ptr _error;
            std::deque<std::shared_ptr<Prefix>> _issued;
            StreamEvent _previous;
            std::recursive_mutex _mutex;
            bool _issuing = false;

            void Poll()
            {
                while (!_issued.empty())
                {
                    try
                    {
                        if (!_issued.front()->Ready())
                            return;
                    }
                    catch (...)
                    {
                        if (!_error)
                            _error = std::current_exception();
                    }
                    _issued.pop_front();
                }
            }

        public:
            explicit Launcher(Device device) : _context(CreateSYCLQueue(device)) {}
            ~Launcher() override
            {
                if (!_context.stream->IsRetired())
                    try
                    {
                        Drain();
                    }
                    catch (...)
                    {
                    }
            }
            ExecutionContext Context() const override { return _context; }
            Buffer Allocate(size_t bytes) override { return Allocate(bytes, {}); }
            Buffer Allocate(size_t bytes, AllocationOptions options) override
            {
                // USM setup allocation returns zero-initialized, ready storage.
                return Buffer::Allocate(bytes, _context.device, options);
            }
            void Submit(Work work, CommandKind kind, Retirement retire) override
            {
                std::lock_guard lock(_mutex);
                DNDS_check_throw_info(work && !_issuing, "empty or recursive SYCL submission");
                Poll();
                _batch.commands.push_back({std::move(work), std::move(retire), kind, _error});
                auto &command = _batch.commands.back();
                if (kind == CommandKind::Cleanup || (_error && kind != CommandKind::OrderedRelease))
                    return;
                _issuing = true;
                try
                {
                    command.work(_context);
                }
                catch (...)
                {
                    command.error = _error = std::current_exception();
                }
                _issuing = false;
            }
            void WaitFor(StreamEvent event) override
            {
                std::lock_guard lock(_mutex);
                DNDS_check_throw_info(!_issuing, "recursive SYCL dependency import");
                const auto prefix = std::dynamic_pointer_cast<const Prefix>(event.Provider());
                if (!prefix)
                {
                    DNDS_check_throw_info(event.Ready(), "pending foreign dependency needs an explicit host wait before SYCL submission");
                    return;
                }
                DNDS_check_throw_info(prefix->device == _context.device, "SYCL dependency belongs to another Runtime context");
                if (prefix->origin.lock() == _context.stream)
                    return;
                _batch.dependencies.push_back(std::move(event));
                try
                {
                    SYCLQueue(_context).ext_oneapi_submit_barrier({prefix->native});
                }
                catch (...)
                {
                    _batch.dependencies.pop_back();
                    throw;
                }
            }
            StreamEvent Record() override
            {
                std::lock_guard lock(_mutex);
                DNDS_check_throw_info(!_issuing, "recursive SYCL prefix recording");
                Poll();
                if (_batch.commands.empty() && _batch.dependencies.empty())
                    return _previous;
                if (_previous.Provider())
                    _batch.dependencies.push_back(_previous);
                // Reserve the owner before taking the batch from the launcher.
                _issued.emplace_back();
                try
                {
                    _issued.back() = std::make_shared<Prefix>(_context, _batch);
                }
                catch (...)
                {
                    _issued.pop_back();
                    throw;
                }
                _previous = StreamEvent(_issued.back());
                return _previous;
            }
            void Drain() override
            {
                std::lock_guard lock(_mutex);
                DNDS_check_throw_info(!_issuing, "recursive SYCL drain");
                // Also covers direct queue borrows and a failure to record an
                // event. The engine has already progressed external requests.
                try
                {
                    SYCLDetail::Synchronize(_context);
                }
                catch (...)
                {
                    if (!_error)
                        _error = std::current_exception();
                }
                for (const auto &prefix : _issued)
                    try
                    {
                        prefix->Wait();
                    }
                    catch (...)
                    {
                        if (!_error)
                            _error = std::current_exception();
                    }
                _issued.clear();
                _batch.Finish(_context, _error);
                if (_error)
                    std::rethrow_exception(_error);
            }
        };
    }

    std::shared_ptr<StreamLauncher> CreateSYCLLauncher(Device device)
    {
        return std::make_shared<Launcher>(device);
    }
}
