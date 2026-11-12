#pragma once

#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include <atomic>
#include <thread>

namespace DNDS::Runtime
{
    /// Portable host progress. MPI is called only on this object's creating
    /// thread. Completion notifies waiting worker streams without entering MPI
    /// on those workers. Drain this
    /// object before synchronizing dependent worker streams or submitting a
    /// dependent CUDA stream (portable MPI is not a native CUDA stream node).
    /// Caller initializes MPI with at least FUNNELED for worker execution and
    /// uses the MPI main thread at that level. Drain/import/destruction belong
    /// to the creating thread before MPI_Finalize. Do not rely on MPI background
    /// progress or synchronize a dependent worker while withholding host progress.
    class MPIProgress
    {
        struct Pending final : StreamEvent::State
        {
            mutable TransferOperation operation;
            const std::thread::id owner = std::this_thread::get_id();
            mutable std::atomic<bool> done{false};
            mutable std::exception_ptr error;
            explicit Pending(TransferOperation op) : operation(std::move(op)) {}
            void Finish(bool wait) const
            {
                DNDS_check_throw_info(std::this_thread::get_id() == owner, "MPI progress requires its creating thread");
                if (done.load(std::memory_order_acquire))
                    return;
                bool complete = false;
                try
                {
                    if (wait)
                    {
                        operation.Wait();
                        complete = true;
                    }
                    else
                        complete = operation.Test();
                }
                catch (...)
                {
                    MPI::AbortIfSolverFailure("MPI progress failed");
                    error = std::current_exception();
                    complete = true;
                }
                if (complete)
                {
                    operation = {}; // Release MPI-owned state on the MPI thread.
                    done.store(true, std::memory_order_release);
                    NotifyCompletion();
                }
            }
            bool Ready() const override
            {
                if (std::this_thread::get_id() == owner)
                    Finish(false);
                if (!done.load(std::memory_order_acquire))
                    return false;
                if (error)
                    std::rethrow_exception(error);
                return true;
            }
            void Wait() const override
            {
                if (!done.load(std::memory_order_acquire))
                    Finish(true);
                if (error)
                    std::rethrow_exception(error);
            }
        };
        std::thread::id _owner = std::this_thread::get_id();
        std::vector<std::shared_ptr<Pending>> _pending;
        std::vector<StreamEvent> _managed;
        void CheckThread() const
        {
            DNDS_check_throw_info(std::this_thread::get_id() == _owner, "MPI progress requires its creating thread");
        }

    public:
        MPIProgress()
        {
            int level = 0, main = 0;
            MPI_Query_thread(&level);
            MPI_Is_thread_main(&main);
            DNDS_check_throw_info(level >= MPI_THREAD_SERIALIZED || main, "MPI thread level requires main-thread progress");
        }
        MPIProgress(const MPIProgress &) = delete;
        MPIProgress &operator=(const MPIProgress &) = delete;
        ~MPIProgress()
        {
            if (std::this_thread::get_id() != _owner)
                std::terminate();
            if (std::uncaught_exceptions())
                MPI::AbortIfSolverFailure("solver unwinding before MPI progress shutdown");
            if (std::uncaught_exceptions())
                for (const auto &pending : _pending)
                    if (!pending->done.load(std::memory_order_acquire))
                        DNDS::assert_false_info("no exception with pending MPI progress", __FILE__, __LINE__,
                                                "cannot drain MPI progress during exception unwinding");
            try
            {
                Drain();
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("MPI progress failed");
            } // Explicit Drain/Record observation reports errors.
        }
        StreamEvent Import(TransferOperation operation)
        {
            CheckThread();
            Poll(); // Bound retained completion records in repeated import/wait loops.
            // Managed begins already live in the engine inventory. Export its
            // event directly: a worker observes publication without another
            // standalone progress object needing to be polled first.
            auto managed = operation.ManagedCompletion();
            if (managed.Provider())
            {
                _managed.push_back(managed);
                return managed;
            }
            auto order = operation.CheckedOrder();
            auto pending = std::make_shared<Pending>(std::move(operation));
            _pending.push_back(pending);
            StreamEvent event(std::move(pending));
            event._order = std::move(order);
            return event;
        }
        void Poll()
        {
            CheckThread();
            std::exception_ptr error;
            for (auto it = _managed.begin(); it != _managed.end();)
                if (it->Ready())
                    it = _managed.erase(it);
                else
                    ++it;
            for (auto it = _pending.begin(); it != _pending.end();)
            {
                (*it)->Finish(false);
                if ((*it)->done.load(std::memory_order_acquire))
                {
                    if ((*it)->error && !error)
                        error = (*it)->error;
                    it = _pending.erase(it);
                }
                else
                    ++it;
            }
            if (error)
                std::rethrow_exception(error);
        }
        void Drain()
        {
            CheckThread();
            std::exception_ptr error;
            for (const auto &event : _managed)
                event.WaitHost();
            _managed.clear();
            for (const auto &pending : _pending)
            {
                pending->Finish(true);
                if (pending->error && !error)
                    error = pending->error;
            }
            _pending.clear();
            if (error)
                std::rethrow_exception(error);
        }
    };
}
