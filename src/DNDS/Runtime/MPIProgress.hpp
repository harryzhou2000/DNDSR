#pragma once

#include "RowTransfer.hpp"
#include "EagerEngine.hpp"
#include <atomic>
#include <thread>

namespace DNDS::Runtime
{
    /// Portable host progress. MPI is called only on this object's creating
    /// thread. Worker streams poll completion without entering MPI. Drain this
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
                    error = std::current_exception();
                    complete = true;
                }
                if (complete)
                {
                    operation = {}; // Release MPI-owned state on the MPI thread.
                    done.store(true, std::memory_order_release);
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
            try
            {
                Drain();
            }
            catch (...)
            {
            } // Explicit Drain/Record observation reports errors.
        }
        StreamEvent Import(TransferOperation operation)
        {
            CheckThread();
            Poll();
            auto pending = std::make_shared<Pending>(std::move(operation));
            _pending.push_back(pending);
            return StreamEvent(std::move(pending));
        }
        void Poll()
        {
            CheckThread();
            std::exception_ptr error;
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
