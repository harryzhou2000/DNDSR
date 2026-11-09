#pragma once

#include "DNDS/MPI/Failure.hpp"
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace DNDS::Runtime
{
    /// Optional serialized communication executor. Requires MPI_THREAD_MULTIPLE:
    /// raw controller MPI calls are not covered by a provider-local mutex.
    /// Submit matching transfers in the same order on participating ranks.
    /// Retain operations until completion; destroy this owner before MPI_Finalize.
    /// The default transfer path creates neither this thread nor its queue.
    class CommunicationWorker
    {
        std::mutex _mutex;
        std::condition_variable _ready;
        std::deque<std::function<void()>> _queue;
        bool _closing = false;
        std::thread _thread;
        void Run()
        {
            for (;;)
            {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(_mutex);
                    _ready.wait(lock, [&]
                                { return _closing || !_queue.empty(); });
                    if (_queue.empty())
                        return;
                    task = std::move(_queue.front());
                    _queue.pop_front();
                }
                try
                {
                    task();
                }
                catch (...)
                {
                    MPI::Abort(MPI_COMM_WORLD, "communication worker failed during active MPI execution");
                }
            }
        }

    public:
        CommunicationWorker()
        {
            int provided = MPI_THREAD_SINGLE;
            MPI_Query_thread(&provided);
            DNDS_check_throw_info(provided == MPI_THREAD_MULTIPLE, "communication worker requires MPI_THREAD_MULTIPLE");
            _thread = std::thread([this]
                                  { Run(); });
        }
        CommunicationWorker(const CommunicationWorker &) = delete;
        CommunicationWorker &operator=(const CommunicationWorker &) = delete;
        ~CommunicationWorker()
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _closing = true;
            }
            _ready.notify_one();
            _thread.join();
        }
        /// Internal transfer submission; ownership and leases are reserved by
        /// the binding before queuing. Work must not recursively wait on this queue.
        void Submit(std::function<void()> work)
        {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                DNDS_check_throw_info(!_closing, "communication worker is closing");
                _queue.push_back(std::move(work));
            }
            _ready.notify_one();
        }
    };
}
