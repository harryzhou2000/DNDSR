#pragma once

#include <exception>
#include <functional>
#include <memory>

namespace DNDS::Cauchy
{
    /** @brief Bounded, ordered file worker with controller-side capture.
     * @details Enqueue waits for capacity BEFORE invoking prepare, so the new
     * immutable snapshot plus queued/running snapshots never exceed capacity.
     * prepare runs on the creating thread and returns a self-contained writer;
     * it may stage domain MPI/DMA and must not leave a live solver borrow in
     * that writer. The writer runs once, in FIFO order, on the I/O thread.
     *
     * This class performs no MPI itself. Production collective writers require
     * verified thread support and independent I/O communicators; their failure
     * handler MUST abort immediately with only rank-local diagnostic work.
     * Only coordinated UTs may return from the handler and observe/rethrow the
     * stored failure. No pending writer runs after a failure. Drain/Close are
     * durability boundaries; enqueue is not a successful-file publication.
     */
    class OutputQueue
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        using Action = std::function<void()>;
        using Prepare = std::function<Action()>;
        using Failure = std::function<void(std::exception_ptr)>;
        explicit OutputQueue(Failure terminalFailure, unsigned capacity = 2);
        ~OutputQueue();
        OutputQueue(const OutputQueue &) = delete;
        OutputQueue &operator=(const OutputQueue &) = delete;
        /// Controller only. Capacity includes the currently executing writer.
        void Enqueue(Prepare prepare);
        /// Controller observation; waits for every submitted file operation.
        void Drain();
        /// Drain and join. Idempotent; rejects all subsequent submissions.
        void Close();
    };
}
