#pragma once

#include "EagerEngine.hpp"

namespace DNDS::Runtime
{
    /// Native CUDA eager launcher. Submission invokes a launch function on the
    /// submitting host thread. Captures/retirements remain owned through GPU
    /// completion. Host cleanup executes when a completed prefix is observed.
    class CUDALauncher final : public StreamLauncher
    {
        struct State;
        std::unique_ptr<State> _state;

    public:
        explicit CUDALauncher(int deviceOrdinal);
        ~CUDALauncher() override;
        ExecutionContext Context() const override;
        void Submit(Work work, CommandKind kind, Retirement retire = {}) override;
        void WaitFor(StreamEvent event) override;
        StreamEvent Record() override;
        void Drain() override;
        Buffer Allocate(size_t bytes) override;
        /// Diagnostics for event-boundary tests, not an execution dependency.
        uint64_t NativeEventsRecorded() const;
    };
}
