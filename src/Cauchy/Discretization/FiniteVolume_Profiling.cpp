#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"

namespace DNDS::Cauchy
{
    std::shared_ptr<FiniteVolume::Workspace::QueuedPhaseSample>
    FiniteVolume::Workspace::BeginQueuedPhase(Runtime::EagerEngine &engine,
                                              const Runtime::ExecutionStream &stream, ProfilePhase phase)
    {
        if (!profiling || stream.Context().device != Runtime::Device::Host())
            return {};
        auto sample = std::make_shared<QueuedPhaseSample>(phase);
        queuedPhaseSamples.push_back(sample);
        engine.Submit(stream, [sample](Runtime::ExecutionContext)
                      { sample->start = std::chrono::steady_clock::now(); });
        return sample;
    }
    void FiniteVolume::Workspace::EndQueuedPhase(Runtime::EagerEngine &engine,
                                                 const Runtime::ExecutionStream &stream,
                                                 const std::shared_ptr<QueuedPhaseSample> &sample)
    {
        if (!sample)
            return;
        engine.Submit(stream, [sample](Runtime::ExecutionContext)
                      {
            sample->seconds = std::chrono::duration<real>(std::chrono::steady_clock::now() - sample->start).count();
            sample->complete.store(true, std::memory_order_release); });
    }
    void FiniteVolume::Workspace::PublishQueuedPhases()
    {
        for (const auto &sample : queuedPhaseSamples)
        {
            DNDS_check_throw_info(sample->complete.load(std::memory_order_acquire),
                                  "queued FV timing requires the completed segment");
            profileSeconds[sample->phase] += sample->seconds;
            ++profileCalls[sample->phase];
        }
        queuedPhaseSamples.clear();
    }
}
