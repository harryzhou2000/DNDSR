#include "Cauchy/Core/DomainExecutionContext.hpp"

#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"

#include <mutex>

namespace DNDS::Cauchy
{
    namespace
    {
        class EvaluationPublication
        {
            QueuedEvaluationDependency _request;
            mutable std::mutex _mutex;
            bool _published = false;

        public:
            explicit EvaluationPublication(QueuedEvaluationDependency request)
                : _request(std::move(request)) {}
            Runtime::StreamEvent Completion() const { return _request.completion; }
            bool Published() const
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _published;
            }
            void Publish()
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (_published)
                    return;
                DNDS_check_throw_info(_request.completion.Ready(),
                                      "interface publication preceded its event");
                _request.publish();
                _published = true;
            }
        };
        /// Progress an imported runtime on this runtime's controller thread.
        /// A compute worker may observe the event but cannot poll FUNNELED MPI.
        /// Retaining the operation also pins the foreign engine and buffers.
        class ImportedEvaluationProgress final : public Runtime::ExecutionProgress
        {
            Runtime::StreamEvent _completion;
            std::shared_ptr<void> _owner;

        public:
            explicit ImportedEvaluationProgress(const QueuedEvaluationDependency &dependency)
                : _completion(dependency.completion), _owner(dependency.owner) {}
            bool Poll() override { return _completion.Ready(); }
        };
        size_t ResolveHostThreads(const ExecutionSettings &settings)
        {
            DNDS_check_throw_info(settings.device == "cpu" || settings.device == "cuda",
                                  "domain execution context has no resolved backend for " + settings.device);
            DNDS_check_throw_info(settings.hostThreads >= 0 && settings.hostThreads <= 1024,
                                  "invalid domain host worker count");
            DNDS_check_throw_info(settings.device != "cuda" || settings.deviceOrdinal >= 0,
                                  "invalid domain CUDA device ordinal");
            return size_t(settings.hostThreads);
        }
        Runtime::ExecutionStream CreateDomainStream(Runtime::EagerEngine &engine, const ExecutionSettings &settings)
        {
            if (settings.device == "cpu")
                return engine.CreateCPUStream();
            // The engine owns the launcher/native stream. These handles share
            // the context; they do not create a second stream lifetime system.
            return engine.CreateStream(std::make_shared<Runtime::CUDALauncher>(settings.deviceOrdinal));
        }
    }
    struct DomainExecutionContext::Data
    {
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        Runtime::EagerEngine engine;
        Runtime::ExecutionStream compute, communication, copy;
        std::mutex dependencyMutex;
        std::shared_ptr<EvaluationPublication> publication;

        Data(const ExecutionSettings &settings,
             std::optional<Physics::Chemistry::ChemicalWorkerPool> workerCapacity)
            : workers(std::move(workerCapacity)),
              engine(ResolveHostThreads(settings),
                     settings.checkAccess
                         ? std::make_shared<Runtime::AccessTrackingContext>()
                         : nullptr),
              compute(CreateDomainStream(engine, settings)),
              communication(CreateDomainStream(engine, settings)),
              copy(CreateDomainStream(engine, settings))
        {
            DNDS_check_throw_info(settings.device == "cpu" || !workers,
                                  "CUDA perfect-gas context cannot retain CPU chemistry workers");
        }
        ~Data()
        {
            // A posted interface dependency is accepted work even when no
            // spatial evaluation consumed it. Finish it before dropping its
            // packet/resource owners; never recover during distributed teardown.
            try
            {
                // Drain earlier compute work before a controller-side fallback
                // binds packet metadata. Registered event progress prevents a
                // worker wait from starving the foreign MPI operation.
                engine.Synchronize();
                if (publication)
                {
                    publication->Completion().WaitHost();
                    publication->Publish();
                    publication.reset();
                }
            }
            catch (...)
            {
                MPI::AbortCaught(MPI_COMM_WORLD, "domain execution shutdown failed");
            }
        }
    };

    DomainExecutionContext::DomainExecutionContext(
        const ExecutionSettings &settings,
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
        : _data(std::make_shared<Data>(settings, std::move(workers)))
    {
    }

    Runtime::EagerEngine &DomainExecutionContext::Engine() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        return _data->engine;
    }

    const Runtime::ExecutionStream &DomainExecutionContext::ComputeStream() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        return _data->compute;
    }

    const Runtime::ExecutionStream &DomainExecutionContext::CommunicationStream() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        return _data->communication;
    }

    const Runtime::ExecutionStream &DomainExecutionContext::CopyStream() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        return _data->copy;
    }

    const std::optional<Physics::Chemistry::ChemicalWorkerPool> &
    DomainExecutionContext::ChemicalWorkers() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        return _data->workers;
    }

    void DomainExecutionContext::Synchronize() const
    {
        Engine().Synchronize();
    }
    void DomainExecutionContext::SetEvaluationDependency(
        QueuedEvaluationDependency dependency) const
    {
        DNDS_check_throw_info(_data && dependency.publish,
                              "invalid queued evaluation dependency");
        std::lock_guard<std::mutex> lock(_data->dependencyMutex);
        DNDS_check_throw_info(!_data->publication || _data->publication->Published(),
                              "previous interface publication has not completed");
        _data->engine.RegisterProgress(
            std::make_shared<ImportedEvaluationProgress>(dependency));
        _data->publication = std::make_shared<EvaluationPublication>(std::move(dependency));
    }
    std::optional<QueuedEvaluationDependency>
    DomainExecutionContext::AcquireEvaluationDependency() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        std::lock_guard<std::mutex> lock(_data->dependencyMutex);
        const auto publication = _data->publication;
        if (!publication)
            return {};
        return QueuedEvaluationDependency{
            publication->Completion(),
            [publication]
            { publication->Publish(); }, publication,
            !publication->Published()};
    }
    void DomainExecutionContext::DrainEvaluationDependency() const
    {
        auto dependency = AcquireEvaluationDependency();
        if (!dependency)
            return;
        dependency->completion.WaitHost();
        dependency->publish();
    }
    void DomainExecutionContext::ClearEvaluationDependency() const
    {
        DrainEvaluationDependency();
        std::lock_guard<std::mutex> lock(_data->dependencyMutex);
        _data->publication.reset();
    }
}
