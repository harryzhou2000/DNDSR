#include "Cauchy/Core/DomainExecutionContext.hpp"

#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        size_t ResolveHostThreads(const ExecutionSettings &settings)
        {
            DNDS_check_throw_info(settings.device == "cpu",
                                  "domain execution context has no resolved backend for " + settings.device);
            return size_t(settings.hostThreads);
        }
    }
    struct DomainExecutionContext::Data
    {
        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers;
        Runtime::EagerEngine engine;
        Runtime::ExecutionStream primary;

        Data(const ExecutionSettings &settings,
             std::optional<Physics::Chemistry::ChemicalWorkerPool> workerCapacity)
            : workers(std::move(workerCapacity)),
              engine(ResolveHostThreads(settings),
                     settings.checkAccess
                         ? std::make_shared<Runtime::AccessTrackingContext>()
                         : nullptr),
              primary(engine.CreateCPUStream())
        {
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

    const Runtime::ExecutionStream &DomainExecutionContext::PrimaryStream() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        return _data->primary;
    }

    const std::optional<Physics::Chemistry::ChemicalWorkerPool> &
    DomainExecutionContext::ChemicalWorkers() const
    {
        DNDS_check_throw_info(_data, "empty domain execution context");
        return _data->workers;
    }

    void DomainExecutionContext::JoinPrimary() const
    {
        PrimaryStream().Record().WaitHost();
    }
}
