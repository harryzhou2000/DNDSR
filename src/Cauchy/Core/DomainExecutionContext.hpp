#pragma once

#include "Cauchy/Physics/PhysicsResources.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

#include <memory>
#include <functional>
#include <optional>

namespace DNDS::Cauchy
{
    /// One externally posted interface publication acquired by spatial FV
    /// evaluations. `publish` runs on the compute lane after `completion` and
    /// installs the newly ready boundary packet. The callback must be rank-local
    /// and retain all operand owners; it must not enter MPI or capture a facade
    /// whose lifetime is only implied by the surrounding controller stack.
    struct QueuedEvaluationDependency
    {
        Runtime::StreamEvent completion;
        std::function<void()> publish;
        std::shared_ptr<void> owner;
        /// Controller-side snapshot when acquired, not a completion query.
        /// False permits reuse of caches already tied to this publication;
        /// true requires the consumer to import the event before binding it.
        bool publicationPending = true;
    };
    struct ExecutionSettings;

    /// @brief Shared owner of one domain's backend execution resources.
    ///
    /// Copies retain the engine runtime, named compute/communication streams and optional chemistry
    /// worker capacity as one lifetime unit. The last owner destroys the stream
    /// before the engine; engine shutdown drains registered CPU/CUDA/MPI/NCCL
    /// completion records before worker resources are released. Numerical state
    /// and operator workspaces are intentionally not owned here.
    ///
    /// Resolves inline/worker CPU or CUDA compute, communication and copy lanes.
    /// Context creation alone does not establish solver kernel availability:
    /// ResolvedExecutionPlan must validate the complete requested path first.
    /// Orchestration must not branch on the native launcher details.
    /// Release the last context on its creating controller thread. Teardown
    /// drains the engine, then publishes an unconsumed interface dependency;
    /// dependency callbacks must not retain this context (directly or through a
    /// solver owner), which would create an ownership cycle.
    class DomainExecutionContext
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        DomainExecutionContext() = default;
        DomainExecutionContext(const ExecutionSettings &settings,
                               std::optional<Physics::Chemistry::ChemicalWorkerPool> workers = {});

        /// @brief Whether this handle retains a resolved execution context.
        [[nodiscard]] explicit operator bool() const { return bool(_data); }
        /// @brief Owning eager engine. Callers submit only on retained streams.
        Runtime::EagerEngine &Engine() const;
        /// @brief Domain compute stream, retained by this context.
        const Runtime::ExecutionStream &ComputeStream() const;
        /// @brief Communication/progress lane. A backend may map its native
        /// transport here; CPU MPI uses it for checked access ordering while
        /// controller progress remains explicit.
        const Runtime::ExecutionStream &CommunicationStream() const;
        /// @brief Context-owned snapshot/copy lane. Producers and subsequent
        /// writers order their dependencies explicitly; this is not a host sync.
        const Runtime::ExecutionStream &CopyStream() const;
        /// @brief Optional shared chemistry worker capacity.
        const std::optional<Physics::Chemistry::ChemicalWorkerPool> &ChemicalWorkers() const;
        /// @brief Explicit quiescent boundary for every retained lane and
        /// registered external operation in this domain runtime.
        void Synchronize() const;
        /// Install the latest interface publication for spatial evaluations.
        /// Reconstruction-only donor preparation deliberately does not consume
        /// this slot. Registration imports controller-side progress as well as
        /// the event, so a worker waiting on a different runtime cannot starve
        /// that runtime's FUNNELED MPI progress. Call on the controller thread.
        void SetEvaluationDependency(QueuedEvaluationDependency dependency) const;
        /// Retain the latest publication for each evaluation, including nested
        /// finest pMG and corrected DITR rate evaluations. Publication is
        /// idempotent; the policy alone decides when donors are republished.
        std::optional<QueuedEvaluationDependency> AcquireEvaluationDependency() const;
        /// Explicit controller barrier for a publication not reached by a
        /// no-op/capped operator path. Already-published records are unchanged;
        /// the record remains available to subsequent spatial evaluations.
        void DrainEvaluationDependency() const;
        /// Stage/step boundary: retire the current publication and remove its
        /// time/slot association before preparing a different mathematical group.
        void ClearEvaluationDependency() const;
    };
}
