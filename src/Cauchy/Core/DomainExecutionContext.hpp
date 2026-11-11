#pragma once

#include "Cauchy/Physics/EquationSetup.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

#include <memory>

namespace DNDS::Cauchy
{
    struct ExecutionSettings;

    /// @brief Shared owner of one domain's backend execution resources.
    ///
    /// Copies retain the engine runtime, primary stream and optional chemistry
    /// worker capacity as one lifetime unit. The last owner destroys the stream
    /// before the engine; engine shutdown drains registered CPU/CUDA/MPI/NCCL
    /// completion records before worker resources are released. Numerical state
    /// and operator workspaces are intentionally not owned here.
    ///
    /// The current resolver supports the CPU backend only and fails during
    /// construction for an unavailable backend. This ownership shape is shared
    /// by inline and worker CPU execution and is the future device-context
    /// boundary; orchestration must not branch on those launcher details.
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
        /// @brief Domain primary stream, retained by this context.
        const Runtime::ExecutionStream &PrimaryStream() const;
        /// @brief Optional shared chemistry worker capacity.
        const std::optional<Physics::Chemistry::ChemicalWorkerPool> &ChemicalWorkers() const;
        /// @brief Wait for all work ordered before the primary-stream record.
        void JoinPrimary() const;
    };
}
