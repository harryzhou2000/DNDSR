#pragma once

#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "DNDS/Core/OMP.hpp"
#include <optional>
#include <vector>

namespace DNDS::Cauchy
{
    /// Build private CPU point-evaluator/scratch slots at operator execution,
    /// before entering an OMP region. Kernels receive only the resulting span.
    /// Perfect-gas/RANS slots follow DNDS distributed OMP controls. Mixtures
    /// retain one leased worker until team reservation is supported by the
    /// bounded chemistry pool; never acquire a team that can starve itself.
    /// Each Worker owns the evaluator and any mutable point-sized scratch.
    template <class Worker, class... Args>
    std::vector<Worker> MakeCPUPointWorkers(
        const EquationSystem &equation, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &pool,
        Physics::Thermodynamics::MixturePropertyRequest properties, const Args &...args)
    {
        const index count = equation.HasMixture() ? 1 : OMP::DistributedThreads();
        std::vector<Worker> workers;
        workers.reserve(size_t(count));
        for (index iWorker = 0; iWorker < count; ++iWorker)
            workers.emplace_back(equation.MakePointEvaluator(
                                     pool ? pool->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, properties),
                                 args...);
        return workers;
    }
}
