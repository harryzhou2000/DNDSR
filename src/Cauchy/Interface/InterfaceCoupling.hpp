#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Interface/InterfaceDiagnostics.hpp"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"

namespace DNDS::Cauchy
{
    /// Optional sharing donor bundle. Nonmembers leave values empty; all
    /// coupling ranks still enter the same matching schedule.
    struct InterfaceDomainInput
    {
        std::optional<CFV::DonorInputs> values;
    };

    /// Physical adapter over Geom matching and CFV point interpolation. It owns
    /// ready receiver packets, posed matching/sampling plans and stable scratch.
    /// Every coupling rank calls in the same connection order, including ranks
    /// outside both domains. Domain inner iterations never enter this protocol.
    /// No conservative interface flux is imposed by this point-value coupling.
    class InterfaceCoupling
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        InterfaceCoupling(const InterfaceCoupling &) = delete;
        InterfaceCoupling &operator=(const InterfaceCoupling &) = delete;
        InterfaceCoupling(InterfaceCoupling &&) noexcept = default;
        InterfaceCoupling &operator=(InterfaceCoupling &&) noexcept = default;
        InterfaceCoupling(const SimulationConfig &config, std::shared_ptr<Runtime::CommunicationContext> coupling,
                          const std::vector<std::optional<FiniteVolume>> &localDomains);
        /// Rebuild maps only when stage-time poses change. Two refreshes at one
        /// time reuse the exact map, sampling workspace and persistent return
        /// binding. Source inputs are frozen for the complete round before any
        /// receiver packet is published, preserving block-Jacobi semantics.
        void Refresh(const std::vector<InterfaceDomainInput> &domains, real time, StateEpoch epoch);
        std::optional<BoundaryInputs> Inputs(index domain) const;
        /// Collective diagnostic reduction on the coupling communicator.
        /// Does not refresh traces, synchronize kernels or mutate trajectories.
        InterfaceDiagnostics Diagnostics() const;
        /// Retain the currently published packets as accepted traces. The next
        /// round uses separate scratch allocations; rejected trials can restore
        /// accepted packets without copying their payload or changing its epoch.
        void Commit() noexcept;
        void Discard() noexcept;
    };
}
