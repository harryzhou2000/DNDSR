#pragma once

#include "Cauchy/Discretization/FiniteVolume.hpp"

namespace DNDS::Cauchy
{
    /// Borrowed synchronous sampling inputs. Nonmembers provide null pointers;
    /// participating domains provide native fields with completed halos. The
    /// caller retains their owners and releases conflicting writers throughout
    /// Refresh. Kernels never follow these host pointers on a device.
    struct InterfaceDomainInput
    {
        const FiniteVolume *discrete = nullptr;
        const CFV::FVField *mean = nullptr, *coefficients = nullptr;
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
        InterfaceCoupling(const SimulationConfig &config, MPI_Comm coupling,
                          const std::vector<const FiniteVolume *> &localDomains);
        /// Rebuild maps only when stage-time poses change. Two refreshes at one
        /// time reuse the exact map, sampling workspace and persistent return
        /// binding. Source inputs are frozen for the complete round before any
        /// receiver packet is published, preserving block-Jacobi semantics.
        void Refresh(const std::vector<InterfaceDomainInput> &domains, real time, StateEpoch epoch);
        const BoundaryInputs *Inputs(index domain) const;
        /// Retain the currently published packets as accepted traces. The next
        /// round uses separate scratch allocations; rejected trials can restore
        /// accepted packets without copying their payload or changing its epoch.
        void Commit() noexcept;
        void Discard() noexcept;
    };
}
