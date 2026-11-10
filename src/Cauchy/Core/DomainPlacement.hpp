#pragma once

#include "DNDS/Runtime/Communication/CommunicationContext.hpp"
#include <string>
#include <vector>

namespace DNDS::Cauchy
{
    struct DomainRankGroup
    {
        std::string name;
        /// Simulation-rank IDs in domain-rank order. Preserve this order even
        /// for noncontiguous or reversed groups.
        std::vector<int> ranks;
    };
    /// Immutable control/domain communicator publication. Every simulation rank
    /// constructs the same descriptor list collectively. Nonmembers retain only
    /// metadata, never a dummy mesh/domain runtime. Destruction of the last
    /// handle occurs on an MPI-permitted thread before MPI finalization.
    class DomainPlacement
    {
        struct State;
        std::shared_ptr<const State> _state;

    public:
        DomainPlacement(MPI_Comm simulation, std::vector<DomainRankGroup> groups);
        MPI_Comm Control() const;
        std::shared_ptr<Runtime::CommunicationContext> ControlContext() const;
        int Rank() const;
        int Size() const;
        index DomainNumber() const;
        const DomainRankGroup &Group(index domain) const;
        int DomainRank(index domain) const;
        int SimulationRank(index domain, int domainRank) const;
        std::shared_ptr<Runtime::CommunicationContext> Context(index domain) const;
        std::vector<index> LocalDomains() const;
    };
}
