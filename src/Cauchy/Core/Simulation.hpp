#pragma once

#include "Cauchy/Core/Domain.hpp"

namespace DNDS::Cauchy
{
    class InterfaceCoupling;
    /// Coordinated CPU frontend. Only local domains have runtime objects.
    /// Local domain phases run serially on shared ranks and concurrently on
    /// disjoint communicators. Physical stages share one clock and consume a
    /// frozen interface round. Caller serializes stepping and external mutation.
    class Simulation
    {
        SimulationConfig _config;
        DomainPlacement _placement;
        std::vector<std::unique_ptr<Domain>> _domains;
        std::unique_ptr<InterfaceCoupling> _interfaces;
        real _time = 0;
        uint64_t _step = 0;

    public:
        using MeshFactory = std::function<ssp<Geom::UnstructuredMesh>(index, const DomainConfig &, const MPIInfo &)>;
        Simulation(SimulationConfig config, MPI_Comm comm, const MeshFactory &factory = {});
        ~Simulation();
        const SimulationConfig &Config() const { return _config; }
        const DomainPlacement &Placement() const { return _placement; }
        bool Local(index domain) const;
        Domain &GetDomain(index domain);
        const Domain &GetDomain(index domain) const;
        real Time() const { return _time; }
        uint64_t Step() const { return _step; }
        /// Prepare all stage inputs, refresh frozen traces, then solve local
        /// domains. All accepted destinations are reserved before any domain
        /// publishes. Process failure recovery is unsupported. DITR currently
        /// retains its single-domain coupled-iteration driver.
        void Advance(real dt);
        /// Initial single-domain matching-partition restart. Restore the
        /// simulation clock together with the domain's accepted state.
        void ReadCheckpoint(const std::string &path);
    };
}
