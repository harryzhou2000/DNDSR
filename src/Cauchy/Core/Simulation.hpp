#pragma once

#include "Cauchy/Core/Domain.hpp"

namespace DNDS::Cauchy
{
    /// Coordinated CPU frontend. Only local domains have runtime objects.
    /// Initial physical advancement supports one domain; shared/split multi-
    /// domain setup and mock scheduling are available, while physical coupling
    /// and multi-domain commit transactions belong to the later package.
    class Simulation
    {
        SimulationConfig _config;
        DomainPlacement _placement;
        std::vector<std::unique_ptr<Domain>> _domains;
        real _time = 0;
        uint64_t _step = 0;

    public:
        using MeshFactory = std::function<ssp<Geom::UnstructuredMesh>(index, const DomainConfig &, const MPIInfo &)>;
        Simulation(SimulationConfig config, MPI_Comm comm, const MeshFactory &factory = {});
        const SimulationConfig &Config() const { return _config; }
        const DomainPlacement &Placement() const { return _placement; }
        bool Local(index domain) const;
        Domain &GetDomain(index domain);
        const Domain &GetDomain(index domain) const;
        real Time() const { return _time; }
        uint64_t Step() const { return _step; }
        /// Single-domain physical SSPRK3. Multi-domain physical advancement
        /// rejects explicitly; use CoordinatedSchedule for placement mocks.
        /// Rank/process failure recovery is unsupported.
        void Advance(real dt);
        /// Initial single-domain matching-partition restart. Restore the
        /// simulation clock together with the domain's accepted state.
        void ReadCheckpoint(const std::string &path);
    };
}
