#pragma once
#include "Cauchy/Config/SimulationConfig.hpp"

namespace DNDS::Cauchy
{
    struct DomainSetupTimings;
    /// Load the domain-owned mesh and apply its solver-frame geometry transform once.
    ssp<Geom::UnstructuredMesh> ReadDomainMesh(const DomainConfig &, const MPIInfo &,
                                               const std::shared_ptr<Runtime::CommunicationContext> &, DomainSetupTimings &, Diagnostics::SetupProgress);
    /// Reuse archived nodal vectors or compute them using this domain's no-slip zones.
    ssp<Geom::UnstructuredMesh> EnsureDomainWallDistance(ssp<Geom::UnstructuredMesh>, const DomainSettings &, Diagnostics::SetupProgress);
    /// Collective mesh-only frontend. Each domain writes one H5 archive before solver-frame transforms.
    void WritePartitionedMeshes(const SimulationConfig &, MPI_Comm);
}
