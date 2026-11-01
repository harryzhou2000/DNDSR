#pragma once

#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "CFV/Plans/GradientPlan.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "Geom/Access/EntityWorkPartition.hpp"

namespace DNDS::Cauchy
{
    /// CPU cell-centered discretization. Preparation retains immutable geometry,
    /// incidence, quadrature and polynomial data. Each simultaneous residual
    /// requires a distinct Workspace. No solver clock/history/IO policy here.
    class FiniteVolume
    {
        struct Data;
        std::shared_ptr<const Data> _data;

    public:
        struct Workspace;
        FiniteVolume(ssp<Geom::UnstructuredMesh> mesh, const DomainSettings &settings, const EquationSystem &equation);
        const CFV::FiniteVolumePlan &Metrics() const;
        const Geom::UnstructuredMesh &Mesh() const;
        std::shared_ptr<Workspace> BuildWorkspace(CFV::FVField &state) const;
        /// Integrated d(V*U)/dt, owned rows only. Mean/rec/gradient exchanges are
        /// explicit inside this domain-level routine; numerical kernels use
        /// owned-only core inputs while ghost transfers are active. The caller
        /// serializes residuals on this workspace and coordinates domain errors.
        void Evaluate(CFV::FVField &state, CFV::FVField &residual, Workspace &workspace,
                      Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream) const;
        real StableStep(const CFV::FVField &state, real cfl) const;
        /// Collective owned-only admissibility; never repairs a state silently.
        bool Admissible(const CFV::FVField &state) const;
        index LastFaceFallbacks(const Workspace &workspace) const;
    };
}
