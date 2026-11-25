#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    OptionalRef<const EquationDofs> FiniteVolume::LastSpatialResidual(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        if (!work.boundaryMomentumCurrent || !work.lastSpatialResidual)
            return {};
        for (index f = 0; f < work.lastSpatialResidual->PhysicalFieldNumber(); ++f)
            if (!work.lastSpatialRows[f].Current(work.lastSpatialResidual->PhysicalField(f).Owned()))
                return {};
        return std::cref(*work.lastSpatialResidual);
    }
    void FiniteVolume::InvalidateSpatialObservation(Workspace &work) const
    {
        work.boundaryMomentumCurrent = work.boundaryFacialFluxCurrent = false;
        work.lastSpatialResidual.reset();
        work.lastSpatialRows.clear();
    }
    Vector3 FiniteVolume::LastBoundaryForce(const Workspace &work, const std::vector<int> &zones,
                                            std::optional<R::ExecutionStream> stream) const
    {
        if (work.fixedFlowRHS)
        {
            DNDS_check_throw_info(work.owner == _data && work.boundaryMomentumCurrent,
                                  "fixed-flow boundary force requires a completed RHS evaluation");
            auto force = work.diagnostics.BoundaryForce(zones);
            Reductions().Reduce<Runtime::Reduction::Sum>(
                SpanView<real>{force.data(), 3});
            return force;
        }
        Vector3 force{};
        auto engine = stream ? stream->LockEngine() : R::EagerEngine{};
        const auto lane = stream ? *stream : engine.CreateCPUStream();
        Metrics().Communication()->LocalStage([&]
                                              {
            DNDS_check_throw_info(work.owner == _data && work.boundaryMomentumCurrent,
                "equation boundary force requires a completed RHS evaluation");
            std::vector<int> walls;
            for (const int zone : zones)
            {
                DNDS_check_throw_info(_data->settings.boundaries.count(std::to_string(zone)), "force zone has no boundary condition");
                const auto kind = _data->boundary.At(zone).kind;
                if (kind == BoundaryKind::SlipWall || kind == BoundaryKind::NoSlipWall)
                    walls.push_back(zone);
            }
            DNDS_check_throw_info(lane.Context().device == Device(), "boundary force needs a matching execution stream");
            if (!work.boundaryForceReduction) work.boundaryForceReduction.emplace(lane.Context());
            force = work.boundaryForceReduction->Evaluate(engine, lane, _data->stencil, _data->boundaryFaces, work.boundaryMomentumFlux, walls); });
        Reductions().Reduce<Runtime::Reduction::Sum>(SpanView<real>{force.data(), 3});
        return force;
    }

    index FiniteVolume::LastFaceFallbacks(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        if (work.fixedFlowRHS)
        {
            auto count = work.diagnostics.FaceFallbacks();
            count = Reductions().Value<Runtime::Reduction::Sum>(count);
            return count;
        }
        return Reductions().Value<Runtime::Reduction::Sum>(work.faceFallbacks[0] + work.faceFallbacks[1]);
    }

    index FiniteVolume::LastCharacteristicFallbacks(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        return work.characteristicFallbacks;
    }

    const EquationDofs &FiniteVolume::LowOrderResidual(const Workspace &work) const
    {
        DNDS_check_throw_info(work.owner == _data, "foreign equation workspace");
        DNDS_check_throw_info(work.lowOrderResidual, "low-order residual requires the positivity workspace");
        return *work.lowOrderResidual;
    }

}
