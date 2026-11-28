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

    std::vector<BoundaryFluxIntegral> FiniteVolume::LastBoundaryFluxIntegrals(
        const Workspace &work, const std::vector<int> &zones,
        std::optional<R::ExecutionStream> stream) const
    {
        if (zones.empty())
            return {};
        const index components = work.componentNumber;
        const index stride = components + 4;
        std::vector<real> sums(zones.size() * size_t(stride), real(0));
        const bool available = Reductions().Value<Runtime::Reduction::Min>(
                                   int(work.owner == _data && work.boundaryFacialFluxCurrent)) != 0;
        if (!available)
            return {};

        auto engine = stream ? stream->LockEngine() : R::EagerEngine{};
        const auto lane = stream ? *stream : engine.CreateCPUStream();
        Metrics().Communication()->LocalStage([&]
                                              {
            for (const int zone : zones)
                DNDS_check_throw_info(_data->settings.boundaries.count(std::to_string(zone)),
                                      "integrated boundary zone has no boundary condition");
            DNDS_check_throw_info(lane.Context().device == Device(),
                                  "boundary flux integration needs a matching execution stream");
            if (!work.boundaryFluxReduction)
                work.boundaryFluxReduction.emplace(lane.Context());
            sums = work.boundaryFluxReduction->EvaluateLocal(
                engine, lane, _data->boundaryOutputRows,
                work.faceFlux, work.boundaryMomentumFlux, Metrics().FaceQuadrature(), zones); });

        Reductions().Reduce<Runtime::Reduction::Sum>(SpanView<real>{sums.data(), index(sums.size())});

        std::vector<BoundaryFluxIntegral> result;
        result.reserve(zones.size());
        for (size_t i = 0; i < zones.size(); ++i)
        {
            const size_t base = i * size_t(stride);
            BoundaryFluxIntegral entry;
            entry.zone = zones[i];
            entry.area = sums[base];
            entry.flux.assign(sums.begin() + std::ptrdiff_t(base + 1),
                              sums.begin() + std::ptrdiff_t(base + 1 + size_t(components)));
            for (int d = 0; d < 3; ++d)
                entry.tangentialForce[d] = sums[base + 1 + size_t(components) + size_t(d)];
            result.push_back(std::move(entry));
        }
        return result;
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
