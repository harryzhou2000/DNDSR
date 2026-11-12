#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeRHSKernels.hpp"
#include "DNDS/Core/OMP.hpp"
#include <cstdio>
#include <cstdlib>
namespace DNDS::Cauchy
{
    namespace
    {
        /// Fatal entry policy covers descriptor preparation as well as the
        /// numerical leaf. In particular, frame validation must not throw
        /// before a fatal kernel gets control. No recovery or MPI is allowed
        /// here; checked UT entries keep their ordinary exception semantics.
        template <bool Fatal, class Action>
        decltype(auto) WithRHSFailurePolicy(Action &&action)
        {
            if constexpr (!Fatal)
                return action();
            else
            {
                try
                {
                    return action();
                }
                catch (const std::exception &error)
                {
                    std::fprintf(stderr, "Cauchy fatal RHS operator: %s\n", error.what());
                    std::abort();
                }
                catch (...)
                {
                    std::fprintf(stderr, "Cauchy fatal RHS operator: unknown failure\n");
                    std::abort();
                }
            }
        }
    }
    struct FiniteVolume::RHSOperator::ReadData
    {
        std::shared_ptr<const FiniteVolume::Data> data;
        CFV::FiniteVolumeRead geometry;
        FiniteVolumeStencilStorage::ReadAccess stencil;
        std::optional<CFV::ReconstructionBasisRead> basis;
        explicit ReadData(std::shared_ptr<const FiniteVolume::Data> owner)
            : data(std::move(owner)), geometry(data->metrics.AcquireRead()), stencil(data->stencil.Read())
        {
            if (data->settings.numerics.reconstruction == "variational")
            {
                DNDS_check_throw_info(data->reconstruction,
                                      "polynomial RHS needs variational reconstruction");
                basis.emplace(data->reconstruction->AcquireBasisRead());
            }
            data->partition.Validate();
        }
        CPU::FlowRHSRequest Request(real time) const
        {
            CPU::FlowRHSRequest request;
            request.stencil = stencil.View();
            request.metric = geometry.HostView();
            if (basis)
                request.basis = basis->EvaluationView();
            request.conditions = {data->boundaryConditions.data(), index(data->boundaryConditions.size())};
            request.fluxPolicy = data->fluxPolicy;
            request.thermalBounds = data->FixedFlowThermalBounds();
            request.frame = data->motion.At(time);
            request.ownedCells = data->metrics.NumCell();
            request.allCells = data->metrics.NumCellProc();
            request.forceSlots = index(data->forceZones.size());
            request.dimension = data->metrics.Dimension();
            request.modes = data->reconstruction ? data->reconstruction->NRecDOF() : 0;
            const auto &settings = data->settings.numerics;
            request.direct = settings.reconstruction != "variational";
            request.linear = settings.reconstruction == "gradient";
            request.firstOrder = settings.firstOrderConvection;
            request.polynomialViscosity = settings.firstOrderConvection || settings.viscousGradient == "polynomial";
            request.hasLocalInterfaceRows = data->hasLocalInterfaceRows;
            request.useStencil = settings.riemann.useStencil;
            request.time = time;
            return request;
        }
    };

    FiniteVolume::RHSOperator FiniteVolume::AcquireRHSOperator() const
    {
        return RHSOperator(std::make_shared<RHSOperator::ReadData>(_data));
    }

    RHSKernelDiagnostics FiniteVolume::RHSOperator::ApplyBorrowedChecked(const RHSReadViews &input, const RHSWriteViews &output,
                                                                         CFV::ReconstructionRegion region, real time) const
    {
        auto diagnostics = ApplyFluxImpl<FailurePolicy::Throw>(input, output, region, time);
        AddSourceImpl<FailurePolicy::Throw>(input, output, region, time);
        return diagnostics;
    }

    RHSKernelDiagnostics FiniteVolume::RHSOperator::ApplyBorrowedFatal(const RHSReadViews &input, const RHSWriteViews &output,
                                                                       CFV::ReconstructionRegion region, real time) const
    {
        auto diagnostics = ApplyFluxImpl<FailurePolicy::Abort>(input, output, region, time);
        AddSourceImpl<FailurePolicy::Abort>(input, output, region, time);
        return diagnostics;
    }

    RHSKernelDiagnostics FiniteVolume::RHSOperator::ApplyFluxBorrowedFatal(const RHSReadViews &input, const RHSWriteViews &output,
                                                                           CFV::ReconstructionRegion region, real time) const
    {
        auto diagnostics = ApplyFluxImpl<FailurePolicy::Abort>(input, output, region, time);
        const auto &settings = _read->data->settings.numerics;
        if (!((settings.faceFallback && !settings.positivity) || diagnostics.FaceFallbacks() == 0))
            std::abort();
        return diagnostics;
    }

    RHSKernelDiagnostics FiniteVolume::RHSOperator::ApplyFlux(const RHSReadViews &input, const RHSWriteViews &output,
                                                              CFV::ReconstructionRegion region, real time) const
    {
        return ApplyFluxImpl<FailurePolicy::Throw>(input, output, region, time);
    }

    template <FiniteVolume::RHSOperator::FailurePolicy policy>
    RHSKernelDiagnostics FiniteVolume::RHSOperator::ApplyFluxImpl(
        const RHSReadViews &input, const RHSWriteViews &output,
        CFV::ReconstructionRegion region, real time) const
    {
        return WithRHSFailurePolicy<policy == FailurePolicy::Abort>([&]
                                                                    {
            DNDS_check_throw_info(_read, "RHS operator was moved");
            Runtime::ValidateViewDevices(Runtime::Device::Host(), input, output);
            const auto request = _read->Request(time);
            const index count = region == CFV::ReconstructionRegion::All ? request.ownedCells :
                (region == CFV::ReconstructionRegion::Core ? request.stencil.coreCells.RowCount() : request.stencil.dependentCells.RowCount());
            const index threads = output.faceFlux ? 1 : std::min<index>(OMP::DistributedThreads(), std::max<index>(1, count));
            std::vector<RHSKernelDiagnostics> diagnostics;
            diagnostics.reserve(size_t(threads));
            for (index worker = 0; worker < threads; ++worker)
                diagnostics.emplace_back(request.forceSlots);
            std::vector<NumericalStatus> status(size_t(threads), NumericalStatus{});
            RequireNumericalSuccess(CPU::EvaluateFlowFlux(request, _read->data->equation, _read->data->flux,
                input, output, region, {{diagnostics.data(), threads}, {status.data(), threads}}), "EvaluateFlowFlux");
            for (index worker = 1; worker < threads; ++worker)
                diagnostics[0].Merge(diagnostics[worker]);
            return std::move(diagnostics[0]); });
    }
    void FiniteVolume::RHSOperator::AddSource(const RHSReadViews &input, const RHSWriteViews &output,
                                              CFV::ReconstructionRegion region, real time) const
    {
        AddSourceImpl<FailurePolicy::Throw>(input, output, region, time);
    }
    template <FiniteVolume::RHSOperator::FailurePolicy policy>
    void FiniteVolume::RHSOperator::AddSourceImpl(
        const RHSReadViews &input, const RHSWriteViews &output,
        CFV::ReconstructionRegion region, real time) const
    {
        WithRHSFailurePolicy<policy == FailurePolicy::Abort>([&]
                                                             {
            DNDS_check_throw_info(_read, "RHS operator was moved");
            Runtime::ValidateViewDevices(Runtime::Device::Host(), input, output);
            RequireNumericalSuccess(CPU::AddFlowRotationSource(_read->Request(time), input, output, region), "AddFlowRotationSource"); });
    }
}
