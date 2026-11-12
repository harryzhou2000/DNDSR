#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;

    void FiniteVolume::Workspace::FillBoundary(
        const EquationDofs &state, bool polynomial, real time,
        PointEvaluator &evaluator, RHSScalarView incoming)
    {
        const PhaseTime timing(*this, BoundaryPhase);
        const auto mean = state.ReadOwned();
        std::optional<EquationDofs::ReadScope> rec;
        if (polynomial)
            rec.emplace(Coefficients().ReadOwned());
        std::vector<EquationBlockReadView> meanBlocks, recBlocks;
        for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
        {
            meanBlocks.push_back(mean.Block(iBlock));
            if (rec)
                recBlocks.push_back(rec->Block(iBlock));
        }
        const R::MetadataBuffer<EquationBlockReadView> meanMetadata(meanBlocks),
            coefficientMetadata(recBlocks);
        std::optional<CFV::ReconstructionBasisRead> basisRead;
        const bool direct =
            owner->settings.numerics.reconstruction != "variational";
        if (!direct && polynomial)
            basisRead.emplace(owner->reconstruction->AcquireBasisRead());
        const auto *ransModel =
            owner->equation.HasRANS() ? &owner->equation.RANS() : nullptr;
        const auto ransMean =
            ransModel ? mean.Block("rans") : EquationBlockReadView{};
        std::optional<WallDistanceRead> wallRead;
        if (ransModel && ransModel->ComponentNumber() == 2)
            wallRead.emplace(owner->wallDistance->AcquireRead());
        const auto geometry = owner->boundaryTraceGeometry.Read();
        const auto rowRead = owner->gradientBoundaryRows.Read();
        const auto sourceRows = rowRead.View<R::FixedRows<1>>();
        const SpanView<const BoundaryCondition> conditions{
            owner->boundaryConditions.data(), index(owner->boundaryConditions.size())};
        CPU::BoundaryTraceRequest request;
        request.points = geometry.View<R::FixedRows<1>>();
        request.means = meanMetadata.HostView();
        request.coefficients = coefficientMetadata.HostView();
        request.logicalFirst = logicalFirst.HostView();
        request.componentNumber = state.ComponentNumber();
        request.polynomial = bool(basisRead);
        if (basisRead)
            request.basis = basisRead->EvaluationView();
        request.rans = ransModel
                           ? RANSStateView{ransModel->Kind(),
                                           ransModel->ComponentNumber()}
                           : RANSStateView{};
        request.ransMeans = ransMean;
        request.wallDistance = wallRead ? wallRead->HostView()
                                        : WallDistanceView{};
        request.frame = owner->motion.At(time);
        request.incoming = incoming;
        request.ransFirst = ransFirst;
        request.modes = owner->reconstruction
                            ? owner->reconstruction->NRecDOF()
                            : 0;
        request.time = time;
        {
            auto outputWrite = boundary.WriteOwned();
            std::vector<EquationBlockArrayView<real>> outputBlocks;
            outputBlocks.reserve(size_t(boundary.BlockNumber()));
            for (index iBlock = 0; iBlock < boundary.BlockNumber(); ++iBlock)
                outputBlocks.push_back(outputWrite.Block(iBlock));
            const R::MetadataBuffer<EquationBlockArrayView<real>> outputs(outputBlocks);
            request.output = outputs.HostView();
            RequireNumericalSuccess(CPU::FillBoundaryTraces(request, conditions,
                                                            owner->boundaryPoint, evaluator),
                                    "FillBoundaryTraces");
        }

        for (auto &field : fields)
            if (field.gradientBoundary)
            {
                const auto source = field.boundaryValues.Read();
                auto target = field.gradientBoundary->Write();
                CPU::CopyBoundaryRows(
                    source.View<R::UniformTableRows>(),
                    target.View<R::UniformTableRows>(),
                    sourceRows);
            }
        if (primitiveBoundary)
        {
            auto target = primitiveBoundary->Write();
            CPU::PrimitiveBoundaryRequest primitiveRequest;
            primitiveRequest.points = request.points;
            primitiveRequest.sourceRows = sourceRows;
            primitiveRequest.means = request.means;
            primitiveRequest.componentNumber = state.ComponentNumber();
            primitiveRequest.output = target.View<R::UniformTableRows>();
            primitiveRequest.gas = owner->equation.Gas();
            primitiveRequest.frame = request.frame;
            primitiveRequest.incoming = incoming;
            primitiveRequest.flowFirst = flowFirst;
            primitiveRequest.time = time;
            CPU::FillPrimitiveBoundary(primitiveRequest, conditions);
        }
    }
}
