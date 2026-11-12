#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumePointKernels.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CFV/Math/PeriodicFrame.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;

    void FiniteVolume::Workspace::FillBoundary(const EquationDofs &state, bool polynomial, real time, EquationPointEvaluator &evaluator,
                                               RHSScalarView incoming)
    {
        const PhaseTime timing(*this, BoundaryPhase);
        const auto mean = state.ReadOwned(), rec = Coefficients().ReadOwned();
        const auto meanBlocks = BlockViews(state, mean), recBlocks = BlockViews(Coefficients(), rec);
        std::optional<CFV::ReconstructionBasisRead> basisRead;
        const bool direct = owner->settings.numerics.reconstruction != "variational";
        if (!direct)
            basisRead.emplace(owner->reconstruction->AcquireBasisRead());
        CFV::Math::SmallMatrix scratch;
        PointScratch point(state.ComponentNumber()), cellMean(state.ComponentNumber());
        const auto *ransModel = owner->equation.HasRANS() ? &owner->equation.RANS() : nullptr;
        const auto ransMean = ransModel ? mean.Block("rans") : EquationBlockReadView{};
        std::optional<WallDistanceRead> wallRead;
        if (ransModel && ransModel->ComponentNumber() == 2)
            wallRead.emplace(owner->wallDistance->AcquireRead());
        const auto wallValues = wallRead ? wallRead->HostView() : WallDistanceView{};
        const auto frame = owner->motion.At(time);
        const int modes = owner->reconstruction ? owner->reconstruction->NRecDOF() : 0;
        {
            auto output = boundary.WriteOwned();
            for (index iPoint = 0; iPoint < index(owner->boundaryEntries.size()); ++iPoint)
            {
                const auto &sample = owner->samples[owner->boundaryEntries[iPoint]];
                Gather(meanBlocks, sample.cell, cellMean);
                point.values = cellMean.values;
                if (polynomial)
                {
                    const auto basis = basisRead->FaceBasis(sample.face, sample.side, sample.point, 0, scratch);
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto row = recBlocks[iBlock][sample.cell];
                        for (index j = 0; j < meanBlocks[iBlock].ComponentNumber(); ++j)
                            for (int mode = 0; mode < modes; ++mode)
                                point.values[logicalFirst[iBlock] + j] += row[j * modes + mode] * basis(0, mode);
                    }
                }
                if (ransModel)
                    RANSStateView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairReconstruction(ransMean[sample.cell], point.values.data() + ransFirst);
                owner->boundaryPoint.Evaluate(
                    point.values, sample.boundary, sample.boundaryPoint,
                    {sample.normal[0], sample.normal[1], sample.normal[2]},
                    {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]},
                    evaluator, frame, time,
                    BoundaryPointStateKind::FaceTrace,
                    OptionalRef<const std::vector<real>>(cellMean.values),
                    wallRead ? wallValues.cellMean(sample.cell, 0) : 0,
                    incoming);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                // Euler's reconstruction callback uses fixUL=true: its
                // returned SA target is zero, rather than a negative ghost.
                if (ransModel && ransModel->Kind() == Physics::RANS::ModelKind::SA && sample.boundary.kind == BoundaryKind::NoSlipWall)
                    point.values[ransFirst] = 0;
#endif
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto target = output.Block(iBlock);
                    for (index j = 0; j < target.ComponentNumber(); ++j)
                        target(iPoint, j) = point.values[logicalFirst[iBlock] + j];
                }
            }
        }
        // GradientPlan may select another face quadrature order. Its setup
        // already maps these rows to the canonical boundary quadrature.
        for (auto &field : fields)
            if (field.gradientBoundary)
            {
                const auto source = field.boundaryValues.Read();
                auto target = field.gradientBoundary->Write();
                const auto a = source.View<R::UniformTableRows>();
                const auto b = target.View<R::UniformTableRows>();
                for (index iPoint = 0; iPoint < b.RowCount(); ++iPoint)
                    for (index j = 0; j < b.Width(); ++j)
                        b(iPoint, j) = a(owner->gradientBoundaryPoints[iPoint], j);
            }
        if (primitiveBoundary)
        {
            auto target = primitiveBoundary->Write();
            const auto values = target.View<R::UniformTableRows>();
            for (index iPoint = 0; iPoint < owner->gradient.NumBoundaryQuadrature(); ++iPoint)
            {
                const auto &sample = owner->samples[owner->boundaryEntries[owner->gradientBoundaryPoints[iPoint]]];
                Gather(meanBlocks, sample.cell, cellMean);
                const auto state = FiniteVolumeDetail::BoundaryGradientValue(
                    sample, owner->equation.Gas(), cellMean.Flow(flowFirst), time, frame, incoming);
                const auto value = owner->equation.Gas().Primitive(state);
                const real converted[5]{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                for (int j = 0; j < 5; ++j)
                    values(iPoint, j) = converted[j];
            }
        }
    }
}
