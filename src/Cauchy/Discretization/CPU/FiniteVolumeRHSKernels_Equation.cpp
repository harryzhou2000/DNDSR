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

    void FiniteVolume::Workspace::ApplyFixedFlowRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                                                    real time, OptionalRef<const AuxiliaryValues> auxiliary,
                                                    OptionalRef<R::Array<real>> faceOutput,
                                                    RHSScalarView incoming)
    {
        DNDS_assert_info(fixedFlowRHS && state.PhysicalFieldNumber() == 1 && result.PhysicalFieldNumber() == 1 &&
                             state.ComponentNumber() == 5 && result.ComponentNumber() == 5,
                         "fixed-flow RHS leaf requires one flow5 physical field");
        const bool ghosts = region != CFV::ReconstructionRegion::Core;
        RHSAccessScope scope(ghosts);
        auto &input = scope.input;
        input.mean = scope.ReadFlowField(state.PhysicalField(0));
        input.includeViscosity = !meanOnly;
        input.meanOnly = meanOnly;
        if (owner->settings.numerics.reconstruction == "variational")
            input.coefficients = scope.ReadField(FluxCoefficients().PhysicalField(0));
        else if (!meanOnly && !owner->settings.numerics.firstOrderConvection &&
                 owner->settings.numerics.reconstruction == "gradient")
        {
            DNDS_assert_info(directGradients.size() == 1, "fixed-flow direct gradient is unavailable");
            input.conservativeGradient = scope.ReadField(directGradients.front());
        }
        if (!owner->settings.numerics.viscousLimited &&
            (owner->settings.numerics.limiter == "WBAP" || limitedVR || owner->settings.numerics.positivity))
        {
            if (owner->settings.numerics.reconstruction == "variational")
                input.viscousCoefficients = scope.ReadField(Coefficients().PhysicalField(0));
            else if (!directViscousGradients.empty())
                input.viscousConservativeGradient = scope.ReadField(directViscousGradients.front());
        }
        if (primitiveGradient)
            input.primitiveGradient = scope.ReadField(*primitiveGradient);
        if (auxiliary)
        {
            const auto &values = *auxiliary;
            if (values.meanThermodynamics)
            {
                scope.thermodynamics.emplace(values.meanThermodynamics->Read());
                input.auxiliary.meanThermodynamics.emplace(scope.thermodynamics->View<R::UniformTableRows>());
            }
            else if (values.derivedOwned)
            {
                const auto typed = EquationDofs::Coupled(owner->equation.Schema(), state.PhysicalField(0));
                scope.derivedOwned.emplace(values.derivedOwned->ReadCurrent(typed, owner->equation));
                if (ghosts && values.derivedGhost)
                    scope.derivedGhost.emplace(values.derivedGhost->ReadCurrent(typed, owner->equation));
                input.auxiliary.meanThermodynamics.emplace(MeanThermodynamicView::Derived(
                    *scope.derivedOwned,
                    scope.derivedGhost
                        ? OptionalRef<const DerivedFields::ReadScope>(*scope.derivedGhost)
                        : OptionalRef<const DerivedFields::ReadScope>{}));
            }
            if (values.cellWaveJump)
                input.auxiliary.cellWaveJump = scope.ReadField(*values.cellWaveJump);
            if (values.cellPressureMinimum)
                input.auxiliary.cellPressureMinimum = scope.ReadField(*values.cellPressureMinimum);
        }
        if (incoming.data)
            input.boundary = incoming;
        scope.result.emplace(result.PhysicalField(0).Owned().Write());
        scope.output.residual = scope.result->View<RHSFlowRows>();
        if (lowOrderResidual)
        {
            scope.lowOrder.emplace(lowOrderResidual->PhysicalField(0).Owned().Write());
            scope.output.lowOrder = scope.lowOrder->View<RHSFlowRows>();
        }
        if (faceOutput)
        {
            scope.face.emplace(faceOutput->Write());
            scope.output.faceFlux = scope.face->View<RHSFlowRows>();
        }
        diagnostics.Merge(fixedFlowRHS->ApplyFluxBorrowedFatal(input, scope.output, region, time));
    }
    void FiniteVolume::Workspace::ApplyRHS(const EquationDofs &state, EquationDofs &result, CFV::ReconstructionRegion region,
                                           EquationPointEvaluator &evaluator, real time,
                                           OptionalRef<const AuxiliaryValues> auxiliary,
                                           OptionalRef<R::Array<real>> faceOutput,
                                           RHSScalarView incoming)
    {
        const PhaseTime timing(*this, RHSPhase);
        if (fixedFlowRHS)
        {
            ApplyFixedFlowRHS(state, result, region, time, auxiliary, faceOutput, incoming);
            return;
        }
        const auto mean = region != CFV::ReconstructionRegion::Core ? state.ReadWithGhost() : state.ReadOwned();
        auto output = result.WriteOwned();
        const EquationViewLayout<const real> owned(state, mean), remote(state, mean, true);
        const EquationViewLayout<real> destination(result, output);
        if (lowOrderResidual)
        {
            auto lowOutput = lowOrderResidual->WriteOwned();
            const EquationViewLayout<real> lowDestination(*lowOrderResidual, lowOutput);
            const auto kernel = [&](auto u, auto ghost, auto rhs, auto low)
            { ApplyRHSRows(state, region, evaluator, time, auxiliary, mean, u, ghost, rhs, low, true, faceOutput, incoming); };
            DispatchCPUEquationLayouts(kernel, owned, remote, destination,
                                       lowDestination);
            return;
        }
        // The disabled low-order view deliberately aliases the high-order
        // descriptor. ApplyRHSRows receives produceLow=false and never
        // reads or writes it, avoiding a dummy allocation in non-PP runs.
        const auto kernel = [&](auto u, auto ghost, auto rhs, auto unusedLow)
        { ApplyRHSRows(state, region, evaluator, time, auxiliary, mean, u, ghost, rhs, unusedLow, false, faceOutput, incoming); };
        // RHS opts into prepared combined physical layout families.
        // Arbitrary block bindings and uncommon tiles keep the same generic
        // numerical loop.
        // Ghost shape may be absent during Core; no ghost value is read.
        DispatchCPUEquationLayouts(kernel, owned, remote, destination,
                                   destination);
    }
    template <class Input, class Output>
    void FiniteVolume::Workspace::ApplyRHSRows(const EquationDofs &state, CFV::ReconstructionRegion region,
                                               EquationPointEvaluator &evaluator, real time,
                                               OptionalRef<const AuxiliaryValues> auxiliary,
                                               const EquationDofs::ReadScope &mean, Input owned, Input remote, Output destination,
                                               Output lowDestination, bool produceLow,
                                               OptionalRef<R::Array<real>> faceOutput, RHSScalarView incoming)
    {
        index localFallbacks = 0;
        auto geometry = owner->metrics.AcquireRead();
        const auto metric = geometry.HostView();
        std::optional<CFV::ReconstructionBasisRead> basisRead;
        const bool direct = owner->settings.numerics.reconstruction != "variational";
        if (!direct)
            basisRead.emplace(owner->reconstruction->AcquireBasisRead());
        const bool ghosts = region != CFV::ReconstructionRegion::Core;
        // Preparation owns writes; the point kernel holds only numeric
        // read views. These means are not reconstructed thermodynamics.
        std::optional<R::Array<real>::ReadLease> meanProperties;
        std::optional<RHSScalarView> preparedMean;
        if (auxiliary && auxiliary->meanThermodynamics)
        {
            meanProperties.emplace(auxiliary->meanThermodynamics->Read());
            preparedMean.emplace(meanProperties->View<R::UniformTableRows>());
        }
        std::array<std::optional<R::Array<real>::ReadLease>, 4> stencilReads;
        RHSFieldView waveJump, pressureMinimum;
        const auto readStencil = [&](const std::optional<CFV::FVField> &field, int slot, RHSFieldView &view)
        {
            if (!field)
                return;
            stencilReads[slot].emplace(field->Owned().Read());
            if (ghosts)
                stencilReads[slot + 1].emplace(field->Ghost().Read());
            view = {stencilReads[slot]->View<R::UniformTableRows>(),
                    ghosts ? stencilReads[slot + 1]->View<R::UniformTableRows>() : RHSScalarView{}};
        };
        if (auxiliary)
        {
            readStencil(auxiliary->cellWaveJump, 0, waveJump);
            readStencil(auxiliary->cellPressureMinimum, 2, pressureMinimum);
        }
        std::optional<EquationDofs::ReadScope> rec, viscousRec;
        std::vector<EquationBlockReadView> recBlocks, viscousRecBlocks;
        if (!direct)
        {
            rec.emplace(ghosts ? FluxCoefficients().ReadWithGhost() : FluxCoefficients().ReadOwned());
            recBlocks = BlockViews(FluxCoefficients(), *rec);
            if (owner->equation.Viscous() && !owner->settings.numerics.viscousLimited &&
                (owner->settings.numerics.limiter == "WBAP" || limitedVR || owner->settings.numerics.positivity))
            {
                viscousRec.emplace(ghosts ? Coefficients().ReadWithGhost() : Coefficients().ReadOwned());
                viscousRecBlocks = BlockViews(Coefficients(), *viscousRec);
            }
        }
        const auto meanBlocks = BlockViews(state, mean);
        std::vector<EquationBlockLocation> directLocations;
        for (index iBlock = 0; direct && iBlock < state.BlockNumber(); ++iBlock)
            directLocations.push_back(state.LocateBlock(iBlock));
        std::vector<R::Array<real>::ReadLease> directOwned, directGhost, directViscousOwned, directViscousGhost;
        std::vector<RHSFieldView> directViews, directViscousViews;
        if (direct && owner->settings.numerics.reconstruction == "gradient" && !owner->settings.numerics.firstOrderConvection)
        {
            DNDS_check_throw_info(directGradients.size() == size_t(state.PhysicalFieldNumber()), "direct FV requires prepared physical gradients");
            for (const auto &gradient : directGradients)
            {
                directOwned.emplace_back(gradient.Owned().Read());
                if (ghosts)
                    directGhost.emplace_back(gradient.Ghost().Read());
                directViews.push_back({directOwned.back().View<R::UniformTableRows>(), ghosts ? directGhost.back().View<R::UniformTableRows>() : RHSScalarView{}});
            }
            for (const auto &gradient : directViscousGradients)
            {
                directViscousOwned.emplace_back(gradient.Owned().Read());
                if (ghosts)
                    directViscousGhost.emplace_back(gradient.Ghost().Read());
                directViscousViews.push_back({directViscousOwned.back().View<R::UniformTableRows>(),
                                              ghosts ? directViscousGhost.back().View<R::UniformTableRows>() : RHSScalarView{}});
            }
        }
        std::optional<R::Array<real>::ReadLease> primitiveOwned, primitiveGhost;
        RHSFieldView primitiveGradientValues;
        if (primitiveGradient)
        {
            primitiveOwned.emplace(primitiveGradient->Owned().Read());
            if (ghosts)
                primitiveGhost.emplace(primitiveGradient->Ghost().Read());
            primitiveGradientValues = {primitiveOwned->View<R::UniformTableRows>(),
                                       primitiveGhost ? primitiveGhost->View<R::UniformTableRows>() : RHSScalarView{}};
        }
        auto boundaryWrite = boundaryMomentumFlux.Write();
        const auto wallFlux = boundaryWrite.View<R::FixedRows<3>>();
        auto faceWrite = (faceOutput ? *faceOutput : faceFlux).Write();
        const auto faceValues = faceWrite.View<R::UniformTableRows>();
        std::optional<R::Array<real>::WriteLease> lowFaceWrite;
        decltype(faceWrite.View<R::UniformTableRows>()) lowFaceValues;
        if (produceLow)
        {
            DNDS_check_throw_info(lowOrderFaceFlux && lowOrderResidual &&
                                      state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1,
                                  "low-order equation flux requires the coupled flow5 PP profile");
            lowFaceWrite.emplace(lowOrderFaceFlux->Write());
            lowFaceValues = lowFaceWrite->View<R::UniformTableRows>();
        }
        const auto gatherMean = [&](index iCell, PointScratch &point)
        {
            for (index j = 0; j < owned.ComponentNumber(); ++j)
                point.values[j] = iCell < owned.RowCount() ? owned(iCell, j) : remote(iCell - owned.RowCount(), j);
            point.gradient.setZero();
        };
        const int modes = owner->reconstruction ? owner->reconstruction->NRecDOF() : 0;
        const int dim = owner->metrics.Dimension();
        const bool viscous = !meanOnly && owner->equation.Viscous();
        const auto &fluxPolicy = owner->fluxPolicy;
        const bool noRsOnWall = fluxPolicy.noRsOnWall;
        const bool noGRPOnWall = fluxPolicy.noGRPOnWall;
        const bool rusanovFlux = fluxPolicy.rusanov;
        const auto &fluxRequirements = fluxPolicy.required;
        const auto &baseFluxSettings = fluxPolicy.settings;
        const bool useWaveJump = owner->settings.numerics.riemann.useStencil && fluxPolicy.transverseSignal;
        const bool usePressureMinimum = owner->settings.numerics.riemann.useStencil && fluxPolicy.pressureStencil;
        const bool preparedWaveJump = auxiliary && bool(auxiliary->cellWaveJump);
        const bool preparedPressureMinimum = auxiliary && bool(auxiliary->cellPressureMinimum);
        DNDS_check_throw_info((!useWaveJump || preparedWaveJump) && (!usePressureMinimum || preparedPressureMinimum),
                              "required equation RHS stencil auxiliary is absent");
        const int species = owner->equation.HasMixture() ? owner->equation.Mixture().SpeciesNumber() : 0;
        const int independent = std::max(0, species - 1);
        const index width = state.ComponentNumber();
        const auto frame = owner->motion.At(time);
        PointScratch left(width), right(width), face(width), cellMean(width);
        const auto *ransModel = owner->equation.HasRANS() ? &owner->equation.RANS() : nullptr;
        const auto thermalBounds = thermal.View();
        const auto ransMean = ransModel ? mean.Block("rans") : EquationBlockReadView{};
        std::vector<real> integrated(size_t(width), 0), flux(size_t(width), 0), speciesFlux(size_t(independent), 0);
        std::vector<real> integratedLow(produceLow ? size_t(width) : 0, 0), lowFlux(produceLow ? size_t(width) : 0, 0);
        EquationTransportScratch transport(species);
        std::optional<WallDistanceRead> wallRead;
        if (owner->wallDistance)
            wallRead.emplace(owner->wallDistance->AcquireRead());
        const auto wallValues = wallRead ? wallRead->HostView() : WallDistanceView{};
        CFV::Math::SmallMatrix basisScratch;
        const auto reconstruct = [&](index iCell, const FiniteVolumeDetail::Sample &sample, int side, PointScratch &point)
        {
            gatherMean(iCell, point);
            // P0 uses mean states and only the normal face jump for
            // viscosity. This does not suppress volume-source gradients:
            // RANS/chemistry sampling retains the configured derivatives.
            if (!owner->settings.numerics.firstOrderConvection && (!direct || owner->settings.numerics.reconstruction == "gradient"))
            {
                if (direct)
                {
                    const auto displacement = FiniteVolumeDetail::DirectFVDisplacement(metric, sample, iCell != sample.cell || side != sample.side);
                    for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                    {
                        const auto location = directLocations[iBlock];
                        const auto row = directViews[location.physicalField][iCell];
                        const auto viscousRow = directViscousViews.empty() ? row : directViscousViews[location.physicalField][iCell];
                        for (index j = 0; j < location.componentNumber; ++j)
                            for (int d = 0; d < dim; ++d)
                            {
                                const real value = row[(location.firstComponent + j) * dim + d];
                                point.values[logicalFirst[iBlock] + j] += value * displacement[d];
                                if (viscous)
                                    point.gradient(logicalFirst[iBlock] + j, d) = viscousRow[(location.firstComponent + j) * dim + d];
                            }
                    }
                }
                else
                {
                    const auto basis = basisRead->FaceBasis(sample.face, side, sample.point, viscous ? 1 : 0, basisScratch);
                    for (index iBlock = 0; iBlock < index(meanBlocks.size()); ++iBlock)
                    {
                        const auto row = recBlocks[iBlock][iCell];
                        const auto viscousRow = viscousRecBlocks.empty() ? row : viscousRecBlocks[iBlock][iCell];
                        for (index j = 0; j < meanBlocks[iBlock].ComponentNumber(); ++j)
                            for (int mode = 0; mode < modes; ++mode)
                            {
                                const real value = row[j * modes + mode];
                                const index logical = logicalFirst[iBlock] + j;
                                point.values[logical] += value * basis(0, mode);
                                if (viscous)
                                    for (int d = 0; d < dim; ++d)
                                        point.gradient(logical, d) += viscousRow[j * modes + mode] * basis(1 + d, mode);
                            }
                    }
                }
            }
            if (ransModel)
                RANSStateView{ransModel->Kind(), ransModel->ComponentNumber()}.RepairReconstruction(ransMean[iCell], point.values.data() + ransFirst);
            if (primitiveGradient)
            {
                const auto row = primitiveGradientValues[iCell];
                for (int j = 0; j < 5; ++j)
                    for (int d = 0; d < dim; ++d)
                        point.gradient(flowFirst + j, d) = row[j * dim + d];
            }
            // PP POLICY -- point-local physical fallback. Reconstruction
            // deliberately retains raw rhoY for conservative advection and
            // diffusion; its final limiter check only evaluates a temporary
            // clipped composition matching EOS/T. No simplex condition is
            // imposed on the polynomial. If the actual point preparation
            // still fails, Cauchy may replace this temporary point by the
            // valid mean without mutating coefficients. Euler's separate
            // point policy is unchanged. See CauchyPositivity.md.
            const bool pointAdmissible = owner->settings.numerics.positivity
                                             ? thermalBounds.ClippedReferenceAdmissible(point.values)
                                             : thermalBounds.Admissible(point.values);
            if ((owner->settings.numerics.faceFallback || owner->settings.numerics.positivity) && !pointAdmissible)
            {
                ++localFallbacks;
                gatherMean(iCell, point);
            }
        };
        const bool allowPointFallback = owner->settings.numerics.faceFallback ||
                                        owner->settings.numerics.positivity;
        const auto evaluateReconstructedPoint = [&](index iCell, PointScratch &point, FlowState &flow)
        {
            const auto evaluate = [&]
            {
                return EquationPointEvaluator::FluxState(
                    flow, evaluator.Evaluate(flow, point.Species(speciesFirst, independent), 0, {}));
            };
            try
            {
                return evaluate();
            }
            catch (const std::exception &)
            {
                if (!allowPointFallback)
                    throw;
                // PP POLICY -- the limiter has already validated the same
                // clipped-reference input. A remaining EOS/T inversion
                // failure is point-local numerical failure: retry from the
                // admissible mean without changing raw polynomial storage.
                ++localFallbacks;
                gatherMean(iCell, point);
                flow = point.Flow(flowFirst);
                return evaluate();
            }
        };
        const auto computeFace = [&](index iAux)
        {
            const index first = owner->faceEntries[iAux];
            const auto &anchor = owner->samples[first];
            const index iCell = anchor.cell;
            const index end = first + metric.FaceQuadratureEnd(anchor.face) - metric.FaceQuadratureBegin(anchor.face);
            std::fill(integrated.begin(), integrated.end(), 0);
            if (produceLow)
                std::fill(integratedLow.begin(), integratedLow.end(), 0);
            for (index entry = first; entry < end; ++entry)
            {
                const auto &sample = owner->samples[entry];
                const bool internal = sample.other != UnInitIndex;
                const auto boundaryKind = sample.boundary.kind;
                const real weight = metric.GetFaceWeight(sample.point);
                reconstruct(iCell, sample, sample.side, left);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                real wallSAViscousInterior = 0;
#endif
                if (internal)
                {
                    reconstruct(sample.other, sample, 1 - sample.side, right);
                    if (sample.changesFrame)
                        RotatePoint(right, sample.rotation, vectorStarts);
                }
                else
                {
                    right.values = left.values;
                    right.gradient = left.gradient;
                    OptionalRef<const std::vector<real>> boundaryMean;
                    if (ransModel && ransModel->ComponentNumber() == 2)
                    {
                        gatherMean(sample.cell, cellMean);
                        boundaryMean.Emplace(cellMean.values);
                    }
                    owner->boundaryPoint.Evaluate(
                        right.values, sample.boundary, sample.boundaryPoint,
                        {sample.normal[0], sample.normal[1], sample.normal[2]},
                        {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]},
                        evaluator, frame, time,
                        BoundaryPointStateKind::ExteriorState, boundaryMean,
                        wallRead ? wallValues.cellMean(sample.cell, 0) : 0,
                        incoming);
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    if (ransModel && ransModel->Kind() == Physics::RANS::ModelKind::SA && sample.boundary.kind == BoundaryKind::NoSlipWall)
                    {
                        // Euler retains ULxyUnlim before its BC zeros the
                        // convective SA pair. Preserve that interior value
                        // for the viscous jump without mutating cell means.
                        wallSAViscousInterior = left.values[ransFirst];
                        left.values[ransFirst] = right.values[ransFirst] = 0;
                    }
#endif
                }
                auto leftFlow = left.Flow(flowFirst), rightFlow = right.Flow(flowFirst);
                const bool wallFluxState = noRsOnWall && !internal && boundaryKind == BoundaryKind::NoSlipWall;
                const auto prepareWallState = [&]
                {
                    face.values = left.values;
                    const auto original = left.Flow(flowFirst);
                    const Physics::Chemistry::SpeciesBufferView wallSpecies{independent ? face.values.data() + speciesFirst : nullptr, independent};
                    const auto wall = sample.boundary.FaceValue(evaluator, original, wallSpecies,
                                                                {sample.normal[0], sample.normal[1], sample.normal[2]},
                                                                {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time,
                                                                OptionalRef<const RigidFrame>(frame));
                    for (int j = 0; j < 5; ++j)
                        face.values[flowFirst + j] = wall[j];
                    if (ransModel)
                        for (int j = 0; j < ransModel->ComponentNumber(); ++j)
                            face.values[ransFirst + j] *= wall[0] / original[0];
                    return wall;
                };
                if (wallFluxState)
                    leftFlow = rightFlow = prepareWallState();
                const auto leftData = wallFluxState
                                          ? EquationPointEvaluator::FluxState(
                                                leftFlow, evaluator.Evaluate(leftFlow, face.Species(speciesFirst, independent), 0, {}))
                                          : evaluateReconstructedPoint(iCell, left, leftFlow);
                const auto rightData = wallFluxState
                                           ? EquationPointEvaluator::FluxState(
                                                 rightFlow, evaluator.Evaluate(rightFlow, face.Species(speciesFirst, independent), 0, {}))
                                       : internal
                                           ? evaluateReconstructedPoint(sample.other, right, rightFlow)
                                           : EquationPointEvaluator::FluxState(
                                                 rightFlow, evaluator.Evaluate(rightFlow, right.Species(speciesFirst, independent), 0, {}));
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                const auto grid = frame.GridVelocity({sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]});
                auto meanLeftData = leftData, meanRightData = rightData;
                if (fluxRequirements.meanStates || produceLow)
                {
                    // Mean eigenvalue preparation follows the same path as
                    // flow5. Storage/block traversal stays in this adapter.
                    Gather(meanBlocks, iCell, cellMean);
                    const auto meanLeft = cellMean.Flow(flowFirst);
                    auto meanRight = meanLeft;
                    if (internal)
                    {
                        Gather(meanBlocks, sample.other, cellMean);
                        meanRight = sample.changesFrame ? FiniteVolumeDetail::RotateState(cellMean.Flow(flowFirst), sample.rotation) : cellMean.Flow(flowFirst);
                    }
                    else
                        meanRight = sample.boundary.kind == BoundaryKind::Interface
                                        ? FiniteVolumeDetail::Load(incoming, sample.boundaryPoint)
                                        : FiniteVolumeDetail::BoundaryExterior(sample, owner->equation.Gas(), meanLeft, time, frame, {});
                    meanLeftData = preparedMean
                                       ? Riemann::AcousticState{{meanLeft, (*preparedMean)(iCell, 0)}, (*preparedMean)(iCell, 1)}
                                       : owner->equation.PrepareFluxState(meanLeft);
                    meanRightData = preparedMean && internal
                                        ? Riemann::AcousticState{{meanRight, (*preparedMean)(sample.other, 0)}, (*preparedMean)(sample.other, 1)}
                                        : owner->equation.PrepareFluxState(meanRight);
                }
                auto fluxSettings = baseFluxSettings;
                if (useWaveJump)
                {
                    real jump = waveJump(iCell, 0);
                    if (internal)
                        jump = std::max(jump, waveJump(sample.other, 0));
                    fluxSettings.dLambda = std::max(fluxSettings.dLambda, jump);
                }
                if (usePressureMinimum)
                {
                    real minimum = pressureMinimum(iCell, 0);
                    if (internal)
                        minimum = std::min(minimum, pressureMinimum(sample.other, 0));
                    fluxSettings.pressureStencilMin = std::min(fluxSettings.pressureStencilMin, minimum);
                }
                const auto convection = owner->equation.EvaluateFlowFlux(owner->flux, leftData, rightData, meanLeftData, meanRightData,
                                                                         normal, grid, fluxSettings);
                const auto &flowFlux = convection.flux;
                FlowState lowFlowFlux{};
                if (produceLow)
                    lowFlowFlux = Riemann::Rusanov(wallFluxState ? leftData : meanLeftData,
                                                   wallFluxState ? rightData : meanRightData, normal, grid);
                real speedLeft = 0, speedRight = 0;
                for (int d = 0; d < 3; ++d)
                {
                    speedLeft += (leftFlow[1 + d] / leftFlow[0] - grid[d]) * normal[d];
                    speedRight += (rightFlow[1 + d] / rightFlow[0] - grid[d]) * normal[d];
                }
                // The scalar kernel receives a fixed speed, not the flow5
                // acoustic flux. Roe's prepared contact magnitude is the
                // Euler velocity/entropy-fix policy. Rusanov's diagnostic
                // contact is acoustic, so use an explicit velocity speed.
                const real alpha = rusanovFlux
                                       ? .5 * (std::abs(speedLeft) + std::abs(speedRight))
                                       : convection.contact;
                for (index j = 0; j < width; ++j)
                    if (j < flowFirst || j >= flowFirst + 5)
                        flux[j] = wallFluxState ? face.values[j] * speedLeft : AdvectedScalarFlux(left.values[j], right.values[j], speedLeft, speedRight, alpha);
                for (int j = 0; j < 5; ++j)
                {
                    flux[flowFirst + j] = flowFlux[j];
                    if (produceLow)
                        lowFlux[flowFirst + j] = lowFlowFlux[j];
                }
                if (ransModel && ransModel->ComponentNumber() == 2)
                {
                    FlowState average;
                    for (int j = 0; j < 5; ++j)
                        average[j] = .5 * (leftFlow[j] + rightFlow[j]);
                    const auto stress = TurbulentNormalFlux(average, wallFluxState ? face.values[ransFirst] : .5 * (left.values[ransFirst] + right.values[ransFirst]), normal);
                    for (int j = 0; j < 5; ++j)
                        flux[flowFirst + j] += stress[j];
                }
                const bool wall = !internal && boundaryKind == BoundaryKind::NoSlipWall;
                if (viscous && (internal || wall))
                {
                    if (primitiveGradient)
                    {
                        DNDS_assert_info(!ransModel && independent == 0 && flowFirst == 0 && width == 5,
                                         "primitive-gradient leaf requires pure flow5");
                        const auto &gas = owner->equation.Gas();
                        auto primitive = gas.Primitive(leftFlow);
                        FlowGradient gradient{};
                        for (int j = 0; j < 5; ++j)
                            for (int d = 0; d < 3; ++d)
                                gradient[j][d] = left.gradient(flowFirst + j, d);
                        Gather(meanBlocks, iCell, cellMean);
                        const auto ownMean = cellMean.Flow(flowFirst);
                        const auto ownPrimitive = gas.Primitive(ownMean);
                        FlowState rightMean;
                        if (internal)
                        {
                            Gather(meanBlocks, sample.other, cellMean);
                            rightMean = sample.changesFrame
                                            ? FiniteVolumeDetail::RotateState(cellMean.Flow(flowFirst), sample.rotation)
                                            : cellMean.Flow(flowFirst);
                        }
                        else
                            rightMean = FiniteVolumeDetail::BoundaryGradientValue(sample, gas, ownMean, time, frame, incoming);
                        const auto rightPrimitive = gas.Primitive(rightMean);
                        const Geom::tPoint distance = !internal
                                                          ? Geom::tPoint(sample.ownPoint - metric.GetCellBary(iCell))
                                                          : Geom::tPoint(sample.otherPoint - metric.GetCellBary(iCell));
                        const real lengthSquared = distance.squaredNorm();
                        DNDS_assert_info(lengthSquared > 0, "zero primitive viscous stencil displacement");
                        const std::array<real, 5> own{ownPrimitive.density, ownPrimitive.velocity[0], ownPrimitive.velocity[1],
                                                      ownPrimitive.velocity[2], ownPrimitive.temperature};
                        const std::array<real, 5> neighbor{rightPrimitive.density, rightPrimitive.velocity[0], rightPrimitive.velocity[1],
                                                           rightPrimitive.velocity[2], rightPrimitive.temperature};
                        for (int j = 0; j < 5; ++j)
                        {
                            real projected = 0;
                            for (int d = 0; d < 3; ++d)
                            {
                                if (internal)
                                    gradient[j][d] = .5 * (gradient[j][d] + right.gradient(flowFirst + j, d));
                                projected += gradient[j][d] * distance[d];
                            }
                            for (int d = 0; d < 3; ++d)
                                gradient[j][d] += (neighbor[j] - own[j] - projected) * distance[d] / lengthSquared;
                        }
                        if (internal)
                        {
                            const auto other = gas.Primitive(rightFlow);
                            for (int d = 0; d < 3; ++d)
                                primitive.velocity[d] = .5 * (primitive.velocity[d] + other.velocity[d]);
                        }
                        else
                        {
                            FlowState faceMean{};
                            for (int j = 0; j < 5; ++j)
                                faceMean[j] = .5 * (leftFlow[j] + rightFlow[j]);
                            primitive = gas.Primitive(faceMean);
                            if (!sample.boundary.isothermal)
                            {
                                real heat = 0;
                                for (int d = 0; d < 3; ++d)
                                    heat += gradient[4][d] * normal[d];
                                for (int d = 0; d < 3; ++d)
                                    gradient[4][d] -= heat * normal[d];
                            }
                        }
                        const auto diffusion = gas.ViscousFlux(primitive, gradient, normal);
                        for (int j = 0; j < 5; ++j)
                        {
                            flux[j] -= diffusion[j];
                            if (produceLow)
                                lowFlux[j] -= diffusion[j];
                        }
                    }
                    else
                    {
                        const real inverseDistance = wall && noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                        for (index j = 0; j < width; ++j)
                        {
                            face.values[j] = .5 * (left.values[j] + right.values[j]);
                            for (int d = 0; d < 3; ++d)
                                face.gradient(j, d) = .5 * (left.gradient(j, d) + right.gradient(j, d)) +
                                                      normal[d] * (right.values[j] - left.values[j]) * inverseDistance;
                        }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                        if (wallSAViscousInterior != 0)
                            for (int d = 0; d < 3; ++d)
                                face.gradient(ransFirst, d) -= normal[d] * wallSAViscousInterior * inverseDistance;
#endif
                        if (wallFluxState)
                            prepareWallState();
                        const auto faceFlow = face.Flow(flowFirst);
                        const auto point = evaluator.Evaluate(faceFlow, face.Species(speciesFirst, independent), 0, {true, true, true, false, false});
                        struct SpeciesGradient
                        {
                            const PointScratch &point;
                            index first;
                            real operator()(int k, int d) const { return point.gradient(first + k, d); }
                        };
                        EquationTransportOptions transportOptions{wall && !sample.boundary.isothermal, wall};
                        if (ransFirst >= 0)
                        {
                            const auto &model = owner->equation.RANS();
                            const index iQ = sample.point - metric.FaceQuadratureBegin(sample.face);
                            // Transport does not consume the DES source length.
                            const auto ransPoint = RANSPoint(face, flowFirst, ransFirst, model, point, wallValues.faces(sample.face, iQ), 0);
                            const auto turbulent = model.Transport(ransPoint);
                            transportOptions.turbulentViscosity = turbulent.viscosity;
                            transportOptions.turbulentConductivity = turbulent.conductivity;
                            transportOptions.turbulentSpeciesDiffusivity = turbulent.speciesDiffusivity;
                            transportOptions.useQCR = model.Settings().useQCR;
                            const auto diffusionRANS = model.DiffusionFlux(ransPoint, Eigen::Vector3d{normal[0], normal[1], normal[2]}, turbulent.viscosity);
                            for (int j = 0; j < model.ComponentNumber(); ++j)
                                flux[ransFirst + j] -= diffusionRANS[j];
                        }
                        const auto diffusion = PrepareViscousFlux(faceFlow, face.FlowDerivative(flowFirst), point, face.Species(speciesFirst, independent),
                                                                  SpeciesGradient{face, speciesFirst}, normal, speciesFlux, transport, transportOptions);
                        for (int j = 0; j < 5; ++j)
                        {
                            flux[flowFirst + j] -= diffusion[j];
                            if (produceLow)
                                lowFlux[flowFirst + j] -= diffusion[j];
                        }
                        for (int k = 0; k < independent; ++k)
                            flux[speciesFirst + k] -= speciesFlux[k];
                    }
                }
                if (sample.boundaryPoint != UnInitIndex)
                    for (int d = 0; d < 3; ++d)
                        wallFlux(sample.boundaryPoint, d) = wall || boundaryKind == BoundaryKind::SlipWall
                                                                ? weight * flux[flowFirst + 1 + d]
                                                                : 0;
                for (index j = 0; j < width; ++j)
                {
                    integrated[j] -= weight * flux[j];
                    if (produceLow)
                        integratedLow[j] -= weight * lowFlux[j];
                }
            }
            for (index j = 0; j < width; ++j)
            {
                faceValues(iAux, j) = integrated[j];
                if (produceLow)
                    lowFaceValues(iAux, j) = integratedLow[j];
            }
        };
        if (region == CFV::ReconstructionRegion::All)
            for (index iAux = 0; iAux < index(owner->faceEntries.size()); ++iAux)
                computeFace(iAux);
        else
            for (const auto iAux : region == CFV::ReconstructionRegion::Core ? owner->rhsCoreFaces : owner->rhsDependentFaces)
                computeFace(iAux);
        const auto compute = [&](index iCell)
        {
            std::fill(integrated.begin(), integrated.end(), 0);
            if (produceLow)
                std::fill(integratedLow.begin(), integratedLow.end(), 0);
            for (index entry = owner->offsets[iCell]; entry < owner->offsets[iCell + 1];)
            {
                const auto &sample = owner->samples[entry];
                const index iAux = sample.auxiliaryFace;
                const auto &anchor = owner->samples[owner->faceEntries[iAux]];
                const bool opposite = sample.side != anchor.side;
                for (index j = 0; j < width; ++j)
                {
                    flux[j] = (opposite ? -1 : 1) * faceValues(iAux, j);
                    if (produceLow)
                        lowFlux[j] = (opposite ? -1 : 1) * lowFaceValues(iAux, j);
                }
                // Scalar fluxes change sign; vector fluxes additionally
                // rotate from the other parent's periodic component frame.
                // Side, not cell identity, distinguishes self-periodic faces.
                if (opposite && sample.changesFrame)
                    for (const index first : vectorStarts)
                    {
                        const Geom::tPoint value{flux[first], flux[first + 1], flux[first + 2]};
                        const Geom::tPoint rotated = sample.rotation * value;
                        for (int d = 0; d < 3; ++d)
                            flux[first + d] = rotated[d];
                        if (produceLow)
                        {
                            const Geom::tPoint lowValue{lowFlux[first], lowFlux[first + 1], lowFlux[first + 2]};
                            const Geom::tPoint lowRotated = sample.rotation * lowValue;
                            for (int d = 0; d < 3; ++d)
                                lowFlux[first + d] = lowRotated[d];
                        }
                    }
                for (index j = 0; j < width; ++j)
                {
                    integrated[j] += flux[j];
                    if (produceLow)
                        integratedLow[j] += lowFlux[j];
                }
                entry += metric.FaceQuadratureEnd(sample.face) - metric.FaceQuadratureBegin(sample.face);
            }
            for (index j = 0; j < destination.ComponentNumber(); ++j)
            {
                destination(iCell, j) = integrated[j];
                if (produceLow)
                    lowDestination(iCell, j) = integratedLow[j];
            }
        };
        if (region == CFV::ReconstructionRegion::All)
            for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
                compute(iCell);
        else
            for (const auto iCell : region == CFV::ReconstructionRegion::Core ? owner->partition.Core() : owner->partition.HaloDependent())
                compute(iCell);
        const size_t slot = region == CFV::ReconstructionRegion::HaloDependent ? 1 : 0;
        faceFallbacks[slot] = localFallbacks;
        if (region == CFV::ReconstructionRegion::All)
            faceFallbacks[1] = 0;
        if (!allowPointFallback && localFallbacks)
            DNDS_check_throw_info(false, "inadmissible reconstructed face with faceFallback disabled");
    }
}
