#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/ViscousFlux.hpp"
#include <cstdio>
#include <cstdlib>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        using FiniteVolumeDetail::Load;
        using FiniteVolumeDetail::RotateState;
        FlowGradient RotateGradient(const FlowGradient &gradient, const Geom::tGPoint &rotation)
        {
            FlowGradient result{};
            for (int component = 0; component < 5; ++component)
                for (int d = 0; d < 3; ++d)
                    for (int k = 0; k < 3; ++k)
                        if (component == 0 || component == 4)
                            result[component][d] += rotation(d, k) * gradient[component][k];
                        else
                            for (int v = 0; v < 3; ++v)
                                result[component][d] += rotation(component - 1, v) * gradient[1 + v][k] * rotation(d, k);
            return result;
        }
        template <bool fatal>
        void RequireKernel(bool condition, const char *message)
        {
            if (condition)
                return;
            if constexpr (fatal)
            {
                std::fprintf(stderr, "Cauchy fatal RHS kernel invariant: %s\n", message);
                std::fflush(stderr);
                std::abort();
            }
            else
                DNDS_check_throw_info(false, message);
        }
    }

    struct FiniteVolume::RHSOperator::ReadData
    {
        std::shared_ptr<const FiniteVolume::Data> data;
        CFV::FiniteVolumeRead geometry;
        std::optional<CFV::ReconstructionBasisRead> basis;
        explicit ReadData(std::shared_ptr<const FiniteVolume::Data> owner)
            : data(std::move(owner)), geometry(data->metrics.AcquireRead())
        {
            if (data->settings.numerics.reconstruction == "variational")
                basis.emplace(data->reconstruction.AcquireBasisRead());
            data->partition.Validate();
        }
    };

    FiniteVolume::RHSOperator FiniteVolume::AcquireRHSOperator() const
    {
        return RHSOperator(std::make_shared<RHSOperator::ReadData>(_data));
    }

    RHSKernelDiagnostics FiniteVolume::RHSOperator::Apply(const RHSReadViews &input, const RHSWriteViews &output,
                                                          CFV::ReconstructionRegion region, real time) const
    {
        return ApplyBorrowedChecked(input, output, region, time);
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

    RHSKernelDiagnostics FiniteVolume::RHSOperator::ApplyFlux(const RHSReadViews &input, const RHSWriteViews &output,
                                                              CFV::ReconstructionRegion region, real time) const
    {
        return ApplyFluxImpl<FailurePolicy::Throw>(input, output, region, time);
    }

    template <FiniteVolume::RHSOperator::FailurePolicy policy>
    RHSKernelDiagnostics FiniteVolume::RHSOperator::ApplyFluxImpl(const RHSReadViews &input, const RHSWriteViews &output,
                                                                  CFV::ReconstructionRegion region, real time) const
    {
        constexpr bool fatal = policy == FailurePolicy::Abort;
        RequireKernel<fatal>(_read && std::isfinite(time), "invalid RHS operator or time");
        const auto &data = _read->data;
        const auto metric = _read->geometry.HostView();
        const int modes = data->reconstruction.NRecDOF(), dim = data->metrics.Dimension();
        const auto &gas = data->equation.Gas();
        const auto &thermalBounds = data->PerfectGasBounds();
        const auto frame = data->motion.At(time);
        RequireKernel<fatal>(!data->hasInterfaces || input.boundary, "RHS requires incoming interface values");
        const auto incoming = input.boundary.value_or(RHSScalarView{});
        if (input.boundary)
            RequireKernel<fatal>(incoming.RowCount() == index(data->boundaryEntries.size()) && incoming.Width() == 5, "RHS interface input shape differs");
        const bool viscous = data->equation.Viscous() && input.includeViscosity;
        const bool direct = data->settings.numerics.reconstruction != "variational";
        const auto physicalGradient = input.conservativeGradient.value_or(RHSFieldView{});
        const auto physicalViscousGradient = input.viscousConservativeGradient.value_or(physicalGradient);
        // P0 always uses the conservative normal jump, without reconstructed
        // face derivatives; higher orders retain their selected formulation.
        const bool polynomialViscosity = viscous && (data->settings.numerics.firstOrderConvection || data->settings.numerics.viscousGradient == "polynomial");
        const auto viscousCoefficient = input.viscousCoefficients.value_or(input.coefficients);
        const auto &mean = input.mean, &coefficient = input.coefficients, &gradientInput = input.primitiveGradient;
        const auto result = output.residual;
        const bool produceLow = bool(output.lowOrder), hasThermo = bool(input.auxiliary.meanThermodynamics);
        const bool useWave = bool(input.auxiliary.cellWaveJump), usePressure = bool(input.auxiliary.cellPressureMinimum);
        const auto lowOutput = output.lowOrder.value_or(RHSOutputView{});
        const auto thermo = input.auxiliary.meanThermodynamics.value_or(RHSScalarView{});
        const auto waveInput = input.auxiliary.cellWaveJump.value_or(RHSFieldView{});
        const auto pressureInput = input.auxiliary.cellPressureMinimum.value_or(RHSFieldView{});
        RHSKernelDiagnostics diagnostics(index(data->forceZones.size()));
        const auto &fluxPolicy = data->fluxPolicy;
        const auto baseFluxSettings = fluxPolicy.settings;
        const bool writeFace = bool(output.faceFlux);
        const bool noRsOnWall = fluxPolicy.noRsOnWall;
        const bool noGRPOnWall = fluxPolicy.noGRPOnWall;
        const bool distinctViscousState = direct ? bool(input.viscousConservativeGradient) : bool(input.viscousCoefficients);
        RequireKernel<fatal>(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                 region == CFV::ReconstructionRegion::HaloDependent,
                             "invalid RHS region");
        const bool needGhost = region != CFV::ReconstructionRegion::Core && !data->partition.HaloDependent().empty();
        const auto checkField = [&](const RHSFieldView &field, index width)
        {
            RequireKernel<fatal>(field.owned.RowCount() == data->metrics.NumCell() && field.owned.Width() == width,
                                 "RHS owned input shape differs");
            if (needGhost)
                RequireKernel<fatal>(field.ghost.RowCount() == data->metrics.NumCellProc() - data->metrics.NumCell() && field.ghost.Width() == width,
                                     "RHS ghost input is missing or has wrong shape");
        };
        checkField(mean, 5);
        if (!direct)
            checkField(coefficient, 5 * modes);
        else if (!input.meanOnly && !data->settings.numerics.firstOrderConvection && data->settings.numerics.reconstruction == "gradient")
        {
            RequireKernel<fatal>(bool(input.conservativeGradient), "direct P1 RHS requires physical conservative gradients");
            checkField(physicalGradient, 5 * dim);
            if (polynomialViscosity)
                checkField(physicalViscousGradient, 5 * dim);
        }
        if (viscous && !polynomialViscosity)
            checkField(gradientInput, 5 * dim);
        if (polynomialViscosity && !direct)
            checkField(viscousCoefficient, 5 * modes);
        if (useWave)
            checkField(waveInput, 1);
        if (usePressure)
            checkField(pressureInput, 1);
        RequireKernel<fatal>(result.RowCount() == data->metrics.NumCell() && result.Width() == 5, "RHS output shape differs");
        if (produceLow)
            RequireKernel<fatal>(lowOutput.RowCount() == result.RowCount() && lowOutput.Width() == 5, "low-order RHS shape differs");
        if (output.faceFlux)
            RequireKernel<fatal>(output.faceFlux->RowCount() == index(data->stencilFaces.size()) && output.faceFlux->Width() == 5,
                                 "RHS face output shape differs");
        if (hasThermo)
            RequireKernel<fatal>(thermo.Width() == 3 && thermo.RowCount() >= (needGhost ? data->metrics.NumCellProc() : data->metrics.NumCell()),
                                 "RHS thermodynamic cache shape differs");
        RequireKernel<fatal>(!data->settings.numerics.riemann.useStencil ||
                                 ((!fluxPolicy.transverseSignal || useWave) && (!fluxPolicy.pressureStencil || usePressure)),
                             "required RHS wave or pressure auxiliary input is absent");
        auto compute = [&](index iCell)
        {
            FlowState integrated{};
            FlowState integratedLow{};
            for (index entry = data->offsets[iCell]; entry < data->offsets[iCell + 1]; ++entry)
            {
                const auto &sample = data->samples[entry];
                const bool internal = sample.other != UnInitIndex;
                const auto boundaryKind = sample.boundary.kind;
                const real weight = metric.GetFaceWeight(sample.point);
                if (writeFace && sample.point == metric.FaceQuadratureBegin(sample.face))
                    for (int j = 0; j < 5; ++j)
                        (*output.faceFlux)(sample.auxiliaryFace, j) = 0;
                auto reconstruct = [&](index cell, int parent)
                {
                    auto state = Load(mean, cell);
                    auto viscousState = state;
                    FlowGradient gradient{};
                    if (direct && !input.meanOnly && data->settings.numerics.reconstruction == "gradient" && !data->settings.numerics.firstOrderConvection)
                    {
                        const auto displacement = FiniteVolumeDetail::DirectFVDisplacement(metric, sample, parent != 0);
                        const auto row = physicalGradient[cell];
                        const auto viscousRow = physicalViscousGradient[cell];
                        for (int component = 0; component < 5; ++component)
                            for (int d = 0; d < dim; ++d)
                            {
                                state[component] += row[component * dim + d] * displacement[d];
                                if (polynomialViscosity)
                                {
                                    viscousState[component] += viscousRow[component * dim + d] * displacement[d];
                                    gradient[component][d] = viscousRow[component * dim + d];
                                }
                            }
                    }
                    else if (!direct && !input.meanOnly && !data->settings.numerics.firstOrderConvection)
                    {
                        CFV::Math::SmallMatrix basisScratch;
                        const auto basis = _read->basis->FaceBasis(sample.face, parent == 0 ? sample.side : 1 - sample.side,
                                                                   sample.point, polynomialViscosity ? 1 : 0, basisScratch);
                        const auto row = coefficient[cell];
                        const auto viscousRow = polynomialViscosity ? viscousCoefficient[cell] : decltype(viscousCoefficient[cell]){};
                        for (int component = 0; component < 5; ++component)
                            for (int mode = 0; mode < modes; ++mode)
                            {
                                const real value = row[component * modes + mode];
                                state[component] += value * basis(0, mode);
                                if (polynomialViscosity)
                                {
                                    const real viscousValue = input.viscousCoefficients ? viscousRow[component * modes + mode] : value;
                                    if (input.viscousCoefficients)
                                        viscousState[component] += viscousValue * basis(0, mode);
                                    for (int d = 0; d < dim; ++d)
                                        gradient[component][d] += viscousValue * basis(1 + d, mode);
                                }
                            }
                    }
                    if (!distinctViscousState)
                        viscousState = state;
                    // Explicit local first-order fallback keeps the mean
                    // unchanged when a high-order face value is inadmissible.
                    if (!ThermallyAdmissible(state, thermalBounds))
                    {
                        diagnostics.AddFaceFallback();
                        state = Load(mean, cell);
                    }
                    if (viscous && !polynomialViscosity)
                        for (int component = 0; component < 5; ++component)
                            for (int d = 0; d < dim; ++d)
                                gradient[component][d] = gradientInput(cell, component * dim + d);
                    struct PointState
                    {
                        FlowState state, viscousState;
                        FlowGradient gradient;
                    };
                    return PointState{state, viscousState, gradient};
                };
                const auto left = reconstruct(iCell, 0);
                FlowState right{}, viscousRight{};
                FlowGradient rightGradient{};
                if (internal)
                {
                    auto other = reconstruct(sample.other, 1);
                    right = sample.changesFrame ? RotateState(other.state, sample.rotation) : other.state;
                    if (viscous)
                    {
                        viscousRight = sample.changesFrame ? RotateState(other.viscousState, sample.rotation) : other.viscousState;
                        rightGradient = sample.changesFrame ? RotateGradient(other.gradient, sample.rotation) : other.gradient;
                    }
                }
                else
                {
                    right = FiniteVolumeDetail::BoundaryExterior(sample, gas, left.state, time, frame, incoming);
                    viscousRight = right;
                    rightGradient = left.gradient;
                }
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                const Vector3 point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                const bool wallFluxState = noRsOnWall && !internal && boundaryKind == BoundaryKind::NoSlipWall;
                const auto convectiveLeft = wallFluxState ? FiniteVolumeDetail::BoundaryFaceValue(sample, gas, left.state, time, frame, incoming) : left.state;
                const auto convectiveRight = wallFluxState ? convectiveLeft : right;
                const auto leftFlux = data->equation.PrepareFluxState(convectiveLeft), rightFlux = data->equation.PrepareFluxState(convectiveRight);
                const auto gridVelocity = frame.GridVelocity(point);
                const auto leftMean = Load(mean, iCell);
                const auto otherMean = internal ? Load(mean, sample.other) : FlowState{};
                const auto rightMean = !internal             ? FiniteVolumeDetail::BoundaryExterior(sample, gas, leftMean, time, frame, incoming)
                                       : sample.changesFrame ? RotateState(otherMean, sample.rotation)
                                                             : otherMean;
                const auto leftMeanData = hasThermo ? Riemann::AcousticState{{leftMean, thermo(iCell, 0)}, thermo(iCell, 1)} : data->equation.PrepareFluxState(leftMean);
                const auto rightMeanData = !internal || !hasThermo ? data->equation.PrepareFluxState(rightMean) : Riemann::AcousticState{{rightMean, thermo(sample.other, 0)}, thermo(sample.other, 1)};
                auto fluxSettings = baseFluxSettings;
                if (useWave)
                {
                    real jump = waveInput(iCell, 0);
                    if (internal)
                        jump = std::max(jump, waveInput(sample.other, 0));
                    fluxSettings.dLambda = std::max(fluxSettings.dLambda, jump);
                }
                if (usePressure)
                {
                    real minimum = pressureInput(iCell, 0);
                    if (internal)
                        minimum = std::min(minimum, pressureInput(sample.other, 0));
                    fluxSettings.pressureStencilMin = std::min(fluxSettings.pressureStencilMin, minimum);
                }
                auto flux = data->equation.EvaluateFlowFlux(data->flux, leftFlux, rightFlux, leftMeanData, rightMeanData, normal, gridVelocity, fluxSettings).flux;
                FlowState lowFlux{};
                if (produceLow)
                    lowFlux = wallFluxState ? Riemann::Rusanov(leftFlux, rightFlux, normal, gridVelocity) : Riemann::Rusanov(leftMeanData, rightMeanData, normal, gridVelocity);
                if (polynomialViscosity && (internal || boundaryKind == BoundaryKind::NoSlipWall || boundaryKind == BoundaryKind::Interface))
                {
                    FlowState rightLinearization = convectiveRight;
                    FlowState faceMean{};
                    for (int j = 0; j < 5; ++j)
                        faceMean[j] = .5 * (convectiveLeft[j] + rightLinearization[j]);
                    RequireKernel<fatal>(ThermallyAdmissible(faceMean, thermalBounds), "inadmissible polynomial viscous linearization");
                    const bool wall = !internal && boundaryKind == BoundaryKind::NoSlipWall;
                    const real inverseDistance = wall && noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                    const auto diffusion = PolynomialViscousFlux(data->equation, convectiveLeft, rightLinearization, left.viscousState, viscousRight,
                                                                 left.gradient, rightGradient, normal, inverseDistance, wall && !sample.boundary.isothermal);
                    for (int component = 0; component < 5; ++component)
                    {
                        flux[component] -= diffusion[component];
                        lowFlux[component] -= diffusion[component];
                    }
                }
                else if (viscous && !polynomialViscosity &&
                         (internal || boundaryKind == BoundaryKind::NoSlipWall || boundaryKind == BoundaryKind::Interface))
                {
                    auto primitive = gas.Primitive(left.state);
                    auto gradient = left.gradient;
                    const auto ownPrimitive = gas.Primitive(Load(mean, iCell));
                    const auto rightPrimitive = gas.Primitive(!internal             ? FiniteVolumeDetail::BoundaryGradientValue(sample, gas, Load(mean, iCell), time, frame, incoming)
                                                              : sample.changesFrame ? RotateState(Load(mean, sample.other), sample.rotation)
                                                                                    : Load(mean, sample.other));
                    // Evaluate into an owning vector: an Eigen expression
                    // would retain a dangling temporary cell barycenter.
                    const Geom::tPoint distance = !internal ? Geom::tPoint(sample.ownPoint - metric.GetCellBary(iCell)) : Geom::tPoint(sample.otherPoint - metric.GetCellBary(iCell));
                    const real lengthSquared = distance.squaredNorm();
                    RequireKernel<fatal>(lengthSquared > 0, "zero viscous stencil displacement");
                    const std::array<real, 5> own{ownPrimitive.density, ownPrimitive.velocity[0], ownPrimitive.velocity[1], ownPrimitive.velocity[2], ownPrimitive.temperature};
                    const std::array<real, 5> neighbor{rightPrimitive.density, rightPrimitive.velocity[0], rightPrimitive.velocity[1], rightPrimitive.velocity[2], rightPrimitive.temperature};
                    for (int component = 0; component < 5; ++component)
                    {
                        real projected = 0;
                        for (int d = 0; d < 3; ++d)
                        {
                            if (internal)
                                gradient[component][d] = .5 * (gradient[component][d] + rightGradient[component][d]);
                            projected += gradient[component][d] * distance[d];
                        }
                        for (int d = 0; d < 3; ++d)
                            gradient[component][d] += (neighbor[component] - own[component] - projected) * distance[d] / lengthSquared;
                    }
                    if (internal)
                    {
                        const auto p = gas.Primitive(right);
                        for (int d = 0; d < 3; ++d)
                            primitive.velocity[d] = .5 * (primitive.velocity[d] + p.velocity[d]);
                    }
                    else if (boundaryKind == BoundaryKind::SlipWall || boundaryKind == BoundaryKind::NoSlipWall)
                    {
                        FlowState faceMean{};
                        for (int j = 0; j < 5; ++j)
                            faceMean[j] = .5 * (convectiveLeft[j] + convectiveRight[j]);
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
                    const auto viscous = gas.ViscousFlux(primitive, gradient, normal);
                    for (int component = 0; component < 5; ++component)
                    {
                        flux[component] -= viscous[component];
                        lowFlux[component] -= viscous[component];
                    }
                }
                if (sample.forceSlot != UnInitIndex)
                    for (int d = 0; d < 3; ++d)
                        diagnostics.AddBoundaryForce(sample.forceSlot, d, weight * flux[1 + d]);
                FlowState canonical{};
                if (writeFace)
                {
                    const auto &anchor = data->samples[data->faceEntries[sample.auxiliaryFace]];
                    canonical = sample.side == anchor.side || !sample.changesFrame ? flux : RotateState(flux, sample.rotation.transpose());
                    if (sample.side != anchor.side)
                        for (auto &value : canonical)
                            value = -value;
                }
                for (int component = 0; component < 5; ++component)
                {
                    integrated[component] -= weight * flux[component];
                    if (writeFace)
                        (*output.faceFlux)(sample.auxiliaryFace, component) -= weight * canonical[component];
                    if (produceLow)
                        integratedLow[component] -= weight * lowFlux[component];
                }
            }
            for (int component = 0; component < 5; ++component)
            {
                result(iCell, component) = integrated[component];
                if (produceLow)
                    lowOutput(iCell, component) = integratedLow[component];
            }
        };
        if (region == CFV::ReconstructionRegion::All)
            for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
                compute(iCell);
        else
            for (auto iCell : region == CFV::ReconstructionRegion::Core ? data->partition.Core() : data->partition.HaloDependent())
                compute(iCell);
        return diagnostics;
    }

    void FiniteVolume::RHSOperator::AddSource(const RHSReadViews &input, const RHSWriteViews &output,
                                              CFV::ReconstructionRegion region, real time) const
    {
        AddSourceImpl<FailurePolicy::Throw>(input, output, region, time);
    }

    template <FiniteVolume::RHSOperator::FailurePolicy policy>
    void FiniteVolume::RHSOperator::AddSourceImpl(const RHSReadViews &input, const RHSWriteViews &output,
                                                  CFV::ReconstructionRegion region, real time) const
    {
        constexpr bool fatal = policy == FailurePolicy::Abort;
        RequireKernel<fatal>(_read && std::isfinite(time), "invalid source operator or time");
        const auto &data = _read->data;
        RequireKernel<fatal>(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                 region == CFV::ReconstructionRegion::HaloDependent,
                             "invalid source region");
        RequireKernel<fatal>(input.mean.owned.RowCount() == data->metrics.NumCell() && input.mean.owned.Width() == 5 &&
                                 output.residual.RowCount() == data->metrics.NumCell() && output.residual.Width() == 5,
                             "source input/output shape differs");
        if (output.lowOrder)
            RequireKernel<fatal>(output.lowOrder->RowCount() == output.residual.RowCount() && output.lowOrder->Width() == 5, "source low-order output shape differs");
        const auto frame = data->motion.At(time);
        if (!frame.moving)
            return;
        const auto metric = _read->geometry.HostView();
        const auto compute = [&](index iCell)
        {
            const auto source = frame.RotationSource(Load(input.mean, iCell));
            for (int j = 0; j < 5; ++j)
            {
                const real value = metric.GetCellVol(iCell) * source[j];
                output.residual(iCell, j) += value;
                if (output.lowOrder)
                    (*output.lowOrder)(iCell, j) += value;
            }
        };
        if (region == CFV::ReconstructionRegion::All)
            for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
                compute(iCell);
        else
            for (const auto iCell : region == CFV::ReconstructionRegion::Core ? data->partition.Core() : data->partition.HaloDependent())
                compute(iCell);
    }
}
