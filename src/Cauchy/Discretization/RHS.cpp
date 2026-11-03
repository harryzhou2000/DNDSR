#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Discretization/ViscousFlux.hpp"

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
    }

    struct FiniteVolume::RHSOperator::ReadData
    {
        std::shared_ptr<const FiniteVolume::Data> data;
        CFV::FiniteVolumeRead geometry;
        CFV::ReconstructionBasisRead basis;
        explicit ReadData(std::shared_ptr<const FiniteVolume::Data> owner)
            : data(std::move(owner)), geometry(data->metrics.AcquireRead()), basis(data->reconstruction.AcquireBasisRead())
        {
            data->partition.Validate();
        }
    };

    FiniteVolume::RHSOperator FiniteVolume::AcquireRHSOperator() const
    {
        return RHSOperator(std::make_shared<RHSOperator::ReadData>(_data));
    }

    void FiniteVolume::RHSOperator::Apply(const RHSReadViews &input, const RHSWriteViews &output,
                                          CFV::ReconstructionRegion region, real time) const
    {
        DNDS_check_throw_info(_read && std::isfinite(time), "invalid RHS operator or time");
        const auto &data = _read->data;
        const auto metric = _read->geometry.HostView();
        const int modes = data->reconstruction.NRecDOF(), dim = data->metrics.Dimension();
        const auto &gas = data->equation.Gas();
        const auto frame = data->motion.At(time);
        const bool viscous = data->equation.Viscous() && input.includeViscosity;
        const bool polynomialViscosity = viscous && data->settings.numerics.viscousGradient == "polynomial";
        const auto viscousCoefficient = input.viscousCoefficients.value_or(input.coefficients);
        const auto &mean = input.mean, &coefficient = input.coefficients, &gradientInput = input.primitiveGradient;
        const auto result = output.residual;
        const bool produceLow = bool(output.lowOrder), hasThermo = bool(input.auxiliary.meanThermodynamics);
        const bool useWave = bool(input.auxiliary.cellWaveJump), usePressure = bool(input.auxiliary.cellPressureMinimum);
        const auto lowOutput = output.lowOrder.value_or(RHSOutputView{});
        const auto thermo = input.auxiliary.meanThermodynamics.value_or(RHSScalarView{});
        const auto waveInput = input.auxiliary.cellWaveJump.value_or(RHSFieldView{});
        const auto pressureInput = input.auxiliary.cellPressureMinimum.value_or(RHSFieldView{});
        auto *forces = output.boundaryForces;
        auto *diagnostics = output.faceFallbacks;
        DNDS_check_throw_info(region == CFV::ReconstructionRegion::All || region == CFV::ReconstructionRegion::Core ||
                                  region == CFV::ReconstructionRegion::HaloDependent,
                              "invalid RHS region");
        const bool needGhost = region != CFV::ReconstructionRegion::Core && !data->partition.HaloDependent().empty();
        const auto checkField = [&](const RHSFieldView &field, index width)
        {
            DNDS_check_throw_info(field.owned.RowCount() == data->metrics.NumCell() && field.owned.Width() == width,
                                  "RHS owned input shape differs");
            if (needGhost)
                DNDS_check_throw_info(field.ghost.RowCount() == data->metrics.NumCellProc() - data->metrics.NumCell() && field.ghost.Width() == width,
                                      "RHS ghost input is missing or has wrong shape");
        };
        checkField(mean, 5);
        checkField(coefficient, 5 * modes);
        if (viscous && !polynomialViscosity)
            checkField(gradientInput, 5 * dim);
        if (polynomialViscosity)
            checkField(viscousCoefficient, 5 * modes);
        if (useWave)
            checkField(waveInput, 1);
        if (usePressure)
            checkField(pressureInput, 1);
        DNDS_check_throw_info(result.RowCount() == data->metrics.NumCell() && result.Width() == 5, "RHS output shape differs");
        if (produceLow)
            DNDS_check_throw_info(lowOutput.RowCount() == result.RowCount() && lowOutput.Width() == 5, "low-order RHS shape differs");
        if (hasThermo)
            DNDS_check_throw_info(thermo.Width() == 3 && thermo.RowCount() >= (needGhost ? data->metrics.NumCellProc() : data->metrics.NumCell()),
                                  "RHS thermodynamic cache shape differs");
        DNDS_check_throw_info(diagnostics || (data->settings.numerics.faceFallback && !data->settings.numerics.positivity),
                              "strict face admissibility requires explicit fallback output");
        DNDS_check_throw_info(!data->settings.numerics.riemann.useStencil ||
                                  ((!data->flux.UsesTransverseSignal() || useWave) && (!data->flux.UsesPressureStencil() || usePressure)),
                              "required RHS wave or pressure auxiliary input is absent");
        auto compute = [&](index iCell)
        {
            FlowState integrated{};
            FlowState integratedLow{};
            for (index entry = data->offsets[iCell]; entry < data->offsets[iCell + 1]; ++entry)
            {
                const auto &sample = data->samples[entry];
                auto reconstruct = [&](index cell, int parent)
                {
                    CFV::Math::SmallMatrix basisScratch;
                    const auto basis = _read->basis.FaceBasis(sample.face, parent == 0 ? sample.side : 1 - sample.side,
                                                              sample.point, polynomialViscosity ? 1 : 0, basisScratch);
                    auto state = Load(mean, cell);
                    auto viscousState = state;
                    FlowGradient gradient{};
                    for (int component = 0; component < 5; ++component)
                        for (int mode = 0; mode < modes; ++mode)
                        {
                            const real value = coefficient(cell, component * modes + mode);
                            state[component] += value * basis(0, mode);
                            if (polynomialViscosity)
                            {
                                const real viscousValue = input.viscousCoefficients ? viscousCoefficient(cell, component * modes + mode) : value;
                                if (input.viscousCoefficients)
                                    viscousState[component] += viscousValue * basis(0, mode);
                                for (int d = 0; d < dim; ++d)
                                    gradient[component][d] += viscousValue * basis(1 + d, mode);
                            }
                        }
                    if (!input.viscousCoefficients)
                        viscousState = state;
                    // Explicit local first-order fallback keeps the mean
                    // unchanged when a high-order face value is inadmissible.
                    if (!ThermallyAdmissible(state, data->thermalBounds))
                    {
                        if (diagnostics)
                            diagnostics->fetch_add(1, std::memory_order_relaxed);
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
                FlowState right, viscousRight;
                FlowGradient rightGradient{};
                if (sample.other != UnInitIndex)
                {
                    auto other = reconstruct(sample.other, 1);
                    right = RotateState(other.state, sample.rotation);
                    viscousRight = RotateState(other.viscousState, sample.rotation);
                    rightGradient = RotateGradient(other.gradient, sample.rotation);
                }
                else
                {
                    right = sample.boundary.Exterior(gas, left.state, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time, &frame);
                    viscousRight = right;
                    rightGradient = left.gradient;
                }
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                const auto leftFlux = data->equation.PrepareFluxState(left.state), rightFlux = data->equation.PrepareFluxState(right);
                const Vector3 point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                const auto gridVelocity = frame.GridVelocity(point);
                const auto leftMean = Load(mean, iCell), rightMean = sample.other == UnInitIndex ? sample.boundary.Exterior(gas, leftMean, normal, point, time, &frame) : RotateState(Load(mean, sample.other), sample.rotation);
                const auto leftMeanData = hasThermo ? Riemann::AcousticState{{leftMean, thermo(iCell, 0)}, thermo(iCell, 1)} : data->equation.PrepareFluxState(leftMean);
                const auto rightMeanData = sample.other == UnInitIndex || !hasThermo ? data->equation.PrepareFluxState(rightMean) : Riemann::AcousticState{{rightMean, thermo(sample.other, 0)}, thermo(sample.other, 1)};
                const auto prepared = data->equation.PrepareRiemann(data->flux, leftFlux, rightFlux, leftMeanData, rightMeanData, normal, gridVelocity);
                auto fluxSettings = data->settings.numerics.riemann.KernelSettings();
                if (useWave)
                {
                    real jump = waveInput(iCell, 0);
                    if (sample.other != UnInitIndex)
                        jump = std::max(jump, waveInput(sample.other, 0));
                    fluxSettings.dLambda = std::max(fluxSettings.dLambda, jump);
                }
                if (usePressure)
                {
                    real minimum = pressureInput(iCell, 0);
                    if (sample.other != UnInitIndex)
                        minimum = std::min(minimum, pressureInput(sample.other, 0));
                    fluxSettings.pressureStencilMin = std::min(fluxSettings.pressureStencilMin, minimum);
                }
                auto flux = data->flux.Evaluate(prepared.Input(leftFlux, rightFlux, normal, gridVelocity), fluxSettings).flux;
                FlowState lowFlux{};
                if (produceLow)
                    lowFlux = Riemann::Rusanov(leftMeanData, rightMeanData, normal, gridVelocity);
                if (polynomialViscosity && (sample.other != UnInitIndex || sample.boundary.kind == BoundaryKind::NoSlipWall))
                {
                    FlowState rightLinearization = right;
                    if (sample.other == UnInitIndex)
                    {
                        // Euler generates the boundary exterior from the
                        // filtered convective state even for unlimited viscosity.
                        if (sample.boundary.isothermal)
                        {
                            const auto wall = gas.Primitive(sample.boundary.FaceValue(gas, left.state, normal, point, time, &frame));
                            const auto exterior = gas.Primitive(right);
                            viscousRight = gas.Conservative({wall.density, exterior.velocity[0], exterior.velocity[1], exterior.velocity[2], wall.pressure});
                            rightLinearization = viscousRight;
                        }
                    }
                    FlowState faceMean{};
                    for (int j = 0; j < 5; ++j)
                        faceMean[j] = .5 * (left.state[j] + rightLinearization[j]);
                    DNDS_check_throw_info(ThermallyAdmissible(faceMean, data->thermalBounds), "inadmissible polynomial viscous linearization");
                    const bool wall = sample.other == UnInitIndex;
                    const real inverseDistance = wall && data->settings.numerics.noGRPOnWall ? 0 : 1 / sample.viscousDistance;
                    const auto diffusion = PolynomialViscousFlux(data->equation, left.state, rightLinearization, left.viscousState, viscousRight,
                                                                 left.gradient, rightGradient, normal, inverseDistance, wall && !sample.boundary.isothermal);
                    for (int component = 0; component < 5; ++component)
                    {
                        flux[component] -= diffusion[component];
                        lowFlux[component] -= diffusion[component];
                    }
                }
                else if (viscous && !polynomialViscosity)
                {
                    auto primitive = gas.Primitive(left.state);
                    auto gradient = left.gradient;
                    const auto ownPrimitive = gas.Primitive(Load(mean, iCell));
                    const auto rightPrimitive = gas.Primitive(sample.other == UnInitIndex ? sample.boundary.FaceValue(gas, Load(mean, iCell), normal, point, time, &frame) : RotateState(Load(mean, sample.other), sample.rotation));
                    // Evaluate into an owning vector: an Eigen expression
                    // would retain a dangling temporary cell barycenter.
                    const Geom::tPoint distance = sample.other == UnInitIndex ? Geom::tPoint(sample.ownPoint - metric.GetCellBary(iCell)) : Geom::tPoint(sample.otherPoint - metric.GetCellBary(iCell));
                    const real lengthSquared = distance.squaredNorm();
                    DNDS_check_throw_info(lengthSquared > 0, "zero viscous stencil displacement");
                    const std::array<real, 5> own{ownPrimitive.density, ownPrimitive.velocity[0], ownPrimitive.velocity[1], ownPrimitive.velocity[2], ownPrimitive.temperature};
                    const std::array<real, 5> neighbor{rightPrimitive.density, rightPrimitive.velocity[0], rightPrimitive.velocity[1], rightPrimitive.velocity[2], rightPrimitive.temperature};
                    for (int component = 0; component < 5; ++component)
                    {
                        real projected = 0;
                        for (int d = 0; d < 3; ++d)
                        {
                            if (sample.other != UnInitIndex)
                                gradient[component][d] = .5 * (gradient[component][d] + rightGradient[component][d]);
                            projected += gradient[component][d] * distance[d];
                        }
                        for (int d = 0; d < 3; ++d)
                            gradient[component][d] += (neighbor[component] - own[component] - projected) * distance[d] / lengthSquared;
                    }
                    if (sample.other != UnInitIndex)
                    {
                        const auto p = gas.Primitive(right);
                        for (int d = 0; d < 3; ++d)
                            primitive.velocity[d] = .5 * (primitive.velocity[d] + p.velocity[d]);
                    }
                    else if (sample.boundary.kind == BoundaryKind::SlipWall || sample.boundary.kind == BoundaryKind::NoSlipWall)
                    {
                        primitive = gas.Primitive(sample.boundary.FaceValue(gas, left.state, normal, point, time, &frame));
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
                if (forces && sample.other == UnInitIndex && (sample.boundary.kind == BoundaryKind::SlipWall || sample.boundary.kind == BoundaryKind::NoSlipWall))
                    for (int d = 0; d < 3; ++d)
                    {
                        auto &sum = forces->at(sample.zone)[d];
                        const real contribution = metric.GetFaceWeight(sample.point) * flux[1 + d];
                        real previous = sum.load(std::memory_order_relaxed);
                        while (!sum.compare_exchange_weak(previous, previous + contribution, std::memory_order_relaxed))
                        {
                        }
                    }
                for (int component = 0; component < 5; ++component)
                {
                    integrated[component] -= metric.GetFaceWeight(sample.point) * flux[component];
                    if (produceLow)
                        integratedLow[component] -= metric.GetFaceWeight(sample.point) * lowFlux[component];
                }
            }
            const auto rotationSource = frame.RotationSource(Load(mean, iCell));
            for (int component = 0; component < 5; ++component)
            {
                const real source = metric.GetCellVol(iCell) * rotationSource[component];
                result(iCell, component) = frame.moving ? integrated[component] + source : integrated[component];
                if (produceLow)
                    lowOutput(iCell, component) = frame.moving ? integratedLow[component] + source : integratedLow[component];
            }
        };
        if (region == CFV::ReconstructionRegion::All)
            for (index iCell = 0; iCell < data->metrics.NumCell(); ++iCell)
                compute(iCell);
        else
            for (auto iCell : region == CFV::ReconstructionRegion::Core ? data->partition.Core() : data->partition.HaloDependent())
                compute(iCell);
    }
}
