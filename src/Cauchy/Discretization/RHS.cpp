#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"

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
        R::Array<real>::ReadLease basis;
        explicit ReadData(std::shared_ptr<const FiniteVolume::Data> owner)
            : data(std::move(owner)), geometry(data->metrics.AcquireRead()), basis(data->basis.Read())
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
        const auto basis = _read->basis.View<R::TableRows>();
        const int modes = data->reconstruction.NRecDOF(), dim = data->metrics.Dimension();
        const auto &gas = data->equation.Gas();
        const bool viscous = data->equation.Viscous() && input.includeViscosity;
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
        if (viscous)
            checkField(gradientInput, 5 * dim);
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
                    auto state = Load(mean, cell);
                    for (int component = 0; component < 5; ++component)
                        for (int mode = 0; mode < modes; ++mode)
                        {
                            const real value = coefficient(cell, component * modes + mode);
                            state[component] += value * basis(entry, parent * 4 * modes + mode);
                        }
                    // Explicit local first-order fallback keeps the mean
                    // unchanged when a high-order face value is inadmissible.
                    if (!ThermallyAdmissible(state, data->thermalBounds))
                    {
                        if (diagnostics)
                            diagnostics->fetch_add(1, std::memory_order_relaxed);
                        state = Load(mean, cell);
                    }
                    FlowGradient primitiveGradient{};
                    if (viscous)
                        for (int component = 0; component < 5; ++component)
                            for (int d = 0; d < dim; ++d)
                                primitiveGradient[component][d] = gradientInput(cell, component * dim + d);
                    return std::make_pair(state, primitiveGradient);
                };
                const auto left = reconstruct(iCell, 0);
                FlowState right;
                FlowGradient rightGradient{};
                if (sample.other != UnInitIndex)
                {
                    auto other = reconstruct(sample.other, 1);
                    right = RotateState(other.first, sample.rotation);
                    rightGradient = RotateGradient(other.second, sample.rotation);
                }
                else
                    right = sample.boundary.Exterior(gas, left.first, {sample.normal[0], sample.normal[1], sample.normal[2]}, {sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]}, time);
                const Vector3 normal{sample.normal[0], sample.normal[1], sample.normal[2]};
                const auto leftFlux = data->equation.PrepareFluxState(left.first), rightFlux = data->equation.PrepareFluxState(right);
                const Vector3 point{sample.ownPoint[0], sample.ownPoint[1], sample.ownPoint[2]};
                const auto leftMean = Load(mean, iCell), rightMean = sample.other == UnInitIndex ? sample.boundary.Exterior(gas, leftMean, normal, point, time) : RotateState(Load(mean, sample.other), sample.rotation);
                const auto leftMeanData = hasThermo ? Riemann::AcousticState{{leftMean, thermo(iCell, 0)}, thermo(iCell, 1)} : data->equation.PrepareFluxState(leftMean);
                const auto rightMeanData = sample.other == UnInitIndex || !hasThermo ? data->equation.PrepareFluxState(rightMean) : Riemann::AcousticState{{rightMean, thermo(sample.other, 0)}, thermo(sample.other, 1)};
                const auto prepared = data->equation.PrepareRiemann(data->flux, leftFlux, rightFlux, leftMeanData, rightMeanData, normal);
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
                auto flux = data->flux.Evaluate(prepared.Input(leftFlux, rightFlux, normal), fluxSettings).flux;
                FlowState lowFlux{};
                if (produceLow)
                    lowFlux = Riemann::Rusanov(leftMeanData, rightMeanData, normal);
                if (viscous)
                {
                    auto primitive = gas.Primitive(left.first);
                    auto gradient = left.second;
                    const auto ownPrimitive = gas.Primitive(Load(mean, iCell));
                    const auto rightPrimitive = gas.Primitive(sample.other == UnInitIndex ? sample.boundary.FaceValue(gas, Load(mean, iCell), normal, point, time) : RotateState(Load(mean, sample.other), sample.rotation));
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
                        primitive = gas.Primitive(sample.boundary.FaceValue(gas, left.first, normal, point, time));
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
    }
}
