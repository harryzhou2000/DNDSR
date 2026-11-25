#include "Cauchy/Implicit/ACMPressureGauge.hpp"
#include "Cauchy/Implicit/NonlinearOperators_Internal.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace NonlinearOperatorDetail;
    NonlinearOperators::Data::Data(CFV::FiniteVolumePlan metrics, AdmissibilityPolicy bounds, R::ExecutionContext where, std::optional<NumericalReductions> backend, bool globalAlphaIn)
        : context(where), geometry(std::move(metrics)), policy(std::move(bounds)), reductions(backend.value_or(NumericalReductions(geometry.Communication()))), globalAlpha(globalAlphaIn), meanCheck(geometry.Space(), policy, context)
    {
        DNDS_check_throw_info(context.device == geometry.Device(), "nonlinear operator geometry placement differs");
        const auto view = policy.View();
        flowFirst = view.thermal.flowFirst;
        speciesFirst = view.thermal.speciesFirst;
        independent = view.thermal.independent;
        ransFirst = view.ransFirst;
        if (context.device == R::Device::Host())
            return;
#ifdef DNDS_USE_CUDA
        const int flowWidth = flowFirst < 0 ? 4 : 5;
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && independent == 0 &&
                                  view.thermal.components == flowWidth + view.rans.components,
                              "nonlinear CUDA operators require flow4 or flow5 with optional RANS");
        const index rows = geometry.NumCell();
        partial = R::Array<CUDA::NonlinearSummary>(R::RowLayout::Dense(CUDA::NonlinearScratchRows(rows), 1), context.device);
        summary = R::Array<CUDA::NonlinearSummary>(R::RowLayout::Dense(1, 1), context.device);
        partial.SetName("nonlinear/reduction/partials", DNDS_ALLOCATION_SITE());
        summary.SetName("nonlinear/reduction/summary", DNDS_ALLOCATION_SITE());
        if (globalAlpha)
        {
            limitedSum = reductions.BindInPlace<index>(summary.Storage(), offsetof(CUDA::NonlinearSummary, limited), 1, R::Reduction::Sum);
            alphaMin = reductions.BindInPlace<real>(summary.Storage(), offsetof(CUDA::NonlinearSummary, minimum), 1, R::Reduction::Min);
        }
#else
        Unavailable{}();
#endif
    }
    NonlinearOperators::NonlinearOperators(CFV::FiniteVolumePlan geometry, AdmissibilityPolicy policy, R::ExecutionContext context, std::optional<NumericalReductions> reductions, bool globalAlpha)
        : _data(std::make_shared<Data>(std::move(geometry), std::move(policy), context, std::move(reductions), globalAlpha)) {}
    void NonlinearOperators::Data::Check(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state) const
    {
        state.Validate();
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == context.device && (context.device == R::Device::Host() || stream.Context().stream == context.stream) &&
                                  state.Device() == context.device && state.Space().Same(geometry.Space()) && state.ComponentNumber() == policy.View().thermal.components,
                              "nonlinear operator stream, shape, space or placement differs");
        if (context.device != R::Device::Host())
            DNDS_check_throw_info(state.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "nonlinear CUDA operators require combined AoS");
    }
    void NonlinearOperators::Data::CheckPair(const EquationDofs &input, const EquationDofs &output) const
    {
        input.RequireCompatible(output);
        EquationDofOperations::RequireDistinct(input, output);
        if (context.device != R::Device::Host())
            DNDS_check_throw_info(output.CoupledField().Owned().Layout().Kind() == R::RowLayoutKind::Table,
                                  "nonlinear CUDA output requires combined AoS");
    }
    void NonlinearOperators::Data::CheckScalar(const R::Array<real> &array) const
    {
        DNDS_check_throw_info(array.GetDevice() == context.device && array.RowCount() == geometry.NumCell() &&
                                  !array.Layout().IsVariable() && array.Layout().Width() == 1,
                              "nonlinear scalar field shape or placement differs");
    }
    void NonlinearOperators::ValidateMeans(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state) const
    {
        _data->Check(engine, stream, state);
        RequireNumericalSuccess(_data->meanCheck.Evaluate(engine, stream, state), "ValidateMeanState");
    }

    void NonlinearOperators::SubmitMeanCheck(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state, IterationChecks &checks) const
    {
        _data->Check(engine, stream, state);
        _data->meanCheck.Submit(engine, stream, state, checks);
    }

    std::vector<real> NonlinearOperators::AssembleDefect(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                         const EquationDofs &mean, const EquationDofs &residual, const EquationDofs &constant, EquationDofs &output,
                                                         real inverseDt, real rhsScale, SpanView<const uint8_t> active, int normOrder, bool measureNorm) const
    {
        return SubmitDefect(engine, stream, mean, residual, constant, output, inverseDt, rhsScale, active, normOrder, measureNorm)();
    }
    std::function<std::vector<real>()> NonlinearOperators::SubmitDefect(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                                        const EquationDofs &mean, const EquationDofs &residual, const EquationDofs &constant, EquationDofs &output,
                                                                        real inverseDt, real rhsScale, SpanView<const uint8_t> active, int normOrder, bool measureNorm, std::shared_ptr<NonlinearNormSum> global, OptionalRef<IterationChecks> checks) const
    {
        _data->Check(engine, stream, mean);
        _data->Check(engine, stream, residual);
        _data->Check(engine, stream, constant);
        for (const auto *input : {&mean, &residual, &constant})
            _data->CheckPair(*input, output);
        DNDS_check_throw_info(std::isfinite(inverseDt) && inverseDt >= 0 && std::isfinite(rhsScale) && rhsScale >= 0 && (normOrder == 1 || normOrder == 2),
                              "invalid nonlinear defect coefficients");
        const auto selection = Selection(active, mean.ComponentNumber());
        Inputs inputs;
        Outputs outputs;
        const auto u = AddInput(mean, inputs), r = AddInput(residual, inputs), b = AddInput(constant, inputs), y = AddOutput(output, outputs);
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto accumulation = mean.Schema().AccumulationKind();
        const auto host = [geometry, u, r, b, y, selection, accumulation, inverseDt, rhsScale, normOrder, measureNorm](auto input, auto output, Result &result)
        {
            const auto mean = u.Layout(input), residual = r.Layout(input), constant = b.Layout(input);
            const auto target = y.Layout(output);
            result.status = CPU::AssembleDefect(mean.Descriptor(), residual.Descriptor(), constant.Descriptor(), target.Descriptor(),
                                                geometry->HostView(), accumulation, inverseDt, rhsScale, selection.HostView(), normOrder, measureNorm,
                                                {result.norms.data(), index(result.norms.size())}, {result.workerSums.data(), index(result.workerSums.size())});
        };
#ifdef DNDS_USE_CUDA
        const auto device = [geometry, u, r, b, y, selection, accumulation, inverseDt, rhsScale, normOrder, measureNorm](auto context, auto input, auto output, auto scratch, auto summary)
        { CUDA::AssembleDefect(context, geometry->View().cells, Table(input[u.firstSlot]), Table(input[r.firstSlot]), Table(input[b.firstSlot]),
                               Table(output[y.firstSlot]), DeviceSelection(selection), inverseDt, rhsScale, normOrder, measureNorm, scratch, summary,
                               accumulation); };
#else
        const Unavailable device;
#endif
        const bool continuity = measureNorm && accumulation == PhysicalAccumulationKind::ConstantDensityVelocityPressure;
        auto observe = _data->SubmitReduction(engine, stream, mean.ComponentNumber() + continuity, std::move(inputs), std::move(outputs), host, device, std::move(global), checks);
        return [observe = std::move(observe)]
        { return observe().norms; };
    }
    std::function<std::vector<real>()> NonlinearOperators::SubmitPressureSum(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state, std::shared_ptr<NonlinearNormSum> global) const
    {
        _data->Check(engine, stream, state);
        DNDS_check_throw_info(state.Schema().AccumulationKind() == PhysicalAccumulationKind::ConstantDensityVelocityPressure,
                              "pressure gauge requires an ACM state");
        Inputs reads;
        const auto pressure = AddInput(state, reads);
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto host = [geometry, pressure](auto input, auto, Result &result)
        {
            const auto field = pressure.Components(input);
            const auto metric = geometry->HostView();
            result.status = OMP::SumRows(field.View().RowCount(), SpanView<real>{result.norms.data(), 2},
                                         {result.workerSums.data(), index(result.workerSums.size())}, [&](index i, SpanView<real> sums) -> NumericalStatus
                                         {
                    const real volume = metric.GetCellVol(i);
                    sums[0] += volume * field.View()(i, 3);
                    sums[1] += volume;
                    return {}; });
        };
#ifdef DNDS_USE_CUDA
        const auto device = [geometry, pressure](auto context, auto input, auto, auto scratch, auto summary)
        { CUDA::PressureGaugeSum(context, geometry->View().cells, Table(input[pressure.firstSlot]), scratch, summary); };
#else
        const Unavailable device;
#endif
        auto observe = _data->SubmitReduction(engine, stream, 1, std::move(reads), {}, host, device, std::move(global));
        return [observe = std::move(observe)]
        { return observe().norms; };
    }
    real NonlinearOperators::PressureMean(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &state) const
    {
        if (!_data->pressureSlot)
            _data->pressureSlot = std::make_shared<NonlinearNormSum>(2, _data->reductions, MPIInfo(_data->geometry.Communicator()).size, stream, stream);
        auto sum = SubmitPressureSum(engine, stream, state, _data->pressureSlot)();
        DNDS_check_throw_info(sum[1] > 0 && std::isfinite(sum[1]) && std::isfinite(sum[0] / sum[1]),
                              "invalid global pressure gauge integral");
        return sum[0] / sum[1];
    }
    void NonlinearOperators::ProjectPressureMean(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                 EquationDofs &candidate, real pressureMean, std::shared_ptr<NonlinearNormSum> global) const
    {
        if (global)
        {
            (void)SubmitPressureSum(engine, stream, candidate, global);
            const auto &sum = global->Consume(engine, stream);
            const auto flow = candidate.LocateBlock("flow");
            R::SubmitArrays(engine, stream, [target = pressureMean, column = flow.firstComponent + 3](auto where, auto integral, auto means)
                            {
                                if (where.device == R::Device::Host())
                                {
                                    const real offset = integral(0, 1) > 0 ? target - integral(0, 0) / integral(0, 1) : std::numeric_limits<real>::quiet_NaN();
                                    OMP::ForRows(0, means.RowCount(), [&](index i)
                                                 { means(i, column) += offset; });
                                }
#ifdef DNDS_USE_CUDA
                                else
                                    CUDA::ShiftPressure(where, Table(means), target, Table(integral));
#endif
                            },
                            R::ReadAs<R::UniformTableRows>(sum), R::ReadWriteAs<R::UniformTableRows>(candidate.PhysicalField(flow.physicalField).Owned()));
            return;
        }
        const real offset = pressureMean - PressureMean(engine, stream, candidate);
        SubmitPressureOffset(engine, stream, candidate, offset);
    }
    std::vector<real> NonlinearOperators::MeasureMeanState(R::EagerEngine &engine, const R::ExecutionStream &stream, const EquationDofs &mean, std::shared_ptr<NonlinearNormSum> global) const
    {
        _data->Check(engine, stream, mean);
        Inputs inputs;
        const auto u = AddInput(mean, inputs);
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto host = [geometry, u](auto input, auto, Result &result)
        {
            const auto layout = u.Components<const real>(input);
            CPU::AccumulateMeanScale(layout.View(), geometry->HostView(), index(u.columns.size()),
                                     {result.norms.data(), index(result.norms.size())}, {result.workerSums.data(), index(result.workerSums.size())});
        };
#ifdef DNDS_USE_CUDA
        const auto device = [geometry, u](auto context, auto input, auto, auto scratch, auto summary)
        { CUDA::MeasureMeanState(context, geometry->View().cells, Table(input[u.firstSlot]), scratch, summary); };
#else
        const Unavailable device;
#endif
        return _data->SubmitReduction(engine, stream, mean.ComponentNumber(), std::move(inputs), {}, host, device, std::move(global))().norms;
    }
    std::vector<real> NonlinearOperators::MeasureRateNorm(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                          const EquationDofs &rate, int normOrder, OptionalRef<const EquationDofs> reference,
                                                          OptionalRef<const EquationDofs> rawACMRate) const
    {
        return SubmitRateNorm(engine, stream, rate, normOrder, reference, rawACMRate, {})();
    }
    std::function<std::vector<real>()> NonlinearOperators::SubmitRateNorm(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                                                          const EquationDofs &rate, int normOrder, OptionalRef<const EquationDofs> reference,
                                                                          OptionalRef<const EquationDofs> rawACMRate, std::shared_ptr<NonlinearNormSum> global) const
    {
        _data->Check(engine, stream, rate);
        if (reference)
        {
            _data->Check(engine, stream, *reference);
            rate.RequireCompatible(*reference);
        }
        if (rawACMRate)
        {
            _data->Check(engine, stream, *rawACMRate);
            rate.RequireCompatible(*rawACMRate);
            DNDS_check_throw_info(rate.Schema().AccumulationKind() == PhysicalAccumulationKind::ConstantDensityVelocityPressure,
                                  "independent continuity norm requires ACM");
        }
        DNDS_check_throw_info(normOrder == 1 || normOrder == 2, "invalid nonlinear rate norm order");
        Inputs inputs;
        const auto u = AddInput(rate, inputs);
        const auto ref = reference ? std::optional(AddInput(*reference, inputs)) : std::nullopt;
        const auto raw = rawACMRate ? std::optional(AddInput(*rawACMRate, inputs)) : std::nullopt;
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto host = [geometry, u, ref, raw, normOrder](auto input, auto, Result &result)
        {
            const auto values = u.Components<const real>(input);
            const auto reference = ref ? std::optional(ref->Components<const real>(input)) : std::nullopt;
            const auto spatial = raw ? std::optional(raw->Components<const real>(input)) : std::nullopt;
            result.status = CPU::MeasureRateNorm(values.View(), reference ? reference->View() : Algebra::DofComponentView<const real>{},
                                                 geometry->HostView(), normOrder, {result.norms.data(), index(result.norms.size())},
                                                 {result.workerSums.data(), index(result.workerSums.size())},
                                                 spatial ? spatial->View() : Algebra::DofComponentView<const real>{});
        };
#ifdef DNDS_USE_CUDA
        const auto device = [geometry, u, ref, raw, normOrder](auto context, auto input, auto, auto scratch, auto summary)
        { CUDA::MeasureRateNorm(context, geometry->View().cells, Table(input[u.firstSlot]),
                                ref ? Table(input[ref->firstSlot]) : CUDA::LinearReadRows{}, normOrder, scratch, summary,
                                raw ? Table(input[raw->firstSlot]) : CUDA::LinearReadRows{}); };
#else
        const Unavailable device;
#endif
        auto observe = _data->SubmitReduction(engine, stream, rate.ComponentNumber() + bool(raw), std::move(inputs), {}, host, device, std::move(global));
        return [observe = std::move(observe)]
        { return observe().norms; };
    }
    void NonlinearOperators::SubmitScaleRate(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                             const EquationDofs &input, EquationDofs &output, bool multiply) const
    {
        _data->Check(engine, stream, input);
        _data->CheckPair(input, output);
        Inputs inputs;
        Outputs outputs;
        const auto x = AddInput(input, inputs), y = AddOutput(output, outputs);
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        R::SubmitArrayBatch(engine, stream, [geometry, x, y, multiply](R::ExecutionContext context, auto input, auto output)
                            {
            if (context.device == R::Device::Host())
            {
                const auto from = x.Components<const real>(input);
                const auto to = y.Components<real>(output);
                if (multiply) CPU::ScaleRateByVolume(from.View(), to.View(), geometry->HostView(), index(x.columns.size()));
                else CPU::NormalizeRateByVolume(from.View(), to.View(), geometry->HostView(), index(x.columns.size()));
            }
            else
            {
#ifdef DNDS_USE_CUDA
                CUDA::ScaleRateByVolume(context, geometry->View().cells, Table(input[x.firstSlot]), Table(output[y.firstSlot]), multiply);
#else
                Unavailable{}();
#endif
            } }, std::move(inputs), std::move(outputs));
    }
    void NonlinearOperators::SubmitEquation(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                            const EquationDofs &mean, const EquationDofs &residual, EquationDofs &output, OptionalRef<EquationDofs> rate, real inverseDt, real rhsScale) const
    {
        _data->Check(engine, stream, mean);
        _data->Check(engine, stream, residual);
        _data->CheckPair(mean, output);
        _data->CheckPair(residual, output);
        if (rate)
        {
            _data->CheckPair(mean, *rate);
            _data->CheckPair(residual, *rate);
            _data->CheckPair(output, *rate);
        }
        DNDS_check_throw_info(std::isfinite(inverseDt) && inverseDt >= 0 && std::isfinite(rhsScale) && rhsScale >= 0, "invalid nonlinear equation coefficients");
        Inputs inputs;
        Outputs outputs;
        const auto u = AddInput(mean, inputs), r = AddInput(residual, inputs), y = AddOutput(output, outputs);
        const auto rateBinding = rate ? std::optional(AddOutput(*rate, outputs)) : std::nullopt;
        const auto geometry = std::make_shared<CFV::FiniteVolumeRead>(_data->geometry.AcquireRead(stream.Context()));
        const auto accumulation = mean.Schema().AccumulationKind();
        R::SubmitArrayBatch(engine, stream, [geometry, u, r, y, rateBinding, accumulation, inverseDt, rhsScale](R::ExecutionContext context, auto input, auto output)
                            {
            if (context.device == R::Device::Host())
            {
                const auto mean = u.Components<const real>(input), residual = r.Components<const real>(input);
                const auto target = y.Components<real>(output);
                DispatchStageShape(accumulation, index(u.columns.size()), [&](auto kind, auto width)
                {
                    if (rateBinding)
                    {
                        const auto rate = rateBinding->Components<real>(output);
                        CPU::AssembleEquationRows<StageAssembly::EquationAndRate, decltype(kind)::value, decltype(width)::value>(
                            mean.View(), residual.View(), target.View(), rate.View(), geometry->HostView(), index(u.columns.size()), inverseDt, rhsScale);
                    }
                    else
                        CPU::AssembleEquationRows<StageAssembly::Equation, decltype(kind)::value, decltype(width)::value>(
                            mean.View(), residual.View(), target.View(), nullptr, geometry->HostView(), index(u.columns.size()), inverseDt, rhsScale);
                });
            }
            else
            {
#ifdef DNDS_USE_CUDA
                CUDA::AssembleEquation(context, geometry->View().cells, Table(input[u.firstSlot]), Table(input[r.firstSlot]),
                    Table(output[y.firstSlot]), rateBinding ? Table(output[rateBinding->firstSlot]) : CUDA::LinearWriteRows{}, inverseDt, rhsScale,
                    accumulation);
#else
                Unavailable{}();
#endif
            } }, std::move(inputs), std::move(outputs));
    }
}
