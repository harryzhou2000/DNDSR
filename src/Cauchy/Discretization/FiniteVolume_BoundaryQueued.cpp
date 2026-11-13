#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Discretization/CUDA/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::SubmitBoundaryValues(
        Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
        const EquationDofs &state, OptionalRef<const EquationDofs> coefficients,
        Runtime::Array<real> &output, Runtime::Array<NumericalStatus> &status,
        BoundarySubmission options, OptionalRef<const Runtime::Array<real>> incoming) const
    {
#ifdef DNDS_USE_CUDA
        namespace R = Runtime;
        DNDS_check_throw_info(Device().space == R::MemorySpace::CUDA && engine.Owns(stream) &&
                                  stream.Context().device == Device() && state.Device() == Device(),
                              "queued boundary backend/stream differs from its spatial plan");
        state.RequireSchema(_data->equation.Schema());
        DNDS_check_throw_info(state.PhysicalFieldNumber() == 1 && state.Space().Same(Metrics().Space()) && std::isfinite(options.time),
                              "queued boundary needs combined state in its prepared row space");
        const bool mapped = options.target != BoundaryTarget::Quadrature;
        const bool primitive = options.target == BoundaryTarget::PrimitiveGradient;
        const index rows = mapped ? _data->gradientBoundaryRows.RowCount() : _data->boundaryTraceGeometry.RowCount();
        DNDS_check_throw_info(output.RowCount() == rows && output.Layout().Kind() == R::RowLayoutKind::Table &&
                                  output.Layout().Width() == (primitive ? 5 : state.ComponentNumber()) && status.RowCount() == rows &&
                                  status.Layout().Kind() == R::RowLayoutKind::Table && status.Layout().Width() == 1 &&
                                  output.GetDevice() == Device() && status.GetDevice() == Device(),
                              "queued boundary output/status shape or placement differs");
        DNDS_check_throw_info(!coefficients || (!mapped && _data->reconstruction),
                              "polynomial boundary values require the quadrature reconstruction target");
        const auto &mean = state.CoupledField().Owned();
        const auto &rec = coefficients ? coefficients->CoupledField().Owned() : _data->deviceEmpty;
        const auto &packet = incoming ? *incoming : _data->deviceEmpty;
        if (coefficients)
            DNDS_check_throw_info(coefficients->PhysicalFieldNumber() == 1 && coefficients->Space().Same(state.Space()) &&
                                      rec.RowCount() == mean.RowCount() && rec.GetDevice() == Device() &&
                                      rec.Layout().Width() == state.ComponentNumber() * _data->reconstruction->NRecDOF(),
                                  "queued boundary coefficients differ from its combined polynomial layout");
        for (const auto *input : {&mean, &rec, &packet})
            DNDS_check_throw_info(!output.Storage().Overlaps(input->Storage()) && !status.Storage().Overlaps(input->Storage()),
                                  "queued boundary output/status aliases an input");
        DNDS_check_throw_info(!output.Storage().Overlaps(status.Storage()), "queued boundary output aliases status");

        // Immutable geometry and small physics descriptors are pinned once at
        // submission. Mutable operands below use the engine's scheduled access
        // reservations, so a prior queued producer requires no host observation.
        struct Inputs
        {
            std::shared_ptr<const Data> owner;
            CFV::FiniteVolumeRead metric;
            R::Array<BoundaryTraceGeometry>::ReadLease points;
            R::Array<index>::ReadLease rows;
            R::ReadAccess policy, conditions;
            std::optional<CFV::ReconstructionResidentRead> basis;
            std::optional<WallDistanceRead> walls;
            Inputs(std::shared_ptr<const Data> data, bool polynomial, bool primitive)
                : owner(std::move(data)), metric(owner->metrics.AcquireRead({owner->metrics.Device(), nullptr})),
                  points(owner->boundaryTraceGeometry.Read({owner->metrics.Device(), nullptr})),
                  rows(owner->gradientBoundaryRows.Read({owner->metrics.Device(), nullptr})),
                  policy(owner->deviceBoundaryPolicy.Read({owner->metrics.Device(), nullptr})),
                  conditions(owner->deviceBoundaryConditions.Read({owner->metrics.Device(), nullptr}))
            {
                if (polynomial)
                    basis.emplace(owner->reconstruction->AcquireResidentRead());
                if (!primitive && owner->equation.HasRANS() && owner->equation.RANS().ComponentNumber() == 2)
                {
                    DNDS_check_throw_info(owner->wallDistance, "two-equation boundary requires wall distance");
                    walls.emplace(owner->wallDistance->AcquireRead({owner->metrics.Device(), nullptr}));
                }
            }
        };
        const auto pin = std::make_shared<Inputs>(_data, bool(coefficients), primitive);
        const auto frame = _data->motion.At(options.time).KernelParameters();
        const int components = int(state.ComponentNumber());
        const bool hasIncoming = bool(incoming);
        R::SubmitArrays(engine, stream, [pin, options, frame, components, mapped, primitive, hasIncoming](R::ExecutionContext context, auto mean, auto rec, auto incoming, auto target, auto status)
                        {
                CUDA::BoundaryTraceRequest request;
                request.parameters = {pin->policy.View<CUDA::BoundaryKernelParameters>().data, nullptr, 1, 1, 1, nullptr, R::RowLayoutKind::Table, context.device};
                request.conditions = {pin->conditions.View<PerfectGasBoundaryParameters>().data, nullptr,
                    index(pin->owner->deviceBoundaryConditions.size()), 1, 1, nullptr, R::RowLayoutKind::Table, context.device};
                request.points = pin->points.View<R::FixedRows<1>>();
                request.sourceRows = pin->rows.View<R::FixedRows<1>>();
                request.means = mean; request.coefficients = rec; request.incoming = incoming;
                request.output = target; request.status = status;
                request.time = options.time; request.frame = frame;
                request.components = components; request.useSourceRows = mapped;
                request.exterior = options.exterior; request.repairRANS = options.repairRANS;
                request.hasIncoming = hasIncoming; request.polynomial = bool(pin->basis);
                if (pin->basis)
                {
                    request.basis = pin->basis->PointBasisView();
                    request.modes = request.basis.polynomial.size - 1;
                }
                if (pin->walls)
                {
                    const auto wall = pin->walls->View().cellMean;
                    request.wallMean = {wall.data, wall.offsets, wall.rows, wall.width, wall.stride, wall.sizes, wall.kind, wall.device};
                }
                if (primitive) CUDA::FillPrimitiveBoundary(context, request);
                else CUDA::FillBoundaryTraces(context, request); }, R::ReadAs<R::TableRows>(mean), R::ReadAs<R::TableRows>(rec), R::ReadAs<R::TableRows>(packet), R::WriteAs<R::TableRows>(output), R::WriteAs<R::FixedRows<1>>(status));
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
    void FiniteVolume::Workspace::BuildBoundaryStatus()
    {
        if (owner->metrics.Device() == Runtime::Device::Host())
            return;
#ifdef DNDS_USE_CUDA
        const auto device = owner->metrics.Device();
        const index quadrature = owner->boundaryTraceGeometry.RowCount();
        const index primitive = primitiveBoundary ? primitiveBoundary->RowCount() : 0;
        boundaryStatus.emplace(BoundaryStatusStorage{
            R::Array<NumericalStatus>(R::RowLayout::Dense(quadrature, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(primitive, 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(CUDA::StateReductionScratchRows(std::max(quadrature, primitive) + 1), 1), device),
            R::Array<NumericalStatus>(R::RowLayout::Dense(1, 1), device)});
        boundaryStatus->quadrature.SetName("Cauchy/boundaryStatus", DNDS_ALLOCATION_SITE());
        boundaryStatus->primitive.SetName("Cauchy/primitiveBoundaryStatus", DNDS_ALLOCATION_SITE());
        boundaryStatus->partial.SetName("Cauchy/boundaryStatusPartials", DNDS_ALLOCATION_SITE());
        boundaryStatus->result.SetName("Cauchy/boundaryStatusResult", DNDS_ALLOCATION_SITE());
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
    void FiniteVolume::Workspace::ResetBoundaryStatus(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(boundaryStatus && !boundaryStatus->pending, "boundary status workspace is unavailable or still pending");
        R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto result)
                        { CUDA::ClearNumericalStatus(context, result); }, R::WriteAs<R::FixedRows<1>>(boundaryStatus->result));
        boundaryStatus->pending = true;
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
    void FiniteVolume::Workspace::CheckBoundaryStatus()
    {
        DNDS_check_throw_info(boundaryStatus && boundaryStatus->pending && boundaryStatus->result.Storage().IsIdle(),
                              "boundary status requires a completed numerical segment");
        const auto result = boundaryStatus->result.To(R::Device::Host());
        boundaryStatus->pending = false;
        RequireNumericalSuccess(result.Read().View()(0, 0), "queued boundary evaluation");
    }
    void FiniteVolume::Workspace::SubmitDeviceBoundary(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, bool polynomial, real time,
        OptionalRef<const R::Array<real>> incoming)
    {
#ifdef DNDS_USE_CUDA
        DNDS_check_throw_info(boundaryStatus && boundaryStatus->pending && fields.size() == 1,
                              "queued boundary needs a combined device workspace and an initialized status segment");
        const FiniteVolume plan(owner);
        const bool mirrored = owner->settings.numerics.reconstruction == "variational" &&
                              owner->settings.numerics.vrBoundaryValue == "physicalMirrored";
        auto &field = fields.front();
        const bool gradientBoundary = bool(field.gradientBoundary);
        const auto append = [&](R::Array<NumericalStatus> &rows)
        {
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto rows, auto partial, auto result)
                            { CUDA::ReduceNumericalStatus(context, rows, result.ConstView(), partial, result); }, R::ReadAs<R::FixedRows<1>>(rows), R::WriteAs<R::FixedRows<1>>(boundaryStatus->partial), R::ReadWriteAs<R::FixedRows<1>>(boundaryStatus->result));
        };
        const auto fill = [&](bool exterior)
        {
            plan.SubmitBoundaryValues(engine, stream, state,
                                      polynomial ? OptionalRef<const EquationDofs>(Coefficients()) : OptionalRef<const EquationDofs>{},
                                      field.boundaryValues, boundaryStatus->quadrature,
                                      {BoundaryTarget::Quadrature, exterior, true, time}, incoming);
            append(boundaryStatus->quadrature);
        };
        fill(mirrored && !gradientBoundary);
        if (gradientBoundary)
            R::SubmitArrays(engine, stream, [](R::ExecutionContext context, auto source, auto output, auto rows)
                            { CUDA::CopyBoundaryRows(context, source, output, rows); }, R::ReadAs<R::TableRows>(field.boundaryValues), R::WriteAs<R::TableRows>(*field.gradientBoundary), R::ReadAs<R::FixedRows<1>>(owner->gradientBoundaryRows));
        if (mirrored && gradientBoundary)
            fill(true);
        if (primitiveBoundary)
        {
            plan.SubmitBoundaryValues(engine, stream, state, {}, *primitiveBoundary, boundaryStatus->primitive,
                                      {BoundaryTarget::PrimitiveGradient, false, false, time}, incoming);
            append(boundaryStatus->primitive);
        }
#else
        DNDS_check_throw_info(false, "CUDA boundary execution is not built");
#endif
    }
}
