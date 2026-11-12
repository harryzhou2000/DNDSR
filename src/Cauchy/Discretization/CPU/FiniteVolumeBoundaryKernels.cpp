#include "Cauchy/Discretization/CPU/FiniteVolumeBoundaryKernels.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "CFV/CPU/ReconstructionBasisKernels.hpp"
#include "Physics/EvaluationStatus.hpp"

namespace DNDS::Cauchy::CPU
{
    namespace
    {
        void Gather(SpanView<const EquationBlockReadView> blocks, index iCell,
                    std::vector<real> &point)
        {
            index first = 0;
            for (const auto &block : blocks)
                for (const auto value : block[iCell])
                    point[size_t(first++)] = value;
            DNDS_assert(first == index(point.size()));
        }
    }

    NumericalStatus FillBoundaryTraces(const BoundaryTraceRequest &request,
                                       SpanView<const BoundaryCondition> conditions,
                                       const BoundaryPointOperator &boundary,
                                       PointEvaluator &evaluator) noexcept
    {
        DNDS_assert((request.polynomial ? request.means.size() == request.coefficients.size()
                                        : request.coefficients.empty()) &&
                    request.means.size() == request.output.size() &&
                    request.means.size() == request.logicalFirst.size());
        CFV::Math::SmallMatrix scratch;
        std::vector<real> point(size_t(request.componentNumber)),
            cellMean(size_t(request.componentNumber));
        const index count = request.usePointRows ? request.pointRows.RowCount()
                                                 : request.points.RowCount();
        for (index iPoint = 0; iPoint < count; ++iPoint)
        {
            const auto sample = request.points(
                request.usePointRows ? request.pointRows(iPoint, 0) : iPoint, 0);
            const auto &condition = conditions[sample.condition];
            Gather(request.means, sample.cell, cellMean);
            point = cellMean;
            if (request.polynomial)
            {
                const auto basis = CFV::CPU::EvaluateFaceBasis(
                    request.basis, sample.face, sample.side, sample.point,
                    0, scratch);
                for (index iBlock = 0; iBlock < request.means.size(); ++iBlock)
                {
                    const auto row = request.coefficients[iBlock][sample.cell];
                    for (index j = 0;
                         j < request.means[iBlock].ComponentNumber(); ++j)
                        for (int mode = 0; mode < request.modes; ++mode)
                            point[size_t(request.logicalFirst[iBlock] + j)] +=
                                row[j * request.modes + mode] * basis(0, mode);
                }
            }
            // PP POLICY -- VR's warm iterate is deliberately unlimited. Like
            // Euler's FBoundary -> CompressRecPart, repair only the temporary
            // boundary input before evaluating a physical state. The flux
            // polynomial's beta limiter runs separately, after reconstruction.
            // Species bounds are not a polynomial condition: the thermal test
            // uses temporary clipped composition, as in pointwise RHS physics.
            if (request.polynomial && request.allowPointFallback &&
                !request.thermal.ClippedReferenceAdmissible(point))
                point = cellMean;
            if (request.rans.components && request.repairRANS)
                RANSReconstructionView{request.rans.kind, request.rans.components}.RepairPoint(
                    request.ransMeans[sample.cell],
                    point.data() + request.ransFirst);
            {
                const auto evaluate = [&]
                { return boundary.TryEvaluate(
                      point, condition, sample.interfaceRow, sample.normal,
                      sample.location, evaluator, request.frame, request.time,
                      request.stateKind,
                      OptionalRef<const std::vector<real>>(cellMean),
                      request.rans.components == 2
                          ? request.wallDistance.cellMean(sample.cell, 0)
                          : 0,
                      request.incoming); };
                auto status = evaluate();
                if (!status && request.polynomial && request.allowPointFallback &&
                    Physics::AllowsMeanFallback(status))
                {
                    // A reported EOS/T failure may retry from the admissible
                    // mean; malformed BC requests/backend failures must escape.
                    point = cellMean;
                    status = evaluate();
                }
                if (!status)
                    return status.At(sample.cell, iPoint);
            }
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
            // Preserve Euler's fixUL=true reconstruction target. RHS exterior
            // reflection remains a separate boundary-producer operation.
            if (request.rans.components &&
                request.rans.kind == Physics::RANS::ModelKind::SA &&
                condition.kind == BoundaryKind::NoSlipWall)
                point[size_t(request.ransFirst)] = 0;
#endif
            for (index iBlock = 0; iBlock < request.output.size(); ++iBlock)
            {
                const auto target = request.output[iBlock];
                for (index j = 0; j < target.ComponentNumber(); ++j)
                    target(iPoint, j) =
                        point[size_t(request.logicalFirst[iBlock] + j)];
            }
        }
        return {};
    }

    void CopyBoundaryRows(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            source,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            destination,
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            sourceRows)
    {
        DNDS_assert(sourceRows.RowCount() == destination.RowCount());
        for (index iPoint = 0; iPoint < destination.RowCount(); ++iPoint)
            for (index j = 0; j < destination.Width(); ++j)
                destination(iPoint, j) = source(sourceRows(iPoint, 0), j);
    }

    NumericalStatus FillPrimitiveBoundary(const PrimitiveBoundaryRequest &request,
                                          SpanView<const BoundaryCondition> conditions) noexcept
    {
        std::vector<real> cellMean(size_t(request.componentNumber));
        for (index iPoint = 0; iPoint < request.output.RowCount(); ++iPoint)
        {
            const auto sample =
                request.points(request.sourceRows(iPoint, 0), 0);
            const auto &condition = conditions[sample.condition];
            Gather(request.means, sample.cell, cellMean);
            FlowState inside{};
            for (int j = 0; j < 5; ++j)
                inside[j] = cellMean[size_t(request.flowFirst + j)];
            FlowState state;
            if (condition.kind == BoundaryKind::Interface)
            {
                for (int j = 0; j < 5; ++j)
                    state[j] = request.incoming(sample.interfaceRow, j);
            }
            else
            {
                const auto exterior = condition.TryExterior(
                    request.gas, inside, sample.normal, sample.location,
                    request.time, OptionalRef<const RigidFrame>(request.frame));
                if (!exterior)
                    return exterior.status.At(sample.cell, iPoint);
                state = exterior.value;
                for (int j = 0; j < 5; ++j)
                    state[j] = .5 * (inside[j] + state[j]);
            }
            const auto value = request.gas.Primitive(state);
            const real converted[5]{value.density, value.velocity[0],
                                    value.velocity[1], value.velocity[2],
                                    value.temperature};
            for (int j = 0; j < 5; ++j)
                request.output(iPoint, j) = converted[j];
        }
        return {};
    }

    NumericalStatus ValidateInterfacePacketRows(Algebra::DofComponentView<const real> rows, BoundaryPointRows points,
                                                ThermalView thermal, RANSStateView rans, EquationBlockReadView ransValues) noexcept
    {
        for (index iPoint = 0; iPoint < rows.RowCount(); ++iPoint)
            if (points(iPoint, 0).interface)
            {
                if (!(thermal.Admissible(
                        ThermalAdmissibilityDetail::PointRow<decltype(rows)>{rows, iPoint})))
                    return NumericalStatus{NumericalError::InvalidState, 1, iPoint};
                if (rans.components)
                    if (!(rans.Admissible(ransValues[iPoint])))
                        return NumericalStatus{NumericalError::InvalidState, 2, iPoint};
            }
        return {};
    }
    Vector3 AccumulateBoundaryForce(FiniteVolumeStencilView stencil,
                                    Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> values,
                                    SpanView<const int> zones)
    {
        Vector3 force{};
        for (index iPoint = 0; iPoint < stencil.boundaryEntries.RowCount(); ++iPoint)
        {
            const auto &sample = stencil.samples(stencil.boundaryEntries(iPoint, 0), 0);
            if (std::find(zones.begin(), zones.end(), sample.zone) != zones.end())
                for (int d = 0; d < 3; ++d)
                    force[d] += values(iPoint, d);
        }
        return force;
    }
    NumericalStatus ValidateInterfaceAliasRows(Algebra::DofComponentView<const real> rows,
                                               BoundaryPointRows a, BoundaryPointRows b, ThermalView thermal) noexcept
    {
        for (index iPoint = 0; iPoint < a.RowCount(); ++iPoint)
        {
            if (!(a(iPoint, 0).cell == b(iPoint, 0).cell && a(iPoint, 0).face == b(iPoint, 0).face && a(iPoint, 0).quadrature == b(iPoint, 0).quadrature &&
                  a(iPoint, 0).zone == b(iPoint, 0).zone && a(iPoint, 0).point == b(iPoint, 0).point && a(iPoint, 0).normal == b(iPoint, 0).normal && a(iPoint, 0).interface == b(iPoint, 0).interface))
                return NumericalStatus{NumericalError::InvalidState, 1, iPoint};
            if (b(iPoint, 0).interface)
                if (!(
                        thermal.Admissible(
                            ThermalAdmissibilityDetail::PointRow<decltype(rows)>{
                                rows, iPoint})))
                    return NumericalStatus{NumericalError::InvalidState, 2, iPoint};
        }
        return {};
    }

}
