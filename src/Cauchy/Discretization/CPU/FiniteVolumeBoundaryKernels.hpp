#pragma once
#include "DNDS/Core/NumericalStatus.hpp"

#include "Cauchy/Discretization/BoundaryTraceGeometry.hpp"
#include "Cauchy/Boundary/BoundaryPointOperator.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Core/Span.hpp"
#include "Cauchy/Boundary/BoundaryInputs.hpp"
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"

namespace DNDS::Cauchy::CPU
{
    struct BoundaryTraceRequest
    {
        Runtime::ArrayView<const BoundaryTraceGeometry,
                           Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>
            points;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            pointRows;
        SpanView<const EquationBlockReadView> means, coefficients;
        SpanView<const EquationBlockArrayView<real>> output;
        SpanView<const index> logicalFirst;
        CFV::ReconstructionBasisEvaluationView basis;
        index componentNumber = 0;
        bool polynomial = false, usePointRows = false, repairRANS = true;
        /// Point-local repair of an unlimited VR iterate before physical BC
        /// evaluation. Never modifies means or reconstruction coefficients.
        bool allowPointFallback = false;
        ThermalView thermal{};
        /// Resolved by the operator. Mirroring is a VR-only choice, not a
        /// property of polynomial sampling or of the downstream CFV kernel.
        BoundaryPointStateKind stateKind = BoundaryPointStateKind::FaceTrace;
        RANSStateView rans;
        EquationBlockReadView ransMeans;
        WallDistanceView wallDistance;
        RigidFrame frame;
        InterfaceBoundaryRows incoming;
        index ransFirst = -1;
        int modes = 0;
        real time = 0;
    };

    /// CPU physical producer context is explicit; it contains no array access
    /// or scheduling. All sampled field operands are frozen in request.
    NumericalStatus FillBoundaryTraces(const BoundaryTraceRequest &request,
                                       SpanView<const BoundaryCondition> conditions,
                                       const BoundaryPointOperator &boundary,
                                       PointEvaluator &evaluator) noexcept;

    void CopyBoundaryRows(
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            source,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            destination,
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            sourceRows);

    struct PrimitiveBoundaryRequest
    {
        Runtime::ArrayView<const BoundaryTraceGeometry,
                           Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>
            points;
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks,
                           Runtime::FixedRows<1>>
            sourceRows;
        SpanView<const EquationBlockReadView> means;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks,
                           Runtime::UniformTableRows>
            output;
        PerfectGas gas;
        RigidFrame frame;
        InterfaceBoundaryRows incoming;
        index componentNumber = 0;
        index flowFirst = -1;
        real time = 0;
    };

    void FillPrimitiveBoundary(const PrimitiveBoundaryRequest &request,
                               SpanView<const BoundaryCondition> conditions);

    using BoundaryPointRows = Runtime::ArrayView<const BoundaryPoint, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    NumericalStatus ValidateInterfacePacketRows(Algebra::DofComponentView<const real> rows, BoundaryPointRows points,
                                                ThermalView thermal, RANSStateView rans, EquationBlockReadView ransValues) noexcept;
    Vector3 AccumulateBoundaryForce(FiniteVolumeStencilView stencil,
                                    Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> values,
                                    SpanView<const int> zones);
    /// Check an alias between identical point layouts under retained scopes.
    NumericalStatus ValidateInterfaceAliasRows(Algebra::DofComponentView<const real> rows,
                                               BoundaryPointRows a, BoundaryPointRows b, ThermalView thermal) noexcept;
}
