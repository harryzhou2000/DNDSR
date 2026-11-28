#pragma once

#include "Cauchy/IO/BoundaryOutputGeometry.hpp"
#include "DNDS/Core/Span.hpp"

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;

    void AccumulateBoundaryFluxes(
        R::ExecutionContext context,
        R::ArrayView<const BoundaryOutputRow, R::AccessBoundsChecks, R::FixedRows<1>> rows,
        SpanView<const int> rowSlots,
        R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> faceFlux,
        R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<3>> boundaryMomentum,
        R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> faceQuadrature,
        R::ArrayView<real, R::AccessBoundsChecks, R::TableRows> zoneTotals);
}
