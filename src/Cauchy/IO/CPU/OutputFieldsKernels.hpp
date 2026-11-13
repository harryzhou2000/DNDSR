#pragma once

#include "Cauchy/State/EquationViews.hpp"
#include "Cauchy/Physics/PointProperties.hpp"

namespace DNDS::Cauchy::CPU
{
    /// One private point evaluator per CPU worker, allocated outside the loop.
    struct OutputFieldsWorker
    {
        PointEvaluator evaluator;
        explicit OutputFieldsWorker(PointEvaluator point) : evaluator(std::move(point)) {}
    };
    /// View-only output arithmetic. Writes every status row, retains no owners,
    /// performs no communication/leases, and never throws for a physical failure.
    void PrepareOutputFields(EquationLayoutView<const real> state, int species,
                             Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> output,
                             Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> status,
                             SpanView<OutputFieldsWorker> workers) noexcept;
}
