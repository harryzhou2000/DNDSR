#pragma once
#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy
{
    class FiniteVolume;
    /// Analytic flow-five cell-average error over the owned mesh partition.
    /// CPU uses the generic oracle; CUDA caches reference quadrature once and
    /// evaluates the same quadrature/order and rigid-frame transformations.
    /// The plan pins geometry generations through its metric facade. No full
    /// field is downloaded. Evaluate drains, aggregates on the domain comm and
    /// normalizes L1/L2/Linf at this explicit reporting boundary.
    class FlowErrorOperator
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        FlowErrorOperator(FiniteVolume, Runtime::ExecutionContext);
        std::array<FlowState, 3> Evaluate(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                                          const EquationDofs &, const FlowProfileSettings &, const FlowState &primitive, real time);
    };
}
