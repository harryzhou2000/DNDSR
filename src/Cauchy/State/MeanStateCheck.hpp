#pragma once

#include "Cauchy/State/EquationDofs.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "DNDS/Core/NumericalStatus.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy
{
    /** @brief Owned mean-state acceptance on an explicit execution lane.
     * @details The policy, optional mask and reduction result are retained here;
     * each submitted read retains its state allocations. No MPI or repair is
     * performed. Evaluate drains before returning its numerical status. A
     * physically invalid mean is an ordinary InvalidState result, allowing a
     * controller's explicit step rejection; shape/residency misuse still fails
     * at the host wrapper. CPU includes mixture/segmented layouts, CUDA only
     * combined AoS perfect-gas flow5/SA/2EQ. Caller supplies its matching physics
     * policy, serializes evaluations, and agrees acceptance on the domain comm.
     */
    class MeanStateCheck
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        /// Optional resident row mask selects the required states. A zero flag
        /// skips an intentionally unused row; shape/device are checked once.
        MeanStateCheck(Algebra::DofSpace, AdmissibilityPolicy, Runtime::ExecutionContext,
                       std::optional<Runtime::Array<uint8_t>> activeRows = std::nullopt);
        NumericalStatus Evaluate(Runtime::EagerEngine &, const Runtime::ExecutionStream &, const EquationDofs &) const;
    };
}
