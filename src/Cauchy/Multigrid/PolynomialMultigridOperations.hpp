#pragma once
#include "Cauchy/State/QueuedEquationOperations.hpp"

namespace DNDS::Cauchy
{
    /// pMG's selected-component algebra and explicit equal-cell-order transfer.
    /// All value operations use the caller's compute lane; only CheckFrozen
    /// observes a compact result. Array ownership/leases stay in this layer.
    /// Full-state cycles allocate no frozen-component reduction scratch.
    class PolynomialMultigridOperations : public QueuedEquationOperations
    {
        struct Data;
        std::shared_ptr<Data> _data;

    public:
        PolynomialMultigridOperations() = default;
        explicit PolynomialMultigridOperations(Runtime::ExecutionStream);
        static index ComponentNumber(const EquationDofs &state) { return state.ComponentNumber(); }
        void CheckFrozen(const EquationDofs &, const EquationDofs &, const std::vector<index> &) const;
        void ProjectCorrection(EquationDofs &, const std::vector<index> &) const;
        void AddSelected(EquationDofs &, const EquationDofs &, const std::vector<index> &) const;
        /// Caller proves the levels have identical cell identity AND order.
        /// Distinct row-space handles are allowed only at this explicit edge;
        /// ordinary equation algebra continues requiring the same DofSpace.
        void CopyIdentityRows(const EquationDofs &input, EquationDofs &output) const;
    };
}
