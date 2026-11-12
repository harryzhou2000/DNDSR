#pragma once

#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;

    struct FiniteVolume::ReconstructionState
    {
        R::SessionAccess session;
        std::shared_ptr<const Data> owner;
        EquationDofOperations::Binding binding;
        /// Durable warm/unlimited modal coefficients. Empty for direct P0/P1;
        /// physical gradients live in Workspace and means are separate inputs.
        std::optional<EquationDofs> coefficients;
        /// A distinct finalized allocation is required only when a limiter or
        /// reconstruction PP mutates a polynomial derived from the warm guess.
        /// Otherwise FinalizedValues selects coefficients without another handle.
        bool distinctFinalized = false;
        /// Durable finalized variational polynomial consumed after limiter/PP.
        /// Empty for direct P0/P1 and unfiltered VR. DITR stages need independent
        /// copies while serially sharing all temporary workspace allocation.
        std::optional<EquationDofs> fluxCoefficients;
        /// Provenance of the finalized owned polynomial. Limiter/PP profiles
        /// require an Evaluate/PrepareReconstruction after the warm guess
        /// changes; unfiltered profiles can republish by a direct owned copy.
        std::optional<EquationDofOperations::Binding> finalizedFrom;
        std::optional<EquationDofOperations::Binding> finalizedPublication;
        std::vector<Algebra::DofHalo<>> meanHalo;
        uint64_t meanEpoch = 0;
        std::vector<R::TransferBinding> coefficientHalo, fluxHalo;
        std::optional<R::Array<real>> limitedVRHistory;

        bool FinalizedCurrent() const;
        void PublishFinalized();
        const EquationDofs &WarmValues() const;
        EquationDofs &WarmValues();
        const EquationDofs &FinalizedValues() const;
        EquationDofs &FinalizedValues();
        EquationDofs &MutableFinalized();
        std::vector<R::TransferBinding> &FinalizedHalo();
        static EquationDofs BuildCoefficients(const Data &data, const EquationDofs &state);
        ReconstructionState(std::shared_ptr<const Data> data, EquationDofs &state);
    };
}
