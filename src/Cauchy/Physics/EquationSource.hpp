#pragma once

#include "Cauchy/State/DerivedFields.hpp"

namespace DNDS::Cauchy
{
    /// Synchronous owned-only CPU point-source operator. Replaces values with
    /// positive dU/dt per code volume; no volume weighting, transport, MPI or
    /// integration policy. Flow mass/momentum/full thermochemical energy sources
    /// and any nonchemical blocks (including RANS) are zero; chemistry fills
    /// the named independent species block. This replaces, rather than adds to,
    /// the complete output. Sum separate physical source contributions outside.
    ///
    /// Optional Jacobian rows have N*N logical columns, column-major within
    /// each row, storing positive dS/dU. An implicit residual subtracts this
    /// derivative. No spectrum filter is applied (the active Euler policy).
    /// Separate/coupled physical inputs and outputs may have different layouts;
    /// they must use the same schema/row space and distinct storage. Every output
    /// is reserved before writing. On evaluation failure scratch may be partial;
    /// discard it rather than treating it as a valid operator result.
    ///
    /// A compatible published temperature bundle may be stale: it seeds the UV
    /// solve only. State/ghost values are not changed and ghosts are not read.
    /// Retain handles and serialize this call; move the worker lease with CPU
    /// work if launched asynchronously. No implicit device/host fallback.
    void EvaluateChemicalSource(const EquationSystem &equation, const EquationDofs &state,
                                EquationDofs &values, Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                                Runtime::Array<real> *jacobian = nullptr, real sourceScale = 1,
                                const DerivedFields *temperatureGuess = nullptr);
}
