#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::CFV::Detail
{
    /// Optional bounded CPU value kernel. All views borrow enclosing host
    /// leases; tables have unit component stride and may have row padding.
    /// Returns false for an unsupported width/mode count without writing.
    /// Keep this out of the generic derivative sampler's translation unit:
    /// value specializations must not enlarge its numerical loop body.
    bool TryEvaluateCachedOwnedTable(
        Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> cells,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> mean,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> coefficients,
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::TableRows> basis,
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> output);
}
