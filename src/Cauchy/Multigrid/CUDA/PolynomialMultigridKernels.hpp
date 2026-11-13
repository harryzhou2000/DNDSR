#pragma once
#include "DNDS/Runtime/Array/Array.hpp"
#include <array>

namespace DNDS::Cauchy::CUDA
{
    using FASRead = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using FASWrite = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using FASIndexWrite = Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    struct FASSelection
    {
        std::array<uint8_t, 7> active{};
    };
    index FrozenCheckScratchRows(index rows);
    /// Bitwise comparison of frozen components, including signed zero. Result
    /// is first bad local row or -1; deterministic bounded integer reduction.
    void CheckFrozenRows(Runtime::ExecutionContext, FASRead, FASRead, FASSelection, FASIndexWrite partial, FASIndexWrite result);
    /// Explicit in-out correction projection or selected addition. With
    /// add=false the input view is absent and no input payload is consumed.
    void UpdateSelectedRows(Runtime::ExecutionContext, FASWrite, FASRead, FASSelection, bool add);
}
