#pragma once

#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"

namespace DNDS::Cauchy::CUDA
{
    /// Pack the retained owned-owned principal pattern as block CSR values. sourceBlocks maps each local block to the global operator's
    /// packed off-diagonal block, or -1 for this cell's diagonal. blockOffsets
    /// describes that local pattern. No off-rank column enters this provider
    /// matrix; the global matrix action remains unchanged in DNDS. Each physical
    /// cell remains one block row, with dense row-major 5/6/7-component blocks.
    void PackLocalAMGMatrix(Runtime::ExecutionContext, LinearIndexRows blockOffsets,
                            LinearIndexRows sourceBlocks, LinearReadRows diagonal,
                            LinearReadRows offDiagonal, LinearWriteRows blockValues,
                            int components, bool fullDiagonal);
    /// Convert AoS rows (possibly row-padded) to/from compact scalar vectors.
    /// Padding is never read/written. Distinct inputs/outputs are mandatory.
    void PackLocalAMGVector(Runtime::ExecutionContext, LinearReadRows source, LinearWriteRows scalar);
    void UnpackLocalAMGVector(Runtime::ExecutionContext, LinearReadRows scalar, LinearWriteRows target);
}
