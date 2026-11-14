#include "Cauchy/Implicit/CUDA/LocalAMGKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        __global__ void PackMatrix(LinearIndexRows offsets, LinearIndexRows sourceBlocks,
                                   LinearReadRows diagonal, LinearReadRows off, LinearWriteRows values, int n, bool full)
        {
            for (index iScalarRow = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iScalarRow < diagonal.RowCount() * n; iScalarRow += index(gridDim.x) * blockDim.x)
            {
                const index iCell = iScalarRow / n, i = iScalarRow % n;
                const index first = offsets(iCell, 0), count = offsets(iCell + 1, 0) - first;
                for (index k = 0; k < count; ++k)
                {
                    const index block = sourceBlocks(first + k, 0);
                    for (index j = 0; j < n; ++j)
                        values((first + k) * n * n + i * n + j, 0) = block >= 0 ? off(block, i + j * n)
                                                                     : full     ? diagonal(iCell, i + j * n)
                                                                     : i == j   ? diagonal(iCell, i)
                                                                                : 0;
                }
            }
        }
        template <bool Pack>
        __global__ void VectorCopy(LinearReadRows source, LinearWriteRows target, index n, index count)
        {
            for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += index(gridDim.x) * blockDim.x)
                if constexpr (Pack)
                    target(i, 0) = source(i / n, i % n);
                else
                    target(i / n, i % n) = source(i, 0);
        }
        unsigned Blocks(index count) { return unsigned(std::min<index>(65535, (count + 127) / 128)); }
        void Copy(Runtime::ExecutionContext context, LinearReadRows source, LinearWriteRows target, bool pack)
        {
            const index n = pack ? source.Width() : target.Width(), rows = pack ? source.RowCount() : target.RowCount();
            DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && n >= 5 && n <= 7 && rows >= 0 && rows <= INT64_MAX / n &&
                                      source.kind == R::RowLayoutKind::Table && target.kind == R::RowLayoutKind::Table &&
                                      (pack ? target.Width() == 1 && target.RowCount() == rows * n : source.Width() == 1 && source.RowCount() == rows * n) &&
                                      (!source.data || source.data != target.data),
                                  "invalid AMG vector packing shape or alias");
            R::ValidateViewDevices(context.device, source, target);
            R::CheckKernelArgumentBudget<LinearReadRows, LinearWriteRows, index, index>();
            if (!rows)
                return;
            const auto stream = static_cast<cudaStream_t>(context.stream);
            if (pack)
                VectorCopy<true><<<Blocks(rows * n), 128, 0, stream>>>(source, target, n, rows * n);
            else
                VectorCopy<false><<<Blocks(rows * n), 128, 0, stream>>>(source, target, n, rows * n);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "AMG vector packing launch failed");
        }
    }
    void PackLocalAMGMatrix(R::ExecutionContext context, LinearIndexRows offsets, LinearIndexRows blocks,
                            LinearReadRows diagonal, LinearReadRows off, LinearWriteRows values, int n, bool full)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && n >= 5 && n <= 7 && offsets.RowCount() == diagonal.RowCount() + 1 &&
                                  diagonal.kind == R::RowLayoutKind::Table && diagonal.Width() == (full ? n * n : n) &&
                                  off.kind == R::RowLayoutKind::Table && off.Width() == n * n && values.kind == R::RowLayoutKind::Table &&
                                  blocks.RowCount() <= INT64_MAX / (n * n) && values.RowCount() == blocks.RowCount() * n * n && values.Width() == 1 &&
                                  (!values.data || (values.data != diagonal.data && values.data != off.data)),
                              "invalid AMG principal matrix shape or alias");
        R::ValidateViewDevices(context.device, offsets, blocks, diagonal, off, values);
        R::CheckKernelArgumentBudget<LinearIndexRows, LinearIndexRows, LinearReadRows, LinearReadRows, LinearWriteRows, int, bool>();
        if (!diagonal.RowCount())
            return;
        PackMatrix<<<Blocks(diagonal.RowCount() * n), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(offsets, blocks, diagonal, off, values, n, full);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "AMG principal matrix packing launch failed");
    }
    void PackLocalAMGVector(R::ExecutionContext context, LinearReadRows source, LinearWriteRows scalar) { Copy(context, source, scalar, true); }
    void UnpackLocalAMGVector(R::ExecutionContext context, LinearReadRows scalar, LinearWriteRows target) { Copy(context, scalar, target, false); }
}
