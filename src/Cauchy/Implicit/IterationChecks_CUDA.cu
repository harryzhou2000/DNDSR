#include "Cauchy/Implicit/IterationChecks.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        __global__ void Latch(FailureView dst, IterationFailure tag, R::ArrayView<const NumericalStatus> status, R::ArrayView<const real> finite, R::ArrayView<const index> invalidRow)
        {
            if (!dst(0, 0).status)
                return;
            if (status.rows)
                tag.status = status(0, 0);
            else if (invalidRow.rows)
            {
                tag.status.row = invalidRow(0, 0);
                if (tag.status.row < 0)
                    return;
            }
            else if (finite(0, 0) == 1)
                return;
            if (!tag.status)
                dst(0, 0) = tag;
        }
        __global__ void Publish(R::ArrayView<const IterationFailure> first, R::ArrayView<const real> src, R::ArrayView<real> dst)
        {
            if (!first(0, 0).status)
                return;
            for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < src.RowCount(); i += index(gridDim.x) * blockDim.x)
                for (index j = 0; j < src.RowSize(i); ++j)
                    dst(i, j) = src(i, j);
        }
    }
    void LatchIterationFailure(R::ExecutionContext context, FailureView dst, IterationFailure tag, R::ArrayView<const NumericalStatus> status, R::ArrayView<const real> finite, R::ArrayView<const index> invalidRow)
    {
        Latch<<<1, 1, 0, R::CUDAStream(context)>>>(dst, tag, status, finite, invalidRow);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "nonlinear status launch failed");
    }
    void CopyValidIterationRows(R::ExecutionContext context, R::ArrayView<const IterationFailure> first, R::ArrayView<const real> src, R::ArrayView<real> dst)
    {
        if (!src.RowCount())
            return;
        const int blocks = int(std::min<index>((src.RowCount() + 255) / 256, 65535));
        Publish<<<blocks, 256, 0, R::CUDAStream(context)>>>(first, src, dst);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "nonlinear candidate publication launch failed");
    }
}
