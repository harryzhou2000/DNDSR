#include "Cauchy/Discretization/CUDA/BoundaryForceKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include "DNDS/Device/ReductionCompletion.hpp"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace DNDS::Cauchy::CUDA
{
    namespace R = Runtime;
    namespace
    {
        constexpr int Threads = 128;
        struct Sum
        {
            __device__ Vector3 operator()(Vector3 a, Vector3 b) const
            {
                for (int d = 0; d < 3; ++d)
                    a[d] += b[d];
                return a;
            }
        };
        __global__ void Rows(ForceSamples samples, ForceEntries entries, ForceInput values,
                             SpanView<const int> zones, ForceOutput partial, ForceOutput result, ForceCounter completed)
        {
            Vector3 value{};
            for (index iPoint = index(blockIdx.x) * Threads + threadIdx.x; iPoint < entries.RowCount(); iPoint += index(gridDim.x) * Threads)
            {
                const auto &sample = samples(entries(iPoint, 0), 0);
                bool selected = false;
                for (index iZone = 0; iZone < zones.size(); ++iZone)
                    selected |= zones[iZone] == sample.zone;
                if (selected)
                    for (int d = 0; d < 3; ++d)
                        value[d] += values(iPoint, d);
            }
            using Reduce = cub::BlockReduce<Vector3, Threads, cub::BLOCK_REDUCE_RAKING>;
            __shared__ typename Reduce::TempStorage scratch;
            __shared__ bool last;
            const auto block = Reduce(scratch).Reduce(value, Sum{});
            if (!threadIdx.x)
            {
                for (int d = 0; d < 3; ++d)
                    partial(blockIdx.x, d) = block[d];
                last = Device::ArriveReductionBlock(completed(0, 0), gridDim.x);
            }
            __syncthreads();
            if (!last)
                return;
            Vector3 tail{};
            for (index iRow = threadIdx.x; iRow < partial.RowCount(); iRow += Threads)
                for (int d = 0; d < 3; ++d)
                    tail[d] += partial(iRow, d);
            const auto total = Reduce(scratch).Reduce(tail, Sum{});
            if (!threadIdx.x)
                for (int d = 0; d < 3; ++d)
                    result(0, d) = total[d];
        }
    }
    index BoundaryForceScratchRows(index points)
    {
        DNDS_check_throw_info(points >= 0, "negative boundary force point count");
        return std::max(index(1), std::min(index(256), points / Threads + (points % Threads != 0)));
    }
    void AccumulateBoundaryForce(R::ExecutionContext context, ForceSamples samples, ForceEntries entries, ForceInput values,
                                 SpanView<const int> zones, ForceOutput partial, ForceOutput result, ForceCounter completed)
    {
        R::ValidateViewDevices(context.device, samples, entries, values, partial, result, completed);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && values.RowCount() == entries.RowCount() &&
                                  partial.RowCount() == BoundaryForceScratchRows(entries.RowCount()) && result.RowCount() == 1 && completed.RowCount() == 1,
                              "invalid CUDA boundary force views");
        const std::array<const void *, 3> outputs{partial.data, result.data, completed.data};
        for (index i = 0; i < index(outputs.size()); ++i)
        {
            for (index j = 0; j < i; ++j)
                DNDS_check_throw_info(outputs[i] != outputs[j], "CUDA boundary force outputs alias");
            for (const void *input : {static_cast<const void *>(samples.data), static_cast<const void *>(entries.data), static_cast<const void *>(values.data)})
                DNDS_check_throw_info(!input || outputs[i] != input, "CUDA boundary force output aliases input");
        }
        R::CheckKernelArgumentBudget<ForceSamples, ForceEntries, ForceInput, SpanView<const int>, ForceOutput, ForceOutput, ForceCounter>();
        const auto stream = static_cast<cudaStream_t>(context.stream);
        DNDS_check_throw_info(cudaMemsetAsync(completed.data, 0, sizeof(unsigned int), stream) == cudaSuccess, "CUDA boundary force counter reset failed");
        Rows<<<unsigned(partial.RowCount()), Threads, 0, stream>>>(samples, entries, values, zones, partial, result, completed);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA boundary force reduction launch failed");
    }
}
