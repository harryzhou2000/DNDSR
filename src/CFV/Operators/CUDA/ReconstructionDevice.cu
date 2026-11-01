#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    __global__ void Jacobi(ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                           R::PairReadView<real> mean, R::PairReadView<real> input,
                           R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        if (status(0, 0))
            return;
        for (index iWork = blockIdx.x; iWork < selection.count; iWork += gridDim.x)
        {
            const auto iCell = selection.Cell(iWork);
            for (index i = threadIdx.x; i < output.width; i += blockDim.x)
            {
                const index mode = i % plan.n, component = i / plan.n;
                real value = input(iCell, i) * (1 - plan.relax);
                for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                {
                    const auto other = plan.neighbors(entry, 0);
                    if (other != UnInitIndex)
                    {
                        real product = 0;
                        for (index j = 0; j < plan.n; ++j)
                            product += plan.matrixAInvB(entry, j * plan.n + mode) * input(other, component * plan.n + j);
                        value += plan.relax * product;
                        const auto first = plan.rotated ? blocks.First(component) : -1;
                        if (first >= 0)
                        {
                            real correction = 0;
                            for (index j = 0; j < plan.n; ++j)
                            {
                                real rotated = 0;
                                for (index d = 0; d < 3; ++d)
                                    rotated += input(other, (first + d) * plan.n + j) *
                                               (plan.rotation(entry, d * 3 + component - first) - (component - first == d ? 1 : 0));
                                correction += plan.matrixAInvB(entry, j * plan.n + mode) * rotated;
                            }
                            value += plan.relax * correction;
                        }
                        real otherMean = mean(other, component);
                        if (first >= 0)
                        {
                            otherMean = 0;
                            for (index d = 0; d < 3; ++d)
                                otherMean += plan.rotation(entry, d * 3 + component - first) * mean(other, first + d);
                        }
                        value += plan.vectorAInvB(entry, mode) * (plan.relax * (otherMean - mean(iCell, component)));
                    }
                    else
                        for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                            value += plan.boundaryInjection(q, mode) * (plan.relax * (boundary(q, component) - mean(iCell, component)));
                }
                output(iCell, i) = value;
            }
        }
    }
    __global__ void ValidateIndicator(R::ArrayView<const real> indicator, R::ArrayView<int> status)
    {
        for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < indicator.rows; i += index(gridDim.x) * blockDim.x)
            if (!isfinite(indicator(i, 0)))
                atomicExch(&status(0, 0), 1);
    }
    __device__ real Biway(real center, real other, real epsilon)
    {
        const real ratio = center / (fabs(other) + epsilon) * ((other > 0) - (other < 0));
        const real r2 = ratio * ratio;
        return other == 0 ? 0 : center * (1 + r2 * ratio) / (1 + r2 * r2);
    }
    __global__ void WBAP(ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                         R::PairReadView<real> input, R::ArrayView<real> output, R::ArrayView<real> scratch,
                         DeviceCharacteristicView transform, R::ArrayView<const real> indicator, R::ArrayView<int> status,
                         index begin, index modes, real threshold, real epsilon)
    {
        const index components = output.width / plan.n, count = modes * components;
        const auto iBlock = index(blockIdx.x);
        for (index iWork = iBlock; iWork < selection.count; iWork += gridDim.x)
        {
            if (status(0, 0))
                return;
            const auto iCell = selection.Cell(iWork);
            for (index i = threadIdx.x; i < output.width; i += blockDim.x)
                output(iCell, i) = input(iCell, i);
            __syncthreads();
            if (indicator.data && indicator(iCell, 0) < threshold)
                continue;
            // Six bounded scratch rows per active block, not per mesh cell.
            const index row = iBlock * 6;
            for (index i = threadIdx.x; i < count; i += blockDim.x)
            {
                scratch(row + 3, i) = plan.wbapCenterWeight;
                scratch(row + 4, i) = plan.wbapCenterWeight;
                scratch(row + 5, i) = 1;
            }
            __syncthreads();
            for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
            {
                const auto neighbor = plan.neighbors(entry, 0);
                if (neighbor == UnInitIndex)
                    continue;
                for (index i = threadIdx.x; i < count; i += blockDim.x)
                {
                    const index mode = i % modes, component = i / modes;
                    scratch(row, i) = input(iCell, component * plan.n + begin + mode);
                    real value = 0;
                    for (index j = 0; j < plan.n; ++j)
                        value += plan.secondary(entry, j * plan.n + begin + mode) * input(neighbor, component * plan.n + j);
                    scratch(row + 1, i) = value;
                }
                __syncthreads();
                if (plan.rotated)
                {
                    for (index i = threadIdx.x; i < count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes, first = vectors.First(component);
                        real value = scratch(row + 1, i);
                        if (first >= 0)
                        {
                            value = 0;
                            for (index d = 0; d < 3; ++d)
                                value += scratch(row + 1, (first + d) * modes + mode) * plan.rotation(entry, d * 3 + component - first);
                        }
                        scratch(row + 2, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < count; i += blockDim.x)
                        scratch(row + 1, i) = scratch(row + 2, i);
                    __syncthreads();
                }
                const auto face = plan.faces(entry, 0);
                if (transform.count)
                {
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes;
                        real value = 0;
                        for (index j = 0; j < transform.count; ++j)
                            value += scratch(row, (transform.first + j) * modes + mode) * transform.forward(face, j * transform.count + component);
                        scratch(row + 2, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                        scratch(row, transform.first * modes + i) = scratch(row + 2, i);
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes;
                        real value = 0;
                        for (index j = 0; j < transform.count; ++j)
                            value += scratch(row + 1, (transform.first + j) * modes + mode) * transform.forward(face, j * transform.count + component);
                        scratch(row + 2, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                        scratch(row + 1, transform.first * modes + i) = scratch(row + 2, i);
                    __syncthreads();
                }
                for (index i = threadIdx.x; i < count; i += blockDim.x)
                    scratch(row + 2, i) = Biway(scratch(row, i), scratch(row + 1, i), epsilon);
                __syncthreads();
                if (transform.count)
                {
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                    {
                        const index mode = i % modes, component = i / modes;
                        real value = 0;
                        for (index j = 0; j < transform.count; ++j)
                            value += scratch(row + 2, (transform.first + j) * modes + mode) * transform.inverse(face, j * transform.count + component);
                        scratch(row + 1, i) = value;
                    }
                    __syncthreads();
                    for (index i = threadIdx.x; i < modes * transform.count; i += blockDim.x)
                        scratch(row + 2, transform.first * modes + i) = scratch(row + 1, i);
                    __syncthreads();
                }
                for (index i = threadIdx.x; i < count; i += blockDim.x)
                {
                    const index mode = i % modes, component = i / modes;
                    const real value = scratch(row + 2, i);
                    const real ratio = input(iCell, component * plan.n + begin + mode) / (fabs(value) + epsilon) * ((value > 0) - (value < 0));
                    scratch(row + 3, i) += ratio * ratio * ratio;
                    scratch(row + 4, i) += ratio * ratio * ratio * ratio;
                    if (value == 0)
                        scratch(row + 5, i) = 0;
                }
                __syncthreads();
            }
            for (index i = threadIdx.x; i < count; i += blockDim.x)
            {
                const index mode = i % modes, component = i / modes;
                output(iCell, component * plan.n + begin + mode) *= scratch(row + 5, i) * scratch(row + 3, i) / scratch(row + 4, i);
            }
            __syncthreads();
        }
    }
    __global__ void Commit(R::ArrayView<const real> input, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        if (status(0, 0))
            return;
        for (index i = index(blockIdx.x) * blockDim.x + threadIdx.x; i < output.rows * output.width; i += index(gridDim.x) * blockDim.x)
            output(i / output.width, i % output.width) = input(i / output.width, i % output.width);
    }
    void LaunchJacobi(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                      R::PairReadView<real> mean, R::PairReadView<real> input, R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        if (!selection.count)
            return;
        Jacobi<<<unsigned(std::min<index>(selection.count, 256)), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(plan, selection, blocks, mean, input, boundary, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA Jacobi launch failed");
    }
    void LaunchWBAP(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection, DevicePeriodicBlocks blocks,
                    R::PairReadView<real> input, R::ArrayView<real> output, R::ArrayView<real> scratch, DeviceCharacteristicView transform,
                    R::ArrayView<const real> indicator, R::ArrayView<int> status, index begin, index modes, real threshold, index blocksCount)
    {
        const auto stream = static_cast<cudaStream_t>(context.stream);
        // Status persists through a queued segment; caller explicitly clears it
        // when recovering after an observed error. No host read per submission.
        if (indicator.rows)
            ValidateIndicator<<<unsigned(std::min<index>(256, (indicator.rows + 127) / 128)), 128, 0, stream>>>(indicator, status);
        if (selection.count)
            WBAP<<<unsigned(blocksCount), 128, 0, stream>>>(plan, selection, blocks, input, output, scratch, transform, indicator, status, begin, modes, threshold, std::pow(verySmallReal, .25));
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA WBAP launch failed");
    }
    void LaunchCommit(R::ExecutionContext context, R::ArrayView<const real> input, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        if (!output.rows || !output.width)
            return;
        Commit<<<unsigned(std::min<index>(256, (output.rows * output.width + 255) / 256)), 256, 0, static_cast<cudaStream_t>(context.stream)>>>(input, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA reconstruction commit launch failed");
    }
}
