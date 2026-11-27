#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Core/CheckedSize.hpp"
#include "CFV/Detail/GradientKernels.hpp"
#include "CFV/Math/GradientEvaluation.hpp"
#include "CFV/Math/GradientLimiting.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    template <int Components>
    __global__ void EvaluateRows(GradientKernelView plan, Runtime::RowSelection selection, DevicePeriodicBlocks vectors,
                                 GradientField mean, GradientInput boundary, GradientOutput output)
    {
        for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x; iWork < selection.RowCount(); iWork += index(gridDim.x) * blockDim.x)
            Math::EvaluateGradientCell(plan, vectors, mean, boundary, output, selection.RowAt(iWork), Components);
    }
    __global__ void EvaluateRowsDynamic(GradientKernelView plan, Runtime::RowSelection selection, DevicePeriodicBlocks vectors,
                                        GradientField mean, GradientInput boundary, GradientOutput output, index components)
    {
        for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x; iWork < selection.RowCount(); iWork += index(gridDim.x) * blockDim.x)
            Math::EvaluateGradientCell(plan, vectors, mean, boundary, output, selection.RowAt(iWork), components);
    }
    __global__ void ConvertRows(GradientCellGeometry cells, GradientInput input, GradientOutput output,
                                index firstCell, int dimension, index modes, bool toCoefficients, index firstComponent, index componentCount)
    {
        for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x; iRow < output.RowCount(); iRow += index(gridDim.x) * blockDim.x)
            Math::ConvertGradientRow(cells, input, output, iRow, firstCell, dimension, modes, toCoefficients, firstComponent, componentCount);
    }
    void LaunchGradient(R::ExecutionContext context, GradientKernelView plan, Runtime::RowSelection selection, DevicePeriodicBlocks vectors,
                        GradientField mean, GradientInput boundary, GradientOutput output)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA gradient needs a CUDA stream");
        R::ValidateViewDevices(context.device, plan, selection, vectors, mean, boundary, output);
        R::CheckKernelArgumentBudget<decltype(plan), decltype(selection), decltype(vectors), decltype(mean), decltype(boundary), decltype(output), index>();
        const index components = mean.owned.Width();
        DNDS_check_throw_info(components > 0 && (plan.dimension == 2 || plan.dimension == 3) &&
                                  output.Width() == components * plan.dimension && output.RowCount() == mean.owned.RowCount() &&
                                  boundary.Width() == components && plan.offsets.RowCount() == output.RowCount() + 1,
                              "CUDA gradient shape/profile mismatch");
        for (const auto kind : {mean.owned.kind, mean.ghost.kind, boundary.kind, output.kind})
            DNDS_check_throw_info(kind == R::RowLayoutKind::Table, "CUDA gradient currently requires AoS tables");
        if (!selection.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>((selection.RowCount() + 127) / 128, 1024));
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        switch (components)
        {
        case 4:
            EvaluateRows<4><<<blocks, 128, 0, stream>>>(plan, selection, vectors, mean, boundary, output);
            break;
        case 5:
            EvaluateRows<5><<<blocks, 128, 0, stream>>>(plan, selection, vectors, mean, boundary, output);
            break;
        case 6:
            EvaluateRows<6><<<blocks, 128, 0, stream>>>(plan, selection, vectors, mean, boundary, output);
            break;
        case 7:
            EvaluateRows<7><<<blocks, 128, 0, stream>>>(plan, selection, vectors, mean, boundary, output);
            break;
        default:
            EvaluateRowsDynamic<<<blocks, 128, 0, stream>>>(plan, selection, vectors, mean, boundary, output, components);
            break;
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA gradient launch failed");
    }
    void LaunchGradientConversion(R::ExecutionContext context, GradientCellGeometry cells, GradientInput input, GradientOutput output,
                                  index firstCell, int dimension, index modes, bool toCoefficients, index firstComponent, index componentCount)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA gradient conversion needs a CUDA stream");
        R::ValidateViewDevices(context.device, cells, input, output);
        R::CheckKernelArgumentBudget<decltype(cells), decltype(input), decltype(output), index, int, index, bool, index, index>();
        DNDS_check_throw_info((dimension == 2 || dimension == 3) && modes >= dimension && input.RowCount() == output.RowCount() &&
                                  firstCell >= 0 && firstCell <= cells.RowCount() && output.RowCount() <= cells.RowCount() - firstCell &&
                                  input.kind == R::RowLayoutKind::Table && output.kind == R::RowLayoutKind::Table,
                              "CUDA gradient conversion layout/shape mismatch");
        const auto gradients = toCoefficients ? input.Width() : output.Width(), coefficients = toCoefficients ? output.Width() : input.Width();
        DNDS_check_throw_info(gradients % dimension == 0 && coefficients == (gradients / dimension) * modes,
                              "CUDA gradient conversion component mismatch");
        if (!output.RowCount())
            return;
        const unsigned blocks = unsigned(std::min<index>((output.RowCount() + 127) / 128, 1024));
        ConvertRows<<<blocks, 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(cells, input, output, firstCell, dimension, modes, toCoefficients, firstComponent, componentCount);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA gradient conversion launch failed");
    }
    template <int Components>
    __global__ void LimitRows(GradientLimiterView plan, Runtime::RowSelection selection, DevicePeriodicBlocks vectors,
                              GradientLimiterValues values, GradientOutput output, R::ArrayView<int> status)
    {
        if (status(0, 0))
            return;
        real scratch[6 * Components * 3];
        for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x; iWork < selection.RowCount(); iWork += index(gridDim.x) * blockDim.x)
            Math::LimitGradientCell(plan, vectors, values, output, selection.RowAt(iWork), {scratch, 6 * Components * 3});
    }
    __global__ void LimitRowsDynamic(GradientLimiterView plan, Runtime::RowSelection selection, DevicePeriodicBlocks vectors,
                                     GradientLimiterValues values, GradientOutput output, R::ArrayView<int> status,
                                     real *scratchPool, index scratchStride)
    {
        if (status(0, 0))
            return;
        const index tid = index(blockIdx.x) * blockDim.x + threadIdx.x;
        real *scratch = scratchPool + tid * scratchStride;
        for (index iWork = tid; iWork < selection.RowCount(); iWork += index(gridDim.x) * blockDim.x)
            Math::LimitGradientCell(plan, vectors, values, output, selection.RowAt(iWork), {scratch, scratchStride});
    }
    __global__ void ValidateGradientIndicator(GradientInput indicator, R::ArrayView<int> status)
    {
        for (index iCell = index(blockIdx.x) * blockDim.x + threadIdx.x; iCell < indicator.RowCount(); iCell += index(gridDim.x) * blockDim.x)
            if (!isfinite(indicator(iCell, 0)))
                atomicExch(&status(0, 0), 1);
    }
    void LaunchGradientLimiter(R::ExecutionContext context, GradientLimiterView plan, Runtime::RowSelection selection,
                               DevicePeriodicBlocks vectors, GradientLimiterValues values, GradientOutput output, R::ArrayView<int> status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA gradient limiter needs a CUDA stream");
        R::ValidateViewDevices(context.device, plan, selection, vectors, values, output, status);
        R::CheckKernelArgumentBudget<decltype(plan), decltype(selection), decltype(vectors), decltype(values), decltype(output), decltype(status), real *, index>();
        DNDS_check_throw_info(status.rows == 1 && status.width == 1, "gradient limiter status shape mismatch");
        DNDS_check_throw_info((plan.dimension == 2 || plan.dimension == 3) && output.Width() % plan.dimension == 0 &&
                                  output.RowCount() == values.input.owned.RowCount() && output.Width() == values.input.owned.Width() &&
                                  plan.offsets.RowCount() == output.RowCount() + 1 && output.kind == R::RowLayoutKind::Table &&
                                  values.input.owned.kind == R::RowLayoutKind::Table && values.input.ghost.kind == R::RowLayoutKind::Table,
                              "CUDA gradient limiter layout/shape mismatch");
        const index components = output.Width() / plan.dimension;
        DNDS_check_throw_info(components > 0 && values.first >= 0 && values.first <= components &&
                                  values.count >= 0 && values.count <= components - values.first &&
                                  std::isfinite(values.threshold) && std::isfinite(values.centerWeight) && values.centerWeight > 0,
                              "CUDA gradient limiter controls/profile mismatch");
        if (values.count)
            DNDS_check_throw_info(values.forward.Width() == values.count * values.count && values.inverse.Width() == values.forward.Width() &&
                                      values.forward.RowCount() == values.inverse.RowCount(),
                                  "CUDA gradient characteristic shape mismatch");
        if (values.activated)
            DNDS_check_throw_info(values.indicator.Width() == 1 && values.indicator.RowCount() == output.RowCount(),
                                  "CUDA gradient indicator shape mismatch");
        const auto stream = ::DNDS::Runtime::CUDAStream(context);
        if (values.activated && values.indicator.RowCount())
            ValidateGradientIndicator<<<unsigned(std::min<index>(256, (values.indicator.RowCount() + 127) / 128)), 128, 0, stream>>>(values.indicator, status);
        if (!selection.RowCount())
        {
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA gradient limiter indicator validation failed");
            return;
        }
        const unsigned blocks = unsigned(std::min<index>((selection.RowCount() + 127) / 128, 1024));
        switch (components)
        {
        case 1:
            LimitRows<1><<<blocks, 128, 0, stream>>>(plan, selection, vectors, values, output, status);
            break;
        case 2:
            LimitRows<2><<<blocks, 128, 0, stream>>>(plan, selection, vectors, values, output, status);
            break;
        case 3:
            LimitRows<3><<<blocks, 128, 0, stream>>>(plan, selection, vectors, values, output, status);
            break;
        case 4:
            LimitRows<4><<<blocks, 128, 0, stream>>>(plan, selection, vectors, values, output, status);
            break;
        case 5:
            LimitRows<5><<<blocks, 128, 0, stream>>>(plan, selection, vectors, values, output, status);
            break;
        case 6:
            LimitRows<6><<<blocks, 128, 0, stream>>>(plan, selection, vectors, values, output, status);
            break;
        case 7:
            LimitRows<7><<<blocks, 128, 0, stream>>>(plan, selection, vectors, values, output, status);
            break;
        default:
        {
            constexpr unsigned dynThreads = 64;
            const unsigned dynBlocks = unsigned(std::min<index>((selection.RowCount() + dynThreads - 1) / dynThreads, 256));
            const index scratchStride = 6 * output.Width();
            const size_t scratchBytes = CheckedSize::Multiply(
                CheckedSize::Multiply(size_t(dynBlocks) * dynThreads, size_t(scratchStride)), sizeof(real));
            real *scratchPool = nullptr;
            DNDS_check_throw_info(cudaMallocAsync(&scratchPool, scratchBytes, stream) == cudaSuccess,
                                  "CUDA dynamic gradient limiter scratch allocation failed");
            LimitRowsDynamic<<<dynBlocks, dynThreads, 0, stream>>>(plan, selection, vectors, values, output, status, scratchPool, scratchStride);
            const auto launchErr = cudaGetLastError();
            const auto freeErr = cudaFreeAsync(scratchPool, stream);
            DNDS_check_throw_info(launchErr == cudaSuccess && freeErr == cudaSuccess,
                                  "CUDA dynamic gradient limiter launch/free failed");
            return;
        }
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA gradient limiter launch failed");
    }
}
