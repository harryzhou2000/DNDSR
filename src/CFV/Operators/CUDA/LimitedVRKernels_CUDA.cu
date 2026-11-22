#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "CFV/Detail/LimitedVRKernels.hpp"
#include "CFV/Math/LimitedVRStencil.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    namespace
    {
        __global__ void BuildLimitedVRSeedsKernel(LimitedVRStencilView stencil, Runtime::RowSelection selection, LVRField scalars, LVRField gates,
                                                  LVRInput previous, LVROutput output, LimitedVRKernelParameters settings)
        {
            for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x; iWork < selection.RowCount(); iWork += index(gridDim.x) * blockDim.x)
                Math::BuildLimitedVRSeedCell(stencil, scalars, gates, previous, output, settings, selection.RowAt(iWork));
        }
        __global__ void LimitedVRMaximumKernel(LimitedVRStencilView stencil, Runtime::RowSelection selection, LVRField input, LVROutput output,
                                               std::array<bool, 4> active, bool excludeBoundary, bool saturatedOnly, std::array<real, 4> caps, LVRInput base, unsigned mergeMask)
        {
            for (index iWork = index(blockIdx.x) * blockDim.x + threadIdx.x; iWork < selection.RowCount(); iWork += index(gridDim.x) * blockDim.x)
                Math::LimitedVRMaximumCell(stencil, input, output, selection.RowAt(iWork), active, excludeBoundary, saturatedOnly, caps, base, mergeMask);
        }
        void ValidateOutput(LimitedVRStencilView stencil, LVROutput output)
        {
            DNDS_check_throw_info(stencil.offsets.RowCount() == output.RowCount() + 1 && stencil.boundary.RowCount() == output.RowCount() && (output.Width() == 3 || output.Width() == 4),
                                  "LVR kernel stencil/output shape differs");
        }
        void ValidateInput(LVRField input, index rows, index width)
        {
            DNDS_check_throw_info(input.owned.RowCount() == rows && input.owned.Width() == width && (!input.ghost.RowCount() || input.ghost.Width() == width),
                                  "LVR kernel input shape differs");
        }
        unsigned Blocks(index rows) { return unsigned(std::min<index>((rows + 127) / 128, 1024)); }
    }
    void LaunchLimitedVRSeeds(R::ExecutionContext context, LimitedVRStencilView stencil, Runtime::RowSelection selection,
                              LVRField scalars, LVRField gates, LVRInput previous, LVROutput output, LimitedVRKernelParameters settings)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA LVR seeds require a CUDA context");
        R::ValidateViewDevices(context.device, stencil, selection, scalars, gates, previous, output);
        ValidateOutput(stencil, output);
        ValidateInput(scalars, output.RowCount(), 2);
        ValidateInput(gates, output.RowCount(), output.Width());
        DNDS_check_throw_info(previous.RowCount() == output.RowCount() && previous.Width() + 1 == output.Width(), "LVR previous seed shape differs");
        R::CheckKernelArgumentBudget<LimitedVRStencilView, Runtime::RowSelection, LVRField, LVRField, LVRInput, LVROutput, LimitedVRKernelParameters>();
        if (!selection.RowCount())
            return;
        BuildLimitedVRSeedsKernel<<<Blocks(selection.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(stencil, selection, scalars, gates, previous, output, settings);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA LVR seed launch failed");
    }
    void LaunchLimitedVRMaximum(R::ExecutionContext context, LimitedVRStencilView stencil, Runtime::RowSelection selection,
                                LVRField input, LVROutput output, std::array<bool, 4> active, bool excludeBoundary,
                                bool saturatedOnly, std::array<real, 4> caps, LVRInput base, unsigned mergeMask)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA, "CUDA LVR maximum requires a CUDA context");
        R::ValidateViewDevices(context.device, stencil, selection, input, output, base);
        ValidateOutput(stencil, output);
        ValidateInput(input, output.RowCount(), output.Width());
        R::CheckKernelArgumentBudget<LimitedVRStencilView, Runtime::RowSelection, LVRField, LVROutput, std::array<bool, 4>, bool, bool, std::array<real, 4>, LVRInput, unsigned>();
        if (!selection.RowCount())
            return;
        LimitedVRMaximumKernel<<<Blocks(selection.RowCount()), 128, 0, ::DNDS::Runtime::CUDAStream(context)>>>(stencil, selection, input, output, active, excludeBoundary, saturatedOnly, caps, base, mergeMask);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "CUDA LVR maximum launch failed");
    }
}
