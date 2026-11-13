#include "CFV/Detail/ReconstructionDeviceKernels.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <mma.h>

namespace DNDS::CFV::Detail
{
    namespace R = Runtime;
    namespace
    {
        /// Sweep-only POD projection; no limiter tables or host descriptors.
        struct QuantizedSweep
        {
            R::ArrayView<const index> offsets, neighbors, boundaryOffsets;
            R::ArrayView<const uint8_t> rotationActive;
            R::ArrayView<const float> matrix, difference, injection;
            R::ArrayView<const real> rotation;
            index modes;
            real relax;
            bool rotated;
        };
        __device__ real Coefficient(QuantizedSweep plan, DevicePeriodicBlocks vectors,
                                    R::PairReadView<real> input, index entry, index other, index component, index mode)
        {
            const index first = plan.rotated && plan.rotationActive(entry, 0) ? vectors.First(component) : -1;
            if (first < 0)
                return input(other, component * plan.modes + mode);
            real value = 0;
            for (index d = 0; d < 3; ++d)
                value += plan.rotation(entry, d * 3 + component - first) * input(other, (first + d) * plan.modes + mode);
            return value;
        }
        __global__ void JacobiTF32(QuantizedSweep plan, DeviceWorkSelection selection, DevicePeriodicBlocks vectors,
                                   R::PairReadView<real> mean, R::PairReadView<real> input,
                                   R::ArrayView<const real> boundary, R::ArrayView<real> output,
                                   R::ArrayView<const int> status)
        {
#if __CUDA_ARCH__ >= 800
            namespace wmma = nvcuda::wmma;
            if (status(0, 0))
                return;
            constexpr int warps = 4;
            __shared__ __align__(32) float matrices[warps][16 * 8];
            __shared__ __align__(32) float coefficients[warps][8 * 16];
            __shared__ __align__(32) float products[warps][16 * 16];
            const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
            for (index iWork = index(blockIdx.x) * warps + warp; iWork < selection.count; iWork += index(gridDim.x) * warps)
            {
                const index iCell = selection.Cell(iWork);
                for (index j = lane; j < output.Width(); j += 32)
                    output(iCell, j) = (1 - plan.relax) * input(iCell, j);
                if (plan.relax == 0)
                    continue;
                for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
                {
                    const index other = plan.neighbors(entry, 0);
                    if (other != UnInitIndex)
                    {
                        for (index firstComponent = 0; firstComponent < mean.owned.Width(); firstComponent += 16)
                            for (index firstMode = 0; firstMode < plan.modes; firstMode += 16)
                            {
                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> product;
                                wmma::fill_fragment(product, 0.f);
                                for (index firstInput = 0; firstInput < plan.modes; firstInput += 8)
                                {
                                    for (int i = lane; i < 16 * 8; i += 32)
                                    {
                                        const index mode = firstMode + i / 8, j = firstInput + i % 8;
                                        const float value = mode < plan.modes && j < plan.modes ? plan.matrix(entry, j * plan.modes + mode) : 0.f;
                                        matrices[warp][i] = wmma::__float_to_tf32(value);
                                    }
                                    for (int i = lane; i < 8 * 16; i += 32)
                                    {
                                        const index component = firstComponent + i / 8, j = firstInput + i % 8;
                                        const float value = component < mean.owned.Width() && j < plan.modes
                                                                ? float(Coefficient(plan, vectors, input, entry, other, component, j))
                                                                : 0.f;
                                        coefficients[warp][i] = wmma::__float_to_tf32(value);
                                    }
                                    __syncwarp();
                                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
                                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
                                    wmma::load_matrix_sync(a, matrices[warp], 8);
                                    wmma::load_matrix_sync(b, coefficients[warp], 8);
                                    wmma::mma_sync(product, a, b, product);
                                    __syncwarp();
                                }
                                wmma::store_matrix_sync(products[warp], product, 16, wmma::mem_row_major);
                                __syncwarp();
                                for (int i = lane; i < 16 * 16; i += 32)
                                {
                                    const index mode = firstMode + i / 16, component = firstComponent + i % 16;
                                    if (mode < plan.modes && component < mean.owned.Width())
                                        output(iCell, component * plan.modes + mode) += plan.relax * real(products[warp][i]);
                                }
                                __syncwarp();
                            }
                    }
                    // Mean and prescribed-boundary contributions retain the
                    // FP32-map/FP64-state arithmetic of the SIMT FP32 anchor.
                    for (index i = lane; i < output.Width(); i += 32)
                    {
                        const index mode = i % plan.modes, component = i / plan.modes;
                        if (other != UnInitIndex)
                        {
                            real otherMean = mean(other, component);
                            const index first = plan.rotated && plan.rotationActive(entry, 0) ? vectors.First(component) : -1;
                            if (first >= 0)
                            {
                                otherMean = 0;
                                for (index d = 0; d < 3; ++d)
                                    otherMean += plan.rotation(entry, d * 3 + component - first) * mean(other, first + d);
                            }
                            output(iCell, i) += plan.difference(entry, mode) * (plan.relax * (otherMean - mean(iCell, component)));
                        }
                        else
                            for (index q = plan.boundaryOffsets(entry, 0); q < plan.boundaryOffsets(entry + 1, 0); ++q)
                                output(iCell, i) += plan.injection(q, mode) * (plan.relax * (boundary(q, component) - mean(iCell, component)));
                    }
                    __syncwarp();
                }
            }
#else
            asm("trap;");
#endif
        }
    }
    bool TF32TensorCoreAvailable(R::Device device)
    {
        if (device.space != R::MemorySpace::CUDA || cudaSetDevice(device.ordinal) != cudaSuccess)
            return false;
        cudaDeviceProp properties;
        cudaFuncAttributes attributes;
        if (cudaGetDeviceProperties(&properties, device.ordinal) != cudaSuccess || cudaFuncGetAttributes(&attributes, JacobiTF32) != cudaSuccess)
            return false;
        // Explicit Ampere/Hopper admission includes RTX sm86. A newer device
        // is not assumed compatible until its kernel image is validated.
        return (properties.major == 8 || properties.major == 9) && attributes.binaryVersion >= 80 && attributes.ptxVersion >= 80;
    }
    void LaunchJacobiTF32(R::ExecutionContext context, ReconstructionDeviceView plan, DeviceWorkSelection selection,
                          DevicePeriodicBlocks vectors, R::PairReadView<real> mean, R::PairReadView<real> input,
                          R::ArrayView<const real> boundary, R::ArrayView<real> output, R::ArrayView<const int> status)
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && plan.quantized,
                              "TF32 Jacobi requires a CUDA context and FP32 sweep maps");
        R::ValidateViewDevices(context.device, plan, selection, vectors, mean, input, boundary, output, status);
        const QuantizedSweep sweep{plan.offsets, plan.neighbors, plan.boundaryOffsets, plan.rotationActive,
                                   plan.matrixAInvB32, plan.vectorAInvB32, plan.boundaryInjection32, plan.rotation, plan.n, plan.relax, plan.rotated};
        R::CheckKernelArgumentBudget<QuantizedSweep, decltype(selection), decltype(vectors), decltype(mean), decltype(input),
                                     decltype(boundary), decltype(output), decltype(status)>();
        if (!selection.count)
            return;
        JacobiTF32<<<unsigned(std::min<index>(256, (selection.count + 3) / 4)), 128, 0, static_cast<cudaStream_t>(context.stream)>>>(
            sweep, selection, vectors, mean, input, boundary, output, status);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "TF32 tensor Jacobi launch failed");
    }
}
