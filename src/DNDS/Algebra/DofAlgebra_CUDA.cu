#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Algebra/DeviceReduction.cuh"
#include "DNDS/Device/CUDADeviceScope.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <limits>

namespace DNDS::Algebra::Detail
{
    namespace
    {
        constexpr int kReduceThreads = 256;
        constexpr unsigned kMaxReduceBlocks = 256;

        __global__ void InitReductionOutputKernel(Runtime::ArrayView<double> result, double identity)
        {
            const size_t comp = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (comp < result.width)
                result(0, comp) = identity;
        }

        template <class T>
        __global__ void ReduceDotTreeKernel(Runtime::ArrayView<const T> x, Runtime::ArrayView<const T> y,
                                            Runtime::ArrayView<double> partials, Runtime::ArrayView<unsigned int> completed,
                                            Runtime::ArrayView<double> result)
        {
            __shared__ double tree[kReduceThreads];
            __shared__ bool lastBlock;
            double sum = 0.0;
            const size_t count = x.rows * x.width;
            for (size_t item = size_t(blockIdx.x) * blockDim.x + threadIdx.x; item < count; item += size_t(gridDim.x) * blockDim.x)
            {
                const size_t row = item / x.width;
                const size_t col = item % x.width;
                sum += double(x(row, col)) * double(y(row, col));
            }
            tree[threadIdx.x] = sum;
            BlockTreeReduce<kReduceThreads>(tree, SumMerge{});
            if (threadIdx.x == 0)
            {
                partials(blockIdx.x, 0) = tree[0];
                lastBlock = Device::ArriveReductionBlock(completed(0, 0), gridDim.x);
            }
            __syncthreads();
            if (!lastBlock)
                return;
            tree[threadIdx.x] = threadIdx.x < gridDim.x ? partials(threadIdx.x, 0) : 0.0;
            BlockTreeReduce<kReduceThreads>(tree, SumMerge{});
            if (threadIdx.x == 0)
                result(0, 0) = tree[0];
        }

        template <class T, bool Squares>
        __global__ void ReduceComponentSumTreeKernel(Runtime::ArrayView<const T> x,
                                                     Runtime::ArrayView<double> partials, Runtime::ArrayView<unsigned int> completed,
                                                     Runtime::ArrayView<double> result)
        {
            __shared__ double tree[kReduceThreads];
            __shared__ bool lastBlock;
            const size_t comp = size_t(blockIdx.y);
            double sum = 0.0;
            for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x; row < x.rows; row += size_t(gridDim.x) * blockDim.x)
            {
                const double item = double(x(row, comp));
                sum += Squares ? item * item : item;
            }
            tree[threadIdx.x] = sum;
            BlockTreeReduce<kReduceThreads>(tree, SumMerge{});
            if (threadIdx.x == 0)
            {
                partials(blockIdx.x, comp) = tree[0];
                lastBlock = Device::ArriveReductionBlock(completed(comp, 0), gridDim.x);
            }
            __syncthreads();
            if (!lastBlock)
                return;
            tree[threadIdx.x] = threadIdx.x < gridDim.x ? partials(threadIdx.x, comp) : 0.0;
            BlockTreeReduce<kReduceThreads>(tree, SumMerge{});
            if (threadIdx.x == 0)
                result(0, comp) = tree[0];
        }

        template <class T, bool IsMin>
        __global__ void ReduceComponentMinMaxKernel(Runtime::ArrayView<const T> x, Runtime::ArrayView<double> result)
        {
            __shared__ double tree[kReduceThreads];
            const size_t comp = size_t(blockIdx.y);
            double value = IsMin ? CUDART_INF : -CUDART_INF;
            for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x; row < x.rows; row += size_t(gridDim.x) * blockDim.x)
            {
                const double item = double(x(row, comp));
                value = IsMin ? fmin(value, item) : fmax(value, item);
            }
            tree[threadIdx.x] = value;
            if constexpr (IsMin)
                BlockTreeReduce<kReduceThreads>(tree, MinMerge{});
            else
                BlockTreeReduce<kReduceThreads>(tree, MaxMerge{});
            if (threadIdx.x == 0)
            {
                if constexpr (IsMin)
                    AtomicMinDouble(&result(0, comp), tree[0]);
                else
                    AtomicMaxDouble(&result(0, comp), tree[0]);
            }
        }
    }

    template <class T>
    __global__ void PointwiseKernel(Pointwise operation, Runtime::ArrayView<T> output,
                                    Runtime::ArrayView<const T> x, Runtime::ArrayView<const T> y, T a, T b)
    {
        const auto count = output.rows * output.width;
        for (size_t item = size_t(blockIdx.x) * blockDim.x + threadIdx.x; item < count; item += size_t(gridDim.x) * blockDim.x)
        {
            const auto row = item / output.width, component = item % output.width;
            switch (operation)
            {
            case Pointwise::Fill:
                output(row, component) = a;
                break;
            case Pointwise::Copy:
                output(row, component) = x(row, component);
                break;
            case Pointwise::Scale:
                output(row, component) *= a;
                break;
            case Pointwise::Axpy:
                output(row, component) += a * x(row, component);
                break;
            case Pointwise::Combination:
            {
                const T left = x(row, component), right = y(row, component);
                output(row, component) = a * left + b * right;
                break;
            }
            }
        }
    }

    template <class T>
    void Launch(Runtime::ExecutionContext context, Pointwise operation, Runtime::ArrayView<T> output,
                Runtime::ArrayView<const T> x, Runtime::ArrayView<const T> y, double a, double b)
    {
        DNDS_check_throw_info(context.device.space == Runtime::MemorySpace::CUDA, "CUDA algebra requires a CUDA context");
        Device::CUDADeviceScope deviceScope(context.device.ordinal);
        Runtime::ValidateViewDevices(context.device, output, x, y);
        Runtime::CheckKernelArgumentBudget<decltype(operation), decltype(output), decltype(x), decltype(y), T, T>();
        const auto count = output.rows * output.width;
        if (!count)
            return;
        const unsigned blocks = unsigned(std::min<size_t>(256, (count + 255) / 256));
        PointwiseKernel<<<blocks, 256, 0, ::DNDS::Runtime::CUDAStream(context)>>>(operation, output, x, y, T(a), T(b));
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "DOF algebra pointwise CUDA launch failed");
    }

    size_t ReduceCUDABlockCount(Reduction operation, size_t rows, size_t width)
    {
        const size_t work = operation == Reduction::Dot ? rows * width : rows;
        if (work == 0)
            return 1;
        return std::min<size_t>(kMaxReduceBlocks, (work + kReduceThreads - 1) / kReduceThreads);
    }

    template <class T>
    void LaunchReductionWithScratch(Runtime::ExecutionContext context, Reduction operation, Runtime::ArrayView<const T> x,
                                    Runtime::ArrayView<const T> y, Runtime::ArrayView<double> result,
                                    Runtime::ArrayView<double> partials, Runtime::ArrayView<unsigned int> completed)
    {
        DNDS_check_throw_info(context.device.space == Runtime::MemorySpace::CUDA, "CUDA reduction requires a CUDA context");
        Device::CUDADeviceScope deviceScope(context.device.ordinal);
        Runtime::ValidateViewDevices(context.device, x, y, result);
        const cudaStream_t stream = ::DNDS::Runtime::CUDAStream(context);
        if (result.width == 0)
            return;
        const double identity = operation == Reduction::Min   ? std::numeric_limits<double>::infinity()
                                : operation == Reduction::Max ? -std::numeric_limits<double>::infinity()
                                                              : 0.0;
        if (x.rows == 0 || x.width == 0)
        {
            const unsigned initBlocks = unsigned((result.width + 127) / 128);
            InitReductionOutputKernel<<<initBlocks, 128, 0, stream>>>(result, identity);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "DOF algebra empty reduction CUDA launch failed");
            return;
        }
        const unsigned blocks = unsigned(ReduceCUDABlockCount(operation, x.rows, x.width));
        if (operation == Reduction::Min || operation == Reduction::Max)
        {
            const unsigned initBlocks = unsigned((result.width + 127) / 128);
            InitReductionOutputKernel<<<initBlocks, 128, 0, stream>>>(result, identity);
            dim3 grid(blocks, unsigned(result.width));
            if (operation == Reduction::Min)
                ReduceComponentMinMaxKernel<T, true><<<grid, kReduceThreads, 0, stream>>>(x, result);
            else
                ReduceComponentMinMaxKernel<T, false><<<grid, kReduceThreads, 0, stream>>>(x, result);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "DOF algebra min/max CUDA launch failed");
            return;
        }
        Runtime::ValidateViewDevices(context.device, partials, completed);
        DNDS_check_throw_info(partials.rows >= blocks && partials.width >= result.width && completed.rows >= result.width,
                              "CUDA reduction scratch buffers are too small");
        DNDS_check_throw_info(cudaMemsetAsync(completed.data, 0, sizeof(unsigned int) * result.width, stream) == cudaSuccess,
                              "CUDA reduction completion counter reset failed");
        if (operation == Reduction::Dot)
        {
            ReduceDotTreeKernel<T><<<blocks, kReduceThreads, 0, stream>>>(x, y, partials, completed, result);
        }
        else if (operation == Reduction::Sum)
        {
            dim3 grid(blocks, unsigned(result.width));
            ReduceComponentSumTreeKernel<T, false><<<grid, kReduceThreads, 0, stream>>>(x, partials, completed, result);
        }
        else
        {
            dim3 grid(blocks, unsigned(result.width));
            ReduceComponentSumTreeKernel<T, true><<<grid, kReduceThreads, 0, stream>>>(x, partials, completed, result);
        }
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "DOF algebra tree reduction CUDA launch failed");
    }

    template <class T>
    void LaunchReduction(Runtime::ExecutionContext context, Reduction operation, Runtime::ArrayView<const T> x,
                         Runtime::ArrayView<const T> y, Runtime::ArrayView<double> result)
    {
        if (operation == Reduction::Min || operation == Reduction::Max || x.rows == 0 || x.width == 0)
        {
            LaunchReductionWithScratch(context, operation, x, y, result, {}, {});
            return;
        }
        const size_t blocks = ReduceCUDABlockCount(operation, x.rows, x.width);
        Runtime::Array<double, Runtime::TableRows> partials(Runtime::RowLayout::Dense(blocks, result.width), context.device);
        Runtime::Array<unsigned int, Runtime::TableRows> completed(Runtime::RowLayout::Dense(result.width, 1), context.device);
        auto pW = partials.Write(context);
        auto cW = completed.Write(context);
        LaunchReductionWithScratch(context, operation, x, y, result, pW.View().DynamicView(), cW.View().DynamicView());
        DNDS_check_throw_info(cudaStreamSynchronize(::DNDS::Runtime::CUDAStream(context)) == cudaSuccess,
                              "CUDA reduction temporary scratch synchronization failed");
    }

    void ApplyCUDA(Runtime::ExecutionContext c, Pointwise op, Runtime::ArrayView<float> out, Runtime::ArrayView<const float> x, Runtime::ArrayView<const float> y, double a, double b) { Launch(c, op, out, x, y, a, b); }
    void ApplyCUDA(Runtime::ExecutionContext c, Pointwise op, Runtime::ArrayView<double> out, Runtime::ArrayView<const double> x, Runtime::ArrayView<const double> y, double a, double b) { Launch(c, op, out, x, y, a, b); }
    void ReduceCUDA(Runtime::ExecutionContext c, Reduction op, Runtime::ArrayView<const float> x, Runtime::ArrayView<const float> y, Runtime::ArrayView<double> result) { LaunchReduction(c, op, x, y, result); }
    void ReduceCUDA(Runtime::ExecutionContext c, Reduction op, Runtime::ArrayView<const double> x, Runtime::ArrayView<const double> y, Runtime::ArrayView<double> result) { LaunchReduction(c, op, x, y, result); }
    void ReduceCUDA(Runtime::ExecutionContext c, Reduction op, Runtime::ArrayView<const float> x, Runtime::ArrayView<const float> y, Runtime::ArrayView<double> result, Runtime::ArrayView<double> partials, Runtime::ArrayView<unsigned int> completed) { LaunchReductionWithScratch(c, op, x, y, result, partials, completed); }
    void ReduceCUDA(Runtime::ExecutionContext c, Reduction op, Runtime::ArrayView<const double> x, Runtime::ArrayView<const double> y, Runtime::ArrayView<double> result, Runtime::ArrayView<double> partials, Runtime::ArrayView<unsigned int> completed) { LaunchReductionWithScratch(c, op, x, y, result, partials, completed); }
}
