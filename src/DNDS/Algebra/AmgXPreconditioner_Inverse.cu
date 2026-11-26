#include "DNDS/Algebra/AmgXPreconditioner.hpp"
#include "Solver/ScaledBlockInverse.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

namespace DNDS::Algebra
{
    namespace
    {
        using namespace Direct::ScaledBlock;

        template <int N>
        __global__ void InvertRows(int rows, const int *diag, const double *values, double *output, int *failed)
        {
            const auto row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= size_t(rows))
                return;
            const auto status = InvertBlock<N>(values + size_t(diag[row]) * N * N, output + row * N * N);
            if (status != Success && atomicCAS(failed, -1, int(row)) == -1)
                failed[1] = status;
        }

        // AmgX's 1x1 BlockJacobi smoother stores the uninverted scalar diagonal
        // in Dinv and divides by it (x + omega * (b - y) / d), unlike width >= 2
        // which multiplies by the inverted block.
        __global__ void CopyScalarDiagonal(int rows, const int *diag, const double *values, double *output, int *failed)
        {
            const auto row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (row >= size_t(rows))
                return;
            const double d = values[size_t(diag[row])];
            if (!std::isfinite(d))
            {
                if (atomicCAS(failed, -1, int(row)) == -1)
                    failed[1] = NonfiniteInput;
                return;
            }
            output[row] = d;
        }
    }

    int ComputeAmgXBlockInverse(int rows, int width, const int *diag, const double *values, double *output)
    {
        const bool supportedWidth = (width >= 1 && width <= 10) || width == 19;
        if (rows <= 0 || !supportedWidth)
            return 1;
        // AmgX setup is already synchronous. Keep its numerical failure visible
        // before returning through the C ABI, including during coarse resetup.
        int *failed = nullptr;
        if (cudaMalloc(&failed, 2 * sizeof(int)) != cudaSuccess)
            return 1;
        int status = 1;
        if (cudaMemset(failed, -1, 2 * sizeof(int)) == cudaSuccess)
        {
            const unsigned blocks = 1 + (unsigned(rows) - 1) / 128;
            switch (width)
            {
            case 1:
                CopyScalarDiagonal<<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 2:
                InvertRows<2><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 3:
                InvertRows<3><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 4:
                InvertRows<4><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 5:
                InvertRows<5><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 6:
                InvertRows<6><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 7:
                InvertRows<7><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 8:
                InvertRows<8><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 9:
                InvertRows<9><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 10:
                InvertRows<10><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            case 19:
                InvertRows<19><<<blocks, 128>>>(rows, diag, values, output, failed);
                break;
            }
            int failure[2];
            if (cudaGetLastError() == cudaSuccess &&
                cudaMemcpy(failure, failed, sizeof(failure), cudaMemcpyDeviceToHost) == cudaSuccess)
            {
                status = failure[0] != -1;
                if (status)
                {
                    const char *reason = failure[1] == NonfiniteInput     ? "nonfinite block input"
                                         : failure[1] == SVDNoConvergence ? "SVD did not converge"
                                                                          : "nonfinite inverse";
                    std::fprintf(stderr, "AmgX ACM inverse failed on row %d of %d (width %d): %s\n",
                                 failure[0], rows, width, reason);
                    int block;
                    std::vector<double> entries(size_t(width) * size_t(width));
                    if (cudaMemcpy(&block, diag + failure[0], sizeof(block), cudaMemcpyDeviceToHost) == cudaSuccess &&
                        cudaMemcpy(entries.data(), values + size_t(block) * width * width, entries.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost) == cudaSuccess)
                    {
                        std::fprintf(stderr, "AmgX diagonal block row-major:");
                        for (size_t i = 0; i < entries.size(); ++i)
                            std::fprintf(stderr, " %.17g", entries[i]);
                        std::fputc('\n', stderr);
                    }
                }
            }
        }
        return cudaFree(failed) == cudaSuccess ? status : 1;
    }
}
