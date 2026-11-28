#include "Cauchy/Discretization/CUDA/BoundaryFluxKernels.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <climits>
#include <cuda_runtime.h>

namespace DNDS::Cauchy::CUDA
{
    namespace
    {
        __global__ void AccumulateRows(
            R::ArrayView<const BoundaryOutputRow, R::AccessBoundsChecks, R::FixedRows<1>> rows,
            SpanView<const int> rowSlots,
            R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> faceFlux,
            R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<3>> boundaryMomentum,
            R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> faceQuadrature,
            R::ArrayView<real, R::AccessBoundsChecks, R::TableRows> zoneTotals)
        {
            const index components = faceFlux.Width();
            for (index iRow = index(blockIdx.x) * blockDim.x + threadIdx.x;
                 iRow < rows.RowCount();
                 iRow += index(gridDim.x) * blockDim.x)
            {
                const auto row = rows(iRow, 0);
                const index slot = index(rowSlots[iRow]);
                atomicAdd(&zoneTotals(slot, 0), row.area);
                if (row.area == 0)
                    continue;
                for (index j = 0; j < components; ++j)
                    atomicAdd(&zoneTotals(slot, 1 + j), faceFlux(row.auxiliary, j));
                Vector3 tangent{};
                for (index iPoint = 0; iPoint < row.count; ++iPoint)
                {
                    Vector3 n = row.normal;
                    if (!row.centerNormal)
                        for (int d = 0; d < 3; ++d)
                            n[d] = faceQuadrature(row.quadratureBegin + iPoint, 3 + d);
                    const auto weighted = boundaryMomentum[row.boundaryBegin + iPoint];
                    real normalMomentum = 0;
                    for (int d = 0; d < 3; ++d)
                        normalMomentum += weighted[d] * n[d];
                    for (int d = 0; d < 3; ++d)
                        tangent[d] -= weighted[d] - normalMomentum * n[d];
                }
                for (int d = 0; d < 3; ++d)
                    atomicAdd(&zoneTotals(slot, 1 + components + d), tangent[d]);
            }
        }
    }

    void AccumulateBoundaryFluxes(
        R::ExecutionContext context,
        R::ArrayView<const BoundaryOutputRow, R::AccessBoundsChecks, R::FixedRows<1>> rows,
        SpanView<const int> rowSlots,
        R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> faceFlux,
        R::ArrayView<const real, R::AccessBoundsChecks, R::FixedRows<3>> boundaryMomentum,
        R::ArrayView<const real, R::AccessBoundsChecks, R::TableRows> faceQuadrature,
        R::ArrayView<real, R::AccessBoundsChecks, R::TableRows> zoneTotals)
    {
        R::ValidateViewDevices(context.device, rows, faceFlux, boundaryMomentum, faceQuadrature, zoneTotals);
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA &&
                                  rows.RowCount() == rowSlots.size() &&
                                  zoneTotals.Width() == faceFlux.Width() + 4,
                              "invalid CUDA boundary flux integration views");
        const auto stream = R::CUDAStream(context);
        DNDS_check_throw_info(cudaSetDevice(context.device.ordinal) == cudaSuccess,
                              "boundary flux integration device selection failed");
        const size_t totalBytes = size_t(zoneTotals.RowCount()) * size_t(zoneTotals.Width()) * sizeof(real);
        if (totalBytes > 0)
            DNDS_check_throw_info(cudaMemsetAsync(zoneTotals.data, 0, totalBytes, stream) == cudaSuccess,
                                  "CUDA boundary flux integration zeroing failed");
        if (!rows.RowCount())
            return;
        R::CheckKernelArgumentBudget<decltype(rows), decltype(rowSlots), decltype(faceFlux),
                                     decltype(boundaryMomentum), decltype(faceQuadrature), decltype(zoneTotals)>();
        constexpr int threads = 128;
        const auto blocks = unsigned(std::min<index>((rows.RowCount() - 1) / threads + 1, INT_MAX));
        AccumulateRows<<<blocks, threads, 0, stream>>>(rows, rowSlots, faceFlux, boundaryMomentum, faceQuadrature, zoneTotals);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess,
                              "CUDA boundary flux integration launch failed");
    }
}
