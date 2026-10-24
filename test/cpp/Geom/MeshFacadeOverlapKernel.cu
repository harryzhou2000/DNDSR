#include "MeshFacadeOverlapKernel.hpp"

namespace DNDS::Geom::OverlapTest
{
    __global__ void CoreKernel(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                               R::ArrayView<const real> owned, R::ArrayView<real> output)
    {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows.Rows(); i += blockDim.x * gridDim.x)
            output(i, 0) = NeighborSum(mesh, rows(i, 0), OwnedSolution{owned});
    }
    __global__ void RingKernel(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                               R::ArrayView<const real> owned, R::ArrayView<const real> ghost, R::ArrayView<real> output)
    {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows.Rows(); i += blockDim.x * gridDim.x)
            output(i, 0) = NeighborSum(mesh, rows(i, 0), SplitSolution{owned, ghost});
    }
    cudaError_t PreloadOverlapKernels()
    {
        cudaFuncAttributes attributes;
        auto status = cudaFuncGetAttributes(&attributes, CoreKernel);
        return status == cudaSuccess ? cudaFuncGetAttributes(&attributes, RingKernel) : status;
    }
    cudaError_t LaunchCore(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                           R::ArrayView<const real> owned, R::ArrayView<real> output, cudaStream_t stream)
    {
        CoreKernel<<<4, 64, 0, stream>>>(mesh, rows, owned, output);
        return cudaGetLastError();
    }
    cudaError_t LaunchRing(UnstructuredMeshView mesh, R::ArrayView<const index> rows,
                           R::ArrayView<const real> owned, R::ArrayView<const real> ghost, R::ArrayView<real> output, cudaStream_t stream)
    {
        RingKernel<<<4, 64, 0, stream>>>(mesh, rows, owned, ghost, output);
        return cudaGetLastError();
    }
}
