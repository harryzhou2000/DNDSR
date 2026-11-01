#pragma once

#include <mma.h>

namespace DNDS::Algebra::CUDA
{
    /// One warp's bounded, aligned staging. A is8x4, B4x8, C8x8, row major.
    /// Callers fill padded tails with zero and synchronize the participating warp.
    struct alignas(32) FP64Tile8
    {
        double a[32], b[32], c[64];
    };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    /// Actual double-input/double-accumulator MMA; all32 lanes participate.
    /// The enclosing launch explicitly checks device and compiled capability.
    class FP64Accumulator8
    {
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 8, 4, double> _value;

    public:
        __device__ FP64Accumulator8() { nvcuda::wmma::fill_fragment(_value, 0.0); }
        __device__ void AddProduct(const FP64Tile8 &tile)
        {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 8, 4, double, nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 8, 4, double, nvcuda::wmma::row_major> b;
            nvcuda::wmma::load_matrix_sync(a, tile.a, 4);
            nvcuda::wmma::load_matrix_sync(b, tile.b, 8);
            nvcuda::wmma::mma_sync(_value, a, b, _value);
        }
        __device__ void Store(FP64Tile8 &tile) const
        {
            nvcuda::wmma::store_matrix_sync(tile.c, _value, 8, nvcuda::wmma::mem_row_major);
        }
    };
#endif
}
