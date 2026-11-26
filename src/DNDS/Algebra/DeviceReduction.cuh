#pragma once

#include "DNDS/Device/ReductionCompletion.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace DNDS::Algebra::Detail
{
    struct SumMerge
    {
        __device__ __forceinline__ double operator()(double a, double b) const { return a + b; }
    };

    struct MinMerge
    {
        __device__ __forceinline__ double operator()(double a, double b) const { return fmin(a, b); }
    };

    struct MaxMerge
    {
        __device__ __forceinline__ double operator()(double a, double b) const { return fmax(a, b); }
    };

    struct FiniteAndMerge
    {
        __device__ __forceinline__ double operator()(double a, double b) const { return (a != 0.0 && b != 0.0) ? 1.0 : 0.0; }
    };

    /// Deterministic binary-tree reduction within one block of @p Threads threads.
    template <int Threads, class MergeOp>
    __device__ __forceinline__ void BlockTreeReduce(double (&tree)[Threads], MergeOp merge)
    {
        __syncthreads();
#pragma unroll
        for (int stride = Threads / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
                tree[threadIdx.x] = merge(tree[threadIdx.x], tree[threadIdx.x + stride]);
            __syncthreads();
        }
    }

    __device__ __forceinline__ void AtomicMinDouble(double *address, double val)
    {
        auto *addrAsUll = reinterpret_cast<unsigned long long int *>(address);
        unsigned long long int old = *addrAsUll;
        unsigned long long int assumed;
        do
        {
            assumed = old;
            const double cur = __longlong_as_double(assumed);
            const double next = fmin(cur, val);
            if (__double_as_longlong(next) == static_cast<long long int>(assumed))
                break;
            old = atomicCAS(addrAsUll, assumed, static_cast<unsigned long long int>(__double_as_longlong(next)));
        } while (assumed != old);
    }

    __device__ __forceinline__ void AtomicMaxDouble(double *address, double val)
    {
        auto *addrAsUll = reinterpret_cast<unsigned long long int *>(address);
        unsigned long long int old = *addrAsUll;
        unsigned long long int assumed;
        do
        {
            assumed = old;
            const double cur = __longlong_as_double(assumed);
            const double next = fmax(cur, val);
            if (__double_as_longlong(next) == static_cast<long long int>(assumed))
                break;
            old = atomicCAS(addrAsUll, assumed, static_cast<unsigned long long int>(__double_as_longlong(next)));
        } while (assumed != old);
    }
}
