#pragma once
#include "CFV/Detail/ReconstructionSweepArguments.hpp"
#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Runtime/Execution/KernelViews.hpp"
#include <cuda_runtime.h>

namespace DNDS::CFV::Detail::V2
{
    template <int W>
    using Compact = R::FixedRows<W, W>;
    template <class T, int W>
    using Rows = R::ArrayView<T, R::AccessBoundsChecks, Compact<W>>;
    template <int W>
    using Pair = R::PairReadView<real, R::AccessBoundsChecks, Compact<W>>;

    template <int M, int C, class Storage>
    struct Sweep
    {
        Rows<const index, 1> offsets, neighbors, faces, boundaryOffsets, vectorStarts;
        Rows<const uint8_t, 1> side, rotationActive;
        Rows<const real, 9> rotation;
        Rows<const Storage, M * M> matrix;
        Rows<const real, M * M> inverse;
        Rows<const float, M * M> packedInverse;
        Rows<const real, M> affine, injection;
        Pair<C> mean;
        Pair<C * M> input;
        Rows<const real, C> boundary;
        Rows<real, C * M> output;
        Rows<const real, C * M> target;
        Rows<const real, 1> alpha;
        Rows<const int, 1> status;
        R::RowSelection selection;
        real relax;
        bool rotated, faceCoupling;

        Sweep(const StagedSweepArguments &a, R::ArrayView<const Storage> coupling)
            : offsets(R::ProjectKernelView<Compact<1>>(a.plan.offsets)),
              neighbors(R::ProjectKernelView<Compact<1>>(a.plan.neighbors)),
              faces(R::ProjectKernelView<Compact<1>>(a.plan.faces)),
              boundaryOffsets(R::ProjectKernelView<Compact<1>>(a.plan.boundaryOffsets)),
              vectorStarts(R::ProjectKernelView<Compact<1>>(a.vectors.starts)),
              side(R::ProjectKernelView<Compact<1>>(a.plan.faceSide)),
              rotationActive(R::ProjectKernelView<Compact<1>>(a.plan.rotationActive)),
              rotation(R::ProjectKernelView<Compact<9>>(a.plan.rotation)),
              matrix(R::ProjectKernelView<Compact<M * M>>(coupling)),
              inverse(R::ProjectKernelView<Compact<M * M>>(a.plan.localInverse)),
              packedInverse(R::ProjectKernelView<Compact<M * M>>(a.plan.localInverse32)),
              affine(R::ProjectKernelView<Compact<M>>(a.plan.vectorAInvB)),
              injection(R::ProjectKernelView<Compact<M>>(a.plan.boundaryInjection)),
              mean(R::ProjectKernelPair<Compact<C>>(a.mean)), input(R::ProjectKernelPair<Compact<C * M>>(a.input)),
              boundary(R::ProjectKernelView<Compact<C>>(a.boundary)), output(R::ProjectKernelView<Compact<C * M>>(a.output)),
              target(R::ProjectKernelView<Compact<C * M>>(a.target)), alpha(R::ProjectKernelView<Compact<1>>(a.alpha)),
              status(R::ProjectKernelView<Compact<1>>(a.status)), selection(a.selection), relax(a.plan.relax),
              rotated(a.plan.rotated), faceCoupling(a.plan.faceCoupling) {}

        template <int Stride, class Row>
        __device__ real Oriented(Row field, index entry, int component, int mode) const
        {
            if (rotated && rotationActive(entry, 0))
                for (index v = 0; v < vectorStarts.RowCount(); ++v)
                {
                    const int first = int(vectorStarts(v, 0)), d = component - first;
                    if (d >= 0 && d < 3)
                    {
                        real value = 0;
                        for (int k = 0; k < 3; ++k)
                            value += rotation(entry, 3 * k + d) * field[(first + k) * Stride + mode];
                        return value;
                    }
                }
            return field[component * Stride + mode];
        }
    };

    __device__ inline void MultiplyAddTile(double a, double b, double &lo, double &hi)
    {
#if __CUDA_ARCH__ >= 800
        asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
            : "+d"(lo), "+d"(hi) : "d"(a), "d"(b));
#endif
    }

    /// Eight lanes cover one small cell, with four independently progressing
    /// cells per warp. Write all modes/components before applying the inverse.
    template <int M, int C, bool Raw, class Storage>
    __global__ void SmallJacobi(Sweep<M, C, Storage> a)
    {
        static_assert(M == 2 || M == 3);
        constexpr int team = 8, width = M * C, pieces = (width + team - 1) / team;
        const int lane = threadIdx.x % team;
        const unsigned mask = 0xffu << ((threadIdx.x % 32) / team * team);
        extern __shared__ real scratch[];
        real *right = scratch + (threadIdx.x / team) * width;
        if (a.status(0, 0))
            return;
        for (index work = (index(blockIdx.x) * blockDim.x + threadIdx.x) / team;
             work < a.selection.RowCount(); work += index(gridDim.x) * blockDim.x / team)
        {
            const index cell = a.selection.RowAt(work);
            const auto own = a.mean[cell];
            real sum[pieces]{};
            for (index entry = a.offsets(cell, 0); entry < a.offsets(cell + 1, 0); ++entry)
            {
                const index other = a.neighbors(entry, 0);
                if (other == UnInitIndex)
                {
                    for (index q = a.boundaryOffsets(entry, 0); q < a.boundaryOffsets(entry + 1, 0); ++q)
#pragma unroll
                        for (int t = 0; t < pieces; ++t)
                        {
                            const int pos = t * team + lane;
                            if (pos < width)
                                sum[t] = fma(a.injection(q, pos % M), a.boundary(q, pos / M) - own[pos / M], sum[t]);
                        }
                    continue;
                }
                const auto mean = a.mean[other];
                const bool diagonal = Raw && other == cell && !(a.rotated && a.rotationActive(entry, 0));
#pragma unroll
                for (int t = 0; t < pieces; ++t)
                {
                    const int pos = t * team + lane, c = pos / M, m = pos % M;
                    if (pos >= width)
                        continue;
                    sum[t] = fma(a.affine(entry, m), a.template Oriented<1>(mean, entry, c, 0) - own[c], sum[t]);
                    if (diagonal)
                        continue;
                    const auto x = a.input[other];
                    const auto block = a.matrix[a.faceCoupling ? a.faces(entry, 0) : entry];
                    const bool transpose = a.faceCoupling && a.side(entry, 0);
#pragma unroll
                    for (int k = 0; k < M; ++k)
                        sum[t] = fma(real(block[transpose ? m * M + k : k * M + m]), a.template Oriented<M>(x, entry, c, k), sum[t]);
                }
            }
            if constexpr (Raw)
            {
#pragma unroll
                for (int t = 0; t < pieces; ++t)
                    if (t * team + lane < width)
                        right[t * team + lane] = sum[t];
                __syncwarp(mask);
                auto apply = [&](auto inverse)
                {
                    const auto row = inverse[cell];
#pragma unroll
                    for (int t = 0; t < pieces; ++t)
                    {
                        const int pos = t * team + lane;
                        if (pos >= width)
                            continue;
                        real value = 0;
#pragma unroll
                        for (int k = 0; k < M; ++k)
                            value = fma(real(row[k * M + pos % M]), right[pos / M * M + k], value);
                        sum[t] = value;
                    }
                };
                if (a.packedInverse.data)
                    apply(a.packedInverse);
                else
                    apply(a.inverse);
                __syncwarp(mask);
            }
#pragma unroll
            for (int t = 0; t < pieces; ++t)
            {
                const int pos = t * team + lane;
                if (pos < width)
                    a.output(cell, pos) = (1 - a.relax) * a.input(cell, pos) + a.relax * sum[t];
            }
        }
    }

    /// One warp per cell. FP64 MMA uses m8n8k4 fragments; missing rows and
    /// components of the last fragment contribute zero.
    template <int M, int C, bool Raw, class Storage, bool Limited, bool MMA>
    __global__ void WarpJacobi(Sweep<M, C, Storage> a)
    {
        static_assert(M == 5 || M == 9 || M == 19);
        constexpr int Team = 32;
        constexpr int W = M * C, Tiles = (M + 7) / 8;
        constexpr int Parts = MMA ? 2 * Tiles : (W + Team - 1) / Team;
        const int lane = threadIdx.x % Team;
        const unsigned mask = (0xffffffffu >> (32 - Team)) << ((threadIdx.x % 32) / Team * Team);
        extern __shared__ real scratch[];
        real *right = scratch + (threadIdx.x / Team) * W;
        if (a.status(0, 0))
            return;
        for (index work = (index(blockIdx.x) * blockDim.x + threadIdx.x) / Team;
             work < a.selection.RowCount(); work += index(gridDim.x) * blockDim.x / Team)
        {
            const index cell = a.selection.RowAt(work);
            const real alpha = Limited ? a.alpha(cell, 0) : 0;
            const real weight = a.relax * (1 - alpha);
            real product[Parts]{}, affine[Parts]{};
            auto position = [&](int part)
            {
                if constexpr (MMA)
                    return ((lane % 4) * 2 + part % 2) * M + (part / 2) * 8 + lane / 4;
                else
                    return part * Team + lane;
            };
            auto valid = [&](int part)
            {
                if constexpr (MMA)
                    return (lane % 4) * 2 + part % 2 < C && (part / 2) * 8 + lane / 4 < M;
                else
                    return position(part) < W;
            };
            if (weight != 0)
            {
                const auto own = a.mean[cell];
                const index end = a.offsets(cell + 1, 0);
                for (index e = a.offsets(cell, 0); e < end; ++e)
                {
                    const index other = a.neighbors(e, 0);
                    if constexpr (MMA)
                        if (e + 1 < end)
                        {
                            const index next = a.neighbors(e + 1, 0);
                            if (next != UnInitIndex && next != cell)
                            {
                                const auto matrix = a.matrix[a.faceCoupling ? a.faces(e + 1, 0) : e + 1];
                                const int elem = lane * (128 / sizeof(Storage));
                                if (elem < M * M)
                                    asm volatile("prefetch.global.L2 [%0];" ::"l"(&matrix[elem]));
                                if (lane * 16 < W)
                                    asm volatile("prefetch.global.L2 [%0];" ::"l"(&a.input[next][lane * 16]));
                            }
                        }
                    if (other == UnInitIndex)
                    {
                        for (index q = a.boundaryOffsets(e, 0); q < a.boundaryOffsets(e + 1, 0); ++q)
#pragma unroll
                            for (int t = 0; t < Parts; ++t)
                                if (valid(t))
                                {
                                    const int pos = position(t);
                                    affine[t] = fma(a.injection(q, pos % M), a.boundary(q, pos / M) - own[pos / M], affine[t]);
                                }
                        continue;
                    }
                    const auto mean = a.mean[other];
#pragma unroll
                    for (int t = 0; t < Parts; ++t)
                        if (valid(t))
                        {
                            const int pos = position(t), c = pos / M;
                            affine[t] = fma(a.affine(e, pos % M), a.template Oriented<1>(mean, e, c, 0) - own[c], affine[t]);
                        }
                    if (Raw && other == cell && !(a.rotated && a.rotationActive(e, 0)))
                        continue;
                    const auto x = a.input[other];
                    const auto b = a.matrix[a.faceCoupling ? a.faces(e, 0) : e];
                    const bool transpose = a.faceCoupling && a.side(e, 0);
                    if constexpr (MMA)
                    {
#pragma unroll
                        for (int k0 = 0; k0 < M; k0 += 4)
                        {
                            const int k = k0 + lane % 4, group = lane / 4;
                            const real bx = k < M && group < C ? a.template Oriented<M>(x, e, group, k) : 0;
#pragma unroll
                            for (int tile = 0; tile < Tiles; ++tile)
                            {
                                const int m = tile * 8 + group;
                                const real ax = m < M && k < M ? real(b[transpose ? m * M + k : k * M + m]) : 0;
                                MultiplyAddTile(ax, bx, product[2 * tile], product[2 * tile + 1]);
                            }
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int t = 0; t < Parts; ++t)
                            if (valid(t))
                            {
                                const int pos = position(t), m = pos % M;
                                real sum = 0;
#pragma unroll
                                for (int k = 0; k < M; ++k)
                                    sum = fma(real(b[transpose ? m * M + k : k * M + m]), a.template Oriented<M>(x, e, pos / M, k), sum);
                                product[t] += sum;
                            }
                    }
                }
                if constexpr (Raw)
                {
#pragma unroll
                    for (int t = 0; t < Parts; ++t)
                        if (valid(t))
                            right[position(t)] = product[t] + affine[t];
                    __syncwarp(mask);
                    auto solve = [&](auto inverse)
                    {
                        const auto inv = inverse[cell];
                        if constexpr (MMA)
                        {
#pragma unroll
                            for (int t = 0; t < Parts; ++t)
                                product[t] = 0;
#pragma unroll
                            for (int k0 = 0; k0 < M; k0 += 4)
                            {
                                const int k = k0 + lane % 4, group = lane / 4;
                                const real bx = k < M && group < C ? right[group * M + k] : 0;
#pragma unroll
                                for (int tile = 0; tile < Tiles; ++tile)
                                {
                                    const int m = tile * 8 + group;
                                    MultiplyAddTile(k < M && m < M ? real(inv[k * M + m]) : 0, bx, product[2 * tile], product[2 * tile + 1]);
                                }
                            }
                        }
                        else
                        {
#pragma unroll
                            for (int t = 0; t < Parts; ++t)
                                if (valid(t))
                                {
                                    const int pos = position(t);
                                    real sum = 0;
#pragma unroll
                                    for (int k = 0; k < M; ++k)
                                        sum = fma(real(inv[k * M + pos % M]), right[(pos / M) * M + k], sum);
                                    product[t] = sum;
                                }
                        }
                    };
                    if (a.packedInverse.data)
                        solve(a.packedInverse);
                    else
                        solve(a.inverse);
                    __syncwarp(mask);
                }
            }
#pragma unroll
            for (int t = 0; t < Parts; ++t)
                if (valid(t))
                {
                    const int pos = position(t);
                    real result = (1 - a.relax) * a.input(cell, pos);
                    if constexpr (Limited)
                        if (alpha > 0)
                            result += a.relax * alpha * a.target(cell, pos);
                    if (weight != 0)
                        result += weight * (Raw ? product[t] : product[t] + affine[t]);
                    a.output(cell, pos) = result;
                }
        }
    }

    template <int M, int C, bool Raw, bool Limited, bool MMA, class Storage>
    ReconstructionSweepResources Launch(const StagedSweepArguments &a, R::ArrayView<const Storage> matrix, int threads)
    {
        constexpr int team = M <= 3 ? 8 : 32;
        const index wide = (a.selection.RowCount() + threads / team - 1) / (threads / team);
        const int grid = int(std::min<index>(a.options.gridBlocks ? a.options.gridBlocks : 2147483647, wide));
        const size_t bytes = Raw ? size_t(threads / team) * M * C * sizeof(real) : 0;
        using Args = Sweep<M, C, Storage>;
        R::CheckKernelArgumentBudget<Args>();
        Args args(a, matrix);
        ReconstructionSweepResources res{index(bytes), 0, 0, grid, threads};
        constexpr auto kernel = []
        {
            if constexpr (M <= 3)
                return SmallJacobi<M, C, Raw, Storage>;
            else
                return WarpJacobi<M, C, Raw, Storage, Limited, MMA>;
        }();
        if (a.execute)
        {
            if (grid)
                kernel<<<grid, threads, bytes, R::CUDAStream(a.context)>>>(args);
            DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "V2 reconstruction launch failed");
        }
        else
        {
            cudaFuncAttributes attrs{};
            DNDS_check_throw_info(cudaFuncGetAttributes(&attrs, kernel) == cudaSuccess &&
                                      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&res.activeBlocksPerSM, kernel, threads, bytes) == cudaSuccess,
                                  "V2 reconstruction resource query failed");
            res.registersPerThread = attrs.numRegs;
        }
        return res;
    }

    template <int M, int C, bool Limited, bool MMA>
    ReconstructionSweepResources StorageRoute(const StagedSweepArguments &a, int threads)
    {
        if (a.plan.packedRawB)
            return Launch<M, C, true, Limited, MMA>(a, a.plan.rawMatrixB32, threads);
        if (a.plan.raw)
            return Launch<M, C, true, Limited, MMA>(a, a.plan.matrixAInvB, threads);
        return Launch<M, C, false, Limited, MMA>(a, a.plan.matrixAInvB, threads);
    }

    template <int M, int C>
    ReconstructionSweepResources Shape(const StagedSweepArguments &a, bool mma, int threads)
    {
        if constexpr (M > 3)
        {
            if (mma && a.limited)
                return StorageRoute<M, C, true, true>(a, threads);
            if (mma)
                return StorageRoute<M, C, false, true>(a, threads);
            if (a.limited)
                return StorageRoute<M, C, true, false>(a, threads);
        }
        return StorageRoute<M, C, false, false>(a, threads);
    }
}
