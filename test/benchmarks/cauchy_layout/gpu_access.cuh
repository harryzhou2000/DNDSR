// Included inside the probe's namespace, after View, Evaluate and Numeric.
// Cooperative candidates retain the same FP64 arithmetic and logical outputs.
#pragma once

template <int L>
__device__ void PhysicalTileCoordinate(int q, int cells, int components, int &row, int &component)
{
    if constexpr (L == 0)
    {
        row = q / components;
        component = q % components;
    }
    else if constexpr (L == 1)
    {
        row = q % cells;
        component = q / cells;
    }
    else if constexpr (L == 8 || L == 32)
    {
        row = (q / (L * components)) * L + q % L;
        component = (q / L) % components;
    }
    else
    {
        if (q < cells * 5)
        {
            row = q / 5;
            component = q % 5;
        }
        else
        {
            q -= cells * 5;
            if constexpr (L == 101)
            {
                row = q % cells;
                component = 5 + q / cells;
            }
            else
            {
                row = (q / (32 * (components - 5))) * 32 + q % 32;
                component = 5 + (q / 32) % (components - 5);
            }
        }
    }
}

struct SharedRows
{
    double *data;
    Index begin;
    int pitch;
    __device__ double operator()(Index iRow, int iComp) const { return data[(iRow - begin) * pitch + iComp]; }
    __device__ void Set(Index iRow, int iComp, double value) const { data[(iRow - begin) * pitch + iComp] = value; }
};

// Load/store in physical order, but compute one cell per thread. Odd FP64 row
// pitch avoids a power-of-two shared row stride; it is not a bank-free proof.
template <int L>
__global__ void SharedCells(int kernel, View<L> u, View<L> out, Index begin, Index end)
{
    constexpr int tile = 64;
    extern __shared__ double storage[];
    const int pitch = u.components | 1;
    const Index first = begin + Index(blockIdx.x) * tile;
    SharedRows input{storage, first, pitch}, output{storage + tile * pitch, first, pitch};
    for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
    {
        int row, component;
        PhysicalTileCoordinate<L>(q, tile, u.components, row, component);
        input.Set(first + row, component, first + row < end ? u(first + row, component) : 0.);
    }
    __syncthreads();
    const Index iRow = first + threadIdx.x;
    if (iRow < end)
        Evaluate(kernel, iRow, u.rows, u.components, input, output);
    __syncthreads();
    for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
    {
        int row, component;
        PhysicalTileCoordinate<L>(q, tile, u.components, row, component);
        if (first + row < end)
            out.Set(first + row, component, output(first + row, component));
    }
}

// Neighbor vectors are staged too: loading only current rows would leave the
// dominant irregular stencil traffic unchanged. Each group computes one row.
template <int L, int Group>
__global__ void SharedStencil(View<L> u, View<L> out, Index begin, Index end)
{
    constexpr int tile = 256 / Group;
    extern __shared__ double storage[];
    const int pitch = u.components | 1, plane = tile * pitch;
    const Index first = begin + Index(blockIdx.x) * tile;
    for (int neighbor = 0; neighbor < 7; ++neighbor)
        for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
        {
            int row, component;
            // Small tiles can be shorter than an AoSoA physical tile. Within
            // that partial tile components are still traversed before rows.
            if constexpr (L == 0)
            {
                row = q / u.components;
                component = q % u.components;
            }
            else
            {
                row = q % tile;
                component = q / tile;
            }
            const Index iRow = first + row;
            const Index source = neighbor == 0 ? iRow : (iRow + 1 + ((iRow * 13 + (neighbor - 1) * 7919) % 16381)) % u.rows;
            storage[neighbor * plane + row * pitch + component] = iRow < end ? u(source, component) : 0.;
        }
    __syncthreads();
    const int row = threadIdx.x / Group, lane = threadIdx.x % Group;
    for (int component = lane; component < u.components; component += Group)
    {
        const double center = storage[row * pitch + component];
        double sum = 0., bound = fabs(center);
        for (int neighbor = 0; neighbor < 6; ++neighbor)
        {
            const double value = storage[(neighbor + 1) * plane + row * pitch + component];
            const double cross = storage[(neighbor + 1) * plane + row * pitch + (component + 1) % u.components];
            sum += (value - center) * (.2 / (neighbor + 1)) + .007 * cross;
            bound = fmax(bound, fabs(value));
        }
        const double limited = fabs(sum) > .2 * bound ? sum * (.2 * bound / fabs(sum)) : sum;
        storage[7 * plane + row * pitch + component] = center + limited;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
    {
        const int row = L == 0 ? q / u.components : q % tile;
        const int component = L == 0 ? q % u.components : q / tile;
        if (first + row < end)
            out.Set(first + row, component, storage[7 * plane + row * pitch + component]);
    }
}

template <int L, int Group>
__global__ void GroupCells(int kernel, View<L> u, View<L> out, Index begin, Index end)
{
    const Index row = begin + Index(blockIdx.x) * (256 / Group) + threadIdx.x / Group;
    const int lane = threadIdx.x % Group;
    const bool active = row < end;
    // Do not return for inactive tail groups: all lanes participate in shuffles.
    if (kernel == 0)
    {
        double rho = 0., mx = 0., p = 0., t = 0., sound = 0., mu = 0., source = 0.;
        if (lane == 0 && active)
        {
            rho = u(row, 0);
            mx = u(row, 1);
            const double my = u(row, 2), mz = u(row, 3);
            p = .4 * (u(row, 4) - .5 * (mx * mx + my * my + mz * mz) / rho);
            t = p / rho;
            sound = sqrt(1.4 * t);
            mu = pow(t, 1.5) / (t + .4);
            if (u.components > 5)
                source = exp(-1. / t);
        }
        rho = __shfl_sync(0xffffffff, rho, 0, Group);
        mx = __shfl_sync(0xffffffff, mx, 0, Group);
        p = __shfl_sync(0xffffffff, p, 0, Group);
        t = __shfl_sync(0xffffffff, t, 0, Group);
        sound = __shfl_sync(0xffffffff, sound, 0, Group);
        mu = __shfl_sync(0xffffffff, mu, 0, Group);
        source = __shfl_sync(0xffffffff, source, 0, Group);
        if (active)
            for (int component = lane; component < u.components; component += Group)
            {
                double value;
                if (component == 0)
                    value = rho * sound;
                else if (component == 1)
                    value = mx * sound + p;
                else if (component == 2)
                    value = u(row, 2) * sound + mu;
                else if (component == 3)
                    value = u(row, 3) * sound + p * .3;
                else if (component == 4)
                    value = (u(row, 4) + p) * mx / rho + mu * t;
                else
                {
                    const double y = u(row, component);
                    value = y * sound + .03 * source * (1. - y);
                }
                out.Set(row, component, value);
            }
    }
    else if (kernel == 1)
    {
        if (active)
            for (int component = lane; component < u.components; component += Group)
            {
                const double center = u(row, component);
                double sum = 0., bound = fabs(center);
                for (int j = 0; j < 6; ++j)
                {
                    const Index nb = (row + 1 + ((row * 13 + j * 7919) % 16381)) % u.rows;
                    const double value = u(nb, component), cross = u(nb, (component + 1) % u.components);
                    sum += (value - center) * (.2 / (j + 1)) + .007 * cross;
                    bound = fmax(bound, fabs(value));
                }
                const double limited = fabs(sum) > .2 * bound ? sum * (.2 * bound / fabs(sum)) : sum;
                out.Set(row, component, center + limited);
            }
    }
    else
    {
        double v[5]{};
        if (lane == 0 && active)
        {
            for (int j = 0; j < 5; ++j)
            {
                double rhs = u(row, j) * 1.1;
                for (int k = 0; k < j; ++k)
                    rhs -= (.02 * (1 + (j + k) % 3)) * v[k];
                v[j] = rhs / (1.4 + .1 * j);
            }
            for (int j = 4; j >= 0; --j)
            {
                double rhs = v[j];
                for (int k = j + 1; k < 5; ++k)
                    rhs -= .025 * v[k];
                v[j] = rhs / 1.2;
            }
        }
        for (int j = 0; j < 5; ++j)
            v[j] = __shfl_sync(0xffffffff, v[j], 0, Group);
        if (active)
            for (int component = lane; component < u.components; component += Group)
            {
                const double x = u(row, component);
                if (component < 5)
                {
                    // Keep the five solved values in registers rather than a
                    // dynamically indexed local-memory array per lane.
                    double solved = 0.;
#pragma unroll
                    for (int j = 0; j < 5; ++j)
                        if (component == j)
                            solved = v[j];
                    out.Set(row, component, x + .1 * solved);
                }
                else
                {
                    const double coupling = .01 * (v[0] + v[4]) + .02 * u(row, component - 1);
                    out.Set(row, component, x + .1 * (x - coupling) / (1.5 + .01 * component));
                }
            }
    }
}

// Four tiles per block bound the pipeline and retain enough grid-level work.
// Async copies go only global->shared. Output still uses cooperative stores.
// Variant Async=false has identical buffers, loop shape and arithmetic, serving
// as the control for attributing any benefit specifically to cp.async.
template <int L, bool Async>
__device__ void PrefetchCells(View<L> u, double *target, int tile, int pitch, Index first, Index end)
{
    for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
    {
        int row, component;
        PhysicalTileCoordinate<L>(q, tile, u.components, row, component);
        auto *destination = target + row * pitch + component;
        if (first + row < end)
        {
            const auto *source = u.data + u.Offset(first + row, component);
            if constexpr (Async)
                __pipeline_memcpy_async(destination, source, sizeof(double));
            else
                *destination = *source;
        }
        else
            *destination = 0.;
    }
    if constexpr (Async)
        __pipeline_commit();
}

template <int L, bool Async>
__global__ void PipelinedCells(int kernel, View<L> u, View<L> out, Index begin, Index end)
{
    constexpr int tile = 64, tiles = 4;
    extern __shared__ double storage[];
    const int pitch = u.components | 1, plane = tile * pitch;
    const Index blockFirst = begin + Index(blockIdx.x) * tile * tiles;
    PrefetchCells<L, Async>(u, storage, tile, pitch, blockFirst, end);
    if constexpr (Async)
        __pipeline_wait_prior(0);
    __syncthreads();
    for (int step = 0; step < tiles; ++step)
    {
        const Index first = blockFirst + step * tile;
        if (first >= end)
            break; // Uniform condition for the entire block.
        if (step + 1 < tiles && first + tile < end)
            PrefetchCells<L, Async>(u, storage + ((step + 1) % 2) * plane, tile, pitch, first + tile, end);
        SharedRows input{storage + (step % 2) * plane, first, pitch};
        SharedRows output{storage + 2 * plane, first, pitch};
        if (first + threadIdx.x < end)
            Evaluate(kernel, first + threadIdx.x, u.rows, u.components, input, output);
        __syncthreads();
        for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
        {
            int row, component;
            PhysicalTileCoordinate<L>(q, tile, u.components, row, component);
            if (first + row < end)
                out.Set(first + row, component, output(first + row, component));
        }
        // Finish reading output/current buffers before next reuse. The async
        // wait is device-side and does not introduce a host synchronization.
        __syncthreads();
        if constexpr (Async)
            __pipeline_wait_prior(0);
        __syncthreads();
    }
}

template <int L, int Group, bool Async>
__device__ void PrefetchStencil(View<L> u, double *target, int pitch, Index first, Index end)
{
    constexpr int tile = 256 / Group;
    const int plane = tile * pitch;
    for (int neighbor = 0; neighbor < 7; ++neighbor)
        for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
        {
            const int row = L == 0 ? q / u.components : q % tile;
            const int component = L == 0 ? q % u.components : q / tile;
            const Index iRow = first + row;
            const Index source = neighbor == 0 ? iRow : (iRow + 1 + ((iRow * 13 + (neighbor - 1) * 7919) % 16381)) % u.rows;
            auto *destination = target + neighbor * plane + row * pitch + component;
            if (iRow < end)
            {
                const auto *value = u.data + u.Offset(source, component);
                if constexpr (Async)
                    __pipeline_memcpy_async(destination, value, sizeof(double));
                else
                    *destination = *value;
            }
            else
                *destination = 0.;
        }
    if constexpr (Async)
        __pipeline_commit();
}

template <int L, int Group, bool Async>
__global__ void PipelinedStencil(View<L> u, View<L> out, Index begin, Index end)
{
    constexpr int tile = 256 / Group, tiles = 4;
    extern __shared__ double storage[];
    const int pitch = u.components | 1, plane = tile * pitch;
    const Index blockFirst = begin + Index(blockIdx.x) * tile * tiles;
    PrefetchStencil<L, Group, Async>(u, storage, pitch, blockFirst, end);
    if constexpr (Async)
        __pipeline_wait_prior(0);
    __syncthreads();
    for (int step = 0; step < tiles; ++step)
    {
        const Index first = blockFirst + step * tile;
        if (first >= end)
            break;
        if (step + 1 < tiles && first + tile < end)
            PrefetchStencil<L, Group, Async>(u, storage + ((step + 1) % 2) * 7 * plane, pitch, first + tile, end);
        const double *input = storage + (step % 2) * 7 * plane;
        double *output = storage + 14 * plane;
        const int row = threadIdx.x / Group, lane = threadIdx.x % Group;
        for (int component = lane; component < u.components; component += Group)
        {
            const double center = input[row * pitch + component];
            double sum = 0., bound = fabs(center);
            for (int neighbor = 0; neighbor < 6; ++neighbor)
            {
                const double value = input[(neighbor + 1) * plane + row * pitch + component];
                const double cross = input[(neighbor + 1) * plane + row * pitch + (component + 1) % u.components];
                sum += (value - center) * (.2 / (neighbor + 1)) + .007 * cross;
                bound = fmax(bound, fabs(value));
            }
            const double limited = fabs(sum) > .2 * bound ? sum * (.2 * bound / fabs(sum)) : sum;
            output[row * pitch + component] = center + limited;
        }
        __syncthreads();
        for (int q = threadIdx.x; q < tile * u.components; q += blockDim.x)
        {
            const int row = L == 0 ? q / u.components : q % tile;
            const int component = L == 0 ? q % u.components : q / tile;
            if (first + row < end)
                out.Set(first + row, component, output[row * pitch + component]);
        }
        __syncthreads();
        if constexpr (Async)
            __pipeline_wait_prior(0);
        __syncthreads();
    }
}

// One block handles exactly one cell. Components and neighbor loads are
// distributed over the block; thread 0 performs the five-equation dependency
// chain and shares its results. This is deliberately the SAME small workload,
// not an inflated matrix introduced to make block-per-cell look favorable.
template <int L>
__global__ void BlockCell(int kernel, View<L> u, View<L> out, Index begin, Index end)
{
    const Index row = begin + blockIdx.x;
    if (row >= end)
        return; // Whole block is inactive, never partial barrier entry.
    extern __shared__ double storage[];
    const int nc = u.components, pitch = nc | 1;
    const int planes = kernel == 1 ? 7 : 1;
    for (int q = threadIdx.x; q < planes * nc; q += blockDim.x)
    {
        const int plane = q / nc, component = q % nc;
        const Index source = plane == 0 ? row : (row + 1 + ((row * 13 + (plane - 1) * 7919) % 16381)) % u.rows;
        storage[plane * pitch + component] = u(source, component);
    }
    double *scalars = storage + 7 * pitch;
    __syncthreads();
    if (kernel == 0)
    {
        if (threadIdx.x == 0)
        {
            const double rho = storage[0], mx = storage[1], my = storage[2], mz = storage[3];
            const double p = .4 * (storage[4] - .5 * (mx * mx + my * my + mz * mz) / rho);
            const double t = p / rho;
            scalars[0] = rho;
            scalars[1] = mx;
            scalars[2] = p;
            scalars[3] = t;
            scalars[4] = sqrt(1.4 * t);
            scalars[5] = pow(t, 1.5) / (t + .4);
            scalars[6] = nc > 5 ? exp(-1. / t) : 0.;
        }
        __syncthreads();
        for (int component = threadIdx.x; component < nc; component += blockDim.x)
        {
            const double rho = scalars[0], mx = scalars[1], p = scalars[2], t = scalars[3], sound = scalars[4], mu = scalars[5];
            double value;
            if (component == 0)
                value = rho * sound;
            else if (component == 1)
                value = mx * sound + p;
            else if (component == 2)
                value = storage[2] * sound + mu;
            else if (component == 3)
                value = storage[3] * sound + p * .3;
            else if (component == 4)
                value = (storage[4] + p) * mx / rho + mu * t;
            else
            {
                const double y = storage[component];
                value = y * sound + .03 * scalars[6] * (1. - y);
            }
            out.Set(row, component, value);
        }
    }
    else if (kernel == 1)
    {
        for (int component = threadIdx.x; component < nc; component += blockDim.x)
        {
            const double center = storage[component];
            double sum = 0., bound = fabs(center);
            for (int neighbor = 0; neighbor < 6; ++neighbor)
            {
                const double value = storage[(neighbor + 1) * pitch + component];
                const double cross = storage[(neighbor + 1) * pitch + (component + 1) % nc];
                sum += (value - center) * (.2 / (neighbor + 1)) + .007 * cross;
                bound = fmax(bound, fabs(value));
            }
            const double limited = fabs(sum) > .2 * bound ? sum * (.2 * bound / fabs(sum)) : sum;
            out.Set(row, component, center + limited);
        }
    }
    else
    {
        if (threadIdx.x == 0)
        {
            for (int j = 0; j < 5; ++j)
            {
                double rhs = storage[j] * 1.1;
                for (int k = 0; k < j; ++k)
                    rhs -= (.02 * (1 + (j + k) % 3)) * scalars[k];
                scalars[j] = rhs / (1.4 + .1 * j);
            }
            for (int j = 4; j >= 0; --j)
            {
                double rhs = scalars[j];
                for (int k = j + 1; k < 5; ++k)
                    rhs -= .025 * scalars[k];
                scalars[j] = rhs / 1.2;
            }
        }
        __syncthreads();
        for (int component = threadIdx.x; component < nc; component += blockDim.x)
        {
            const double x = storage[component];
            if (component < 5)
                out.Set(row, component, x + .1 * scalars[component]);
            else
            {
                const double coupling = .01 * (scalars[0] + scalars[4]) + .02 * storage[component - 1];
                out.Set(row, component, x + .1 * (x - coupling) / (1.5 + .01 * component));
            }
        }
    }
}

struct KernelConfiguration
{
    const void *function;
    int threads, cells;
    size_t shared;
};

template <int L>
KernelConfiguration AccessConfiguration(int variant, int kernel, int components)
{
    if (variant == 0)
        return {reinterpret_cast<const void *>(Numeric<L>), 256, 256, 0};
    if (variant == 8)
        return {reinterpret_cast<const void *>(GroupCells<L, 8>), 256, 32, 0};
    if (variant == 32)
        return {reinterpret_cast<const void *>(GroupCells<L, 32>), 256, 8, 0};
    if (variant == 64 || variant == 128 || variant == 256 || variant == 1256)
        return {reinterpret_cast<const void *>(BlockCell<L>), variant == 1256 ? 256 : variant, 1,
                variant == 1256 ? exclusiveBlockSharedBytes : size_t(7 * (components | 1) + 8) * sizeof(double)};
    if (variant == 2 || variant == 3)
    {
        if (kernel != 1)
            return {variant == 2 ? reinterpret_cast<const void *>(PipelinedCells<L, true>) : reinterpret_cast<const void *>(PipelinedCells<L, false>),
                    64, 256, size_t(3 * 64 * (components | 1)) * sizeof(double)};
        if (components <= 8)
            return {variant == 2 ? reinterpret_cast<const void *>(PipelinedStencil<L, 8, true>) : reinterpret_cast<const void *>(PipelinedStencil<L, 8, false>),
                    256, 128, size_t(15 * 32 * (components | 1)) * sizeof(double)};
        return {variant == 2 ? reinterpret_cast<const void *>(PipelinedStencil<L, 32, true>) : reinterpret_cast<const void *>(PipelinedStencil<L, 32, false>),
                256, 32, size_t(15 * 8 * (components | 1)) * sizeof(double)};
    }
    if (kernel != 1)
        return {reinterpret_cast<const void *>(SharedCells<L>), 64, 64, size_t(2 * 64 * (components | 1)) * sizeof(double)};
    if (components <= 8)
        return {reinterpret_cast<const void *>(SharedStencil<L, 8>), 256, 32, size_t(8 * 32 * (components | 1)) * sizeof(double)};
    return {reinterpret_cast<const void *>(SharedStencil<L, 32>), 256, 8, size_t(8 * 8 * (components | 1)) * sizeof(double)};
}

template <int L>
void ReportAccessResources(int variant, int kernel, int components)
{
    const auto config = AccessConfiguration<L>(variant, kernel, components);
    if (config.shared > 48 * 1024)
        CUDA(cudaFuncSetAttribute(config.function, cudaFuncAttributeMaxDynamicSharedMemorySize, int(config.shared)));
    cudaFuncAttributes attrs;
    CUDA(cudaFuncGetAttributes(&attrs, config.function));
    int blocks;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, config.function, config.threads, config.shared));
    if (variant == 1256)
        Check(blocks == 1, "exclusive block probe must permit exactly one resident block per SM");
    std::fprintf(stderr, "resources variant=%d layout=%d kernel=%d components=%d threads=%d cells=%d regs=%d local=%zu static_shared=%zu dynamic_shared=%zu blocks_per_sm=%d\n",
                 variant, L, kernel, components, config.threads, config.cells, attrs.numRegs, attrs.localSizeBytes, attrs.sharedSizeBytes, config.shared, blocks);
}

template <int L>
void LaunchAccess(int variant, int kernel, View<L> u, View<L> out, Index begin, Index end, cudaStream_t stream)
{
    const auto config = AccessConfiguration<L>(variant, kernel, u.components);
    const unsigned blocks = unsigned((end - begin + config.cells - 1) / config.cells);
    if (variant == 0)
        Numeric<<<blocks, config.threads, 0, stream>>>(kernel, u, out, begin, end);
    else if (variant == 8)
        GroupCells<L, 8><<<blocks, config.threads, 0, stream>>>(kernel, u, out, begin, end);
    else if (variant == 32)
        GroupCells<L, 32><<<blocks, config.threads, 0, stream>>>(kernel, u, out, begin, end);
    else if (variant == 64 || variant == 128 || variant == 256 || variant == 1256)
        BlockCell<<<blocks, config.threads, config.shared, stream>>>(kernel, u, out, begin, end);
    else if (variant == 2 && kernel != 1)
        PipelinedCells<L, true><<<blocks, config.threads, config.shared, stream>>>(kernel, u, out, begin, end);
    else if (variant == 3 && kernel != 1)
        PipelinedCells<L, false><<<blocks, config.threads, config.shared, stream>>>(kernel, u, out, begin, end);
    else if (variant == 2 && u.components <= 8)
        PipelinedStencil<L, 8, true><<<blocks, config.threads, config.shared, stream>>>(u, out, begin, end);
    else if (variant == 3 && u.components <= 8)
        PipelinedStencil<L, 8, false><<<blocks, config.threads, config.shared, stream>>>(u, out, begin, end);
    else if (variant == 2)
        PipelinedStencil<L, 32, true><<<blocks, config.threads, config.shared, stream>>>(u, out, begin, end);
    else if (variant == 3)
        PipelinedStencil<L, 32, false><<<blocks, config.threads, config.shared, stream>>>(u, out, begin, end);
    else if (kernel != 1)
        SharedCells<<<blocks, config.threads, config.shared, stream>>>(kernel, u, out, begin, end);
    else if (u.components <= 8)
        SharedStencil<L, 8><<<blocks, config.threads, config.shared, stream>>>(u, out, begin, end);
    else
        SharedStencil<L, 32><<<blocks, config.threads, config.shared, stream>>>(u, out, begin, end);
    CUDA(cudaGetLastError());
}
