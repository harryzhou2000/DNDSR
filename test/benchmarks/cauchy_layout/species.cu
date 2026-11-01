// Simulated coupled flow/species arithmetic, not a chemical mechanism solver.
#ifndef CAUCHY_SPECIES_CPU
#    include <cuda_runtime.h>
#    include <cuda_pipeline.h>
#    define SPECIES_HD __host__ __device__
#else
#    define SPECIES_HD
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using Index = long long;
Index planePadding = 0;
#ifndef CAUCHY_SPECIES_CPU
void CUDA(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
#endif
void Check(bool ok, const char *s)
{
    if (!ok)
        throw std::runtime_error(s);
}
double Clock() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <int L>
struct Rows
{
    double *data;
    Index n;
    int width;
    Index pitch;
    SPECIES_HD Rows(double *p, Index count, int columns, Index plane = 0)
        : data(p), n(count), width(columns), pitch(plane ? plane : count) {}
    SPECIES_HD Index Offset(Index row, int component) const
    {
        if constexpr (L == 0)
            return row * width + component;
        else if constexpr (L == 1)
            return Index(component) * pitch + row;
        else
            return (row / L) * (width * L) + component * L + row % L;
    }
    SPECIES_HD Index Elements() const
    {
        if constexpr (L == 0)
            return n * width;
        else if constexpr (L == 1)
            return pitch * width;
        else
            return ((n + L - 1) / L) * L * width;
    }
    SPECIES_HD double operator()(Index row, int c) const { return data[Offset(row, c)]; }
    SPECIES_HD void Set(Index row, int c, double x) const { data[Offset(row, c)] = x; }
};

template <int L, bool Split>
struct State
{
    Rows<L> values;
    Rows<0> flow;
    int ns;
    SPECIES_HD double operator()(Index row, int c) const
    {
        if constexpr (Split)
            return c < 5 ? flow(row, c) : values(row, c - 5);
        return values(row, c);
    }
    SPECIES_HD void Set(Index row, int c, double x) const
    {
        if constexpr (Split)
        {
            if (c < 5)
                flow.Set(row, c, x);
            else
                values.Set(row, c - 5, x);
        }
        else
            values.Set(row, c, x);
    }
};

struct Analytic
{
    int ns;
    SPECIES_HD double operator()(Index row, int c) const
    {
        const double phase = double((row * 17) % 1009) / 1009.;
        if (c == 0)
            return 1. + .1 * phase;
        if (c < 4)
            return .12 + .03 * phase;
        if (c == 4)
            return 3. + .1 * phase;
        return (1. + .02 * (c - 5) + .02 * phase) /
               (ns * (1. + .02 * phase) + .01 * ns * (ns - 1));
    }
};

struct Mixture
{
    double rho, mx, pressure, temperature, cp, mol;
};
template <class Input>
SPECIES_HD Mixture Mix(Input u, Index row, int ns)
{
    double cp = 0., mol = 0.;
    for (int k = 0; k < ns; ++k)
    {
        const double y = u(row, 5 + k);
        cp += y * (1. + .03 * k);
        mol += y / (2. + .5 * k);
    }
    const double rho = u(row, 0), mx = u(row, 1), my = u(row, 2), mz = u(row, 3);
    const double pressure = .4 * (u(row, 4) - .5 * (mx * mx + my * my + mz * mz) / rho);
    return {rho, mx, pressure, pressure / (rho * (1. + .2 * cp)), cp, mol};
}
template <class Input>
SPECIES_HD double Component(int kernel, int c, Input u, Index row, int ns, Mixture m)
{
    if (c < 5)
    {
        if (c == 0)
            return m.rho * sqrt(1.4 * m.temperature);
        if (c < 4)
            return u(row, c) * (1. + .01 * m.cp) + m.pressure * (.1 * c);
        return (u(row, 4) + m.pressure) * m.mx / m.rho + .03 * m.cp * m.mol;
    }
    const int k = c - 5;
    const double y = u(row, c);
    if (kernel == 0)
    {
        const double diffusion = .01 * pow(m.temperature, 1.5) / (1. + .05 * k);
        const double rate = exp(-(.5 + .03 * k) / m.temperature);
        return y + diffusion * (u(row, 5 + (k + 1) % ns) - y) + .001 * rate * (1. - y) + .002 * m.mol * y;
    }
    double action = 0.;
    for (int j = 0; j < ns; ++j)
    {
        const double a = (j == k ? 1.5 : 0.) + .002 * (((k * 7 + j * 11) % 13) - 6);
        action += a * u(row, 5 + j);
    }
    return .95 * y + .002 * action + .001 * m.rho * exp(-(1. + .02 * k) / m.temperature) * action;
}

#ifndef CAUCHY_SPECIES_CPU
template <class S>
__global__ void Direct(int kernel, S u, S out, Index n)
{
    const Index row = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n)
        return;
    const auto m = Mix(u, row, u.ns);
    for (int c = 0; c < 5 + u.ns; ++c)
        out.Set(row, c, Component(kernel, c, u, row, u.ns, m));
}
template <int G, class S>
__global__ void Group(int kernel, S u, S out, Index n)
{
    const Index row = Index(blockIdx.x) * (256 / G) + threadIdx.x / G;
    const int lane = threadIdx.x % G;
    Mixture m{};
    if (row < n && lane == 0)
        m = Mix(u, row, u.ns);
    // Full warp remains active for tail groups; readiness is per subgroup.
    m.rho = __shfl_sync(0xffffffff, m.rho, 0, G);
    m.mx = __shfl_sync(0xffffffff, m.mx, 0, G);
    m.pressure = __shfl_sync(0xffffffff, m.pressure, 0, G);
    m.temperature = __shfl_sync(0xffffffff, m.temperature, 0, G);
    m.cp = __shfl_sync(0xffffffff, m.cp, 0, G);
    m.mol = __shfl_sync(0xffffffff, m.mol, 0, G);
    if (row < n)
        for (int c = lane; c < 5 + u.ns; c += G)
            out.Set(row, c, Component(kernel, c, u, row, u.ns, m));
}

struct SharedInput
{
    double *data;
    Index first;
    int pitch;
    __device__ double operator()(Index row, int c) const { return data[(row - first) * pitch + c]; }
};
template <int L, bool Split>
__device__ void TileCoordinates(int q, int tile, int ns, int &row, int &c)
{
    const int width = ns + 5;
    if constexpr (!Split && L == 0)
    {
        row = q / width;
        c = q % width;
    }
    else if constexpr (!Split && L == 1)
    {
        row = q % tile;
        c = q / tile;
    }
    else if constexpr (!Split)
    {
        row = (q / (L * width)) * L + q % L;
        c = (q / L) % width;
    }
    else
    {
        if (q < tile * 5)
        {
            row = q / 5;
            c = q % 5;
        }
        else
        {
            q -= tile * 5;
            if constexpr (L == 0)
            {
                row = q / ns;
                c = 5 + q % ns;
            }
            else if constexpr (L == 1)
            {
                row = q % tile;
                c = 5 + q / tile;
            }
            else
            {
                row = (q / (L * ns)) * L + q % L;
                c = 5 + (q / L) % ns;
            }
        }
    }
}
template <int L, bool Split, bool Async>
__device__ void Load(State<L, Split> u, double *destination, int pitch, Index first, Index n)
{
    for (int q = threadIdx.x; q < 64 * (u.ns + 5); q += blockDim.x)
    {
        int row, c;
        TileCoordinates<L, Split>(q, 64, u.ns, row, c);
        double *target = destination + row * pitch + c;
        if (first + row < n)
        {
            const double *source;
            if constexpr (Split)
                source = c < 5 ? u.flow.data + u.flow.Offset(first + row, c) : u.values.data + u.values.Offset(first + row, c - 5);
            else
                source = u.values.data + u.values.Offset(first + row, c);
            if constexpr (Async)
                __pipeline_memcpy_async(target, source, sizeof(double));
            else
                *target = *source;
        }
        else
            *target = 0.;
    }
    if constexpr (Async)
        __pipeline_commit();
}
template <int L, bool Split, bool Async>
__global__ void Tiled(int kernel, State<L, Split> u, State<L, Split> out, Index n)
{
    constexpr int tiles = Async ? 4 : 1;
    extern __shared__ double storage[];
    const int width = 5 + u.ns, pitch = width | 1, plane = 64 * pitch;
    const Index firstBlock = Index(blockIdx.x) * 64 * tiles;
    Load<L, Split, Async>(u, storage, pitch, firstBlock, n);
    if constexpr (Async)
        __pipeline_wait_prior(0);
    __syncthreads();
    for (int step = 0; step < tiles; ++step)
    {
        const Index first = firstBlock + step * 64;
        if (first >= n)
            break;
        if constexpr (Async)
            if (step + 1 < tiles && first + 64 < n)
                Load<L, Split, true>(u, storage + ((step + 1) % 2) * plane, pitch, first + 64, n);
        SharedInput input{storage + (Async ? step % 2 : 0) * plane, first, pitch};
        double *result = storage + (Async ? 2 : 1) * plane;
        const Index row = first + threadIdx.x;
        if (row < n)
        {
            const auto m = Mix(input, row, u.ns);
            for (int c = 0; c < width; ++c)
                result[threadIdx.x * pitch + c] = Component(kernel, c, input, row, u.ns, m);
        }
        __syncthreads();
        for (int q = threadIdx.x; q < 64 * width; q += blockDim.x)
        {
            int local, c;
            TileCoordinates<L, Split>(q, 64, u.ns, local, c);
            if (first + local < n)
                out.Set(first + local, c, result[local * pitch + c]);
        }
        __syncthreads();
        if constexpr (Async)
            __pipeline_wait_prior(0);
        __syncthreads();
    }
}
template <class S>
__global__ void Block(int kernel, S u, S out, Index n)
{
    const Index row = blockIdx.x;
    if (row >= n)
        return;
    extern __shared__ double storage[];
    for (int c = threadIdx.x; c < u.ns + 5; c += blockDim.x)
        storage[c] = u(row, c);
    __shared__ Mixture mixture;
    __syncthreads();
    SharedInput input{storage, row, u.ns + 5};
    if (threadIdx.x == 0)
        mixture = Mix(input, row, u.ns);
    __syncthreads();
    for (int c = threadIdx.x; c < u.ns + 5; c += blockDim.x)
        out.Set(row, c, Component(kernel, c, input, row, u.ns, mixture));
}

struct Allocation
{
    double *p;
    Allocation(Index n) { CUDA(cudaMalloc(&p, size_t(std::max<Index>(n, 1)) * 8)); }
    ~Allocation() { cudaFree(p); }
};
template <int L, bool Split>
void Run(Index n, int ns, int variant, int samples)
{
    const int width = ns + 5;
    Rows<L> v{nullptr, n, Split ? ns : width, n + planePadding};
    Rows<0> f{nullptr, n, 5};
    Allocation data(v.Elements()), result(v.Elements()), flow(Split ? f.Elements() : 0), resultFlow(Split ? f.Elements() : 0);
    State<L, Split> u{{data.p, n, v.width, v.pitch}, {flow.p, n, 5}, ns}, out{{result.p, n, v.width, v.pitch}, {resultFlow.p, n, 5}, ns};
    std::vector<double> h(size_t(v.Elements()), -12345.), hf(size_t(Split ? f.Elements() : 1), -12345.);
    State<L, Split> host{{h.data(), n, v.width, v.pitch}, {hf.data(), n, 5}, ns};
    for (Index row = 0; row < n; ++row)
        for (int c = 0; c < width; ++c)
            host.Set(row, c, Analytic{ns}(row, c));
    CUDA(cudaMemcpy(data.p, h.data(), h.size() * 8, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(result.p, h.data(), h.size() * 8, cudaMemcpyHostToDevice));
    if constexpr (Split)
    {
        CUDA(cudaMemcpy(flow.p, hf.data(), hf.size() * 8, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(resultFlow.p, hf.data(), hf.size() * 8, cudaMemcpyHostToDevice));
    }
    for (int kernel = 0; kernel < 2; ++kernel)
    {
        const int pitch = width | 1;
        const int threads = variant == 1 || variant == 2 ? 64 : variant == 128 ? 128
                                                                               : 256;
        const int cells = variant == 0 ? 256 : variant == 8 ? 32
                                           : variant == 32  ? 8
                                           : variant == 1   ? 64
                                           : variant == 2   ? 256
                                                            : 1;
        const size_t shared = variant == 1 ? size_t(2 * 64 * pitch) * 8 : variant == 2 ? size_t(3 * 64 * pitch) * 8
                                                                      : variant == 128 ? size_t(width) * 8
                                                                                       : 0;
        const void *function = variant == 0 ? reinterpret_cast<const void *>(Direct<State<L, Split>>) : variant == 8 ? reinterpret_cast<const void *>(Group<8, State<L, Split>>)
                                                                                                    : variant == 32  ? reinterpret_cast<const void *>(Group<32, State<L, Split>>)
                                                                                                    : variant == 1   ? reinterpret_cast<const void *>(Tiled<L, Split, false>)
                                                                                                    : variant == 2   ? reinterpret_cast<const void *>(Tiled<L, Split, true>)
                                                                                                                     : reinterpret_cast<const void *>(Block<State<L, Split>>);
        if (shared > 48 * 1024)
            CUDA(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
        cudaFuncAttributes attrs;
        int blocks;
        CUDA(cudaFuncGetAttributes(&attrs, function));
        CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, function, threads, shared));
        std::fprintf(stderr, "resources layout=%d split=%d variant=%d kernel=%d species=%d regs=%d local=%zu shared=%zu blocks=%d threads=%d\n", L, int(Split), variant, kernel, ns, attrs.numRegs, attrs.localSizeBytes, shared, blocks, threads);
        auto launch = [&]
        {
            const unsigned grid = unsigned((n + cells - 1) / cells);
            if (variant == 0)
                Direct<<<grid, threads>>>(kernel, u, out, n);
            else if (variant == 8)
                Group<8><<<grid, threads>>>(kernel, u, out, n);
            else if (variant == 32)
                Group<32><<<grid, threads>>>(kernel, u, out, n);
            else if (variant == 1)
                Tiled<L, Split, false><<<grid, threads, shared>>>(kernel, u, out, n);
            else if (variant == 2)
                Tiled<L, Split, true><<<grid, threads, shared>>>(kernel, u, out, n);
            else
                Block<<<grid, threads, shared>>>(kernel, u, out, n);
            CUDA(cudaGetLastError());
        };
        launch();
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaMemcpy(h.data(), result.p, h.size() * 8, cudaMemcpyDeviceToHost));
        if constexpr (Split)
            CUDA(cudaMemcpy(hf.data(), resultFlow.p, hf.size() * 8, cudaMemcpyDeviceToHost));
        double error = 0.;
        const Index probes = std::min<Index>(257, n);
        for (Index q = 0; q < probes; ++q)
        {
            const Index row = q * (n - 1) / std::max<Index>(1, probes - 1);
            const auto m = Mix(Analytic{ns}, row, ns);
            for (int c = 0; c < width; ++c)
            {
                const double expected = Component(kernel, c, Analytic{ns}, row, ns, m), got = host(row, c);
                Check(std::isfinite(got), "nonfinite species result");
                error = std::max(error, fabs(got - expected) / (1. + fabs(expected)));
            }
        }
        Check(error < 5e-12, "species indexing/arithmetic mismatch");
        if (n <= 257)
        {
            std::vector<unsigned char> logical(size_t(v.Elements()), 0);
            for (Index iRow = 0; iRow < n; ++iRow)
                for (int c = 0; c < v.width; ++c)
                    logical[size_t(v.Offset(iRow, c))] = 1;
            for (Index position = 0; position < v.Elements(); ++position)
                if (!logical[size_t(position)])
                    Check(h[size_t(position)] == -12345., "species kernel overwrote padding");
        }
        cudaEvent_t start, stop;
        CUDA(cudaEventCreate(&start));
        CUDA(cudaEventCreate(&stop));
        for (int sample = 0; sample < samples; ++sample)
        {
            constexpr int repeats = 4;
            const double wall = Clock();
            CUDA(cudaEventRecord(start));
            for (int r = 0; r < repeats; ++r)
                launch();
            CUDA(cudaEventRecord(stop));
            CUDA(cudaEventSynchronize(stop));
            float milliseconds;
            CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
            std::printf("species,%d,%d,%lld,%d,%d,k%d,%d,%.9g,%.9g,%.3g,%lld,%lld\n", L, int(Split), n, ns, variant, kernel + 1, sample,
                        milliseconds * .001 / repeats, (Clock() - wall) / repeats, error,
                        (long long)(2 * (v.Elements() + (Split ? f.Elements() : 0)) * 8), (long long)(L == 1 ? planePadding : 0));
        }
        CUDA(cudaEventDestroy(start));
        CUDA(cudaEventDestroy(stop));
    }
}

#else
// CPU row-oriented reference implementation. Identical physical layouts and
// arithmetic make CPU/GPU and full/split comparisons explicit; no OMP here.
template <int L, bool Split>
void Run(Index n, int ns, int variant, int samples)
{
    Check(variant == 0, "CPU species supports direct row loops only");
    const int width = ns + 5;
    Rows<L> shape{nullptr, n, Split ? ns : width, n + planePadding};
    std::vector<double> data(size_t(shape.Elements()), -12345.), result(data);
    std::vector<double> flow(size_t(Split ? n * 5 : 1), -12345.), resultFlow(flow);
    State<L, Split> u{{data.data(), n, shape.width, shape.pitch}, {flow.data(), n, 5}, ns};
    State<L, Split> out{{result.data(), n, shape.width, shape.pitch}, {resultFlow.data(), n, 5}, ns};
    for (Index iRow = 0; iRow < n; ++iRow)
        for (int c = 0; c < width; ++c)
            u.Set(iRow, c, Analytic{ns}(iRow, c));
    for (int kernel = 0; kernel < 2; ++kernel)
    {
        auto launch = [&]
        {
            for (Index iRow = 0; iRow < n; ++iRow)
            {
                const auto mixture = Mix(u, iRow, ns);
                for (int c = 0; c < width; ++c)
                    out.Set(iRow, c, Component(kernel, c, u, iRow, ns, mixture));
            }
        };
        launch();
        double error = 0.;
        const Index probes = std::min<Index>(257, n);
        for (Index q = 0; q < probes; ++q)
        {
            const Index iRow = q * (n - 1) / std::max<Index>(1, probes - 1);
            const auto mixture = Mix(Analytic{ns}, iRow, ns);
            for (int c = 0; c < width; ++c)
            {
                const double expected = Component(kernel, c, Analytic{ns}, iRow, ns, mixture), got = out(iRow, c);
                Check(std::isfinite(got), "nonfinite CPU species result");
                error = std::max(error, fabs(got - expected) / (1. + fabs(expected)));
            }
        }
        Check(error < 5e-12, "CPU species indexing/arithmetic mismatch");
        if (n <= 257)
        {
            std::vector<unsigned char> logical(size_t(shape.Elements()), 0);
            for (Index iRow = 0; iRow < n; ++iRow)
                for (int c = 0; c < shape.width; ++c)
                    logical[size_t(shape.Offset(iRow, c))] = 1;
            for (Index position = 0; position < shape.Elements(); ++position)
                if (!logical[size_t(position)])
                    Check(result[size_t(position)] == -12345., "CPU species kernel overwrote padding");
        }
        for (int sample = 0; sample < samples; ++sample)
        {
            const double start = Clock();
            for (int repeat = 0; repeat < 2; ++repeat)
                launch();
            const double seconds = (Clock() - start) / 2;
            std::printf("species-cpu,%d,%d,%lld,%d,%d,k%d,%d,0,%.9g,%.3g,%lld,%lld\n", L, int(Split), n, ns, variant, kernel + 1, sample,
                        seconds, error, (long long)(2 * (shape.Elements() + (Split ? n * 5 : 0)) * 8), (long long)(L == 1 ? planePadding : 0));
            std::fflush(stdout);
        }
    }
}
#endif

int main(int argc, char **argv)
{
    try
    {
        Check(argc == 8 || argc == 9, "usage: species rows species layout split variant samples device [plane-padding]");
        const Index n = std::stoll(argv[1]);
        const int ns = std::stoi(argv[2]), layout = std::stoi(argv[3]), split = std::stoi(argv[4]);
        const int variant = std::stoi(argv[5]), samples = std::stoi(argv[6]);
        planePadding = argc == 9 ? std::stoll(argv[8]) : 0;
        Check(planePadding >= 0 && planePadding <= 4096, "invalid species plane padding");
        Check(n > 0 && (ns == 16 || ns == 32) && (split == 0 || split == 1) && samples > 0, "invalid species dimensions");
        Check(variant == 0 || variant == 1 || variant == 2 || variant == 8 || variant == 32 || variant == 128, "invalid species mapping");
#ifndef CAUCHY_SPECIES_CPU
        CUDA(cudaSetDevice(std::stoi(argv[7])));
#endif
        if (split)
        {
            if (layout == 0)
                Run<0, true>(n, ns, variant, samples);
            else if (layout == 1)
                Run<1, true>(n, ns, variant, samples);
            else if (layout == 8)
                Run<8, true>(n, ns, variant, samples);
            else
            {
                Check(layout == 32, "invalid species layout");
                Run<32, true>(n, ns, variant, samples);
            }
        }
        else
        {
            if (layout == 0)
                Run<0, false>(n, ns, variant, samples);
            else if (layout == 1)
                Run<1, false>(n, ns, variant, samples);
            else if (layout == 8)
                Run<8, false>(n, ns, variant, samples);
            else
            {
                Check(layout == 32, "invalid species layout");
                Run<32, false>(n, ns, variant, samples);
            }
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
