// Standalone simulated kernels: no production Array/CFV changes are needed.
// Layout, arithmetic and transfer costs are deliberately measured separately.
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#ifdef __CUDACC__
#    include <cuda_runtime.h>
#    define HD __host__ __device__
#else
#    define HD
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    using Index = int64_t;
    void Check(bool ok, const char *message)
    {
        if (!ok)
            throw std::runtime_error(message);
    }
#ifdef __CUDACC__
    void CUDA(cudaError_t error)
    {
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
#endif
    double Clock()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    const char *Name(int layout)
    {
        switch (layout)
        {
        case 0:
            return "aos";
        case 1:
            return "soa";
        case 8:
            return "aosoa8";
        case 32:
            return "aosoa32";
        case 101:
            return "hybrid_soa";
        default:
            return "hybrid_aosoa32";
        }
    }
    template <int Layout>
    struct View
    {
        double *data;
        Index rows;
        int components;
        HD Index Offset(Index iRow, int iComp) const
        {
            if constexpr (Layout == 0)
                return iRow * components + iComp;
            if constexpr (Layout == 1)
                return Index(iComp) * rows + iRow;
            if constexpr (Layout == 8 || Layout == 32)
                return (iRow / Layout) * (components * Layout) + iComp * Layout + iRow % Layout;
            if constexpr (Layout >= 100)
            {
                if (iComp < 5)
                    return iRow * 5 + iComp;
                if constexpr (Layout == 101)
                    return rows * 5 + Index(iComp - 5) * rows + iRow;
                return rows * 5 + (iRow / 32) * ((components - 5) * 32) + (iComp - 5) * 32 + iRow % 32;
            }
            return 0;
        }
        HD double operator()(Index iRow, int iComp) const { return data[Offset(iRow, iComp)]; }
        HD void Set(Index iRow, int iComp, double value) const { data[Offset(iRow, iComp)] = value; }
        Index Elements() const
        {
            if constexpr (Layout == 8 || Layout == 32)
                return ((rows + Layout - 1) / Layout) * Layout * components;
            if constexpr (Layout == 132)
                return rows * 5 + ((rows + 31) / 32) * 32 * (components - 5);
            return rows * components;
        }
    };
    HD double Initial(Index iRow, int iComp)
    {
        const double x = double((iRow * 17 + iComp * 23) % 1009) / 1009.;
        if (iComp == 0)
            return 1. + .1 * x;
        if (iComp < 4)
            return .12 + .03 * x;
        if (iComp == 4)
            return 3. + .1 * x;
        return .01 * (iComp + 1) + .005 * x;
    }
    struct Reference
    {
        HD double operator()(Index iRow, int iComp) const { return Initial(iRow, iComp); }
    };
    // Shared row arithmetic. Matrix coefficients are a small reproducible pool,
    // not a full edge-matrix allocation; this isolates field-layout effects.
    template <class Input, class Output>
    HD void Evaluate(int kernel, Index iRow, Index n, int nc, Input u, Output out)
    {
        if (kernel == 0)
        {
            const double rho = u(iRow, 0), mx = u(iRow, 1), my = u(iRow, 2), mz = u(iRow, 3);
            const double p = .4 * (u(iRow, 4) - .5 * (mx * mx + my * my + mz * mz) / rho);
            const double t = p / rho;
            const double sound = sqrt(1.4 * t), mu = pow(t, 1.5) / (t + .4);
            out.Set(iRow, 0, rho * sound);
            out.Set(iRow, 1, mx * sound + p);
            out.Set(iRow, 2, my * sound + mu);
            out.Set(iRow, 3, mz * sound + p * .3);
            out.Set(iRow, 4, (u(iRow, 4) + p) * mx / rho + mu * t);
            for (int iComp = 5; iComp < nc; ++iComp)
            {
                const double y = u(iRow, iComp);
                out.Set(iRow, iComp, y * sound + .03 * exp(-1. / t) * (1. - y));
            }
        }
        else if (kernel == 1)
        {
            for (int iComp = 0; iComp < nc; ++iComp)
            {
                const double center = u(iRow, iComp);
                double sum = 0., bound = fabs(center);
                for (int j = 0; j < 6; ++j)
                {
                    const Index nb = (iRow + 1 + ((iRow * 13 + j * 7919) % 16381)) % n;
                    const double value = u(nb, iComp), cross = u(nb, (iComp + 1) % nc);
                    sum += (value - center) * (.2 / (j + 1)) + .007 * cross;
                    bound = fmax(bound, fabs(value));
                }
                const double limited = fabs(sum) > .2 * bound ? sum * (.2 * bound / fabs(sum)) : sum;
                out.Set(iRow, iComp, center + limited);
            }
        }
        else
        {
            double v[5];
            for (int j = 0; j < 5; ++j)
            {
                double rhs = u(iRow, j) * 1.1;
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
                out.Set(iRow, j, u(iRow, j) + .1 * v[j]);
            }
            for (int iComp = 5; iComp < nc; ++iComp)
            {
                const double x = u(iRow, iComp);
                const double coupling = .01 * (v[0] + v[4]) + .02 * u(iRow, iComp - 1);
                out.Set(iRow, iComp, x + .1 * (x - coupling) / (1.5 + .01 * iComp));
            }
        }
    }
#ifdef __CUDACC__
    template <int L>
    __global__ void Numeric(int kernel, View<L> u, View<L> out, Index begin, Index end)
    {
        const Index iRow = begin + Index(blockIdx.x) * blockDim.x + threadIdx.x;
        if (iRow < end)
            Evaluate(kernel, iRow, u.rows, u.components, u, out);
    }
    template <int L>
    __global__ void Gather(View<L> u, double *packet, Index count, const Index *rows)
    {
        const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count * u.components)
            packet[i] = u(rows[i / u.components], int(i % u.components));
    }
    template <int L>
    __global__ void Scatter(View<L> u, const double *packet, Index count)
    {
        const Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < count * u.components)
            u.Set(i / u.components, int(i % u.components), packet[i]);
    }
#endif
    template <int L>
    void Launch(bool gpu, int kernel, View<L> u, View<L> out, Index begin, Index end, void *stream = nullptr)
    {
#ifdef __CUDACC__
        if (gpu)
        {
            Numeric<<<unsigned((end - begin + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(kernel, u, out, begin, end);
            CUDA(cudaGetLastError());
            return;
        }
#else
        (void)gpu;
        (void)stream;
#endif
        for (Index iRow = begin; iRow < end; ++iRow)
            Evaluate(kernel, iRow, u.rows, u.components, u, out);
    }
    void Fence(bool gpu, void *stream = nullptr)
    {
#ifdef __CUDACC__
        if (gpu)
            CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
#else
        (void)gpu;
        (void)stream;
#endif
    }
    struct Memory
    {
        double *data = nullptr;
        bool device;
        Memory(Index elements, bool gpu) : device(gpu)
        {
#ifdef __CUDACC__
            if (gpu)
            {
                CUDA(cudaMalloc(&data, size_t(elements) * sizeof(double)));
                return;
            }
#endif
            data = static_cast<double *>(std::malloc(size_t(elements) * sizeof(double)));
            Check(data, "allocation failed");
        }
        ~Memory()
        {
#ifdef __CUDACC__
            if (device)
            {
                cudaFree(data);
                return;
            }
#endif
            std::free(data);
        }
        void Upload(const std::vector<double> &host)
        {
#ifdef __CUDACC__
            if (device)
            {
                CUDA(cudaMemcpy(data, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
                return;
            }
#endif
            std::memcpy(data, host.data(), host.size() * sizeof(double));
        }
    };
    struct SmallOutput
    {
        double *data;
        HD void Set(Index, int iComp, double value) const { data[iComp] = value; }
    };
    template <int L>
    void Compute(bool gpu, Index n, int nc, int samples)
    {
        View<L> shape{nullptr, n, nc};
        Memory source(shape.Elements(), gpu), destination(shape.Elements(), gpu);
        View<L> u{source.data, n, nc}, out{destination.data, n, nc};
        std::vector<double> host(size_t(shape.Elements()), -12345.);
        View<L> h{host.data(), n, nc};
        for (Index iRow = 0; iRow < n; ++iRow)
            for (int iComp = 0; iComp < nc; ++iComp)
                h.Set(iRow, iComp, Initial(iRow, iComp));
        source.Upload(host);
        destination.Upload(host);
        for (int kernel = 0; kernel < 3; ++kernel)
        {
            Launch(gpu, kernel, u, out, 0, n);
            Fence(gpu);
            std::vector<double> values;
            std::vector<Index> probes;
            for (Index j = 0; j < std::min<Index>(257, n); ++j)
                probes.push_back(j * (n - 1) / std::max<Index>(1, std::min<Index>(257, n) - 1));
#ifdef __CUDACC__
            if (gpu)
            {
                Index *indices;
                CUDA(cudaMalloc(&indices, probes.size() * sizeof(Index)));
                CUDA(cudaMemcpy(indices, probes.data(), probes.size() * sizeof(Index), cudaMemcpyHostToDevice));
                Memory compact(Index(probes.size()) * nc, true);
                Gather<<<unsigned((probes.size() * nc + 255) / 256), 256>>>(out, compact.data, Index(probes.size()), indices);
                CUDA(cudaGetLastError());
                values.resize(probes.size() * nc);
                CUDA(cudaMemcpy(values.data(), compact.data, values.size() * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA(cudaFree(indices));
            }
#endif
            double error = 0.;
            for (Index j = 0; j < Index(probes.size()); ++j)
            {
                std::vector<double> expected(size_t(nc), 0.);
                Evaluate(kernel, probes[j], n, nc, Reference{}, SmallOutput{expected.data()});
                for (int iComp = 0; iComp < nc; ++iComp)
                {
                    const double got = gpu ? values[size_t(j * nc + iComp)] : out(probes[j], iComp);
                    Check(std::isfinite(got), "nonfinite kernel result");
                    error = std::max(error, fabs(got - expected[iComp]) / (1. + fabs(expected[iComp])));
                }
            }
            Check(error < 5e-12, "kernel reference mismatch");
#ifdef __CUDACC__
            cudaEvent_t eventStart{}, eventStop{};
            if (gpu)
            {
                CUDA(cudaEventCreate(&eventStart));
                CUDA(cudaEventCreate(&eventStop));
            }
#endif
            for (int sample = 0; sample < samples; ++sample)
            {
                const int repetitions = gpu ? 12 : 2;
                const double start = Clock();
#ifdef __CUDACC__
                if (gpu)
                    CUDA(cudaEventRecord(eventStart));
#endif
                for (int j = 0; j < repetitions; ++j)
                    Launch(gpu, kernel, u, out, 0, n);
                double deviceSeconds = 0.;
#ifdef __CUDACC__
                if (gpu)
                {
                    CUDA(cudaEventRecord(eventStop));
                    CUDA(cudaEventSynchronize(eventStop));
                    float milliseconds;
                    CUDA(cudaEventElapsedTime(&milliseconds, eventStart, eventStop));
                    deviceSeconds = milliseconds * .001 / repetitions;
                }
#endif
                const double elapsed = (Clock() - start) / repetitions;
                std::printf("compute,%s,%s,%lld,%d,k%d,%d,%.9g,%.3g,%lld,%.9g\n", gpu ? "cuda" : "cpu", Name(L), (long long)n, nc, kernel + 1, sample, elapsed, error, (long long)(shape.Elements() * 16), deviceSeconds);
                std::fflush(stdout);
            }
#ifdef __CUDACC__
            if (gpu)
            {
                CUDA(cudaEventDestroy(eventStart));
                CUDA(cudaEventDestroy(eventStop));
            }
#endif
        }
    }

    HD double Payload(int rank, Index iRow, int iComp) { return double(rank) * 1.e9 + double(iRow) * 64. + iComp; }
    struct PayloadReference
    {
        int rank;
        const Index *rows;
        HD double operator()(Index iRow, int iComp) const { return Payload(rank, rows ? rows[iRow] : iRow, iComp); }
    };
    template <int L>
    void Pack(bool gpu, View<L> u, Memory &packet, const std::vector<Index> &rows, const Index *deviceRows, void *stream = nullptr)
    {
#ifdef __CUDACC__
        if (gpu)
        {
            Gather<<<unsigned((rows.size() * u.components + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(u, packet.data, Index(rows.size()), deviceRows);
            CUDA(cudaGetLastError());
            return;
        }
#else
        (void)gpu;
        (void)deviceRows;
        (void)stream;
#endif
        for (Index j = 0; j < Index(rows.size()); ++j)
            for (int iComp = 0; iComp < u.components; ++iComp)
                packet.data[j * u.components + iComp] = u(rows[j], iComp);
    }
    template <int L>
    void Unpack(bool gpu, View<L> u, Memory &packet, Index count, void *stream = nullptr)
    {
#ifdef __CUDACC__
        if (gpu)
        {
            Scatter<<<unsigned((count * u.components + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(u, packet.data, count);
            CUDA(cudaGetLastError());
            return;
        }
#else
        (void)gpu;
        (void)stream;
#endif
        for (Index j = 0; j < count; ++j)
            for (int iComp = 0; iComp < u.components; ++iComp)
                u.Set(j, iComp, packet.data[j * u.components + iComp]);
    }
    template <int L>
    void Communication(bool gpu, Index n, int nc, int samples, int rank, int size, bool pipeline)
    {
        Check(size >= 2, "communication needs at least two ranks");
        int dims[3]{}, periodic[3]{1, 1, 1};
        MPI_Dims_create(size, 3, dims);
        MPI_Comm cart;
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periodic, 0, &cart);
        std::vector<int> peers;
        for (int axis = 0; axis < 3; ++axis)
        {
            int left, right;
            MPI_Cart_shift(cart, axis, 1, &left, &right);
            for (int peer : {left, right})
                if (peer != rank && std::find(peers.begin(), peers.end(), peer) == peers.end())
                    peers.push_back(peer);
        }
        MPI_Comm_free(&cart);
        char hostname[MPI_MAX_PROCESSOR_NAME];
        int hostnameSize;
        MPI_Get_processor_name(hostname, &hostnameSize);
        std::string topology = "topology rank=" + std::to_string(rank) + " host=" + hostname + " dims=" +
                               std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x" + std::to_string(dims[2]) + " peers=";
        for (int peer : peers)
            topology += std::to_string(peer) + ":";
        std::fprintf(stderr, "%s\n", topology.c_str());
        View<L> shape{nullptr, n, nc};
        Memory source(shape.Elements(), gpu), destination(shape.Elements(), gpu), numeric(shape.Elements(), gpu);
        View<L> u{source.data, n, nc}, ghost{destination.data, n, nc}, rhs{numeric.data, n, nc};
        std::vector<double> host(size_t(shape.Elements()), -12345.);
        View<L> h{host.data(), n, nc};
        for (Index iRow = 0; iRow < n; ++iRow)
            for (int iComp = 0; iComp < nc; ++iComp)
                h.Set(iRow, iComp, Payload(rank, iRow, iComp));
        source.Upload(host);
        destination.Upload(host);
        void *computeStream = nullptr, *commStream = nullptr;
#ifdef __CUDACC__
        cudaStream_t computeNative{}, commNative{};
        if (gpu)
        {
            CUDA(cudaStreamCreateWithFlags(&computeNative, cudaStreamNonBlocking));
            CUDA(cudaStreamCreateWithFlags(&commNative, cudaStreamNonBlocking));
            computeStream = computeNative;
            commStream = commNative;
        }
#endif
        for (Index requestedBytes : {Index(65536), Index(8388608)})
        {
            const Index count = std::min(n / (4 * Index(peers.size())), std::max<Index>(1, requestedBytes / (nc * 8)));
            Check(count > 0, "not enough rows for Cartesian halo");
            const Index haloRows = count * Index(peers.size());
            Memory send(count * nc, gpu), receive(haloRows * nc, gpu);
            for (int scattered = 0; scattered < 2; ++scattered)
            {
                std::vector<Index> rows(size_t(count), 0);
                for (Index j = 0; j < count; ++j)
                    rows[size_t(j)] = scattered ? (j * 7919) % n : j;
                Index *deviceRows = nullptr;
#ifdef __CUDACC__
                if (gpu)
                {
                    CUDA(cudaMalloc(&deviceRows, size_t(count) * sizeof(Index)));
                    CUDA(cudaMemcpy(deviceRows, rows.data(), size_t(count) * sizeof(Index), cudaMemcpyHostToDevice));
                }
#endif
                std::vector<int> sb;
                std::vector<MPI_Aint> sd;
                // Compile equivalent logical (row,component) pairs in physical
                // source order. This gives SoA/AoSoA the same opportunity as AoS
                // to coalesce contiguous sections, without changing field order.
                auto append = [](auto &blocks, auto &offsets, Index offset)
                {
                    if (!blocks.empty() && offsets.back() + MPI_Aint(blocks.back()) * 8 == offset * 8)
                        ++blocks.back();
                    else
                    {
                        blocks.push_back(1);
                        offsets.push_back(MPI_Aint(offset * 8));
                    }
                };
                const double setupStart = Clock();
                std::vector<std::pair<Index, Index>> entries;
                entries.reserve(size_t(count * nc));
                for (Index j = 0; j < count; ++j)
                    for (int iComp = 0; iComp < nc; ++iComp)
                        entries.emplace_back(u.Offset(rows[size_t(j)], iComp), j * nc + iComp);
                std::sort(entries.begin(), entries.end());
                for (const auto &entry : entries)
                    append(sb, sd, entry.first);
                MPI_Datatype st;
                std::vector<MPI_Datatype> rt(peers.size());
                MPI_Type_create_hindexed(int(sb.size()), sb.data(), sd.data(), MPI_DOUBLE, &st);
                MPI_Type_commit(&st);
                for (Index ip = 0; ip < Index(peers.size()); ++ip)
                {
                    std::vector<int> rb;
                    std::vector<MPI_Aint> rd;
                    for (const auto &entry : entries)
                        append(rb, rd, ghost.Offset(ip * count + entry.second / nc, int(entry.second % nc)));
                    MPI_Type_create_hindexed(int(rb.size()), rb.data(), rd.data(), MPI_DOUBLE, &rt[size_t(ip)]);
                    MPI_Type_commit(&rt[size_t(ip)]);
                }
                const double setup = Clock() - setupStart;
                std::vector<MPI_Request> requests(peers.size() * 2);
                for (int packed = 0; packed < 2; ++packed)
                {
                    for (int overlap = 0; overlap < (pipeline ? 2 : 1); ++overlap)
                    {
                        // Warmup + exact check, then matched samples; full completion
                        // protects source reuse and destination reads on every path.
                        for (int sample = -1; sample < samples; ++sample)
                        {
                            MPI_Barrier(MPI_COMM_WORLD);
                            const double start = Clock();
                            if (packed)
                            {
                                Pack(gpu, u, send, rows, deviceRows, commStream);
                                Fence(gpu, commStream);
                            }
                            const double packedAt = Clock();
                            for (Index ip = 0; ip < Index(peers.size()); ++ip)
                            {
                                MPI_Irecv(packed ? receive.data + ip * count * nc : ghost.data, packed ? int(count * nc) : 1,
                                          packed ? MPI_DOUBLE : rt[size_t(ip)], peers[size_t(ip)], 71, MPI_COMM_WORLD, &requests[size_t(ip * 2)]);
                                MPI_Isend(packed ? send.data : u.data, packed ? int(count * nc) : 1,
                                          packed ? MPI_DOUBLE : st, peers[size_t(ip)], 71, MPI_COMM_WORLD, &requests[size_t(ip * 2 + 1)]);
                            }
                            auto core = [&]
                            {
                                if (gpu)
                                    Launch(gpu, 1, u, rhs, haloRows, n, computeStream);
                                else
                                    for (Index begin = haloRows; begin < n; begin += 4096)
                                    {
                                        Launch(false, 1, u, rhs, begin, std::min(n, begin + 4096));
                                        if (overlap)
                                        {
                                            int done;
                                            MPI_Testall(int(requests.size()), requests.data(), &done, MPI_STATUSES_IGNORE);
                                        }
                                    }
                            };
                            if (pipeline && overlap)
                                core();
                            MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
                            const double transferredAt = Clock();
                            if (packed)
                            {
                                Unpack(gpu, ghost, receive, haloRows, commStream);
                                Fence(gpu, commStream);
                            }
                            if (pipeline)
                            {
                                if (!overlap)
                                    core();
                                Launch(gpu, 2, ghost, rhs, 0, haloRows, computeStream);
                                Fence(gpu, computeStream);
                            }
                            const double end = Clock();
                            if (sample == -1)
                            {
                                std::vector<double> got(size_t(shape.Elements()));
#ifdef __CUDACC__
                                if (gpu)
                                    CUDA(cudaMemcpy(got.data(), ghost.data, got.size() * 8, cudaMemcpyDeviceToHost));
                                else
#endif
                                    std::memcpy(got.data(), ghost.data, got.size() * 8);
                                View<L> check{got.data(), n, nc};
                                for (Index j = 0; j < haloRows; ++j)
                                    for (int iComp = 0; iComp < nc; ++iComp)
                                        Check(check(j, iComp) == Payload(peers[size_t(j / count)], rows[size_t(j % count)], iComp), "halo row/component mismatch");
                                if (pipeline)
                                {
#ifdef __CUDACC__
                                    if (gpu)
                                        CUDA(cudaMemcpy(got.data(), rhs.data, got.size() * 8, cudaMemcpyDeviceToHost));
                                    else
#endif
                                        std::memcpy(got.data(), rhs.data, got.size() * 8);
                                    for (Index j = 0; j < 257; ++j)
                                    {
                                        const Index iRow = j * (n - 1) / 256;
                                        std::vector<double> expected(size_t(nc), 0.);
                                        if (iRow < haloRows)
                                            Evaluate(2, iRow % count, n, nc, PayloadReference{peers[size_t(iRow / count)], rows.data()}, SmallOutput{expected.data()});
                                        else
                                            Evaluate(1, iRow, n, nc, PayloadReference{rank, nullptr}, SmallOutput{expected.data()});
                                        for (int iComp = 0; iComp < nc; ++iComp)
                                            Check(fabs(check(iRow, iComp) - expected[iComp]) < 1e-11 * (1. + fabs(expected[iComp])), "pipeline numeric mismatch");
                                    }
                                }
                            }
                            else
                            {
                                double times[4]{end - start, packedAt - start, transferredAt - packedAt, end - transferredAt}, maximum[4];
                                MPI_Reduce(times, maximum, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                                if (rank == 0)
                                    std::printf("%s,%s,%s,%lld,%d,%s_%s%s,%d,%.9g,%.9g,%.9g,%.9g,%lld,%lld,%.9g,%d\n", pipeline ? "pipeline" : "comm", gpu ? "cuda" : "cpu", Name(L), (long long)n, nc,
                                                scattered ? "scattered" : "coalesced", packed ? "packed" : "direct", pipeline ? (overlap ? "_overlap" : "_serial") : "", sample, maximum[0], maximum[1], maximum[2], maximum[3], (long long)(count * nc * 8), (long long)sb.size(), setup, size);
                                std::fflush(stdout);
                            }
                        }
                    }
                }
                MPI_Type_free(&st);
                for (auto &type : rt)
                    MPI_Type_free(&type);
#ifdef __CUDACC__
                if (gpu)
                    CUDA(cudaFree(deviceRows));
#endif
            }
        }
#ifdef __CUDACC__
        if (gpu)
        {
            CUDA(cudaStreamDestroy(computeNative));
            CUDA(cudaStreamDestroy(commNative));
        }
#endif
    }
    template <class F>
    void Dispatch(int layout, F &&f)
    {
        switch (layout)
        {
        case 0:
            f(std::integral_constant<int, 0>{});
            break;
        case 1:
            f(std::integral_constant<int, 1>{});
            break;
        case 8:
            f(std::integral_constant<int, 8>{});
            break;
        case 32:
            f(std::integral_constant<int, 32>{});
            break;
        case 101:
            f(std::integral_constant<int, 101>{});
            break;
        case 132:
            f(std::integral_constant<int, 132>{});
            break;
        default:
            Check(false, "unknown layout");
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try
    {
        Check(argc >= 7, "usage: probe compute|comm cpu|cuda rows components samples layout|-1 [device]");
        const std::string mode = argv[1];
        const bool gpu = std::string(argv[2]) == "cuda";
        const Index n = std::stoll(argv[3]);
        const int nc = std::stoi(argv[4]), samples = std::stoi(argv[5]), selected = std::stoi(argv[6]);
        Check(n > 0 && nc >= 5 && nc <= 64 && samples > 0, "invalid dimensions");
#ifdef __CUDACC__
        if (gpu)
        {
            const int device = argc > 7 ? std::stoi(argv[7]) : rank;
            CUDA(cudaSetDevice(device));
            cudaDeviceProp prop;
            CUDA(cudaGetDeviceProperties(&prop, device));
            std::fprintf(stderr, "rank=%d device=%d name=%s\n", rank, device, prop.name);
        }
#else
        Check(!gpu, "CUDA not compiled");
#endif
        for (int layout : {0, 1, 8, 32, 101, 132})
        {
            if (selected != -1 && selected != layout)
                continue;
            Dispatch(layout, [&](auto policy)
                     {
                constexpr int L = decltype(policy)::value;
                if (mode == "compute") { Check(size == 1, "compute probe uses one rank"); Compute<L>(gpu, n, nc, samples); }
                else { Check(mode == "comm" || mode == "pipeline", "invalid mode"); Communication<L>(gpu, n, nc, samples, rank, size, mode == "pipeline"); } });
        }
        MPI_Finalize();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
