#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/CUDAExecution.hpp"
#include "DNDS/Runtime/EagerArray.hpp"
#include "DNDS/Runtime/EagerCopy.hpp"
#include "DNDS/Runtime/AllocationResource.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <chrono>

namespace R = DNDS::Runtime;

struct Gate
{
    std::shared_ptr<std::atomic<bool>> open = std::make_shared<std::atomic<bool>>(false);
    ~Gate() { *open = true; }
    void Enqueue(R::EagerEngine &engine, const R::ExecutionStream &stream)
    {
        engine.Submit(stream, [owner = open](R::ExecutionContext context)
                      {
            auto status = cudaLaunchHostFunc(static_cast<cudaStream_t>(context.stream), [](void *p)
            {
                auto &value = *static_cast<std::atomic<bool> *>(p);
                while (!value.load()) std::this_thread::sleep_for(std::chrono::microseconds(50));
            }, owner.get());
            DNDS_check_throw_info(status == cudaSuccess, "test gate launch failed"); });
    }
};

__global__ void Add(R::ArrayView<double> u, double value)
{
    for (size_t row = blockIdx.x; row < u.Rows(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < u.RowSize(row); col += blockDim.x)
            u(row, col) += value;
}
__global__ void Scale(R::ArrayView<const double> in, R::ArrayView<double> out, double factor)
{
    for (size_t row = blockIdx.x; row < out.Rows(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < out.RowSize(row); col += blockDim.x)
            out(row, col) = in(row, col) * factor;
}
__global__ void CoreStencil(R::ArrayView<const int64_t> neighbors, R::ArrayView<const double> owned, R::ArrayView<double> out)
{
    for (size_t row = blockIdx.x; row < neighbors.Rows(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < out.RowSize(row); col += blockDim.x)
        {
            double sum = 0;
            for (size_t j = 0; j < neighbors.RowSize(row); ++j)
                sum += owned(neighbors(row, j), col);
            out(row, col) = sum;
        }
}
__global__ void RingUpdate(R::ArrayView<const double> ghost, R::ArrayView<double> out)
{
    for (size_t row = blockIdx.x; row < ghost.Rows(); row += gridDim.x)
        for (size_t col = threadIdx.x; col < ghost.RowSize(row); col += blockDim.x)
            out(out.Rows() - ghost.Rows() + row, col) = ghost(row, col);
}
static void LaunchAdd(R::ExecutionContext context, R::ArrayView<double> u)
{
    Add<<<7, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(u, 1);
    DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test Add launch failed");
}

static void PreloadKernels()
{
    // CUDA lazy first-use loading may synchronize the context. Load before
    // deliberately blocking a stream; otherwise the test gate and first launch
    // can deadlock independently of the engine's ordering.
    cudaFuncAttributes attributes{};
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, Add) == cudaSuccess, "test Add preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, Scale) == cudaSuccess, "test Scale preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, CoreStencil) == cudaSuccess, "test stencil preload failed");
    DNDS_check_throw_info(cudaFuncGetAttributes(&attributes, RingUpdate) == cudaSuccess, "test ring preload failed");
}

static R::Buffer Pinned(size_t bytes, std::atomic<int> &releases)
{
    void *pointer = nullptr;
    DNDS_check_throw_info(cudaMallocHost(&pointer, bytes) == cudaSuccess, "test pinned allocation failed");
    auto owner = std::shared_ptr<void>(pointer, [&releases](void *p)
                                       {
        if (cudaFreeHost(p) != cudaSuccess) std::terminate();
        ++releases; });
    return R::Buffer::FromAllocation(R::Allocation::Import(std::move(owner), bytes), 0, bytes);
}

TEST_CASE("Eager CUDA uses explicit prefix events with retained DMA owners and a three stream join")
{
    std::atomic<int> uploadedReleased{0}, downloadedReleased{0};
    R::EagerEngine engine(2);
    auto producer = std::make_shared<R::CUDALauncher>(0);
    auto reader1 = std::make_shared<R::CUDALauncher>(0);
    auto reader2 = std::make_shared<R::CUDALauncher>(0);
    auto compute = engine.CreateStream(producer);
    auto left = engine.CreateStream(reader1), right = engine.CreateStream(reader2);
    CHECK(compute.Context().stream != nullptr);
    CHECK(compute.Context().stream != left.Context().stream);
    auto shape = R::RowLayout::Dense(257, 6, 8);
    R::Array<double, R::FixedRows<6>> state(shape, R::Device::CUDA(0));
    R::Array<double> first(R::RowLayout::Ragged(std::vector<size_t>(257, 6)), R::Device::CUDA(0));
    std::vector<size_t> lengths(257);
    for (size_t i = 0; i < lengths.size(); ++i)
        lengths[i] = i % 7;
    R::Array<double> second(R::RowLayout::Padded(lengths, 9), R::Device::CUDA(0));
    auto upload = Pinned(shape.Elements() * sizeof(double), uploadedReleased);
    auto download = Pinned(first.Storage().Bytes(), downloadedReleased);
    {
        auto write = upload.Write();
        auto values = write.View<double>();
        for (size_t i = 0; i < values.size; ++i)
            values.data[i] = double(i);
    }
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, compute);
    auto stateBytes = state.Storage();
    R::CopyAsync(engine, compute, upload, stateBytes);
    stateBytes = {};
    CHECK_THROWS(upload.Write());
    upload = {};
    CHECK(uploadedReleased == 0);
    for (int i = 0; i < 17; ++i)
        R::SubmitArrays(engine, compute, LaunchAdd, R::ReadWrite(state));
    CHECK(producer->NativeEventsRecorded() == 0);
    auto ready = compute.Record();
    CHECK(producer->NativeEventsRecorded() == 1);
    CHECK_FALSE(ready.Ready());
    CHECK_THROWS(state.Read(compute.Context()));
    left.WaitFor(ready);
    right.WaitFor(ready);
    R::SubmitArrays(engine, left, [](R::ExecutionContext context, R::ArrayView<const double> in, R::ArrayView<double> out)
                    {
        Scale<<<9, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(in, out, 2);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test Scale launch failed"); }, R::Read(state), R::Write(first));
    R::SubmitArrays(engine, right, [](R::ExecutionContext context, R::ArrayView<const double> in, R::ArrayView<double> out)
                    {
        Scale<<<9, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(in, out, 3);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test Scale launch failed"); }, R::Read(state), R::Write(second));
    left.WaitFor(right.Record()); // All independent readers precede release here.
    std::weak_ptr<const R::ArrayStorage> original = state.SharedStorage();
    engine.ReleaseOn(std::move(state), left);
    R::CopyAsync(engine, left, first.Storage(), download);
    auto joined = left.Record();
    CHECK(reader1->NativeEventsRecorded() == 1);
    CHECK(original.expired()); // CPU handles are gone; native free is queued after the joined readers.
    CHECK_THROWS(download.Read());
    auto cpu = engine.CreateCPUStream();
    std::atomic<bool> observed{false};
    cpu.WaitFor(joined);
    engine.Submit(cpu, [&observed](R::ExecutionContext)
                  { observed = true; });
    CHECK_FALSE(observed);
    *gate.open = true;
    engine.Synchronize();
    CHECK(observed);
    CHECK(original.expired());
    CHECK(uploadedReleased == 1);
    auto downloadedAccess = download.Read();
    auto result = downloadedAccess.View<double>();
    for (size_t row = 0; row < 257; ++row)
        for (size_t col = 0; col < 6; ++col)
            CHECK(result.data[row * 6 + col] == 2 * double(row * 8 + col + 17));
    auto host = second.To(R::Device::Host(), right.Context());
    auto read = host.Read();
    for (size_t row = 0; row < 257; ++row)
        for (size_t col = 0; col < lengths[row]; ++col)
            CHECK(read.At(row, col) == 3 * double(row * 8 + col + 17));
    CHECK(ready.Ready());
}

TEST_CASE("CUDA owned CSR stencil finishes while ghost writes are pending then joins ring work")
{
    constexpr size_t rows = 257, ghosts = 19, coreRows = rows - ghosts;
    std::atomic<int> releases{0};
    R::EagerEngine engine;
    auto compute = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto comm = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    R::Array<double> host(R::RowLayout::Dense(rows, 6, 8));
    std::vector<size_t> widths(coreRows);
    for (size_t i = 0; i < coreRows; ++i)
        widths[i] = i % 7;
    R::Array<int64_t> hostNeighbors(R::RowLayout::Ragged(widths));
    {
        auto u = host.Write();
        auto c2c = hostNeighbors.Write();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < 6; ++j)
                u.At(i, j) = 10 * i + j;
        for (size_t i = 0; i < coreRows; ++i)
            for (size_t j = 0; j < widths[i]; ++j)
                c2c.At(i, j) = (17 * i + 13 * j) % rows;
    }
    auto owned = host.To(R::Device::CUDA(0));
    auto neighbors = hostNeighbors.To(R::Device::CUDA(0));
    R::Array<double> ghost(R::RowLayout::Dense(ghosts, 6, 8), R::Device::CUDA(0));
    R::Array<double> output(host.Layout(), R::Device::CUDA(0));
    auto downloaded = Pinned(output.Storage().Bytes(), releases);
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, comm);
    R::SubmitArrays(engine, comm, LaunchAdd, R::Write(ghost));
    auto ghostReady = comm.Record();
    R::SubmitArrays(engine, compute, [](R::ExecutionContext context, auto c2c, auto u, auto rhs)
                    {
        CoreStencil<<<11, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(c2c, u, rhs);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test stencil launch failed"); }, R::Read(neighbors), R::Read(owned), R::Write(output));
    engine.ReleaseOn(std::move(neighbors), compute); // Device metadata follows the last stencil use.
    R::CopyAsync(engine, compute, output.Storage(), downloaded);
    auto coreReady = compute.Record();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!coreReady.Ready() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool completedIndependently = coreReady.Ready();
    CHECK_FALSE(ghostReady.Ready());
    CHECK_THROWS(ghost.Read(comm.Context()));
    if (!completedIndependently)
        *gate.open = true;
    REQUIRE(completedIndependently);
    {
        auto read = downloaded.Read();
        auto values = read.View<double>();
        for (size_t i = 0; i < coreRows; ++i)
            for (size_t j = 0; j < 6; ++j)
            {
                double expected = 0;
                for (size_t k = 0; k < widths[i]; ++k)
                    expected += 10 * ((17 * i + 13 * k) % rows) + j;
                CHECK(values.data[i * 8 + j] == expected);
            }
    }
    compute.WaitFor(ghostReady);
    R::SubmitArrays(engine, compute, [](R::ExecutionContext context, auto g, auto rhs)
                    {
        RingUpdate<<<3, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(g, rhs);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test ring launch failed"); }, R::Read(ghost), R::ReadWrite(output));
    *gate.open = true;
    engine.Synchronize();
    auto result = output.To(R::Device::Host());
    auto read = result.Read();
    for (size_t i = coreRows; i < rows; ++i)
        for (size_t j = 0; j < 6; ++j)
            CHECK(read.At(i, j) == 1);
}

TEST_CASE("CUDA fixed prefixes retire prior batches and preserve delayed ownership")
{
    R::EagerEngine engine;
    auto backend = std::make_shared<R::CUDALauncher>(0);
    auto stream = engine.CreateStream(backend);
    R::Array<double> array(R::RowLayout::Dense(73, 5, 8), R::Device::CUDA(0));
    R::SubmitArrays(engine, stream, LaunchAdd, R::Write(array));
    auto first = stream.Record();
    first.WaitHost();
    auto before = array.Version();
    Gate gate;
    gate.Enqueue(engine, stream);
    R::SubmitArrays(engine, stream, LaunchAdd, R::ReadWrite(array));
    auto second = stream.Record();
    CHECK(first.Ready());
    CHECK_FALSE(second.Ready());
    CHECK(array.Version() == before);
    *gate.open = true;
    second.WaitHost();
    CHECK(array.Version() == before + 1);
    CHECK(array.To(R::Device::Host(), stream.Context()).Read().At(72, 4) == 2);
}

TEST_CASE("CUDA launch exceptions drain submitted uses and invalidate partial and skipped writes")
{
    std::atomic<int> released{0};
    R::EagerEngine engine;
    auto backend = std::make_shared<R::CUDALauncher>(0);
    auto stream = engine.CreateStream(backend);
    R::Array<double> partial(R::RowLayout::Dense(131, 7, 9), R::Device::CUDA(0));
    R::Array<double> skipped(partial.Layout(), R::Device::CUDA(0));
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, stream);
    R::SubmitArrays(engine, stream, [](R::ExecutionContext context, R::ArrayView<double> values)
                    {
        LaunchAdd(context, values);
        throw std::runtime_error("host failure after launch"); }, R::Write(partial));
    bool launched = false;
    R::SubmitArrays(engine, stream, [&launched](R::ExecutionContext context, R::ArrayView<double> values)
                    {
        launched = true;
        LaunchAdd(context, values); }, R::Write(skipped));
    auto owner = std::shared_ptr<int>(new int(3), [&](int *p)
                                      { delete p; ++released; });
    engine.ReleaseOn(std::move(owner), stream);
    auto tail = stream.Record();
    CHECK_FALSE(launched);
    CHECK_FALSE(tail.Ready());
    CHECK(released == 0);
    *gate.open = true;
    CHECK_THROWS(engine.Synchronize());
    CHECK_FALSE(partial.IsValid());
    CHECK_FALSE(skipped.IsValid());
    CHECK(partial.Storage().IsIdle());
    CHECK(released == 1);
}

TEST_CASE("Concurrent CUDA host submitters preserve one native stream order without kernel events")
{
    R::EagerEngine engine;
    auto backend = std::make_shared<R::CUDALauncher>(0);
    auto stream = engine.CreateStream(backend);
    R::Array<double> state(R::RowLayout::Dense(43, 6, 8), R::Device::CUDA(0));
    std::vector<std::thread> threads;
    std::atomic<bool> failed{false};
    for (int i = 0; i < 4; ++i)
        threads.emplace_back([&]
                             {
            try { for (int j = 0; j < 17; ++j) R::SubmitArrays(engine, stream, LaunchAdd, R::ReadWrite(state)); }
            catch (...) { failed = true; } });
    for (auto &thread : threads)
        thread.join();
    CHECK_FALSE(failed);
    CHECK(backend->NativeEventsRecorded() == 0);
    stream.Record().WaitHost();
    CHECK(backend->NativeEventsRecorded() == 1);
    CHECK(state.Version() == 68);
    CHECK(state.To(R::Device::Host(), stream.Context()).Read().At(42, 5) == 68);
}

TEST_CASE("CUDA dependency only prefixes retire earlier batches and completed history is discarded")
{
    R::StreamEvent survivor;
    R::Array<double> a(R::RowLayout::Dense(53, 6, 8), R::Device::CUDA(0));
    R::Array<double> b(a.Layout(), R::Device::CUDA(0));
    {
        R::EagerEngine engine;
        auto first = std::make_shared<R::CUDALauncher>(0);
        auto second = std::make_shared<R::CUDALauncher>(0);
        auto left = engine.CreateStream(first), right = engine.CreateStream(second);
        PreloadKernels();
        Gate gate;
        gate.Enqueue(engine, left);
        R::SubmitArrays(engine, left, LaunchAdd, R::Write(a));
        auto earlier = left.Record();
        std::weak_ptr<const R::StreamEvent::State> old = earlier.Provider();
        R::SubmitArrays(engine, right, LaunchAdd, R::Write(b));
        left.WaitFor(right.Record());
        auto join = left.Record(); // Contains a wait, no kernel.
        CHECK_FALSE(join.Ready());
        *gate.open = true;
        join.WaitHost();
        CHECK(a.Storage().IsIdle());
        CHECK(b.Storage().IsIdle());
        CHECK(a.Version() == 1);
        earlier = {};
        join = {};
        for (int iteration = 0; iteration < 97; ++iteration)
        {
            R::SubmitArrays(engine, left, LaunchAdd, R::ReadWrite(a));
            left.Record().WaitHost();
        }
        CHECK(old.expired());
        CHECK(first->NativeEventsRecorded() == 99);
        survivor = left.Record(); // Same immutable prefix; no extra event.
        CHECK(first->NativeEventsRecorded() == 99);
        struct Unready : R::StreamEvent::State
        {
            bool Ready() const override { return false; }
            void Wait() const override { throw std::runtime_error("must not host-wait implicitly"); }
        };
        CHECK_THROWS(left.WaitFor(R::StreamEvent(std::make_shared<Unready>())));
        engine.Synchronize();
    }
    CHECK(survivor.Ready()); // Native stream/event ownership outlives the engine.
    survivor.WaitHost();
    CHECK(a.To(R::Device::Host()).Read().At(52, 5) == 98);
}

TEST_CASE("CUDA async allocation supports dynamic tables and prepared padded CSR metadata behind pending work")
{
    R::EagerEngine engine;
    auto producer = std::make_shared<R::CUDALauncher>(0), consumer = std::make_shared<R::CUDALauncher>(0);
    auto first = engine.CreateStream(producer), last = engine.CreateStream(consumer);
    std::vector<size_t> lengths(193);
    for (size_t i = 0; i < lengths.size(); ++i)
        lengths[i] = i % 9;
    R::Array<double> paddedPrototype(R::RowLayout::Padded(lengths, 11), R::Device::CUDA(0));
    R::Array<double> csrPrototype(R::RowLayout::Ragged(lengths), R::Device::CUDA(0));
    R::Array<double> result(R::RowLayout::Dense(257, 6, 8), R::Device::CUDA(0));
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, first);
    auto table = R::AllocateArrayAsync<double, R::FixedRows<6>>(engine, first, result.Layout());
    auto padded = R::AllocateLikeAsync(engine, first, paddedPrototype);
    auto csr = R::AllocateLikeAsync(engine, first, csrPrototype);
    CHECK(padded.SharedStorage()->SharedLayout() == paddedPrototype.SharedStorage()->SharedLayout());
    CHECK(csr.SharedStorage()->SharedLayout() == csrPrototype.SharedStorage()->SharedLayout());
    CHECK_THROWS(table.Read(first.Context()));
    CHECK_THROWS(R::AllocateArrayAsync<double>(engine, first, R::RowLayout::Ragged({1, 3})));
    CHECK_THROWS((R::AllocateArrayAsync<double, R::FixedRows<5>>(engine, first, result.Layout())));
    auto alias = table;
    for (int i = 0; i < 7; ++i)
    {
        R::SubmitArrays(engine, first, LaunchAdd, R::ReadWrite(table));
        R::SubmitArrays(engine, first, LaunchAdd, R::ReadWrite(padded));
        R::SubmitArrays(engine, first, LaunchAdd, R::ReadWrite(csr));
    }
    engine.ReleaseOn(std::move(table), first); // Non-final: alias still owns values.
    last.WaitFor(first.Record());
    R::SubmitArrays(engine, last, [](R::ExecutionContext context, R::ArrayView<const double> in, R::ArrayView<double> out)
                    {
        Scale<<<9, 32, 0, static_cast<cudaStream_t>(context.stream)>>>(in, out, 3);
        DNDS_check_throw_info(cudaGetLastError() == cudaSuccess, "test scale failed"); }, R::Read(alias), R::Write(result));
    auto values = std::weak_ptr<const R::ArrayStorage>(alias.SharedStorage());
    engine.ReleaseOn(std::move(alias), last); // Final free queued on the consumer stream.
    CHECK(values.expired());
    auto empty = engine.AllocateBuffer(first, 0);
    engine.ReleaseOn(std::move(empty), first);
    CHECK(producer->NativeEventsRecorded() == 1);
    CHECK(consumer->NativeEventsRecorded() == 0);
    *gate.open = true;
    engine.Synchronize();
    CHECK(result.To(R::Device::Host(), last.Context()).Read().At(256, 5) == 21);
    auto hostPadded = padded.To(R::Device::Host(), first.Context());
    auto hostCSR = csr.To(R::Device::Host(), first.Context());
    auto p = hostPadded.Read(), c = hostCSR.Read();
    for (size_t row = 0; row < lengths.size(); ++row)
        for (size_t col = 0; col < lengths[row]; ++col)
        {
            CHECK(p.At(row, col) == 7);
            CHECK(c.At(row, col) == 7);
        }
    CHECK(padded.Version() == 8); // Async zero initialization plus seven updates.
    engine.ReleaseOn(std::move(padded), last);
    engine.ReleaseOn(std::move(csr), last);
    engine.Synchronize();
}

TEST_CASE("Backend allocation resources receive the final selected free stream across independent handles")
{
    struct Observation
    {
        int releases = 0;
        void *stream = nullptr;
    } observed;
    struct Resource : R::AllocationResource
    {
        void *pointer = nullptr;
        Observation &observed;
        explicit Resource(Observation &o) : observed(o)
        {
            DNDS_check_throw_info(cudaMalloc(&pointer, Bytes()) == cudaSuccess, "test allocation failed");
        }
        ~Resource() override { Release(nullptr); }
        void *Data() const noexcept override { return pointer; }
        size_t Bytes() const noexcept override { return 31 * 7 * sizeof(double); }
        R::Device GetDevice() const noexcept override { return R::Device::CUDA(0); }
        bool CanReleaseOrdered(R::ExecutionContext c) const noexcept override { return c.device == GetDevice(); }
        void Release(const R::ExecutionContext *c) noexcept override
        {
            if (!pointer)
                return;
            observed.stream = c ? c->stream : nullptr;
            auto status = c ? cudaFreeAsync(pointer, static_cast<cudaStream_t>(c->stream)) : cudaFree(pointer);
            if (status != cudaSuccess)
                std::terminate();
            pointer = nullptr;
            ++observed.releases;
        }
    };
    R::EagerEngine engine;
    auto a = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto b = engine.CreateStream(std::make_shared<R::CUDALauncher>(0));
    auto buffer = R::Buffer::FromResource(std::make_unique<Resource>(observed));
    auto array = R::Array<double>::FromStorage(R::RowLayout::Dense(31, 5, 7), std::move(buffer));
    auto alias = array;
    PreloadKernels();
    Gate gate;
    gate.Enqueue(engine, a);
    R::SubmitArrays(engine, a, [](R::ExecutionContext c, R::ArrayView<double> u)
                    {
        DNDS_check_throw_info(cudaMemsetAsync(u.data, 0, u.Rows() * u.stride * sizeof(double), static_cast<cudaStream_t>(c.stream)) == cudaSuccess,
                              "test initialization failed");
        LaunchAdd(c, u); }, R::Write(array));
    engine.ReleaseOn(std::move(array), a);
    CHECK(observed.releases == 0);
    b.WaitFor(a.Record());
    R::SubmitArrays(engine, b, LaunchAdd, R::ReadWrite(alias));
    engine.ReleaseOn(std::move(alias), b);
    CHECK(observed.releases == 1);
    CHECK(observed.stream == b.Context().stream);
    *gate.open = true;
    engine.Synchronize();
    CHECK(observed.releases == 1);
}
