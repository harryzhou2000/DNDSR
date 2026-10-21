#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Buffer.hpp"
#include "DNDS/Runtime/Array.hpp"
#include "DNDS/Array.hpp"
#include <cuda_runtime.h>

using namespace DNDS::Runtime;

__global__ void FillRows(ArrayView<double> values)
{
    for (size_t i = blockIdx.x; i < values.Rows(); i += gridDim.x)
        for (size_t j = threadIdx.x; j < values.RowSize(i); j += blockDim.x)
            values(i, j) = 100 * i + j;
}

TEST_CASE("Native ragged Array CUDA views retain metadata through replacement")
{
    DNDS::Runtime::Array<double> values(RowLayout::Ragged({2, 0, 513}), Device::CUDA(0));
    auto retained = values;
    auto shared = Array<double>::FromSharedStorage(values.SharedStorage());
    CHECK(shared.SharedStorage() == values.SharedStorage());
    CHECK(shared.SharedStorage()->SharedLayout() == values.SharedStorage()->SharedLayout());
    CHECK(shared.SharedStorage()->SharedLayout()->DeviceOffsetIdentity() == values.SharedStorage()->SharedLayout()->DeviceOffsetIdentity());
    auto ctx = ExecutionContext::CUDA(0);
    {
        auto write = values.Write(ctx);
        FillRows<<<3, 128>>>(write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(Completion::Record(ctx));
        values.Reset(RowLayout::Dense(1, 1));
        CHECK_THROWS(retained.Read(ctx));
        write.Wait();
    }
    auto host = retained.To(Device::Host(), ctx);
    CHECK(host.Read().At(2, 512) == 712);
    CHECK(host.RowSize(1) == 0);
    auto copy = retained.Clone(ctx);
    CHECK(copy.Identity() != retained.Identity());
    CHECK(copy.To(Device::Host(), ctx).Read().At(0, 1) == 1);
}

TEST_CASE("Padded CUDA views use logical row sizes and preserve capacity padding")
{
    cudaStream_t stream;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = ExecutionContext::CUDA(0, stream);
    {
        Array<double> values(RowLayout::Padded({0, 5, 7, 33, 0, 19}, 40), Device::CUDA(0));
        auto retained = values;
        auto write = values.Write(context);
        FillRows<<<3, 64, 0, stream>>>(write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(Completion::Record(context));
        values.Reset(RowLayout::Dense(0, 0));
        write.Wait();
        auto host = retained.To(Device::Host(), context);
        CHECK(host.Layout().IsPadded());
        CHECK(host.Read().At(3, 32) == 332);
        CHECK_THROWS(host.Read().At(3, 33));
        auto raw = host.Storage().Read();
        auto flat = raw.View<double>();
        for (size_t i = 0; i < host.Rows(); ++i)
            for (size_t j = host.RowSize(i); j < 40; ++j)
                CHECK(flat[i * 40 + j] == 0);
    }
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

__global__ void Scale(Span<const double> input, Span<double> output)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input.size)
        output[i] = 3 * input[i];
}

TEST_CASE("External CUDA allocation and shared CSR metadata survive pending kernels")
{
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    const auto context = ExecutionContext::CUDA(0, stream);
    int releases = 0;
    bool releaseOK = true;
    {
        const auto shape = RowLayout::Ragged({0, 5, 19, 0, 513, 7});
        void *pointer = nullptr;
        const auto bytes = shape.Elements() * sizeof(double);
        REQUIRE(cudaMalloc(&pointer, bytes) == cudaSuccess);
        auto owner = std::shared_ptr<void>(pointer, [&releases, &releaseOK](void *p)
                                           { ++releases; releaseOK = cudaFree(p) == cudaSuccess; });
        auto allocation = Allocation::Import(owner, bytes, Device::CUDA(0));
        auto array = Array<double>::FromStorage(shape, Buffer::FromAllocation(allocation, 0, bytes));
        auto second = Array<double>::FromSharedStorage(array.SharedStorage()->WithValues(Buffer::Allocate(bytes, Device::CUDA(0)), sizeof(double)));
        CHECK(second.SharedStorage()->SharedLayout() == array.SharedStorage()->SharedLayout());
        CHECK(second.SharedStorage()->SharedLayout()->DeviceOffsetIdentity() == array.SharedStorage()->SharedLayout()->DeviceOffsetIdentity());
        auto write = array.Write(context);
        FillRows<<<7, 128, 0, stream>>>(write.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        write.Complete(Completion::Record(context));
        auto retained = array;
        array.Reset(RowLayout::Dense(0, 0));
        owner.reset();
        allocation = {};
        CHECK(releases == 0);
        write.Wait();
        auto host = retained.To(Device::Host(), context);
        CHECK(host.Read().At(4, 512) == 912);
        CHECK(host.RowSize(0) == 0);
        retained.Reset(RowLayout::Dense(0, 0));
        CHECK(releases == 1);
        auto edit = second.Write(context);
        FillRows<<<3, 64, 0, stream>>>(edit.View());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        edit.Complete(Completion::Record(context));
        edit.Wait();
        CHECK(second.To(Device::Host(), context).Read().At(5, 6) == 506);
    }
    CHECK(releaseOK);
    CHECK(releases == 1);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("Runtime CUDA residency, cross-stream transfer and retained leases")
{
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    cudaStream_t stream1, stream2;
    REQUIRE(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking) == cudaSuccess);
    {
        constexpr size_t count = 513;
        auto h = Buffer::Allocate(count * sizeof(double));
        {
            auto w = h.Write();
            for (size_t i = 0; i < count; ++i)
                w.View<double>()[i] = double(i);
        }
        auto d = Buffer::Allocate(h.Bytes(), Device::CUDA(0));
        auto out = Buffer::Allocate(h.Bytes(), Device::CUDA(0));
        auto c1 = ExecutionContext::CUDA(0, stream1);
        auto c2 = ExecutionContext::CUDA(0, stream2);
#if CUDART_VERSION >= 13000
        int deviceCount = 0;
        REQUIRE(cudaGetDeviceCount(&deviceCount) == cudaSuccess);
        if (deviceCount > 1)
        {
            auto second = Buffer::Allocate(8, Device::CUDA(1));
            CHECK_THROWS(second.Read(ExecutionContext::CUDA(1, stream1)));
        }
#endif
        {
            auto write = d.Write(c1);
            CHECK_THROWS(write.Complete(Completion::Record(c2)));
        }
        CHECK_THROWS(d.Read());
        CHECK_THROWS(d.Write(ExecutionContext::CUDA(1)));
        auto upload = Copy(h, d, c1);
        CHECK_THROWS(h.Write());
        CHECK_THROWS(d.Read(c1));
        upload.Wait();
        auto r = d.Read(c2);
        auto w = out.Write(c2);
        CHECK_THROWS(r.View<double>()[0]);
        Scale<<<3, 256, 0, stream2>>>(r.View<double>(), w.View<double>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        auto done = Completion::Record(c2);
        r.Complete(done);
        w.Complete(done);
        CHECK_THROWS(w.View<double>());
        d = {}; // r still owns the in-flight input allocation
        r.Wait();
        w.Wait();
        auto result = Buffer::Allocate(h.Bytes());
        auto download = Copy(out, result, c1);
        out = {}; // download retains both allocations
        download.Wait();
        auto read = result.Read();
        for (size_t i = 0; i < count; ++i)
            CHECK(read.View<double>()[i] == 3 * double(i));
    }
    REQUIRE(cudaStreamDestroy(stream1) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(stream2) == cudaSuccess);
}

TEST_CASE("Legacy device mirror cannot supply stale native host access")
{
    DNDS::Array<double, 1> array;
    array.Resize(2);
    array(0, 0) = 5;
    array.to_device(DNDS::DeviceBackend::CUDA);
    CHECK_THROWS(array.HostReadData());
    CHECK_THROWS(array.HostWriteData());
    auto before = array.HostDataVersion();
    array.to_host();
    array.clear_device();
    CHECK(array.HostDataVersion() > before);
    CHECK(array.HostReadData().View<double>()[0] == 5);
}

TEST_CASE("Runtime CUDA destructor fences an unsubmitted launch")
{
    auto a = Buffer::Allocate(128 * sizeof(double), Device::CUDA(0));
    auto b = Buffer::Allocate(a.Bytes(), Device::CUDA(0));
    auto context = ExecutionContext::CUDA(0);
    {
        auto r = a.Read(context);
        auto w = b.Write(context);
        Scale<<<1, 128>>>(r.View<double>(), w.View<double>());
        REQUIRE(cudaGetLastError() == cudaSuccess);
        // No Complete: lease destructors must fence before allowing buffer reuse.
    }
    CHECK(a.IsIdle());
    CHECK(b.Version() == 1);
    auto empty = Buffer::Allocate(0, Device::CUDA(0));
    auto emptyHost = Buffer::Allocate(0);
    Copy(empty, emptyHost, context).Wait();
}
