#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/RowTransfer.hpp"
#include <optional>

using namespace DNDS::Runtime;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    int result = ctx.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Native direct ragged halo retains leases and reusable routing")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size;
    Array<double> owned(RowLayout::Ragged({2, 0, 3}));
    {
        auto write = owned.Write();
        for (size_t i = 0; i < owned.Rows(); ++i)
            for (size_t j = 0; j < owned.RowSize(i); ++j)
                write.At(i, j) = rank * 100 + i * 10 + j;
    }
    RowTransferPlan route(MPI_COMM_WORLD, 3, 3, {{next, 2, 0}, {next, 1, 1}, {rank, 0, 2}});
    Array<double> ghost(route.DestinationLayout(owned.Layout()));
    PayloadPlan payload(route, owned.Layout(), ghost.Layout(), ElementSchema::Arithmetic<double>());
    auto binding = payload.Bind(owned, ghost);
    const auto version = ghost.Version();
    {
        auto operation = binding.Begin();
        auto coreRead = owned.Read();
        CHECK(coreRead.At(0, 0) == rank * 100);
        CHECK_THROWS(owned.Write());
        CHECK_THROWS(ghost.Read());
        CHECK_THROWS(binding.Begin());
        CHECK(ghost.Version() == version);
        operation.WaitDestinationReady();
        CHECK(ghost.Version() == version + 1);
        auto ringRead = ghost.Read();
        CHECK(ringRead.At(0, 2) == next * 100 + 22);
        CHECK(ringRead.At(2, 1) == rank * 100 + 1);
        CHECK_THROWS(owned.Write()); // Send lease remains owned until full completion.
        operation.Wait();
    }
    binding.Begin().Wait();
    payload.Bind(owned, ghost, TransferTransport::HostPacked).Begin().Wait();
    CHECK(ghost.Read().At(0, 2) == next * 100 + 22);
    auto badShape = Array<double>(RowLayout::Dense(1, owned.Layout().Elements()));
    CHECK_THROWS(payload.Bind(badShape, ghost));
    {
        std::optional<Array<double>::ReadLease> conflict;
        if (rank == 0)
            conflict.emplace(ghost.Read());
        CHECK_THROWS(binding.Begin());
    }
    binding.Begin().Wait();
    Array<int32_t> second(RowLayout::Dense(3, 4));
    Array<int32_t> secondGhost(route.DestinationLayout(second.Layout()));
    PayloadPlan(route, second.Layout(), secondGhost.Layout(), ElementSchema::Arithmetic<int32_t>())
        .Bind(second, secondGhost)
        .Begin()
        .Wait();
}

TEST_CASE("Direct permutation honors imported buffer origins and padded state strides")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const auto peer = (rank + 1) % size;
    constexpr size_t rows = 31, width = 7, prefix = 13, suffix = 17;
    auto sourceOwner = std::make_shared<std::vector<double>>(prefix + rows * 8 + suffix, -9);
    auto targetOwner = std::make_shared<std::vector<double>>(prefix + rows * 9 + suffix, -19);
    auto sourceAllocation = Allocation::Import(std::shared_ptr<void>(sourceOwner, sourceOwner->data()), sourceOwner->size() * sizeof(double));
    auto targetAllocation = Allocation::Import(std::shared_ptr<void>(targetOwner, targetOwner->data()), targetOwner->size() * sizeof(double));
    auto source = Array<double>::FromStorage(RowLayout::Dense(rows, width, 8),
                                             Buffer::FromAllocation(sourceAllocation, prefix * sizeof(double), rows * 8 * sizeof(double)));
    auto target = Array<double>::FromStorage(RowLayout::Dense(rows, width, 9),
                                             Buffer::FromAllocation(targetAllocation, prefix * sizeof(double), rows * 9 * sizeof(double)));
    {
        auto write = source.Write();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < width; ++j)
                write.At(i, j) = rank * 10000 + i * 100 + j;
    }
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < rows; ++i)
        requests.push_back({peer, int64_t(i * 7 % rows), i});
    RowTransferPlan route(MPI_COMM_WORLD, rows, rows, requests);
    PayloadPlan payload(route, source.Layout(), target.Layout(), ElementSchema::Arithmetic<double>());
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        payload.Bind(source, target, transport).Begin().Wait();
        auto read = target.Read();
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < width; ++j)
                CHECK(read.At(i, j) == peer * 10000 + (i * 7 % rows) * 100 + j);
            CHECK((*targetOwner)[prefix + i * 9 + 7] == -19);
            CHECK((*targetOwner)[prefix + i * 9 + 8] == -19);
            CHECK((*sourceOwner)[prefix + i * 8 + 7] == -9);
        }
        for (size_t i = 0; i < prefix; ++i)
            CHECK((*targetOwner)[i] == -19);
        for (size_t i = prefix + rows * 9; i < targetOwner->size(); ++i)
            CHECK((*targetOwner)[i] == -19);
    }
}

TEST_CASE("One route transfers padded reconstruction rows into a packed CSR destination")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr size_t count = 37;
    std::vector<size_t> lengths(count);
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < count; ++i)
    {
        lengths[i] = i % 11 == 0 ? 0 : (i % 7 + 1) * 5;
        requests.push_back({(rank + 1) % size, int64_t(i * 5 % count), i});
    }
    Array<double> source(RowLayout::Padded(lengths, 40));
    {
        auto write = source.Write();
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < lengths[i]; ++j)
                write.At(i, j) = rank * 10000 + i * 100 + j;
    }
    RowTransferPlan route(MPI_COMM_WORLD, count, count, requests);
    Array<double> destination(route.DestinationLayout(source.Layout()));
    CHECK(destination.Layout().IsRagged());
    auto payload = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>());
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        payload.Bind(source, destination, transport).Begin().Wait();
        auto read = destination.Read();
        for (size_t i = 0; i < count; ++i)
        {
            const auto original = i * 5 % count;
            CHECK(destination.RowSize(i) == lengths[original]);
            for (size_t j = 0; j < lengths[original]; ++j)
                CHECK(read.At(i, j) == ((rank + 1) % size) * 10000 + original * 100 + j);
        }
    }
}

TEST_CASE("Native reverse fan-in rejects copy and supports reductions")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Array<int32_t> root(RowLayout::Dense(rank == 0 ? 1 : 0, 2));
    Array<int32_t> leaves(RowLayout::Dense(2, 2));
    RowTransferPlan halo(MPI_COMM_WORLD, root.Rows(), 2, {{0, 0, 0}, {0, 0, 1}});
    PayloadPlan reverse(halo.Reverse(), leaves.Layout(), root.Layout(), ElementSchema::Arithmetic<int32_t>());
    CHECK_THROWS(reverse.Bind(leaves, root));
    {
        auto write = leaves.Write();
        write.At(0, 0) = rank + 1;
        write.At(1, 0) = rank + 2;
        write.At(0, 1) = 1;
        write.At(1, 1) = 2;
    }
    for (auto op : {Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::BitOr, Reduction::BitXor, Reduction::BitAnd})
    {
        reverse.Bind(leaves, root, Reducer::Builtin<int32_t>(op, DestinationInit::Identity)).Begin().Wait();
        if (rank == 0)
        {
            auto result = root.Read();
            if (op == Reduction::Sum)
                CHECK(result.At(0, 0) == size * (size + 2));
            if (op == Reduction::Min)
                CHECK(result.At(0, 0) == 1);
            if (op == Reduction::Max)
                CHECK(result.At(0, 0) == size + 1);
            if (op == Reduction::BitOr)
                CHECK(result.At(0, 1) == 3);
            if (op == Reduction::BitXor)
                CHECK(result.At(0, 1) == (size % 2 ? 3 : 0));
            if (op == Reduction::BitAnd)
                CHECK(result.At(0, 1) == 0);
        }
    }
    if (rank == 0)
        root.Write().At(0, 0) = 7;
    reverse.Bind(leaves, root, Reducer::Builtin<int32_t>(Reduction::Sum)).Begin().Wait();
    if (rank == 0)
        CHECK(root.Read().At(0, 0) == 7 + size * (size + 2));
}

TEST_CASE("Native POD transfers use MPI extent and support custom reductions")
{
    struct Metadata
    {
        int32_t tag;
        double weight;
    };
    MPI_Datatype raw, pod;
    int blocks[] = {1, 1};
    MPI_Aint offsets[] = {offsetof(Metadata, tag), offsetof(Metadata, weight)};
    MPI_Datatype types[] = {MPI_INT32_T, MPI_DOUBLE};
    MPI_Type_create_struct(2, blocks, offsets, types, &raw);
    MPI_Type_create_resized(raw, 0, sizeof(Metadata), &pod);
    MPI_Type_commit(&pod);
    ElementSchema schema(pod, sizeof(Metadata), "test.metadata.v1");
    MPI_Type_free(&raw);
    MPI_Type_free(&pod);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Array<Metadata> src(RowLayout::Dense(3, 2)), dst(RowLayout::Dense(3, 2));
    {
        auto write = src.Write();
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 2; ++j)
                write.At(i, j) = {rank, double(i * 2 + j)};
    }
    RowTransferPlan route(MPI_COMM_WORLD, 3, 3, {{(rank + 1) % size, 0, 0}, {(rank + 1) % size, 1, 1}, {(rank + 1) % size, 2, 2}});
    PayloadPlan payload(route, src.Layout(), dst.Layout(), schema);
    CHECK(payload.SendSegments() == 1);
    CHECK(payload.ReceiveSegments() == 1);
    payload.Bind(src, dst).Begin().Wait();
    CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
    CHECK(dst.Read().At(2, 1).weight == 5);
    payload.Bind(src, dst, TransferTransport::HostPacked).Begin().Wait();
    CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
    auto add = Reducer::Custom<Metadata>(schema, "metadata.sum", [](Metadata &a, const Metadata &b)
                                         { a.tag += b.tag; a.weight += b.weight; }, Metadata{}, DestinationInit::Identity);
    payload.Bind(src, dst, add).Begin().Wait();
    CHECK(dst.Read().At(2, 1).weight == 5);
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        payload.Bind(src, dst, transport, {TransferMode::Transient, {}, 0}).Begin().Wait();
        CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
        CHECK(dst.Read().At(2, 1).weight == 5);
        payload.Bind(src, dst, add, transport, {TransferMode::Transient, {}, 0}).Begin().Wait();
        CHECK(dst.Read().At(2, 1).tag == (rank + 1) % size);
        CHECK(dst.Read().At(2, 1).weight == 5);
    }
#ifdef DNDS_USE_CUDA
    auto deviceSource = src.To(Device::CUDA(0));
    auto deviceDestination = dst.To(Device::CUDA(0));
    payload.Bind(deviceSource, deviceDestination, DNDS::MPI::isCudaAware() ? TransferTransport::Direct : TransferTransport::HostPacked)
        .Begin(ExecutionContext::CUDA(0))
        .Wait();
    auto deviceResult = deviceDestination.To(Device::Host());
    CHECK(deviceResult.Read().At(2, 1).tag == (rank + 1) % size);
    CHECK(deviceResult.Read().At(2, 1).weight == 5);
#endif
}

TEST_CASE("Native empty routes and generation validation")
{
    VersionTracker tracker;
    RowTransferPlan route(MPI_COMM_WORLD, 0, 0, {}, {tracker.Stamp(Change::Layout)});
    Array<double> src, dst;
    auto binding = PayloadPlan(route, src.Layout(), dst.Layout(), ElementSchema::Arithmetic<double>()).Bind(src, dst);
    CHECK(binding.Begin().Test());
    PayloadPlan(route, src.Layout(), dst.Layout(), ElementSchema::Arithmetic<double>())
        .Bind(src, dst, TransferTransport::HostPacked)
        .Begin()
        .Wait();
    auto mutation = tracker.BeginMutation(Change::Layout);
    mutation.Commit();
    CHECK_THROWS(binding.Begin());
}

TEST_CASE("Native reducer failure drains MPI and invalidates partial output")
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Array<int32_t> a(RowLayout::Dense(1, 1)), b(RowLayout::Dense(1, 1));
    RowTransferPlan route(MPI_COMM_WORLD, 1, 1, {{rank, 0, 0}});
    auto reducer = Reducer::Custom<int32_t>(ElementSchema::Arithmetic<int32_t>(), "throws", [](int32_t &, int32_t)
                                            { throw std::runtime_error("custom reduction failed"); }, 0);
    auto binding = PayloadPlan(route, a.Layout(), b.Layout(), ElementSchema::Arithmetic<int32_t>()).Bind(a, b, reducer);
    auto operation = binding.Begin();
    CHECK_THROWS(operation.Wait());
    CHECK_FALSE(b.IsValid());
    CHECK_THROWS(b.Read());
    CHECK(a.Storage().IsIdle());
    CHECK(b.Storage().IsIdle());
    operation.Wait(); // No repeated combine/destructor failure.
}

#ifdef DNDS_USE_CUDA
TEST_CASE("Native CUDA integer reduction families agree with the CPU reference")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (!DNDS::MPI::isCudaAware())
        return;
    Array<int32_t> source(RowLayout::Dense(1, 2)), cpu(RowLayout::Dense(1, 2));
    source.Write().At(0, 0) = rank + 1;
    source.Write().At(0, 1) = 1 << (rank % 8);
    auto gpuSource = source.To(Device::CUDA(0));
    auto gpu = cpu.To(Device::CUDA(0));
    std::vector<RowRequest> requests;
    for (int peer = 0; peer < size; ++peer)
    {
        requests.push_back({peer, 0, 0});
        requests.push_back({peer, 0, 0});
    }
    RowTransferPlan route(MPI_COMM_WORLD, 1, 1, requests);
    PayloadPlan payload(route, source.Layout(), cpu.Layout(), ElementSchema::Arithmetic<int32_t>());
    for (auto op : {Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::BitOr, Reduction::BitXor, Reduction::BitAnd})
    {
        auto reducer = Reducer::Builtin<int32_t>(op, DestinationInit::Identity);
        payload.Bind(source, cpu, reducer).Begin().Wait();
        payload.Bind(gpuSource, gpu, reducer).Begin(ExecutionContext::CUDA(0)).Wait();
        auto result = gpu.To(Device::Host());
        CHECK(result.Read().At(0, 0) == cpu.Read().At(0, 0));
        CHECK(result.Read().At(0, 1) == cpu.Read().At(0, 1));
    }
}

TEST_CASE("Native CUDA direct MPI and device fan-in reduction")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto ctx = ExecutionContext::CUDA(0);
    Array<double> host(RowLayout::Ragged({0, 3}));
    {
        auto w = host.Write();
        for (size_t j = 0; j < 3; ++j)
            w.At(1, j) = rank + j;
    }
    auto source = host.To(Device::CUDA(0), ctx);
    Array<double> destination(source.Layout(), Device::CUDA(0));
    RowTransferPlan route(MPI_COMM_WORLD, 2, 2, {{(rank + 1) % size, 0, 0}, {(rank + 1) % size, 1, 1}, {rank, 1, 1}});
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>());
    auto reducer = Reducer::Builtin<double>(Reduction::Sum, DestinationInit::Identity);
    payload.Bind(source, destination, reducer, TransferTransport::HostPacked).Begin(ctx).Wait();
    CHECK(destination.To(Device::Host(), ctx).Read().At(1, 2) == rank + (rank + 1) % size + 4);
    if (!DNDS::MPI::isCudaAware())
    {
        CHECK_THROWS(payload.Bind(source, destination, reducer));
        MESSAGE("Direct device MPI unavailable; capability rejection verified");
        return;
    }
    auto binding = payload.Bind(source, destination, reducer);
    binding.Begin(ctx).Wait();
    auto result = destination.To(Device::Host(), ctx);
    CHECK(result.Read().At(1, 2) == rank + (rank + 1) % size + 4);
}
#endif
