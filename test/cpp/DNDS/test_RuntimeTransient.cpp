#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/DynamicRowTransfer.hpp"
#include "DNDS/Runtime/MPIProgress.hpp"
#include "DNDS/Runtime/EagerArray.hpp"
#include "DNDS/Runtime/EagerCopy.hpp"
#include "DNDS/Runtime/CUDAExecution.hpp"

using namespace DNDS::Runtime;

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    int result = context.run();
    MPI_Finalize();
    return result;
}

TEST_CASE("Transient reduction stages collisions and rejects a partial publication")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    Array<int64_t> source(RowLayout::Dense(3, 7, 8));
    {
        auto w = source.Write();
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 7; ++j)
                w.At(i, j) = rank * 10 + i + j;
    }
    RowTransferPlan route(context, 3, 1, {{rank, 0, 0}, {(rank + 1) % size, 2, 0}});
    CHECK_THROWS(DynamicRowTransfer(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(),
                                    TransferTransport::Direct, {TransferMode::Transient, context, 61}));
    for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
    {
        DynamicRowTransfer op(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(),
                              transport, {TransferMode::Transient, context, 61}, Reducer::Builtin<int64_t>(Reduction::Sum, DestinationInit::Identity));
        auto result = Array<int64_t>::FromSharedStorage(op.Commit());
        auto read = result.Read();
        for (size_t j = 0; j < 7; ++j)
            CHECK(read.At(0, j) == rank * 10 + (rank + 1) % size * 10 + 2 + 2 * j);
    }
    auto bad = Reducer::Custom<int64_t>(ElementSchema::Arithmetic<int64_t>(), "test.fail-on-root", [rank](int64_t &a, const int64_t &b)
                                        { if (rank == 0) throw std::runtime_error("injected reduction failure"); a += b; }, int64_t(0));
    DynamicRowTransfer rejected(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(),
                                TransferTransport::Direct, {TransferMode::Transient, context, 62}, bad);
    CHECK_THROWS(rejected.Commit());
    CHECK_THROWS(rejected.Commit());
}

#ifdef DNDS_USE_CUDA
TEST_CASE("CUDA eager producer portable packed MPI and consumer retain ordering")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EagerEngine engine(2);
    auto stream = engine.CreateStream(std::make_shared<CUDALauncher>(0));
    Array<double> host(RowLayout::Dense(67, 6, 8));
    {
        auto w = host.Write();
        for (size_t i = 0; i < 67; ++i)
            for (size_t j = 0; j < 6; ++j)
                w.At(i, j) = rank * 10000 + i * 10 + j;
    }
    auto source = AllocateArrayAsync<double>(engine, stream, host.Layout());
    auto destination = AllocateLikeAsync(engine, stream, source);
    auto sourceBuffer = source.Storage();
    CopyAsync(engine, stream, host.Storage(), sourceBuffer);
    stream.Record().WaitHost(); // Producer observation retires its write reservation.
    std::vector<RowRequest> requests;
    for (size_t i = 0; i < 67; ++i)
        requests.push_back({(rank + 1) % size, int64_t(66 - i), i});
    RowTransferPlan route(MPI_COMM_WORLD, 67, 67, requests);
    auto binding = PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>())
                       .Bind(source, destination, TransferTransport::HostPacked, {TransferMode::Transient, {}, 0});
    MPIProgress progress;
    auto event = progress.Import(binding.BeginScheduled(stream.Context()));
    event.WaitHost();
    stream.WaitFor(event);
    Array<double> output(host.Layout());
    auto outputBuffer = output.Storage();
    CopyAsync(engine, stream, destination.Storage(), outputBuffer);
    stream.Record().WaitHost();
    progress.Drain();
    engine.Synchronize();
    auto read = output.Read();
    for (size_t i = 0; i < 67; ++i)
        for (size_t j = 0; j < 6; ++j)
            CHECK(read.At(i, j) == (rank + 1) % size * 10000 + (66 - i) * 10 + j);
}
#endif

TEST_CASE("Transient and persistent contexts transfer permuted SoA rows")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size, previous = (rank + size - 1) % size;
    constexpr size_t rows = 193;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    std::vector<std::vector<size_t>> send(size), receive(size);
    for (size_t i = 0; i < rows; ++i)
    {
        send[previous].push_back(i * 17 % rows);
        receive[next].push_back(i);
    }
    auto route = RowTransferPlan::Known(context, rows, rows, send, receive);
    std::vector<size_t> lengths(rows);
    for (size_t i = 0; i < rows; ++i)
        lengths[i] = i % 8;
    Array<double> source(RowLayout::Padded(lengths, 11));
    Array<int64_t> labels(RowLayout::Dense(rows, 2, 3));
    {
        auto x = source.Write();
        auto ids = labels.Write();
        for (size_t i = 0; i < rows; ++i)
        {
            ids.At(i, 0) = rank * 10000 + i;
            ids.At(i, 1) = -int64_t(i);
            for (size_t j = 0; j < lengths[i]; ++j)
                x.At(i, j) = rank * 10000 + i * 10 + j;
        }
    }
    Array<double> target(route.DestinationLayout(source.Layout()));
    Array<int64_t> targetIDs(RowLayout::Dense(rows, 2, 4));
    PayloadPlan values(route, source.Layout(), target.Layout(), ElementSchema::Arithmetic<double>());
    PayloadPlan ids(route, labels.Layout(), targetIDs.Layout(), ElementSchema::Arithmetic<int64_t>());
    for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
        for (auto transport : {TransferTransport::Direct, TransferTransport::HostPacked})
        {
            auto first = values.Bind(source, target, transport, {mode, context, 37});
            auto second = ids.Bind(labels, targetIDs, transport, {mode, context, 38});
            CHECK_THROWS(values.Bind(source, target, transport, {mode, context, 37}));
            for (int repeat = 0; repeat < 3; ++repeat)
            {
                auto a = first.Begin();
                auto b = second.Begin();
                CHECK_THROWS(first.Begin());
                b.Wait();
                a.Wait();
                auto y = target.Read();
                auto indices = targetIDs.Read();
                for (size_t i = 0; i < rows; ++i)
                {
                    const auto from = i * 17 % rows;
                    CHECK(target.RowSize(i) == lengths[from]);
                    CHECK(indices.At(i, 0) == next * 10000 + from);
                    CHECK(indices.At(i, 1) == -int64_t(from));
                    for (size_t j = 0; j < lengths[from]; ++j)
                        CHECK(y.At(i, j) == next * 10000 + from * 10 + j);
                }
            }
        }
    auto badSend = send;
    if (rank == 0)
        badSend[previous].pop_back();
    CHECK_THROWS(RowTransferPlan::Known(context, rows, rows, badSend, receive));
}

TEST_CASE("Dynamic CSR migration privately prepares shapes and publishes once")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int peer = (rank + 1) % size;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    for (size_t epoch = 0; epoch < 4; ++epoch)
    {
        const size_t rows = 67 + epoch * 2;
        std::vector<size_t> widths(rows);
        for (size_t i = 0; i < rows; ++i)
            widths[i] = (i + rank + epoch) % 9;
        Array<int64_t> source(RowLayout::Ragged(widths));
        {
            auto w = source.Write();
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < widths[i]; ++j)
                    w.At(i, j) = rank * 100000 + i * 100 + j;
        }
        std::vector<RowRequest> requests;
        for (size_t i = 0; i < rows; ++i)
            requests.push_back({peer, int64_t(rows - i - 1), i});
        RowTransferPlan route(context, rows, rows + 1, requests);
        DynamicRowTransfer operation(route, source.SharedStorage(), ElementSchema::Arithmetic<int64_t>(), Device::Host(),
                                     TransferTransport::Direct, {TransferMode::Transient, context, 51});
        CHECK_THROWS(source.Write());
        source.Reset(RowLayout::Dense(1, 1)); // Operation retains the old snapshot.
        auto destination = Array<int64_t>::FromSharedStorage(operation.Commit());
        CHECK_THROWS(operation.Commit());
        REQUIRE(destination.RowSize(rows) == 0);
        auto read = destination.Read();
        for (size_t i = 0; i < rows; ++i)
        {
            const auto from = rows - i - 1;
            REQUIRE(destination.RowSize(i) == (from + peer + epoch) % 9);
            for (size_t j = 0; j < destination.RowSize(i); ++j)
                CHECK(read.At(i, j) == peer * 100000 + from * 100 + j);
        }
    }
}

TEST_CASE("Main-thread MPI progression joins worker computation with scheduled leases")
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr size_t rows = 257;
    for (size_t workers : {size_t(0), size_t(2)})
    {
        EagerEngine engine(workers);
        auto core = engine.CreateCPUStream(), ring = engine.CreateCPUStream();
        Array<double> owned(RowLayout::Dense(rows, 6, 8)), ghost(RowLayout::Dense(19, 6, 8));
        Array<double> output(RowLayout::Dense(rows, 6));
        {
            auto w = owned.Write();
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < 6; ++j)
                    w.At(i, j) = rank * 10000 + i * 10 + j;
        }
        std::vector<RowRequest> requests;
        for (size_t i = 0; i < 19; ++i)
            requests.push_back({(rank + 1) % size, int64_t(i * 13), i});
        RowTransferPlan route(MPI_COMM_WORLD, rows, 19, requests);
        auto binding = PayloadPlan(route, owned.Layout(), ghost.Layout(), ElementSchema::Arithmetic<double>())
                           .Bind(owned, ghost, TransferTransport::Direct, {TransferMode::Transient, {}, 0});
        MPIProgress progress;
        auto event = progress.Import(binding.BeginScheduled());
        CHECK_THROWS(ghost.Read());
        SubmitArrays(engine, core, [](ExecutionContext, auto u, auto rhs)
                     { for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < 6; ++j) rhs(i, j) = 2 * u(i, j); }, Read(owned), Write(output));
        ring.WaitFor(event);
        ring.WaitFor(core.Record());
        SubmitArrays(engine, ring, [](ExecutionContext, auto g, auto rhs)
                     { for (size_t i = 0; i < 19; ++i) for (size_t j = 0; j < 6; ++j) rhs(i, j) += g(i, j); }, Read(ghost), ReadWrite(output));
        progress.Drain();
        engine.Synchronize();
        auto r = output.Read();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < 6; ++j)
                CHECK(r.At(i, j) == 2 * (rank * 10000 + i * 10 + j) + (i < 19 ? (rank + 1) % size * 10000 + i * 130 + j : 0));
    }
}
