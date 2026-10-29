#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "DNDS/Runtime/RowTransfer.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
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

TEST_CASE("Large permuted routes preserve row order padding and rank-local collision flags")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    constexpr DNDS::index rows = 8193, width = 3, step = 4099;
    const int previous = (rank + size - 1) % size, next = (rank + 1) % size;
    const auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    Array<int64_t> source(RowLayout::Dense(rows, width, 5)), destination(RowLayout::Dense(rows, width, 6));
    {
        auto write = source.Storage().Write();
        std::fill_n(write.View<int64_t>().data, rows * 5, int64_t(-999));
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            for (DNDS::index iCol = 0; iCol < width; ++iCol)
                write.View<int64_t>()[iRow * 5 + iCol] = rank * 1000000 + iRow * 16 + iCol;
    }
    for (bool duplicate : {false, true})
    {
        std::vector<std::vector<DNDS::index>> send(size), receive(size);
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
        {
            send[next].push_back(iRow * step % rows);
            receive[previous].push_back(iRow);
        }
        if (duplicate && rank == 0)
            send[next].back() = send[next].front();
        auto route = RowTransferPlan::Known(context, rows, rows, send, receive);
        CHECK_FALSE(route.HasCollisions());
        CHECK(route.Reverse().HasCollisions() == (duplicate && rank == 0));
        {
            auto write = destination.Storage().Write();
            std::fill_n(write.View<int64_t>().data, rows * 6, int64_t(-777));
        }
        const auto schema = ElementSchema::Arithmetic<int64_t>();
        PayloadPlan payload(route, source.Layout(), destination.Layout(), schema);
        payload.Bind(source, destination).Begin().Wait();
        const auto verify = [&]
        {
            auto input = source.Storage().Read(), output = destination.Storage().Read();
            const auto in = input.View<int64_t>(), out = output.View<int64_t>();
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            {
                const auto original = duplicate && previous == 0 && iRow == rows - 1 ? 0 : iRow * step % rows;
                for (DNDS::index iCol = 0; iCol < width; ++iCol)
                    CHECK(out[iRow * 6 + iCol] == previous * 1000000 + original * 16 + iCol);
                for (DNDS::index iCol = width; iCol < 6; ++iCol)
                    CHECK(out[iRow * 6 + iCol] == -777);
                for (DNDS::index iCol = width; iCol < 5; ++iCol)
                    CHECK(in[iRow * 5 + iCol] == -999);
            }
        };
        verify();
        if (duplicate)
        {
            PayloadPlan reverse(route.Reverse(), destination.Layout(), source.Layout(), schema);
            CHECK_THROWS(reverse.Bind(destination, source));
            // A failed reverse admission must leave the matching namespace
            // reusable and must not alter the forward row correspondence.
            payload.Bind(source, destination).Begin().Wait();
            verify();
        }
    }
}

TEST_CASE("Complete small row relations agree with independent sorted collision oracle")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    for (DNDS::index rows = 0; rows <= 4; ++rows)
    {
        DNDS::index combinations = 1;
        for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            combinations *= rows;
        for (DNDS::index pattern = 0; pattern < combinations; ++pattern)
        {
            std::vector<std::vector<DNDS::index>> send(size), receive(size);
            auto value = pattern;
            for (DNDS::index iRow = 0; iRow < rows; ++iRow)
            {
                send[rank].push_back(value % rows);
                value /= rows;
                receive[rank].push_back(iRow);
            }
            auto oracle = send[rank];
            std::sort(oracle.begin(), oracle.end());
            const bool duplicate = std::adjacent_find(oracle.begin(), oracle.end()) != oracle.end();
            auto route = RowTransferPlan::Known(context, rows, rows, send, receive);
            CHECK_FALSE(route.HasCollisions());
            CHECK(route.Reverse().HasCollisions() == duplicate);
        }
    }
}

TEST_CASE("Typed binding rejects same-size arithmetic schema confusion")
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    RowTransferPlan route(MPI_COMM_WORLD, 31, 31, {{rank, 17, 3}});
    Array<int64_t> source(RowLayout::Dense(31, 7, 9)), destination(source.Layout());
    PayloadPlan wrong(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<double>());
    CHECK_THROWS(wrong.Bind(source, destination));
    CHECK_THROWS(wrong.Bind(source, destination, Reducer::Builtin<double>(Reduction::Sum)));
    CHECK_THROWS(Reducer::Custom<int64_t>(ElementSchema::Arithmetic<double>(), "wrong.scalar", [](int64_t &a, const int64_t &b)
                                          { a += b; }, int64_t(0)));
    CHECK(source.Storage().IsIdle());
    CHECK(destination.Storage().IsIdle());
}

TEST_CASE("Sender ordering derives a checked route without changing peer row correspondence")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size, previous = (rank + size - 1) % size;
    const std::vector<DNDS::index> sourceOrder{4, 1, 5, 0, 3, 2}, destinationOrder{3, 0, 5, 2, 1, 4};
    std::vector<std::vector<DNDS::index>> send(size), receive(size);
    send[next] = sourceOrder;
    receive[previous] = destinationOrder;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    VersionTracker epoch;
    auto route = RowTransferPlan::Known(context, 6, 6, send, receive, {epoch.Stamp(Change::Distribution)});
    auto ordered = route.ReorderSourceByPeer();
    CHECK(ordered.first.Context() == context);
    CHECK(route.SendRows() == send);
    CHECK(ordered.first.ReceiveRows() == receive);
    CHECK_FALSE(ordered.first.HasCollisions());
    CHECK(ordered.first.SendRows()[next] == std::vector<DNDS::index>{0, 1, 2, 3, 4, 5});
    Array<int64_t> original(RowLayout::Ragged({1, 2, 3, 1, 2, 3}));
    std::vector<size_t> widths(6);
    {
        auto write = original.Write();
        const auto view = write.View();
        for (DNDS::index iRow = 0; iRow < 6; ++iRow)
        {
            widths[size_t(ordered.second[size_t(iRow)])] = size_t(view.RowSize(iRow));
            for (DNDS::index iSlot = 0; iSlot < view.RowSize(iRow); ++iSlot)
                view(iRow, iSlot) = rank * 100 + iRow * 10 + iSlot;
        }
    }
    Array<int64_t> permuted(RowLayout::Ragged(widths));
    {
        auto read = original.Read();
        auto write = permuted.Write();
        for (DNDS::index iRow = 0; iRow < 6; ++iRow)
            std::copy(read.View()[iRow].begin(), read.View()[iRow].end(), write.View()[ordered.second[size_t(iRow)]].begin());
    }
    PayloadPlan payload(ordered.first, permuted.Layout(), ElementSchema::Arithmetic<int64_t>());
    Array<int64_t> destination(payload.DestinationLayout());
    payload.Bind(permuted, destination).Begin().Wait();
    auto read = destination.Read();
    for (DNDS::index iEdge = 0; iEdge < 6; ++iEdge)
    {
        const auto iRow = destinationOrder[size_t(iEdge)], sourceRow = sourceOrder[size_t(iEdge)];
        CHECK(read.View().RowSize(iRow) == 1 + sourceRow % 3);
        for (DNDS::index iSlot = 0; iSlot < read.View().RowSize(iRow); ++iSlot)
            CHECK(read.View()(iRow, iSlot) == previous * 100 + sourceRow * 10 + iSlot);
    }
    const auto reverse = route.Reverse().ReorderSourceByPeer();
    CHECK(reverse.first.ReceiveRows() == send);
    if (size > 1)
        CHECK_THROWS((rank == 0 ? route.Reverse() : route).ReorderSourceByPeer());
    auto mutation = epoch.BeginMutation(Change::Distribution);
    mutation.Commit();
    CHECK_FALSE(ordered.first.Valid());
    CHECK_FALSE(reverse.first.Valid());
    CHECK_THROWS(route.ReorderSourceByPeer());

    auto incomplete = RowTransferPlan::Known(context, rank == 0 ? 7 : 6, 6, send, receive);
    CHECK_THROWS(incomplete.ReorderSourceByPeer());
    if (rank == 0)
        send[next].back() = send[next].front();
    auto duplicate = RowTransferPlan::Known(context, 6, 6, send, receive);
    CHECK_THROWS(duplicate.ReorderSourceByPeer());
    send[next] = {1, 0};
    receive[previous] = {0, 0};
    auto fanIn = RowTransferPlan::Known(context, 2, 1, send, receive).ReorderSourceByPeer();
    CHECK(fanIn.first.HasCollisions());
    CHECK_THROWS(fanIn.first.Reverse().ReorderSourceByPeer());
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
        for (size_t i = 0; i < owned.RowCount(); ++i)
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

TEST_CASE("Inferred payload shapes retain sparse holes widths and collective policy checks")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size;
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    Array<int64_t> source(RowLayout::Ragged({0, size_t(rank + 2), 3}));
    {
        auto write = source.Write();
        auto view = write.View();
        for (DNDS::index iRow = 0; iRow < view.RowCount(); ++iRow)
            for (DNDS::index iCol = 0; iCol < view.RowSize(iRow); ++iCol)
                view(iRow, iCol) = rank * 1000 + iRow * 10 + iCol;
    }
    RowTransferPlan route(context, 3, 4, {{next, 1, 3}, {next, 0, 1}, {rank, 2, 0}});
    CHECK(route.Context() == context);
    PayloadPlan payload(route, source.Layout(), ElementSchema::Arithmetic<int64_t>());
    CHECK(payload.DestinationLayout() == route.DestinationLayout(source.Layout()));
    Array<int64_t> destination(payload.DestinationLayout());
    for (auto mode : {TransferMode::Transient, TransferMode::Persistent})
    {
        auto binding = payload.Bind(source, destination, TransferTransport::Direct, {mode, context, 7});
        binding.Begin().Wait();
        auto read = destination.Read();
        CHECK(read.View().RowSize(1) == 0);
        CHECK(read.View().RowSize(2) == 0);
        CHECK(read.View().RowSize(3) == next + 2);
        CHECK(read.At(0, 2) == rank * 1000 + 22);
        CHECK(read.At(3, next + 1) == next * 1000 + 10 + next + 1);
    }
    RowTransferPlan collision(context, 3, 1, {{rank, 0, 0}, {rank, 2, 0}});
    CHECK_THROWS(PayloadPlan(collision, source.Layout(), ElementSchema::Arithmetic<int64_t>()));
    if (size > 1)
    {
        CHECK_THROWS(payload.Bind(source, destination, rank == 0 ? TransferTransport::HostPacked : TransferTransport::Direct,
                                  {TransferMode::Transient, context, 8}));
        auto mixed = [&]
        {
            if (rank == 0)
                return PayloadPlan(route, source.Layout(), ElementSchema::Arithmetic<int64_t>());
            return PayloadPlan(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
        }();
        mixed.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 8}).Begin().Wait();
        CHECK_THROWS([&]
                     {
            if (rank == 0)
                return PayloadPlan(route, source.Layout(), RowLayout::Dense(4, 9), ElementSchema::Arithmetic<int64_t>());
            return PayloadPlan(route, source.Layout(), ElementSchema::Arithmetic<int64_t>()); }());
    }
    payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 7}).Begin().Wait();
}

TEST_CASE("Table width agreement preserves sparse rows and handles rank-varying layouts")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int next = (rank + 1) % size;
    RowTransferPlan route(MPI_COMM_WORLD, 3, 4, {{next, 2, 3}, {rank, 1, 0}});
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        const DNDS::index width = scenario == 1 ? rank + 2 : 5;
        // Common-width tables, different widths on different ranks, and a
        // CSR/table mixture all follow one compatible collective protocol.
        const auto shape = scenario == 2 && rank == 0
                               ? RowLayout::Ragged({5, 5, 5})
                               : RowLayout::Dense(3, width, width + 2);
        Array<int64_t> source(shape);
        {
            auto write = source.Write();
            const auto view = write.View();
            for (DNDS::index iRow = 0; iRow < 3; ++iRow)
                for (DNDS::index iCol = 0; iCol < width; ++iCol)
                    view(iRow, iCol) = 1000 * rank + 100 * iRow + iCol;
        }
        PayloadPlan payload(route, source.Layout(), ElementSchema::Arithmetic<int64_t>());
        Array<int64_t> output(payload.DestinationLayout());
        payload.Bind(source, output).Begin().Wait();
        auto read = output.Read();
        const auto view = read.View();
        CHECK(view.RowSize(0) == width);
        CHECK(view.RowSize(1) == 0);
        CHECK(view.RowSize(2) == 0);
        const DNDS::index remoteWidth = scenario == 1 ? next + 2 : 5;
        CHECK(view.RowSize(3) == remoteWidth);
        for (DNDS::index iCol = 0; iCol < width; ++iCol)
            CHECK(view(0, iCol) == 1000 * rank + 100 + iCol);
        for (DNDS::index iCol = 0; iCol < remoteWidth; ++iCol)
            CHECK(view(3, iCol) == 1000 * next + 200 + iCol);
    }
}

TEST_CASE("Binding rejects reordered communicator and rank-local tag reservation failure")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    RowTransferPlan route(context, 1, 1, {{(rank + 1) % size, 0, 0}});
    Array<int64_t> source(RowLayout::Dense(1, 1)), destination(RowLayout::Dense(1, 1));
    {
        auto write = source.Write();
        write.At(0, 0) = rank + 19;
    }
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
    if (size > 1)
    {
        MPI_Comm reordered = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, 0, size - rank, &reordered);
        {
            auto wrong = std::make_shared<CommunicationContext>(reordered);
            CHECK_THROWS(payload.Bind(source, destination, TransferTransport::Direct,
                                      {TransferMode::Transient, wrong, 12}));
        }
        MPI_Comm_free(&reordered);
    }
    std::shared_ptr<const int> held;
    if (rank == 0)
        held = context->Reserve(12);
    CHECK_THROWS(payload.Bind(source, destination, TransferTransport::Direct,
                              {TransferMode::Transient, context, 12}));
    held.reset();
    // Failed ranks must retire any successful local reservations, so the
    // identical tag remains usable after the collectively rejected binding.
    payload.Bind(source, destination, TransferTransport::Direct,
                 {TransferMode::Transient, context, 12})
        .Begin()
        .Wait();
    CHECK(destination.Read().At(0, 0) == (rank + 1) % size + 19);
}

TEST_CASE("Combined payload and binding preflight reject rank-local mismatches before requests")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    VersionTracker versions;
    RowTransferPlan route(context, 3, 2, {{(rank + 1) % size, 2, 0}, {rank, 0, 1}}, {versions.Stamp()});
    Array<int64_t> source(RowLayout::Dense(3, 5, 7)), destination(RowLayout::Dense(2, 5, 8));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index iCol = 0; iCol < 5; ++iCol)
                write.At(iRow, iCol) = rank * 100 + iRow * 10 + iCol;
    }
    CHECK_THROWS(PayloadPlan(route, RowLayout::Dense(rank == 0 ? 2 : 3, 5), ElementSchema::Arithmetic<int64_t>()));
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
    // Same logical widths, different physical strides only on
    // one rank. Both typed copy and reduction must fail collectively.
    auto wrong = rank == 0 ? Array<int64_t>(RowLayout::Dense(2, 5, 9)) : destination;
    CHECK_THROWS(payload.Bind(source, wrong, TransferTransport::Direct, {TransferMode::Transient, context, 19}));
    CHECK_THROWS(payload.Bind(source, wrong, Reducer::Builtin<int64_t>(Reduction::Sum),
                              TransferTransport::Direct, {TransferMode::Transient, context, 19}));
    CHECK(source.Storage().IsIdle());
    CHECK(destination.Storage().IsIdle());
    // A rejected preflight must not retain the matching tag or any leases.
    payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 19}).Begin().Wait();
    CHECK(destination.Read().At(0, 4) == ((rank + 1) % size) * 100 + 24);
    if (rank == 0)
    {
        auto mutation = versions.BeginMutation(Change::Distribution);
        mutation.Commit();
    }
    CHECK_THROWS(PayloadPlan(route, source.Layout(), ElementSchema::Arithmetic<int64_t>()));
}

TEST_CASE("Typed MPI bindings reject one unpublished shape collectively and recover after commit")
{
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto context = std::make_shared<CommunicationContext>(MPI_COMM_WORLD);
    RowTransferPlan route(context, 3, 2, {{(rank + 1) % size, 2, 0}, {rank, 0, 1}});
    Array<int64_t> source(RowLayout::Dense(3, 5, 7)), destination(RowLayout::Dense(2, 5, 8));
    {
        auto write = source.Write();
        for (DNDS::index iRow = 0; iRow < 3; ++iRow)
            for (DNDS::index iCol = 0; iCol < 5; ++iCol)
                write.At(iRow, iCol) = rank * 100 + iRow * 10 + iCol;
    }
    PayloadPlan payload(route, source.Layout(), destination.Layout(), ElementSchema::Arithmetic<int64_t>());
    std::optional<ExclusiveRows> editor;
    if (rank == 0)
        editor.emplace(destination.EditRows());
    CHECK_THROWS(payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 23}));
    CHECK_THROWS(payload.Bind(source, destination, Reducer::Builtin<int64_t>(Reduction::Sum),
                              TransferTransport::Direct, {TransferMode::Transient, context, 23}));
    if (editor)
        editor->Commit();
    payload.Bind(source, destination, TransferTransport::Direct, {TransferMode::Transient, context, 23}).Begin().Wait();
    {
        auto read = destination.Read();
        for (DNDS::index iCol = 0; iCol < 5; ++iCol)
        {
            CHECK(read.At(0, iCol) == ((rank + 1) % size) * 100 + 20 + iCol);
            CHECK(read.At(1, iCol) == rank * 100 + iCol);
        }
    }
    // Abandoned edits and consumed handles take the same collective path.
    if (rank == 0)
    {
        auto abandoned = destination.EditRows();
    }
    CHECK_THROWS(payload.Bind(source, destination));
    destination.Reset(RowLayout::Dense(2, 5, 8));
    std::optional<Array<int64_t>> retained;
    if (rank == 0)
        retained.emplace(std::move(source));
    CHECK_THROWS(payload.Bind(source, destination));
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
    RowTransferPlan halo(MPI_COMM_WORLD, root.RowCount(), 2, {{0, 0, 0}, {0, 0, 1}});
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
    CHECK_THROWS_WITH(operation.Wait(), "custom reduction failed"); // Reports the stored failure; no repeated combine.
}

#ifdef DNDS_USE_CUDA
TEST_CASE_TEMPLATE("Native CUDA reduction dispatch covers wide rows without requiring device MPI", T,
                   int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double)
{
    const auto ctx = ExecutionContext::CUDA(0);
    Array<T> source(RowLayout::Ragged({129, 129, 129, 257, 257, 257, 257}));
    Array<T> initial(RowLayout::Dense(3, 257, 263));
    std::vector<Contribution> groups;
    {
        auto w = source.Write();
        for (size_t i = 0; i < source.RowCount(); ++i)
        {
            groups.push_back({source.Layout().Offset(i), i < 3 ? size_t(0) : size_t(526), source.RowSize(i)});
            for (size_t j = 0; j < source.RowSize(i); ++j)
                w.At(i, j) = j == 0 ? std::numeric_limits<T>::max() : T(int((i * 5 + j) % 27) - 13);
        }
        auto seed = initial.Storage().Write();
        std::fill_n(seed.template View<T>().data, initial.Layout().Elements(), T(9));
    }
    auto gpuSource = source.To(Device::CUDA(0));
    auto hostGroups = Buffer::Allocate(groups.size() * sizeof(Contribution));
    {
        auto w = hostGroups.Write();
        std::copy(groups.begin(), groups.end(), w.View<Contribution>().data);
    }
    auto deviceGroups = Buffer::Allocate(hostGroups.Bytes(), Device::CUDA(0));
    Copy(hostGroups, deviceGroups, ctx).Wait();
    for (auto op : {Reduction::Sum, Reduction::Min, Reduction::Max, Reduction::BitOr, Reduction::BitXor, Reduction::BitAnd})
    {
        if (!std::is_integral_v<T> && op > Reduction::Max)
            continue;
        for (auto init : {DestinationInit::Identity, DestinationInit::Preserve})
        {
            auto expected = initial.Clone();
            auto result = initial.To(Device::CUDA(0));
            auto reducer = Reducer::Builtin<T>(op, init);
            {
                auto src = source.Read();
                auto dst = expected.Write();
                reducer.host(src.View().data, dst.View().data, groups, ExecutionContext::Host());
            }
            {
                auto src = gpuSource.Read(ctx);
                auto dst = result.Write(ctx);
                auto mapping = deviceGroups.Read(ctx);
                REQUIRE(bool(reducer.device));
                reducer.device(src.View().data, dst.View().data, mapping.View<Contribution>().data, groups.size(), ctx);
                auto done = Completion::Record(ctx);
                src.Complete(done);
                dst.Complete(done);
                mapping.Complete(done);
            }
            auto hostResult = result.To(Device::Host());
            auto actual = hostResult.Storage().Read();
            auto reference = expected.Storage().Read();
            // Includes untouched row, short-row tail and physical stride padding.
            for (size_t j = 0; j < initial.Layout().Elements(); ++j)
                CHECK(actual.template View<T>().data[j] == reference.template View<T>().data[j]);
        }
    }
}

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
