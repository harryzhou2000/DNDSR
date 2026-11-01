#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <atomic>
#include <chrono>
#include <thread>

namespace R = DNDS::Runtime;
TEST_CASE("Checked prefixes reject missing stale and incomplete fan-in dependencies")
{
    R::AccessTrackingContext context;
    const auto producer = context.NewLane(), left = context.NewLane(), right = context.NewLane(), last = context.NewLane();
    auto buffer = R::Buffer::Allocate(8);
    const auto old = context.Record(producer, [] {});
    R::AccessTrackingContext::Submission writer, l, r;
    context.Submit(producer, {{buffer, R::AccessMode::Write}}, [&](auto use)
                   { writer = use; });
    CHECK_THROWS(context.Submit(left, {{buffer, R::AccessMode::Read}}, [](auto) {}));
    context.WaitFor(left, old, [] {});
    CHECK_THROWS(context.Submit(left, {{buffer, R::AccessMode::Read}}, [](auto) {}));
    const auto ready = context.Record(producer, [] {});
    context.WaitFor(left, ready, [] {});
    context.WaitFor(right, ready, [] {});
    context.Submit(left, {{buffer, R::AccessMode::Read}}, [&](auto use)
                   { l = use; });
    context.Submit(right, {{buffer, R::AccessMode::Read}}, [&](auto use)
                   { r = use; });
    context.WaitFor(last, context.Record(left, [] {}), [] {});
    CHECK_THROWS(context.Submit(last, {{buffer, R::AccessMode::Write}}, [](auto) {}));
    context.WaitFor(last, context.Record(right, [] {}), [] {});
    context.Submit(last, {{buffer, R::AccessMode::Write}}, [](auto use)
                   { use.Complete(); });
    writer.Complete();
    l.Complete();
    r.Complete();
}
TEST_CASE("Backend rejection rolls back aliases without holding the ledger across work")
{
    R::AccessTrackingContext context;
    const auto a = context.NewLane(), b = context.NewLane();
    auto buffer = R::Buffer::Allocate(8);
    CHECK_THROWS(context.Submit(a, {{buffer, R::AccessMode::Write}}, [](auto)
                                { throw std::runtime_error("backend rejected"); }));
    context.Submit(b, {{buffer, R::AccessMode::Read}, {buffer, R::AccessMode::Write}}, [](auto use)
                   { use.Complete(); });
    auto other = R::Buffer::Allocate(8);
    // Nested issue on an independent lane must not deadlock on a ledger mutex.
    context.Submit(a, {{buffer, R::AccessMode::Read}}, [&](auto use)
                   {
        context.Submit(b, {{other, R::AccessMode::Write}}, [](auto nested) { nested.Complete(); });
        use.Complete(); });
    CHECK_THROWS(context.Submit(a, {{buffer, R::AccessMode::Write}}, [&](auto)
                                { context.Record(a, [] {}); }));
    context.Submit(b, {{buffer, R::AccessMode::Write}}, [](auto use)
                   { use.Complete(); });
}
TEST_CASE("Explicit cross-context events prevent independent alias authority")
{
    R::AccessTrackingContext first, second;
    const auto a = first.NewLane(), b = second.NewLane();
    auto buffer = R::Buffer::Allocate(16);
    auto alias = buffer;
    R::AccessTrackingContext::Submission use;
    first.Submit(a, {{buffer, R::AccessMode::Write}}, [&](auto token)
                 { use = token; });
    CHECK_THROWS(second.Submit(b, {{alias, R::AccessMode::Read}}, [](auto) {}));
    second.WaitFor(b, first.Record(a, [] {}), [] {});
    second.Submit(b, {{alias, R::AccessMode::Read}}, [](auto token)
                  { token.Complete(); });
    use.Complete();
}
TEST_CASE("CPU worker checks submission before completion and preserves real event execution")
{
    auto context = std::make_shared<R::AccessTrackingContext>();
    R::EagerEngine engine(2, context);
    auto producer = engine.CreateCPUStream(), consumer = engine.CreateCPUStream();
    R::Array<double> values(R::RowLayout::Dense(7, 5)), result(R::RowLayout::Dense(7, 5));
    auto gate = std::make_shared<std::atomic<bool>>(false);
    struct Release
    {
        std::shared_ptr<std::atomic<bool>> gate;
        ~Release() { *gate = true; }
    } cleanup{gate};
    R::SubmitArrays(engine, producer, [gate](auto, auto output)
                    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!gate->load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        for (DNDS::index iRow = 0; iRow < output.rows; ++iRow)
            for (DNDS::index j = 0; j < output.width; ++j) output(iRow, j) = iRow * 10 + j; }, R::Write(values));
    const auto consume = [](auto, auto input, auto output)
    {
        for (DNDS::index iRow = 0; iRow < output.rows; ++iRow)
            for (DNDS::index j = 0; j < output.width; ++j)
                output(iRow, j) = input(iRow, j) + 1;
    };
    CHECK_THROWS(R::SubmitArrays(engine, consumer, consume, R::Read(values), R::Write(result)));
    consumer.WaitFor(producer.Record());
    R::SubmitArrays(engine, consumer, consume, R::Read(values), R::Write(result));
    *gate = true;
    engine.Synchronize();
    const auto read = result.Read();
    for (DNDS::index iRow = 0; iRow < 7; ++iRow)
        for (DNDS::index j = 0; j < 5; ++j)
            CHECK(read.View()(iRow, j) == iRow * 10 + j + 1);
}
TEST_CASE("Checked release consumes aliases only after an explicit fan-in")
{
    auto context = std::make_shared<R::AccessTrackingContext>();
    R::EagerEngine engine(2, context);
    auto left = engine.CreateCPUStream(), right = engine.CreateCPUStream(), last = engine.CreateCPUStream();
    R::Array<double> array(R::RowLayout::Dense(5, 3));
    auto alias = array;
    auto gate = std::make_shared<std::atomic<bool>>(false);
    struct Release
    {
        std::shared_ptr<std::atomic<bool>> gate;
        ~Release() { *gate = true; }
    } cleanup{gate};
    auto heldRead = [gate](auto, auto input)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!gate->load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        DNDS_check_throw_info(input.rows == 5, "retained view is invalid");
    };
    R::SubmitArrays(engine, left, heldRead, R::Read(array));
    R::SubmitArrays(engine, right, heldRead, R::Read(alias));
    last.WaitFor(left.Record());
    CHECK_THROWS(engine.ReleaseOn(std::move(array), last));
    CHECK(array.IsValid()); // Rejected issue returns ownership to the caller.
    last.WaitFor(right.Record());
    engine.ReleaseOn(std::move(array), last);
    CHECK_FALSE(array.IsValid());
    // A release declaration is also a use: independently dropping the alias
    // on left lacks the dependency on last, even if not the last C++ owner.
    CHECK_THROWS(engine.ReleaseOn(std::move(alias), left));
    left.WaitFor(last.Record());
    engine.ReleaseOn(std::move(alias), left);
    *gate = true;
    engine.Synchronize();
}
TEST_CASE("Immediate and unchecked boundaries reject pending conflicting checked accesses")
{
    auto buffer = R::Buffer::Allocate(32);
    auto checked = buffer.Reserve(R::AccessMode::Read, R::Device::Host(), true);
    CHECK_THROWS(buffer.Reserve(R::AccessMode::Write, R::Device::Host()));
    auto ordinaryReader = buffer.Reserve(R::AccessMode::Read, R::Device::Host());
    CHECK_THROWS(buffer.Reserve(R::AccessMode::Write, R::Device::Host(), true));
    ordinaryReader.Complete(true);
    checked.Complete(true);
    auto ordinaryWriter = buffer.Reserve(R::AccessMode::Write, R::Device::Host());
    CHECK_THROWS(buffer.Reserve(R::AccessMode::Read, R::Device::Host(), true));
    ordinaryWriter.Complete(true);
    auto writer = buffer.Reserve(R::AccessMode::Write, R::Device::Host(), true);
    CHECK_THROWS(buffer.Read());
    CHECK_THROWS(buffer.Write());
    writer.Complete(true);
    CHECK_NOTHROW(buffer.Read());
}
TEST_CASE("Inline checked execution publishes immediately and concurrent issue stays ordered")
{
    auto context = std::make_shared<R::AccessTrackingContext>();
    R::EagerEngine inlineEngine(0, context);
    const auto a = inlineEngine.CreateCPUStream(), b = inlineEngine.CreateCPUStream();
    R::Array<double> values(R::RowLayout::Dense(3, 5));
    const auto increment = [](auto, auto output)
    {
        for (DNDS::index iRow = 0; iRow < output.rows; ++iRow)
            for (DNDS::index j = 0; j < output.width; ++j)
                output(iRow, j) += 1;
    };
    R::SubmitArrays(inlineEngine, a, increment, R::ReadWrite(values));
    // CPU-inline completion already observed by retirement: no fabricated wait.
    R::SubmitArrays(inlineEngine, b, increment, R::ReadWrite(values));
    CHECK(values.Read().View()(2, 4) == 2);
    R::EagerEngine workers(2, context);
    const auto lane = workers.CreateCPUStream();
    std::atomic<int> errors{0};
    auto issue = [&]
    {
        try
        {
            for (int i = 0; i < 25; ++i)
                R::SubmitArrays(workers, lane, increment, R::ReadWrite(values));
        }
        catch (...)
        {
            ++errors;
        }
    };
    std::thread first(issue), second(issue);
    first.join();
    second.join();
    workers.Synchronize();
    CHECK(errors == 0);
    CHECK(values.Read().View()(2, 4) == 52);
}
