#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DNDS/Runtime/EagerEngine.hpp"
#include "DNDS/Runtime/EagerArray.hpp"
#include "DNDS/Runtime/EagerCopy.hpp"
#include "DNDS/Runtime/CUDAExecution.hpp"
#include "DNDS/Runtime/RowBuilder.hpp"
#include <atomic>
#include <thread>
#include <stdexcept>

namespace R = DNDS::Runtime;

struct Gate : R::StreamEvent::State
{
    std::atomic<bool> open{false};
    bool Ready() const override { return open.load(); }
    void Wait() const override
    {
        while (!Ready())
            std::this_thread::yield();
    }
};

TEST_CASE("Inline and worker launchers execute the same eager fork join loop")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        std::vector<double> state(67 * 7, 1);
        double norm = 0;
        int iterations = 0;
        for (int iteration = 0; iteration < 19; ++iteration)
        {
            engine.Submit(compute, [&](R::ExecutionContext context)
                          {
                CHECK(context.device == R::Device::Host());
                for (auto &x : state) x *= 0.5; });
            comm.WaitFor(compute.Record());
            engine.Submit(comm, [&](R::ExecutionContext)
                          {
                norm = 0;
                for (auto x : state) norm += x; });
            auto reduced = comm.Record();
            reduced.WaitHost();
            ++iterations;
            if (norm < 1)
                break;
            compute.WaitFor(reduced);
        }
        engine.Synchronize();
        CHECK(iterations == 9);
        CHECK(state[66 * 7 + 6] == doctest::Approx(1.0 / 512));
    }
}

TEST_CASE("Record describes a fixed prefix and a waiting lane cannot exhaust workers")
{
    for (size_t workers : {size_t(0), size_t(1)})
    {
        R::EagerEngine engine(workers);
        auto first = engine.CreateCPUStream();
        auto other = engine.CreateCPUStream();
        std::atomic<int> value{0};
        engine.Submit(first, [&](R::ExecutionContext)
                      { value = 1; });
        auto prefix = first.Record();
        prefix.WaitHost();
        auto gate = std::make_shared<Gate>();
        first.WaitFor(R::StreamEvent(gate));
        engine.Submit(first, [&](R::ExecutionContext)
                      { value = 2; });
        auto tail = first.Record();
        CHECK(prefix.Ready());
        CHECK_FALSE(tail.Ready());
        CHECK(value == 1);
        engine.Submit(other, [gate](R::ExecutionContext)
                      { gate->open = true; });
        engine.Synchronize();
        CHECK(value == 2);
        CHECK(tail.Ready());
    }
}

TEST_CASE("Imported CPU prefixes progress without blocking another executor")
{
    R::EagerEngine producer;
    R::EagerEngine consumer(1);
    auto source = producer.CreateCPUStream();
    auto target = consumer.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    std::atomic<int> state{0};
    source.WaitFor(R::StreamEvent(gate));
    producer.Submit(source, [&](R::ExecutionContext)
                    { state = 7; });
    target.WaitFor(source.Record());
    consumer.Submit(target, [&](R::ExecutionContext)
                    { CHECK(state == 7); state = 9; });
    gate->open = true;
    consumer.Synchronize();
    CHECK(state == 9);
    CHECK_THROWS(consumer.Submit(source, [](R::ExecutionContext) {}));
}

TEST_CASE("Queued temporary ownership and releases follow CPU completion on either lane")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        auto gate = std::make_shared<Gate>();
        std::atomic<int> deletes{0}, sum{0};
        auto owner = std::shared_ptr<std::vector<int>>(new std::vector<int>(31 * 5, 3), [&](auto *p)
                                                       { delete p; ++deletes; });
        std::weak_ptr<std::vector<int>> weak = owner;
        auto alias = owner;
        compute.WaitFor(R::StreamEvent(gate));
        engine.Submit(compute, [owner, &sum](R::ExecutionContext)
                      {
            for (auto x : *owner) sum += x; });
        engine.ReleaseOn(std::move(owner), compute);
        comm.WaitFor(compute.Record());
        engine.Submit(comm, [alias, &sum](R::ExecutionContext)
                      { sum += alias->back(); });
        engine.ReleaseOn(std::move(alias), comm);
        CHECK_FALSE(owner);
        CHECK_FALSE(alias);
        CHECK_FALSE(weak.expired());
        CHECK(deletes == 0);
        gate->open = true;
        engine.Synchronize();
        CHECK(weak.expired());
        CHECK(deletes == 1);
        CHECK(sum == 31 * 5 * 3 + 3);
        engine.ReleaseOn(std::move(owner), comm); // Releasing an empty handle is harmless.
        engine.Synchronize();
        CHECK(deletes == 1);
    }
}

TEST_CASE("Errors skip dependent work but retain ordered cleanup and drain other lanes")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        auto before = compute.Record();
        std::atomic<bool> skipped{true}, cleaned{false}, unrelated{false};
        engine.Submit(compute, [](R::ExecutionContext)
                      { throw std::runtime_error("operator failed"); });
        comm.WaitFor(compute.Record());
        engine.Submit(comm, [&](R::ExecutionContext)
                      { skipped = false; });
        engine.SubmitCleanup(comm, [&](R::ExecutionContext)
                             { cleaned = true; });
        auto independent = engine.CreateCPUStream();
        engine.Submit(independent, [&](R::ExecutionContext)
                      { unrelated = true; });
        CHECK_THROWS(engine.Synchronize());
        CHECK(before.Ready());
        CHECK(skipped);
        CHECK(cleaned);
        CHECK(unrelated);
    }
}

TEST_CASE("Concurrent host submitters linearize commands and preserve every accepted use")
{
    R::EagerEngine engine(2);
    auto a = engine.CreateCPUStream();
    auto b = engine.CreateCPUStream();
    std::atomic<size_t> total{0};
    std::vector<std::thread> submitters;
    for (int thread = 0; thread < 4; ++thread)
        submitters.emplace_back([&, thread]
                                {
            for (int i = 0; i < 257; ++i)
                engine.Submit(thread % 2 ? a : b, [&](R::ExecutionContext) { total += 7; }); });
    for (auto &thread : submitters)
        thread.join();
    engine.Synchronize();
    CHECK(total == 4 * 257 * 7);
}

TEST_CASE("Failed lanes still honor joins before releasing shared resources")
{
    R::EagerEngine engine(2);
    auto failed = engine.CreateCPUStream();
    auto peer = engine.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    std::atomic<int> deletes{0};
    auto owner = std::shared_ptr<int>(new int(17), [&](int *p)
                                      { delete p; ++deletes; });
    std::weak_ptr<int> weak = owner;
    peer.WaitFor(R::StreamEvent(gate));
    engine.Submit(peer, [owner](R::ExecutionContext)
                  { CHECK(*owner == 17); });
    engine.Submit(failed, [](R::ExecutionContext)
                  { throw std::runtime_error("failed before join"); });
    auto error = failed.Record();
    CHECK_THROWS(error.WaitHost());
    failed.WaitFor(peer.Record());
    engine.ReleaseOn(std::move(owner), failed);
    CHECK_FALSE(weak.expired());
    CHECK(deletes == 0);
    gate->open = true;
    CHECK_THROWS(engine.Synchronize());
    CHECK(weak.expired());
    CHECK(deletes == 1);
}

TEST_CASE("Rejected release returns the caller handle and never guesses a stream")
{
    class Reject : public R::StreamLauncher
    {
    public:
        R::ExecutionContext Context() const override { return R::ExecutionContext::Host(); }
        void Submit(Work, R::CommandKind, Retirement) override { throw std::runtime_error("queue rejected"); }
        void WaitFor(R::StreamEvent) override {}
        R::StreamEvent Record() override { return {}; }
        void Drain() override {}
    };
    R::EagerEngine engine;
    auto owner = std::make_shared<int>(42);
    CHECK_THROWS(engine.ReleaseOn(std::move(owner), R::ExecutionStream{}));
    REQUIRE(owner);
    auto rejected = engine.CreateStream(std::make_shared<Reject>());
    CHECK_THROWS(engine.ReleaseOn(std::move(owner), rejected));
    REQUIRE(owner);
    CHECK(*owner == 42);
}

TEST_CASE("Array reservations publish at execution and preserve same stream read write ordering")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto stream = engine.CreateCPUStream();
        auto gate = std::make_shared<Gate>();
        stream.WaitFor(R::StreamEvent(gate));
        R::Array<double, R::FixedRows<6>> state(R::RowLayout::Dense(257, 6, 8));
        auto alias = state;
        const auto initial = state.Version();
        for (int iteration = 0; iteration < 31; ++iteration)
            R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<double> u)
                            {
                for (size_t row = 0; row < u.Rows(); ++row)
                    for (size_t col = 0; col < u.RowSize(row); ++col)
                        u(row, col) += double(row + col + 1); }, R::ReadWrite(state));
        CHECK(state.Version() == initial);
        CHECK_FALSE(state.Storage().IsIdle());
        CHECK_THROWS(alias.Read());
        CHECK_THROWS(alias.Write());
        CHECK_THROWS(state.EditRows());
        gate->open = true;
        stream.Record().WaitHost();
        CHECK(state.Storage().IsIdle());
        CHECK(state.Version() == initial + 31);
        auto read = state.Read();
        auto flatAccess = state.Storage().Read();
        auto flat = flatAccess.View<double>();
        for (size_t row = 0; row < state.Rows(); ++row)
        {
            CHECK(read.At(row, 5) == 31 * double(row + 6));
            CHECK(flat.data[row * 8 + 6] == 0);
            CHECK(flat.data[row * 8 + 7] == 0);
        }
    }
}

TEST_CASE("Owned only CSR stencil can complete while ghost receives remain pending")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto compute = engine.CreateCPUStream();
        auto comm = engine.CreateCPUStream();
        R::ArrayPair<double, R::FixedRows<5>> state{
            R::Array<double, R::FixedRows<5>>(R::RowLayout::Dense(257, 5, 8)),
            R::Array<double, R::FixedRows<5>>(R::RowLayout::Dense(19, 5, 7))};
        {
            auto write = state.owned.Write();
            for (size_t row = 0; row < 257; ++row)
                for (size_t col = 0; col < 5; ++col)
                    write.At(row, col) = row + col;
        }
        std::vector<size_t> lengths(239);
        for (size_t row = 0; row < lengths.size(); ++row)
            lengths[row] = 1 + row % 5;
        R::Array<size_t> adjacency(R::RowLayout::Ragged(lengths));
        {
            auto write = adjacency.Write();
            for (size_t row = 0; row < lengths.size(); ++row)
                for (size_t j = 0; j < lengths[row]; ++j)
                    write.At(row, j) = (row + 7 * j) % 257;
        }
        R::Array<double> residual(R::RowLayout::Dense(239, 5));
        auto gate = std::make_shared<Gate>();
        comm.WaitFor(R::StreamEvent(gate));
        R::SubmitArrays(engine, comm, [](R::ExecutionContext, R::ArrayView<double> ghost)
                        {
            for (size_t row = 0; row < ghost.Rows(); ++row)
                for (size_t col = 0; col < ghost.RowSize(row); ++col) ghost(row, col) = 1000 + row + col; }, R::Write(state.ghost));
        R::SubmitArrays(engine, compute, [](R::ExecutionContext, R::ArrayView<const double> owned, R::ArrayView<const size_t> c2c, R::ArrayView<double> rhs)
                        {
                for (size_t row = 0; row < rhs.Rows(); ++row)
                    for (size_t col = 0; col < rhs.RowSize(row); ++col)
                        for (size_t n = 0; n < c2c.RowSize(row); ++n) rhs(row, col) += owned(c2c(row, n), col); }, R::Read(state.owned), R::Read(adjacency), R::Write(residual));
        compute.Record().WaitHost();
        CHECK_THROWS(state.ghost.Read());
        auto output = residual.Read();
        for (size_t row = 0; row < lengths.size(); ++row)
        {
            double expected = 0;
            for (size_t n = 0; n < lengths[row]; ++n)
                expected += (row + 7 * n) % 257 + 4;
            CHECK(output.At(row, 4) == expected);
        }
        gate->open = true;
        compute.WaitFor(comm.Record());
        R::Array<double> ring(R::RowLayout::Dense(19, 5));
        R::SubmitArrays(engine, compute, [](R::ExecutionContext, R::ArrayView<const double> owned, R::ArrayView<const double> ghost, R::ArrayView<double> rhs)
                        {
                R::PairReadView<double> u{owned, ghost};
                for (size_t row = 0; row < rhs.Rows(); ++row)
                    for (size_t col = 0; col < rhs.RowSize(row); ++col) rhs(row, col) = u(row, col) + u(owned.Rows() + row, col); }, R::Read(state.owned), R::Read(state.ghost), R::Write(ring));
        engine.Synchronize();
        CHECK(ring.Read().At(18, 4) == 1044);
    }
}

TEST_CASE("Pending operands retain old values and buffered row sizes across handle reset")
{
    R::EagerEngine engine(2);
    auto stream = engine.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    stream.WaitFor(R::StreamEvent(gate));
    std::vector<size_t> lengths(97);
    for (size_t i = 0; i < lengths.size(); ++i)
        lengths[i] = i % 8;
    R::Array<int64_t> input(R::RowLayout::Padded(lengths, 11));
    {
        auto write = input.Write();
        for (size_t i = 0; i < lengths.size(); ++i)
            for (size_t j = 0; j < lengths[i]; ++j)
                write.At(i, j) = i * 100 + j;
    }
    std::weak_ptr<const R::ArrayStorage> storage = input.SharedStorage();
    std::weak_ptr<const R::Detail::ArrayLayout> metadata = input.SharedStorage()->SharedLayout();
    R::Array<int64_t> output(R::RowLayout::Ragged(lengths));
    R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<const int64_t> in, R::ArrayView<int64_t> out)
                    {
        for (size_t i = 0; i < out.Rows(); ++i)
            for (size_t j = 0; j < out.RowSize(i); ++j) out(i, j) = in(i, j) + 1; }, R::Read(input), R::Write(output));
    input.Reset(R::RowLayout::Dense(1, 2));
    CHECK_FALSE(storage.expired());
    CHECK_FALSE(metadata.expired());
    gate->open = true;
    stream.Record().WaitHost();
    CHECK(storage.expired());
    CHECK(metadata.expired());
    auto values = output.Read();
    for (size_t i = 0; i < lengths.size(); ++i)
        for (size_t j = 0; j < lengths[i]; ++j)
            CHECK(values.At(i, j) == int64_t(i * 100 + j + 1));
}

TEST_CASE("Rejected reservations cancel while failed or skipped writes stay invalid")
{
    R::EagerEngine engine(2), other;
    auto stream = engine.CreateCPUStream();
    auto gate = std::make_shared<Gate>();
    stream.WaitFor(R::StreamEvent(gate));
    R::Array<double> input(R::RowLayout::Dense(43, 6)), output(input.Layout()), skipped(input.Layout());
    auto kernel = [](R::ExecutionContext, R::ArrayView<const double>, R::ArrayView<double>) {};
    auto initial = output.Version();
    {
        auto pin = input.Write();
        CHECK_THROWS(R::SubmitArrays(engine, stream, kernel, R::Read(input), R::Write(output)));
    }
    CHECK(output.Storage().IsIdle());
    CHECK(output.IsValid());
    CHECK(output.Version() == initial);
    CHECK_THROWS(R::SubmitArrays(other, stream, kernel, R::Read(input), R::Write(output)));
    CHECK(output.Storage().IsIdle());
    {
        auto pin = input.Read();
        R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<const double>) {}, R::Read(input));
        CHECK_THROWS(R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<double>) {}, R::Write(input)));
    }
    CHECK_THROWS(input.Write()); // Pending read excludes immediate writer.
    CHECK_NOTHROW(input.Read());
    R::SubmitArrays(engine, stream, [](R::ExecutionContext, R::ArrayView<double> value)
                    {
        value(17, 4) = 9;
        throw std::runtime_error("partial update"); }, R::Write(output));
    R::SubmitArrays(engine, stream, kernel, R::Read(output), R::Write(skipped));
    gate->open = true;
    CHECK_THROWS(engine.Synchronize());
    CHECK(input.IsValid());
    CHECK_FALSE(output.IsValid());
    CHECK_FALSE(skipped.IsValid());
    CHECK(output.Version() == initial);
    CHECK(output.Storage().IsIdle());
    CHECK_THROWS(output.Read());
    CHECK_THROWS(skipped.Read());
}

TEST_CASE("Eager byte copies use the same pending access contract on inline and worker CPU")
{
    for (size_t workers : {size_t(0), size_t(2)})
    {
        R::EagerEngine engine(workers);
        auto stream = engine.CreateCPUStream();
        auto gate = std::make_shared<Gate>();
        stream.WaitFor(R::StreamEvent(gate));
        auto source = R::Buffer::Allocate(131 * 7 * sizeof(int64_t));
        auto target = R::Buffer::Allocate(source.Bytes());
        {
            auto write = source.Write();
            auto view = write.View<int64_t>();
            for (size_t i = 0; i < view.size; ++i)
                view.data[i] = int64_t(13 * i);
        }
        R::CopyAsync(engine, stream, source, target);
        CHECK_THROWS(source.Write());
        CHECK_THROWS(target.Read());
        CHECK_THROWS(R::CopyAsync(engine, stream, source, source));
        auto small = R::Buffer::Allocate(3);
        CHECK_THROWS(R::CopyAsync(engine, stream, source, small));
        CHECK(small.IsIdle());
        source = {};
        gate->open = true;
        engine.Synchronize();
        auto read = target.Read();
        auto values = read.View<int64_t>();
        for (size_t i = 0; i < values.size; ++i)
            CHECK(values.data[i] == int64_t(13 * i));
        auto empty = R::Buffer::Allocate(0), other = R::Buffer::Allocate(0);
        R::CopyAsync(engine, stream, empty, other);
        engine.Synchronize();
        CHECK(other.Version() == 1);
    }
#ifndef DNDS_USE_CUDA
    CHECK_THROWS(R::CUDALauncher(0));
#endif
}

TEST_CASE("CPU async allocation facade preserves shared variable shape metadata and handle ownership")
{
    R::EagerEngine engine(2);
    auto stream = engine.CreateCPUStream();
    auto table = R::AllocateArrayAsync<double, R::FixedRows<6>>(engine, stream, R::RowLayout::Dense(31, 6, 8));
    CHECK(table.Read().At(30, 5) == 0);
    auto csr = R::AllocateArrayAsync<int64_t>(engine, stream, R::RowLayout::Ragged({0, 7, 19, 5}));
    auto second = R::AllocateLikeAsync(engine, stream, csr);
    CHECK(csr.SharedStorage()->SharedLayout() == second.SharedStorage()->SharedLayout());
    auto alias = table;
    auto gate = std::make_shared<Gate>();
    stream.WaitFor(R::StreamEvent(gate));
    engine.ReleaseOn(std::move(table), stream);
    CHECK(alias.IsValid());
    gate->open = true;
    engine.Synchronize();
    CHECK(alias.Read().At(30, 5) == 0);
    engine.ReleaseOn(std::move(alias), stream);
    engine.ReleaseOn(std::move(alias), stream);
    engine.Synchronize();
}
