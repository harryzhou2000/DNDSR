#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/InterfaceIterationSchedule.hpp"
#include "Cauchy/Core/DomainPlacement.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <numeric>
#include <atomic>
#include <chrono>
#include <thread>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    std::vector<C::DomainRankGroup> Groups(bool shared)
    {
        std::vector<int> first, second;
        for (int rank = 0; rank < mpi.size; ++rank)
        {
            if (shared || mpi.size == 1 || rank % 2 == 0)
                first.push_back(rank);
            if (shared || mpi.size == 1 || rank % 2 != 0)
                second.push_back(rank);
        }
        if (shared)
            std::reverse(second.begin(), second.end());
        return {{"left", first}, {"right", second}};
    }
    struct Exchange
    {
        C::DomainPlacement placement;
        std::shared_ptr<R::CommunicationContext> context;
        std::vector<I> rootsHere;
        R::Array<double> source, packet, scratch;
        R::RowTransferPlan route;
        R::TransferBinding binding;
        std::optional<R::TransferOperation> operation;
        int begins = 0, completes = 0, discards = 0;
        uint64_t sourceVersion = 0;
        bool failBegin = false, failComplete = false;
        std::vector<std::string> events;
        static std::vector<I> LocalRoots(const C::DomainPlacement &placement)
        {
            std::vector<I> result;
            for (auto domain : placement.LocalDomains())
                if (placement.DomainRank(domain) == 0)
                    result.push_back(domain);
            return result;
        }
        R::RowTransferPlan Route() const
        {
            std::vector<std::vector<I>> send(mpi.size), receive(mpi.size);
            for (int peer = 0; peer < mpi.size; ++peer)
                for (I iRow = 0; iRow < I(rootsHere.size()); ++iRow)
                    send[peer].push_back(iRow);
            for (I domain = 0; domain < 2; ++domain)
                receive[placement.SimulationRank(domain, 0)].push_back(domain);
            return R::RowTransferPlan::Known(context, I(rootsHere.size()), 2, send, receive);
        }
        explicit Exchange(bool shared)
            : placement(mpi.comm, Groups(shared)), context(std::make_shared<R::CommunicationContext>(placement.Control())),
              rootsHere(LocalRoots(placement)), source(R::RowLayout::Dense(I(rootsHere.size()), 1)), packet(R::RowLayout::Dense(2, 1)),
              scratch(R::RowLayout::Dense(1, 1)), route(Route()),
              binding(R::PayloadPlan(route, source.Layout(), packet.Layout(), R::ElementSchema::For<double>()).Bind(source, packet)) {}
        void Fill(const std::array<double, 2> &values)
        {
            auto write = source.Write();
            const auto view = write.View<R::FixedRows<1>>();
            for (I iRow = 0; iRow < I(rootsHere.size()); ++iRow)
                view(iRow, 0) = values[rootsHere[iRow]];
        }
        C::InterfaceExchangeHooks Hooks()
        {
            return {
                [this](const auto &context)
                {
                    ++begins;
                    events.push_back("begin" + std::to_string(context.epoch.couplingIteration));
                    sourceVersion = source.Version();
                    operation.emplace(binding.Begin());
                    DNDS::MPI::LocalStage(mpi, [&]
                                          {
                        if (failBegin && mpi.rank == 0)
                            throw std::runtime_error("mock begin failure after posting"); });
                },
                [this](const auto &context)
                {
                    operation->Wait();
                    operation.reset();
                    DNDS_check_throw_info(source.Version() == sourceVersion, "interface source changed during send");
                    DNDS::MPI::LocalStage(mpi, [&]
                                          {
                        if (failComplete && mpi.rank == 0)
                            throw std::runtime_error("mock complete failure before publication"); });
                    ++completes;
                    events.push_back("complete" + std::to_string(context.epoch.couplingIteration));
                },
                [this](const auto &)
                {
                    ++discards;
                    if (operation)
                    {
                        operation->Wait();
                        operation.reset();
                    }
                }};
        }
    };
}
TEST_CASE("Stage and iteration refresh freeze the correct traces across serial and partitioned mock domains")
{
    for (bool shared : {false, true})
        for (auto policy : {C::InterfaceRefreshPolicy::PerStage, C::InterfaceRefreshPolicy::PerIteration})
            for (int workers : {0, 2})
                for (bool overlap : {false, true})
                {
                    Exchange exchange(shared);
                    const C::InterfaceStageContext stage{{7, 13, 2, 0, 11, 17, 1}, .35};
                    C::InterfaceIterationSchedule schedule(exchange.context, stage, policy, exchange.Hooks());
                    R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                    const auto stream = engine.CreateCPUStream();
                    std::array<double, 2> local{1, 2}, expected{1, 2};
                    const auto initial = expected;
                    std::vector<std::string> expectedEvents;
                    CHECK_THROWS(schedule.CompleteInputs());
                    CHECK_THROWS(schedule.EndIteration());
                    CHECK_THROWS(schedule.TraceContext());
                    // Only one participant supplies the wrong first ordinal.
                    // Agreement must reject everywhere before posting requests.
                    CHECK_THROWS(schedule.BeginIteration(mpi.rank == 0 ? 1 : 0));
                    CHECK(exchange.begins == 0);
                    for (I iteration = 0; iteration < 3; ++iteration)
                    {
                        exchange.Fill(local);
                        const bool refresh = policy == C::InterfaceRefreshPolicy::PerIteration || iteration == 0;
                        CHECK(schedule.BeginIteration(iteration) == refresh);
                        if (refresh)
                            expectedEvents.push_back("begin" + std::to_string(iteration));
                        CHECK_THROWS(schedule.BeginIteration(iteration));
                        CHECK_THROWS(schedule.EndIteration());
                        CHECK_THROWS(schedule.TraceContext());
                        if (refresh)
                        {
                            CHECK_THROWS(exchange.packet.Read());
                            if (exchange.source.RowCount())
                                CHECK_THROWS(exchange.source.Write());
                        }
                        std::atomic<bool> started{false}, release{!overlap || !workers};
                        const auto independent = [&]
                        {
                            R::SubmitArrays(engine, stream, [&](auto, auto source, auto output)
                                            {
                                double sum = 0;
                                for (I iRow = 0; iRow < source.RowCount(); ++iRow) sum += source(iRow, 0);
                                output(0, 0) = 2 * sum;
                                started.store(true, std::memory_order_release);
                                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                                while (!release.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
                                DNDS_check_throw_info(release.load(), "mock independent worker gate timed out");
                                exchange.events.push_back("core" + std::to_string(iteration)); }, R::ReadAs<R::FixedRows<1>>(exchange.source), R::WriteAs<R::FixedRows<1>>(exchange.scratch));
                            if (workers && overlap)
                            {
                                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                                while (!started.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
                                    std::this_thread::yield();
                                CHECK(started.load());
                                // The compute task is live with an owned read
                                // while the controller progresses actual MPI.
                                if (refresh)
                                {
                                    bool ready = false;
                                    while (!ready && std::chrono::steady_clock::now() < deadline)
                                        ready = exchange.operation->Test();
                                    CHECK(ready);
                                    if (exchange.source.RowCount())
                                        CHECK_THROWS(exchange.source.Write());
                                }
                                release.store(true, std::memory_order_release);
                            }
                            stream.Record().WaitHost();
                            const auto value = exchange.scratch.Read();
                            double sum = 0;
                            for (auto domain : exchange.rootsHere)
                                sum += local[domain];
                            CHECK(value.View()(0, 0) == 2 * sum);
                        };
                        if (overlap)
                            independent();
                        schedule.CompleteInputs();
                        if (refresh)
                            expectedEvents.push_back("complete" + std::to_string(iteration));
                        if (!overlap)
                            independent();
                        if (overlap)
                            expectedEvents.insert(expectedEvents.end() - (refresh ? 1 : 0), "core" + std::to_string(iteration));
                        else
                            expectedEvents.push_back("core" + std::to_string(iteration));
                        CHECK_THROWS(schedule.CompleteInputs());
                        const auto published = schedule.TraceContext();
                        CHECK(published.stageTime == stage.stageTime);
                        CHECK(published.epoch.trial == 13);
                        CHECK(published.epoch.stage == 2);
                        CHECK(published.epoch.coupledStage == 1);
                        CHECK(published.epoch.geometry == 11);
                        CHECK(published.epoch.schema == 17);
                        CHECK(published.epoch.couplingIteration == uint64_t(policy == C::InterfaceRefreshPolicy::PerStage ? 0 : iteration));
                        const auto donors = policy == C::InterfaceRefreshPolicy::PerStage ? initial : expected;
                        const auto old = expected;
                        for (I domain = 0; domain < 2; ++domain)
                            expected[domain] = .5 * old[domain] + .25 * donors[1 - domain] + .1 * (domain + 1);
                        {
                            const auto read = exchange.packet.Read();
                            const auto packet = read.View<R::FixedRows<1>>();
                            for (I domain = 0; domain < 2; ++domain)
                                CHECK(packet(domain, 0) == donors[domain]);
                            for (auto domain : exchange.placement.LocalDomains())
                            {
                                local[domain] = .5 * local[domain] + .25 * packet(1 - domain, 0) + .1 * (domain + 1);
                                CHECK(local[domain] == expected[domain]);
                            }
                        }
                        schedule.EndIteration();
                    }
                    CHECK(exchange.events == expectedEvents);
                    const int calls = policy == C::InterfaceRefreshPolicy::PerStage ? 1 : 3;
                    CHECK(exchange.begins == calls);
                    CHECK(exchange.completes == calls);
                    schedule.Discard();
                    schedule.Discard();
                    CHECK(exchange.discards == 1);
                    CHECK_THROWS(schedule.TraceContext());
                    CHECK_THROWS(schedule.BeginIteration(3));
                }
}
TEST_CASE("Rank-local exchange failures drain pending transfers and never publish a failed packet")
{
    for (bool failAtBegin : {false, true})
    {
        Exchange exchange(true);
        exchange.Fill({1, 2});
        exchange.failBegin = failAtBegin;
        exchange.failComplete = !failAtBegin;
        C::InterfaceIterationSchedule schedule(exchange.context, {{0, 1, 0, 0, 0, 0}, 0}, C::InterfaceRefreshPolicy::PerIteration, exchange.Hooks());
        const auto version = exchange.source.Version();
        if (failAtBegin)
            CHECK_THROWS(schedule.BeginIteration(0));
        else
        {
            CHECK(schedule.BeginIteration(0));
            CHECK_THROWS(schedule.CompleteInputs());
        }
        CHECK_THROWS(schedule.TraceContext());
        CHECK_THROWS(schedule.BeginIteration(0));
        schedule.Discard();
        schedule.Discard();
        CHECK_FALSE(exchange.operation.has_value());
        CHECK(exchange.discards == 1);
        CHECK(exchange.source.Version() == version);
        CHECK_THROWS(schedule.CompleteInputs());
        exchange.failBegin = exchange.failComplete = false;
        C::InterfaceIterationSchedule retry(exchange.context, {{0, 2, 0, 0, 0, 0}, 0}, C::InterfaceRefreshPolicy::PerStage, exchange.Hooks());
        CHECK(retry.BeginIteration(0));
        retry.CompleteInputs();
        CHECK(retry.TraceContext().epoch.trial == 2);
        retry.EndIteration();
    }
}
TEST_CASE("Retained packet readers permit stage reuse but reject a refresh before MPI posting")
{
    for (auto policy : {C::InterfaceRefreshPolicy::PerStage, C::InterfaceRefreshPolicy::PerIteration})
    {
        Exchange exchange(true);
        exchange.Fill({1, 2});
        C::InterfaceIterationSchedule schedule(exchange.context, {{0, 1, 0, 0, 0, 0}, 0}, policy, exchange.Hooks());
        CHECK(schedule.BeginIteration(0));
        schedule.CompleteInputs();
        schedule.EndIteration();
        const auto version = exchange.packet.Version();
        {
            std::optional<R::Array<double>::ReadLease> retained;
            if (!mpi.rank)
                retained.emplace(exchange.packet.Read());
            exchange.Fill({3, 4});
            if (policy == C::InterfaceRefreshPolicy::PerStage)
            {
                CHECK_FALSE(schedule.BeginIteration(1));
                schedule.CompleteInputs();
                CHECK(schedule.TraceContext().epoch.couplingIteration == 0);
                schedule.EndIteration();
            }
            else
            {
                CHECK_THROWS(schedule.BeginIteration(1));
                CHECK_FALSE(exchange.operation.has_value());
                CHECK_THROWS(schedule.TraceContext());
                schedule.Discard();
            }
            CHECK(exchange.packet.Version() == version);
            if (retained)
            {
                CHECK(retained->View()(0, 0) == 1);
                CHECK(retained->View()(1, 0) == 2);
            }
        }
        C::InterfaceIterationSchedule retry(exchange.context, {{0, 2, 0, 0, 0, 0}, 0}, policy, exchange.Hooks());
        CHECK(retry.BeginIteration(0));
        retry.CompleteInputs();
        const auto packet = exchange.packet.Read();
        CHECK(packet.View()(0, 0) == 3);
        CHECK(packet.View()(1, 0) == 4);
        retry.EndIteration();
    }
}
TEST_CASE("A failed discard stays retryable until every pending transfer is drained")
{
    Exchange exchange(true);
    exchange.Fill({1, 2});
    auto hooks = exchange.Hooks();
    const auto drain = hooks.discard;
    bool fail = true;
    hooks.discard = [&](const auto &context)
    {
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            if (fail && mpi.rank == 0)
                throw std::runtime_error("mock cleanup failure before draining"); });
        drain(context);
    };
    C::InterfaceIterationSchedule schedule(exchange.context, {{0, 1, 0, 0, 0, 0}, 0}, C::InterfaceRefreshPolicy::PerIteration, std::move(hooks));
    CHECK(schedule.BeginIteration(0));
    CHECK_THROWS(schedule.Discard());
    CHECK_THROWS(schedule.TraceContext());
    CHECK_THROWS(schedule.BeginIteration(0));
    if (!mpi.rank)
        CHECK(exchange.operation.has_value());
    fail = false;
    CHECK_NOTHROW(schedule.Discard());
    CHECK_FALSE(exchange.operation.has_value());
    const auto count = exchange.discards;
    CHECK_NOTHROW(schedule.Discard());
    CHECK(exchange.discards == count);
}
TEST_CASE("Stage setup rejects inconsistent identities and caller-ordered hooks retain the same protocol")
{
    Exchange exchange(true);
    const C::InterfaceStageContext stage{{0, 1, 2, 99, 3, 4}, .125};
    auto missing = exchange.Hooks();
    if (!mpi.rank)
        missing.begin = {};
    CHECK_THROWS(C::InterfaceIterationSchedule(exchange.context, stage, C::InterfaceRefreshPolicy::PerStage, std::move(missing)));
    auto invalid = stage;
    if (!mpi.rank)
        invalid.stageTime = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(C::InterfaceIterationSchedule(exchange.context, invalid, C::InterfaceRefreshPolicy::PerStage, exchange.Hooks()));
    if (mpi.size > 1)
    {
        invalid = stage;
        if (!mpi.rank)
            invalid.epoch.trial = std::numeric_limits<uint64_t>::max();
        CHECK_THROWS(C::InterfaceIterationSchedule(exchange.context, invalid, C::InterfaceRefreshPolicy::PerStage, exchange.Hooks()));
        invalid = stage;
        if (!mpi.rank)
            invalid.stageTime = .25;
        CHECK_THROWS(C::InterfaceIterationSchedule(exchange.context, invalid, C::InterfaceRefreshPolicy::PerStage, exchange.Hooks()));
        CHECK_THROWS(C::InterfaceIterationSchedule(exchange.context, stage, mpi.rank ? C::InterfaceRefreshPolicy::PerStage : C::InterfaceRefreshPolicy::PerIteration, exchange.Hooks()));
        CHECK_THROWS(C::InterfaceIterationSchedule(exchange.context, stage, C::InterfaceRefreshPolicy::PerStage, exchange.Hooks(), mpi.rank != 0));
    }
    CHECK(exchange.begins == 0);
    C::InterfaceIterationSchedule eager(exchange.context, stage, C::InterfaceRefreshPolicy::PerIteration, exchange.Hooks(), false);
    for (I iteration = 0; iteration < 2; ++iteration)
    {
        const std::array<double, 2> expected{2. + iteration, 7. - iteration};
        exchange.Fill(expected);
        // With checks off callers make all participants follow the same error
        // path before entering any backend collective; no rank-local shortcut.
        CHECK_THROWS(eager.BeginIteration(iteration + 2));
        CHECK(eager.BeginIteration(iteration));
        eager.CompleteInputs();
        CHECK(eager.TraceContext().epoch.couplingIteration == uint64_t(iteration));
        {
            const auto read = exchange.packet.Read();
            for (I domain = 0; domain < 2; ++domain)
                CHECK(read.View()(domain, 0) == expected[domain]);
        }
        eager.EndIteration();
    }
    CHECK(exchange.begins == 2);
    CHECK(exchange.completes == 2);
}
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    doctest::Context context(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
