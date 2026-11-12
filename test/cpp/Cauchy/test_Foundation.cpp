#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/CoordinatedSchedule.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "DNDS/Algebra/DofBundle.hpp"
#include <atomic>
#include <numeric>
#include <thread>

namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static int rank, size;
static std::vector<int> AllRanks()
{
    std::vector<int> result(size);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
TEST_CASE("Cauchy configuration preflight coordinates invalid or differing rank inputs")
{
    const auto config = C::SimulationConfig::ResolveCollective(C::Json::object(), MPI_COMM_WORLD);
    REQUIRE(config.Domains().size() == 1);
    CHECK(config.Domains()[0].placement.ranks == AllRanks());
    C::Json bad = C::Json::object();
    if (rank == 0)
        bad["defaults"]["execution"]["device"] = "cuda";
    CHECK_THROWS(C::SimulationConfig::ResolveCollective(bad, MPI_COMM_WORLD));
    if (size > 1)
    {
        C::Json different = C::Json::object();
        different["defaults"]["equation"]["gamma"] = rank == 0 ? 1.4 : 1.5;
        CHECK_THROWS(C::SimulationConfig::ResolveCollective(different, MPI_COMM_WORLD));
    }
    CHECK_NOTHROW(C::SimulationConfig::ResolveCollective(C::Json::object(), MPI_COMM_WORLD));
}
TEST_CASE("Domain execution contexts retain named lanes and backend lifetime")
{
    for (const int workers : {0, 2})
    {
        C::ExecutionSettings settings;
        settings.hostThreads = workers;
        settings.checkAccess = true;
        std::optional<C::DomainExecutionContext> retained;
        {
            C::DomainExecutionContext context(settings);
            CHECK(context.Engine().Owns(context.ComputeStream()));
            CHECK(context.Engine().Owns(context.CommunicationStream()));
            CHECK(context.Engine().ChecksAccess());
            retained.emplace(context);
        }
        std::atomic<int> state{0};
        retained->Engine().Submit(retained->ComputeStream(),
                                  [&](R::ExecutionContext)
                                  { state = 1; });
        retained->CommunicationStream().WaitFor(
            retained->ComputeStream().Record());
        retained->Engine().Submit(retained->CommunicationStream(),
                                  [&](R::ExecutionContext)
                                  { state = state == 1 ? 2 : -1; });
        retained->ComputeStream().WaitFor(
            retained->CommunicationStream().Record());
        retained->Engine().Submit(retained->ComputeStream(),
                                  [&](R::ExecutionContext)
                                  { state = state == 2 ? 3 : -2; });
        retained->Synchronize();
        CHECK(state.load() == 3);
    }
    C::ExecutionSettings settings;
    settings.device = "cuda";
    CHECK_THROWS(C::DomainExecutionContext(settings));
}
TEST_CASE("Prepared operator records retain their owners through execution retirement")
{
    struct Prepared
    {
        std::shared_ptr<std::atomic<int>> result;
        std::vector<R::ScheduledBufferAccess> Accesses() const { return {}; }
        void Launch(R::ExecutionContext context)
        {
            DNDS_assert(context.device == R::Device::Host());
            result->store(1);
        }
        void Retire(std::exception_ptr error) noexcept
        {
            result->store(error ? -1 : result->load() == 1 ? 2
                                                           : -2);
        }
    };
    for (const int workers : {0, 2})
    {
        R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
        C::OperatorSchedule schedule(engine, engine.CreateCPUStream(), engine.CreateCPUStream(), true);
        auto result = std::make_shared<std::atomic<int>>(0);
        auto prepared = std::make_shared<Prepared>(Prepared{result});
        std::weak_ptr<Prepared> retained = prepared;
        schedule.SubmitPrepared(std::move(prepared));
        schedule.Join();
        CHECK(result->load() == 2);
        CHECK(retained.expired());
    }
}
TEST_CASE("Imported interface completions progress on the controller and retain receiver resources")
{
    struct ControllerCompletion final : R::StreamEvent::State
    {
        const std::thread::id controller = std::this_thread::get_id();
        std::atomic<bool> allowed{false};
        mutable std::atomic<bool> done{false};
        bool Ready() const override
        {
            if (allowed && std::this_thread::get_id() == controller)
                done = true;
            return done;
        }
        void Wait() const override
        {
            DNDS_check_throw_info(Ready(), "mock completion needs controller progress");
        }
    };
    for (const int workers : {0, 2})
        for (const bool consume : {false, true})
        {
            C::ExecutionSettings settings;
            settings.hostThreads = workers;
            auto event = std::make_shared<ControllerCompletion>();
            auto value = std::make_shared<int>(17);
            const std::weak_ptr<int> lifetime = value;
            std::atomic<int> published{0};
            std::atomic<int> publications{0};
            {
                C::DomainExecutionContext context(settings);
                C::QueuedEvaluationDependency dependency;
                dependency.completion = R::StreamEvent(event);
                dependency.owner = value;
                dependency.publish = [value, &published, &publications]
                { published = *value; ++publications; };
                context.SetEvaluationDependency(std::move(dependency));
                value.reset();
                CHECK_FALSE(lifetime.expired());
                if (consume)
                {
                    auto incoming = context.AcquireEvaluationDependency();
                    REQUIRE(incoming);
                    const auto record = std::make_shared<C::QueuedEvaluationDependency>(std::move(*incoming));
                    context.ComputeStream().WaitFor(record->completion);
                    context.Engine().Submit(context.ComputeStream(),
                                            [record](R::ExecutionContext)
                                            { record->publish(); });
                }
                CHECK(published == 0);
                event->allowed = true;
                if (consume)
                {
                    context.Synchronize();
                    CHECK(published == 17);
                    // A nested finest evaluation obtains the same latest
                    // packet without rerunning its publication side effects.
                    const auto again = context.AcquireEvaluationDependency();
                    REQUIRE(again);
                    again->publish();
                    CHECK(publications == 1);
                }
                // Unconsumed accepted work is completed by context teardown.
            }
            CHECK(published == 17);
            CHECK(publications == 1);
            CHECK(lifetime.expired());
        }
}
TEST_CASE("Cauchy domain placement preserves rank namespaces and isolates shared groups")
{
    auto reverse = AllRanks();
    std::reverse(reverse.begin(), reverse.end());
    C::DomainPlacement placement(MPI_COMM_WORLD, {{"flow", AllRanks()}, {"reverse", reverse}});
    CHECK(placement.LocalDomains() == std::vector<I>{0, 1});
    CHECK(placement.DomainRank(0) == rank);
    CHECK(placement.DomainRank(1) == size - rank - 1);
    for (int r = 0; r < size; ++r)
        CHECK(placement.SimulationRank(1, r) == size - r - 1);
    int relation;
    MPI_Comm_compare(placement.Context(0)->Communicator(), placement.Context(1)->Communicator(), &relation);
    CHECK(relation != MPI_IDENT);
    CHECK_THROWS(placement.Group(-1));
    CHECK_THROWS(placement.SimulationRank(0, size));
    CHECK_THROWS(C::DomainPlacement(MPI_COMM_WORLD, {{"bad", {size}}}));
    CHECK_THROWS(C::DomainPlacement(MPI_COMM_WORLD, {{"duplicate", {0, 0}}}));
    CHECK_THROWS(C::DomainPlacement(MPI_COMM_WORLD, {{"same", {0}}, {"same", {0}}}));
    if (size > 1)
        CHECK_THROWS(C::DomainPlacement(MPI_COMM_WORLD, {{rank == 0 ? "left" : "right", AllRanks()}}));
}
TEST_CASE("Frozen trace rounds give identical shared and disjoint rank numerical schemes")
{
    std::vector<int> even, odd;
    for (int r = 0; r < size; ++r)
        (r % 2 ? odd : even).push_back(r);
    if (odd.empty())
        odd = even;
    C::DomainPlacement shared(MPI_COMM_WORLD, {{"a", AllRanks()}, {"b", AllRanks()}});
    C::DomainPlacement split(MPI_COMM_WORLD, {{"a", even}, {"b", odd}});
    C::CoordinatedSchedule first(shared), second(split);
    for (int step = 0; step < 5; ++step)
    {
        // Independent oracle recurrence: a'=a+2b+1; b'=3a-b+2.
        const auto before = first.Traces();
        const std::vector<double> expected{before[0] + 2 * before[1] + 1, 3 * before[0] - before[1] + 2};
        const auto advance = [&](C::CoordinatedSchedule &schedule, const C::DomainPlacement &placement)
        {
            schedule.Advance(.125, [&](I domain, const auto &trace, const auto &epoch)
                             {
                CHECK(epoch.acceptedStep == uint64_t(step));
                double value = domain == 0 ? trace[0] + 2 * trace[1] + 1 : 3 * trace[0] - trace[1] + 2;
                // Different local inner iteration counts, domain collectives
                // only. Simulation control is entered once both domains finish.
                for (I iteration = 0; iteration < domain + 1; ++iteration)
                {
                    double maximum;
                    MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, placement.Context(domain)->Communicator());
                    value = maximum;
                }
                return value; });
        };
        advance(first, shared);
        advance(second, split);
        CHECK(first.Traces() == expected);
        CHECK(second.Traces() == expected);
        CHECK(first.Time() == doctest::Approx((step + 1) * .125));
    }
    for (I domain = 0; domain < split.DomainNumber(); ++domain)
        if (split.DomainRank(domain) < 0)
            CHECK_THROWS(split.Context(domain));
    const auto before = second.Traces();
    CHECK_THROWS(second.Advance(.125, [&](I, const auto &, const auto &) -> double
                                { return rank == 0 ? std::numeric_limits<double>::quiet_NaN() : 7; }));
    CHECK(second.Traces() == before);
    CHECK(second.Epoch().acceptedStep == 5);
    CHECK_THROWS(second.Advance(rank == 0 ? -1 : .125, [](I, const auto &, const auto &)
                                { return 0.; }));
    second.Advance(.125, [](I domain, const auto &, const auto &epoch)
                   { DNDS_check_throw_info(epoch.trial == 7, "rejected trial epoch was reused"); return double(domain); });
    CHECK(second.Traces() == std::vector<double>{0, 1});
}
TEST_CASE("Named coupled DOF blocks preserve layout independence subset algebra and owned norms")
{
    auto context = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    A::DofSpace space(context, size > 1 && rank == 0 ? 0 : 7, 3, "cells");
    A::DofBundle state(space);
    state.Add("flow", A::DofField<double>(space, R::RowLayout::Dense(0, 5)));
    state.Add("species", A::DofField<double>(space, R::RowLayout::ColMajor(0, 11, {8, 1})));
    state.Fill(2);
    auto copy = state.Clone();
    copy.Fill(9, {"species"});
    copy.Axpy(3, state, {"flow"});
    const double cells = 7 * (size > 1 ? size - 1 : 1);
    CHECK(copy.DotGlobal(state, {"flow"}) == doctest::Approx(cells * 5 * 8 * 2));
    CHECK(copy.DotGlobal(state, {"species"}) == doctest::Approx(cells * 11 * 9 * 2));
    // Poison ghost values numerically; owned norms must remain unchanged.
    {
        auto w = copy.Block("flow").Ghost().Write();
        for (I iRow = 0; iRow < 3; ++iRow)
            for (I j = 0; j < 5; ++j)
                w.View()(iRow, j) = 1e30;
    }
    CHECK(copy.DotGlobal(state, {"flow"}) == doctest::Approx(cells * 5 * 8 * 2));
    CHECK_THROWS(copy.Fill(0, {"flow", "missing"}));
    CHECK_THROWS(copy.Fill(0, {"flow", "flow"}));
    CHECK_THROWS(copy.Add("flow", A::DofField<double>(space, 5)));
    CHECK_THROWS(copy.Add("alias", copy.Block("flow")));
    CHECK(copy.DotGlobal(state, {"flow"}) == doctest::Approx(cells * 5 * 8 * 2));
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
