#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/DomainPlacement.hpp"
#include "Cauchy/Implicit/NonlinearSolve.hpp"
#include "Cauchy/Physics/EquationSetup.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <array>
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
    // Real direct-datatype transfer; the production policy owns the solve loop.
    struct Exchange
    {
        C::DomainPlacement placement;
        std::shared_ptr<R::CommunicationContext> context;
        std::vector<I> rootsHere;
        R::Array<double> source, packet, scratch;
        R::RowTransferPlan route;
        R::TransferBinding binding;
        std::optional<R::TransferOperation> operation;
        int begins = 0, completes = 0;
        uint64_t sourceVersion = 0;
        bool published = false;
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
            : placement(mpi.comm, Groups(shared)), context(placement.ControlContext()),
              rootsHere(LocalRoots(placement)), source(R::RowLayout::Dense(I(rootsHere.size()), 1)),
              packet(R::RowLayout::Dense(2, 1)), scratch(R::RowLayout::Dense(1, 1)), route(Route()),
              binding(R::PayloadPlan(route, source.Layout(), packet.Layout(),
                                     R::ElementSchema::For<double>())
                          .Bind(source, packet)) {}
        void Fill(const std::array<double, 2> &values)
        {
            auto write = source.Write();
            const auto view = write.View<R::FixedRows<1>>();
            for (I iRow = 0; iRow < I(rootsHere.size()); ++iRow)
                view(iRow, 0) = values[size_t(rootsHere[size_t(iRow)])];
        }
        void Begin(const R::ExecutionStream &communication)
        {
            DNDS_check_throw_info(!operation, "mock transfer already active");
            published = false;
            sourceVersion = source.Version();
            operation.emplace(binding.BeginReadyChecked(communication));
            ++begins;
        }
        void Finish()
        {
            if (!operation)
                return;
            operation->Wait();
            operation.reset();
            DNDS_check_throw_info(source.Version() == sourceVersion, "interface source changed during send");
            published = true;
            ++completes;
        }
    };
    C::ImplicitSettings Settings(int iterations)
    {
        C::ImplicitSettings settings;
        settings.iterations = settings.minIterations = iterations;
        settings.absoluteTolerance = settings.relativeTolerance = 0;
        return settings;
    }
}

TEST_CASE("Production single-stage policy preserves frozen/updated traces and real overlap")
{
    for (bool shared : {false, true})
        for (bool refreshEachIteration : {false, true})
            for (int workers : {0, 2})
                for (bool overlap : {false, true})
                {
                    R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                    const auto compute = engine.CreateCPUStream(), communication = engine.CreateCPUStream();
                    Exchange exchange(shared);
                    std::array<double, 2> local{1, 2}, expected{1, 2}, donors{1, 2};
                    const auto initial = expected;
                    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
                    C::NonlinearSolveSetup setup(Settings(3), schema, 0, "singleStageImplicit", {});
                    std::vector<C::NonlinearSession> sessions;
                    const auto localDomains = exchange.placement.LocalDomains();
                    sessions.reserve(localDomains.size());
                    I round = -1;
                    for (const auto domain : localDomains)
                    {
                        C::NonlinearProblem problem;
                        problem.evaluate = [&]
                        {
                            exchange.Finish();
                            CHECK(exchange.published);
                            const auto packet = exchange.packet.Read();
                            for (I iDomain = 0; iDomain < 2; ++iDomain)
                                CHECK(packet.View()(iDomain, 0) == donors[size_t(iDomain)]);
                        };
                        problem.measure = [&, domain](auto &norms)
                        {
                            // Domain norms use the subset, not the simulation communicator.
                            const auto comm = exchange.placement.Context(domain)->Communicator();
                            const double norm = DNDS::MPI::AllreduceValue<DNDS::MPI::Reduction::Max>(1., comm);
                            std::fill(norms.begin(), norms.end(), norm);
                        };
                        problem.reconstruction = []
                        { return std::make_pair(0, 0.); };
                        problem.advanceStage = [&, domain](int slot, auto &, bool apply)
                        {
                            CHECK(slot == 0);
                            CHECK(apply);
                            const auto packet = exchange.packet.Read();
                            local[size_t(domain)] = .5 * local[size_t(domain)] +
                                                    .25 * packet.View()(1 - domain, 0) + .1 * (domain + 1);
                            CHECK(local[size_t(domain)] == expected[size_t(domain)]);
                        };
                        sessions.push_back(C::NonlinearSession::Prepare(setup, 2, std::move(problem)));
                    }
                    C::NonlinearSolveGroup group;
                    group.communication = exchange.context;
                    group.policy = setup.policy;
                    group.collectiveChecks = true;
                    for (auto &session : sessions)
                        group.domains.emplace_back(session);
                    group.refreshInterfaces = [&](int slot, I iteration)
                    {
                        CHECK(slot == 0);
                        CHECK(iteration == ++round);
                        const bool refresh = refreshEachIteration || iteration == 0;
                        donors = refreshEachIteration ? expected : initial;
                        const auto old = expected;
                        // Iteration 3 observes the final residual without an update.
                        if (iteration < 3)
                            for (I domain = 0; domain < 2; ++domain)
                                expected[size_t(domain)] = .5 * old[size_t(domain)] +
                                                           .25 * donors[size_t(1 - domain)] + .1 * (domain + 1);
                        exchange.Fill(local);
                        if (refresh)
                        {
                            exchange.Begin(communication);
                            CHECK_THROWS(exchange.packet.Read());
                            if (exchange.source.RowCount())
                                CHECK_THROWS(exchange.source.Write());
                        }
                        if (!overlap)
                            exchange.Finish();
                        std::atomic<bool> started{false}, release{!overlap || !workers};
                        R::SubmitArrays(engine, compute, [&](auto, auto source, auto output)
                                        {
                            double sum = 0;
                            for (I iRow = 0; iRow < source.RowCount(); ++iRow)
                                sum += source(iRow, 0);
                            output(0, 0) = 2 * sum;
                            started = true;
                            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                            while (!release.load() && std::chrono::steady_clock::now() < deadline)
                                std::this_thread::yield();
                            DNDS_check_throw_info(release.load(), "mock core gate timed out"); }, R::ReadAs<R::FixedRows<1>>(exchange.source), R::WriteAs<R::FixedRows<1>>(exchange.scratch));
                        if (workers && overlap)
                        {
                            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                            while (!started.load() && std::chrono::steady_clock::now() < deadline)
                                std::this_thread::yield();
                            CHECK(started.load());
                            if (refresh)
                            {
                                bool ready = false;
                                while (!ready && std::chrono::steady_clock::now() < deadline)
                                    ready = exchange.operation->Test();
                                CHECK(ready);
                                // The live core read still protects source after MPI completion.
                                if (exchange.source.RowCount())
                                    CHECK_THROWS(exchange.source.Write());
                            }
                            release = true;
                        }
                        compute.Record().WaitHost();
                        const auto scratch = exchange.scratch.Read();
                        double sum = 0;
                        for (const auto domain : exchange.rootsHere)
                            sum += local[size_t(domain)];
                        CHECK(scratch.View()(0, 0) == 2 * sum);
                    };
                    group.finishInterfaces = [&]
                    { exchange.Finish(); };
                    C::SolveNonlinearGroup(group);
                    CHECK(round == 3);
                    CHECK(exchange.begins == (refreshEachIteration ? 4 : 1));
                    CHECK(exchange.completes == exchange.begins);
                    for (const auto &session : sessions)
                    {
                        CHECK(session.Complete());
                        CHECK(session.Report().iterations == 3);
                    }
                    engine.Synchronize();
                }
}

TEST_CASE("Retained packet reads reject refresh before posting and preserve received values")
{
    R::EagerEngine engine(0, std::make_shared<R::AccessTrackingContext>());
    const auto communication = engine.CreateCPUStream();
    Exchange exchange(true);
    exchange.Fill({1, 2});
    exchange.Begin(communication);
    exchange.Finish();
    const auto version = exchange.packet.Version();
    {
        std::optional<R::Array<double>::ReadLease> retained;
        if (!mpi.rank)
            retained.emplace(exchange.packet.Read());
        exchange.Fill({3, 4});
        CHECK_THROWS(exchange.Begin(communication));
        CHECK_FALSE(exchange.operation.has_value());
        CHECK(exchange.packet.Version() == version);
        if (retained)
            CHECK(retained->View()(0, 0) == 1);
    }
    exchange.Begin(communication);
    exchange.Finish();
    const auto packet = exchange.packet.Read();
    CHECK(packet.View()(0, 0) == 3);
    CHECK(packet.View()(1, 0) == 4);
    engine.Synchronize();
}

TEST_CASE("Coordinated DITR policy retains idle ranks and midpoint-before-endpoint ordering")
{
    C::DomainPlacement placement(mpi.comm, {{"one", {0}}});
    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
    const C::UpdatePolicyRequirements requirements{
        2, C::NonlinearDetail::ResidualObservation::AfterUpdate, true};
    C::NonlinearSolveSetup setup(Settings(2), schema, 0, "ditrSequential2Stage", requirements);
    std::vector<std::string> order;
    std::optional<C::NonlinearSession> session;
    C::NonlinearProblem problem;
    problem.evaluate = [&]
    { order.push_back("observe"); };
    problem.measure = [](auto &norms)
    { std::fill(norms.begin(), norms.end(), 1.); };
    problem.reconstruction = []
    { return std::make_pair(0, 0.); };
    problem.advanceStage = [&](int slot, auto &, bool apply)
    {
        CHECK(apply);
        order.push_back("update" + std::to_string(slot));
    };
    if (placement.DomainRank(0) >= 0)
        session.emplace(C::NonlinearSession::Prepare(setup, 0, std::move(problem)));
    C::NonlinearSolveGroup group;
    group.communication = placement.ControlContext();
    group.policy = setup.policy;
    group.collectiveChecks = true;
    if (session)
        group.domains.emplace_back(*session);
    group.refreshInterfaces = [&](int slot, I iteration)
    {
        CHECK(iteration >= 1);
        CHECK(iteration <= 2);
        order.push_back("refresh" + std::to_string(slot));
    };
    group.finishInterfaces = [&]
    { order.push_back("finish"); };
    C::SolveNonlinearGroup(group);
    std::vector<std::string> expected;
    for (int iteration = 1; iteration <= 2; ++iteration)
    {
        for (int slot = 0; slot < 2; ++slot)
        {
            expected.push_back("refresh" + std::to_string(slot));
            if (session)
                expected.push_back("update" + std::to_string(slot));
            expected.push_back("finish");
        }
        if (session)
            expected.push_back("observe");
    }
    CHECK(order == expected);
    if (session)
        CHECK(session->Report().iterations == 2);
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
