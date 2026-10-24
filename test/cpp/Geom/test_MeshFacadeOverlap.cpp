#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "MeshFacadeOverlapFixture.hpp"
#include "MeshFacadeOverlapKernel.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace T = DNDS::Geom::OverlapTest;
namespace R = DNDS::Runtime;
namespace G = DNDS::Geom;

static void CheckCPUOverlap(T::Fixture &fixture, size_t workers)
{
    auto &mesh = *fixture.mesh;
    auto access = mesh.AcquireRead(G::MeshReadRequirements::CellFlux());
    const auto view = access.HostView();
    const auto partition = T::Classify(mesh, view);
    auto coreRows = T::RowList(partition.core), ringRows = T::RowList(partition.ring);
    auto owned = T::OwnedField(mesh);
    R::Array<DNDS::real> ghost(R::RowLayout::Dense(size_t(mesh.NumCellGhost()), 1));
    R::Array<DNDS::real> coreResult(R::RowLayout::Dense(partition.core.size(), 1));
    R::Array<DNDS::real> ringResult(R::RowLayout::Dense(partition.ring.size(), 1));
    const auto entities = mesh.Entities(G::EntityKind::Cell);
    auto route = entities.BuildRowTransferPlan();
    auto binding = R::PayloadPlan(route, owned.Layout(), ghost.Layout(), R::ElementSchema::Arithmetic<DNDS::real>())
                       .Bind(owned, ghost, R::TransferTransport::Direct);
    R::EagerEngine engine(workers);
    auto core = engine.CreateCPUStream(), ring = engine.CreateCPUStream();
    struct WorkerState
    {
        std::atomic<bool> open{false};
        std::atomic<bool> entered{false}, exited{false};
        std::atomic<size_t> passes{0};
    };
    auto state = std::make_shared<WorkerState>();
    // The submitted body owns every atomic it touches, including on exception.
    // Release before engine destruction drains its captured shared state.
    struct ReleaseWorker
    {
        std::shared_ptr<WorkerState> state;
        ~ReleaseWorker() { state->open = true; }
    } releaseWorker{state};
    R::MPIProgress progress;
    auto received = progress.Import(binding.BeginScheduled());
    // This is an active receive lease, independent of MPI's eager/rendezvous
    // message size or whether the network has already delivered its bytes.
    CHECK_THROWS(ghost.Read());
    R::SubmitArrays(engine, core, [state, view, workers](R::ExecutionContext, auto rows, auto u, auto out)
                    {
        state->entered = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        do
        {
            T::CoreHost(view, rows, u, out); // Receives no ghost solution handle.
            ++state->passes;
            if (!workers || state->open.load())
                break;
            DNDS_check_throw_info(std::chrono::steady_clock::now() < deadline, "real mesh CPU/MPI overlap gate timed out");
            std::this_thread::yield();
        } while (true);
        state->exited = true; }, R::Read(coreRows), R::Read(owned), R::Write(coreResult));
    const auto coreDone = core.Record();
    if (workers)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while ((!state->entered.load() || !state->passes.load()) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        CHECK(state->entered.load());
        CHECK(state->passes.load() > 0);
        CHECK_FALSE(state->exited.load());
    }
    CHECK_THROWS(ghost.Read()); // Core work occurred inside the active receive interval.
    progress.Drain();           // Main-thread MPI completes while the core body is live.
    CHECK(received.Ready());
    if (workers)
    {
        CHECK_FALSE(state->exited.load());
        CHECK_FALSE(coreDone.Ready());
    }
    else
    {
        CHECK(state->entered.load());
        CHECK(state->exited.load());
        CHECK(state->passes.load() == 1);
    }
    state->open = true;
    // Submit dependent work only after the interval checks. A stream's Ready
    // query may otherwise acknowledge a small MPI transfer during submission.
    ring.WaitFor(received);
    ring.WaitFor(coreDone);
    R::SubmitArrays(engine, ring, [view](R::ExecutionContext, auto rows, auto u, auto halo, auto out)
                    { T::RingHost(view, rows, u, halo, out); }, R::Read(ringRows), R::Read(owned), R::Read(ghost), R::Write(ringResult));
    engine.Synchronize();
    partition.stamp.Validate();
    const auto coreValues = coreResult.Read(), ringValues = ringResult.Read();
    for (size_t i = 0; i < partition.core.size(); ++i)
        CHECK(coreValues.View()(i, 0) == doctest::Approx(T::Expected(fixture, partition.core[i])).epsilon(1e-12));
    for (size_t i = 0; i < partition.ring.size(); ++i)
        CHECK(ringValues.View()(i, 0) == doctest::Approx(T::Expected(fixture, partition.ring[i])).epsilon(1e-12));
    DNDS::index counts[]{DNDS::index(partition.core.size()), DNDS::index(partition.ring.size())};
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, DNDS::DNDS_MPI_INDEX, MPI_SUM, mesh.mpi.comm);
    CHECK(counts[0] > 0);
    CHECK(counts[0] + counts[1] == DNDS::index(fixture.expected.size()));
    if (mesh.mpi.size > 1)
        CHECK(counts[1] > 0);
    if (mesh.mpi.rank == 0)
        std::cout << "real mesh CPU engine workers=" << workers << " core=" << counts[0]
                  << " ring=" << counts[1] << " original-id oracle cells=" << fixture.expected.size()
                  << "; active-operation interval verified, no speedup claim\n";
}

TEST_CASE("Real mesh finite-volume neighbor sums overlap native halo with inline and worker CPU execution")
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto fixture = T::BuildFixture(mpi);
    for (size_t workers : {size_t(0), size_t(2)})
        CheckCPUOverlap(fixture, workers);
    auto prior = fixture.mesh->Entities(G::EntityKind::Cell);
    auto priorRoute = prior.BuildRowTransferPlan();
    fixture.mesh->ReorderLocalCells(2, 1);
    CHECK_FALSE(prior.Valid());
    CHECK_FALSE(priorRoute.Valid());
    for (size_t workers : {size_t(0), size_t(2)})
        CheckCPUOverlap(fixture, workers);
    fixture.mesh->PublishNative(G::MeshReadRequirements::CellFlux());
    for (size_t workers : {size_t(0), size_t(2)})
        CheckCPUOverlap(fixture, workers);
    T::RedistributeCurrent(fixture);
    for (size_t workers : {size_t(0), size_t(2)})
        CheckCPUOverlap(fixture, workers);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    DNDS_check_throw_info(provided >= MPI_THREAD_FUNNELED, "mesh overlap test requires MPI FUNNELED");
    const auto result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
