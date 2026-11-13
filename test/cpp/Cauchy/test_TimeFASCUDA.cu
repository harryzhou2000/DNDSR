#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/Multigrid/PolynomialMultigridOperations.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Solver/StateFAS.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "Cauchy/IO/StateArchive.hpp"
#include "DNDS/Serializer/SerializerH5.hpp"
#include <filesystem>
#include <unistd.h>
#include <cuda_runtime.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using DNDS::real;
static DNDS::MPIInfo mpi;
namespace
{
    C::EquationDofs MakeState(const A::DofSpace &space, int n, R::Device device)
    {
        C::DomainSettings settings;
        if (n > 5)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .01;
            settings.equation.rans.model = n == 6 ? "SA" : "KOmegaSST";
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        return C::EquationDofs::Coupled(equation.Schema(), A::DofField<real>(space,
                                                                             R::Array<real>(R::RowLayout::Dense(space.OwnedRows(), n, n + 2), device),
                                                                             R::Array<real>(R::RowLayout::Dense(space.GhostRows(), n, n + 2), device)));
    }
    void Equal(const R::Array<real> &expected, const R::Array<real> &actual)
    {
        const auto host = actual.To(R::Device::Host());
        const auto a = expected.Read(), b = host.Read();
        for (I iRow = 0; iRow < expected.RowCount(); ++iRow)
            for (I j = 0; j < expected.Layout().Width(); ++j)
                CHECK(b.View()(iRow, j) == doctest::Approx(a.View()(iRow, j)).epsilon(2e-12).scale(1));
    }
}
TEST_CASE("Resident equation checkpoint copies preserve padded layout and distributed row remapping")
{
    int deviceCount = 0;
    REQUIRE(cudaGetDeviceCount(&deviceCount) == cudaSuccess);
    REQUIRE(deviceCount > 0);
    const auto device = R::Device::CUDA(mpi.rank % deviceCount);
    REQUIRE(cudaSetDevice(device.ordinal) == cudaSuccess);
    R::EagerEngine engine;
    const auto stream = engine.CreateStream(std::make_shared<R::CUDALauncher>(device.ordinal));
    auto communication = std::make_shared<R::CommunicationContext>(mpi.comm);
    constexpr I rows = 13;
    A::DofSpace space(communication, rows, 2, "CUDA checkpoint ingress");
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto path = (std::filesystem::temp_directory_path() / ("cauchy-cuda-archive-" + std::to_string(pid) + ".h5")).string();
    for (const int n : {5, 6, 7})
    {
        auto host = MakeState(space, n, R::Device::Host());
        {
            auto write = host.CoupledField().Owned().Write();
            for (I iCell = 0; iCell < rows; ++iCell)
                for (I j = 0; j < n; ++j)
                    write.View()(iCell, j) = 10000 * mpi.rank + 100 * iCell + j;
        }
        auto resident = C::EquationDofOperations::CopyOwnedTo(host, device, stream.Context());
        auto file = std::make_shared<DNDS::Serializer::SerializerH5>(mpi);
        file->OpenFile(path, false);
        C::WriteEquationArrays(resident, file, "state_", mpi, stream.Context());
        file->CloseFile();
        file->OpenFile(path, true);
        const auto archive = C::StateArchive::Read(file, "state_", C::DescribeEquationStorage(host), mpi, {rows, mpi.rank * rows});
        file->CloseFile();
        auto destination = C::EquationDofOperations::NewVector(resident);
        std::vector<R::RowRequest> requests;
        for (I iCell = 0; iCell < rows; ++iCell)
            requests.push_back({(mpi.rank + 1) % mpi.size, rows - 1 - iCell, iCell});
        R::RowTransferPlan route(communication, rows, rows, requests);
        archive.Restore(destination, route, communication, stream.Context());
        const auto snapshot = C::EquationDofOperations::CopyOwnedTo(destination, R::Device::Host(), stream.Context());
        C::EquationDofOperations::Fill(engine, stream, destination, -1);
        engine.Synchronize();
        const auto read = snapshot.CoupledField().Owned().Read();
        CHECK(snapshot.CoupledField().Owned().Layout().Stride() == n + 2);
        for (I iCell = 0; iCell < rows; ++iCell)
            for (I j = 0; j < n; ++j)
                CHECK(read.View()(iCell, j) == 10000 * ((mpi.rank + 1) % mpi.size) + 100 * (rows - 1 - iCell) + j);
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(path));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("Method-owned stage history uses instance-bound algebra on every execution backend")
{
    const auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    A::DofSpace space(communication, 17, 0, "time CUDA algebra");
    for (const int n : {5, 6, 7})
        for (const std::string name : {"bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"})
        {
            CAPTURE(n);
            CAPTURE(name);
            std::vector<R::Array<real>> anchor;
            for (int backend : {0, 1, 2})
            {
                CAPTURE(backend);
                const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
                R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
                const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
                auto initial = MakeState(space, n, device), trial = MakeState(space, n, device);
                C::QueuedEquationOperations operations(lane);
                operations.Fill(initial, 2);
                operations.Synchronize();
                // Accepted-state publication reserves access before the
                // cross-domain decision. Its copy must run on the same lane
                // without reacquiring those already-held leases.
                {
                    auto write = trial.WriteOwned(lane.Context());
                    auto read = initial.ReadOwned(lane.Context());
                    C::EquationDofOperations::CopyReserved(engine, lane, trial, initial, write, read);
                }
                const auto initialHost = initial.CoupledField().Owned().To(R::Device::Host());
                Equal(initialHost, trial.CoupledField().Owned());
                const auto run = [&](auto &stepper)
                {
                    auto stoppingResidual = C::EquationDofOperations::NewVector(initial);
                    const auto evaluate = [&](auto &u, auto &rate, real, int)
                    { operations.Copy(rate, u); operations.Scale(rate, -.3); operations.Synchronize(); };
                    real time = 0;
                    for (int step = 0; step < 2; ++step)
                    {
                        const real dt = step ? .13 : .1; // exercises variable-step BDF history
                        stepper.BeginStep(initial, trial, time, dt, evaluate);
                        for (int iGroup = 0; iGroup < stepper.GroupNumber(); ++iGroup)
                        {
                            auto group = stepper.PrepareGroup(trial);
                            if (group.implicit)
                                for (int update = 0; update < (group.coupledStages == 2 ? 4 : 1); ++update)
                                {
                                    for (int stage = 0; stage < group.coupledStages; ++stage)
                                    {
                                        auto equation = stepper.PrepareEquation(group, trial, stage);
                                        const auto c = equation.coefficients;
                                        // Exact scalar solve of F(U)=-.3 U for
                                        // each prepared equation; no FV/PP is
                                        // involved in this temporal arithmetic oracle.
                                        operations.Copy(equation.state, equation.resOther);
                                        operations.Scale(equation.state, 1 / (c.inverseDt + .3 * c.rhsScale));
                                        // The synchronous oracle tracks completed
                                        // versions. Queued adapters instead track
                                        // submission epochs and need no drain here.
                                        if (backend == 0)
                                            operations.Synchronize();
                                        stepper.CompleteEquation(group, trial, evaluate);
                                    }
                                    if (group.coupledStages == 2)
                                        stepper.AssembleStoppingResidual(group, trial, stoppingResidual);
                                }
                            stepper.CompleteGroup(group, trial, evaluate);
                        }
                        operations.Copy(initial, trial);
                        operations.Synchronize();
                        stepper.Commit();
                        if (backend == 0)
                            anchor.push_back(initial.CoupledField().Owned().Clone());
                        else
                            Equal(anchor[step], initial.CoupledField().Owned());
                        if (name == "bdf1" && step == 0)
                        {
                            const auto host = initial.CoupledField().Owned().To(R::Device::Host());
                            const auto read = host.Read();
                            CHECK(read.View()(0, 0) == doctest::Approx(2 / 1.03).epsilon(1e-12));
                        }
                        time += dt;
                    }
                };
                if (backend == 0)
                {
                    // Keep the synchronous pre-CUDA arithmetic as the anchor.
                    DNDS::ODE::StateTimeStepper<C::EquationDofs, C::EquationDofOperations> stepper(initial, name);
                    run(stepper);
                }
                else
                {
                    C::EquationTimeStepper stepper(initial, name, {}, {}, operations);
                    run(stepper);
                }
            }
        }
}
TEST_CASE("Selected FAS algebra and cross-space transfer preserve frozen components")
{
    using State = C::EquationDofs;
    using Operations = C::PolynomialMultigridOperations;
    using Cycle = DNDS::Multigrid::StateFASCycle<State, Operations>;
    const auto communication = std::make_shared<R::CommunicationContext>(MPI_COMM_WORLD);
    A::DofSpace fineSpace(communication, 17, 0, "fine CUDA FAS"), coarseSpace(communication, 17, 0, "coarse CUDA FAS");
    for (int backend : {0, 1, 2})
    {
        CAPTURE(backend);
        const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
        R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto lane = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
        Operations operations(lane);
        auto fine = MakeState(fineSpace, 7, device), coarse = MakeState(coarseSpace, 7, device), source = MakeState(fineSpace, 7, device);
        operations.Fill(fine, .4);
        operations.Fill(coarse, .4);
        operations.Fill(source, 4);
        operations.Synchronize();
        const auto level = [&](State state, real damping)
        {
            const auto smooth = [operations, damping](State &u, const State &b, I count, const std::vector<I> &selected)
            {
                auto delta = Operations::NewVector(u);
                for (I pass = 0; pass < count; ++pass)
                {
                    operations.LinearCombination(delta, 1, b, -2, u);
                    operations.Scale(delta, damping / 2);
                    operations.AddSelected(u, delta, selected);
                }
                operations.Synchronize();
                return count;
            };
            return DNDS::Multigrid::StateFASLevel<State>{state,
                                                         [operations](State &u, State &out)
                                                         { operations.Copy(out, u); operations.Scale(out, 2); operations.Synchronize(); },
                                                         [smooth](State &u, const State &b, I count)
                                                         { return smooth(u, b, count, {}); },
                                                         {},
                                                         smooth,
                                                         {}};
        };
        const auto transfer = [operations](const State &input, State &output)
        { operations.CopyIdentityRows(input, output); operations.Synchronize(); };
        Cycle cycle({level(fine, .3), level(coarse, .8)}, {{transfer, transfer, transfer}}, {1, 1, 2, {0, 4}}, [operations](const auto &run)
                    { run(); operations.Synchronize(); }, {}, operations);
        cycle.Apply(source);
        operations.Synchronize();
        const auto host = fine.CoupledField().Owned().To(R::Device::Host());
        const auto read = host.Read();
        for (I iRow = 0; iRow < host.RowCount(); ++iRow)
            for (I j = 0; j < 7; ++j)
                CHECK(read.View()(iRow, j) == doctest::Approx(j == 0 || j == 4 ? 2 - (2 - .4) * .0196 : .4).epsilon(1e-12));
    }
}
int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    mpi = DNDS::MPIInfo(MPI_COMM_WORLD);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    DNDS::MPI::Finalize();
    return result;
}
