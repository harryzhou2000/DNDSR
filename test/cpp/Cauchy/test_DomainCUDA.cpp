#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "CartesianFixture.hpp"
#include <filesystem>
#include <unistd.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Device domains advance nonuniform implicit flow through single grid and pMG with restart")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.SORInstead = false;
    settings.numerics.reconstructionIterations = 12;
    settings.numerics.reconstructionTolerance = 0;
    settings.numerics.faceFallback = false;
    settings.numerics.limiter = "none";
    settings.implicit.iterations = 80;
    settings.implicit.minIterations = 1;
    settings.implicit.relativeTolerance = 1e-8;
    settings.implicit.absoluteTolerance = 1e-11;
    settings.implicit.linear.method = "gmres";
    settings.implicit.linear.preconditioner = "jacobi";
    settings.implicit.linear.relativeTolerance = 1e-3;
    settings.implicit.diagonal = "block";
    settings.implicit.multigrid.cycles = 40;
    settings.implicit.multigrid.preSmooth = 1;
    settings.implicit.multigrid.postSmooth = 1;
    settings.implicit.multigrid.coarseSmooth = 3;
    const auto gas = C::EquationSystem::Resolve(settings.equation).Gas();
    R::Array<double> initial(R::RowLayout::Dense(metrics.NumCell(), 5));
    {
        auto write = initial.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto point = metrics.GetCellCent(iCell);
            const auto u = gas.Conservative({1 + .02 * std::sin(2 * DNDS::pi * (point[0] + point[1])), .3, .2, .1, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = u[j];
        }
    }
    int pid = getpid();
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto checkpoint = (std::filesystem::temp_directory_path() / ("cauchy-device-domain-" + std::to_string(pid) + ".h5")).string();
    for (int profile = 0; profile < 2; ++profile)
    {
        CAPTURE(profile);
        C::TimeSettings time;
        time.method = profile ? "esdirk4" : "bdf2";
        settings.implicit.multigrid.levels = profile ? 2 : 0;
        settings.implicit.linear.method = profile ? "jacobi" : "gmres";
        std::optional<R::Array<double>> reference;
        for (int backend = 0; backend < 3; ++backend)
        {
            CAPTURE(backend);
            settings.execution.device = backend == 2 ? "cuda" : "cpu";
            settings.execution.deviceOrdinal = 0; // np2 launcher exposes one GPU per rank.
            settings.execution.hostThreads = backend == 1 ? 1 : 0;
            C::Domain domain("fluid", mesh, settings, time);
            const auto device = domain.Dofs().Device();
            REQUIRE(device == (backend == 2 ? R::Device::CUDA(0) : R::Device::Host()));
            domain.Dofs().CoupledField().Owned().CopyFrom(initial, {device, nullptr}).Wait();
            for (int step = 0; step < 2; ++step)
            {
                REQUIRE(domain.Advance(1e-4).accepted);
                REQUIRE_FALSE(domain.LastImplicitStages().empty());
                for (const auto &stage : domain.LastImplicitStages())
                    CHECK(stage.converged);
            }
            const auto final = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
            if (!backend)
                reference = final.Clone();
            const auto actual = final.Read(), expected = reference->Read();
            for (I iCell = 0; iCell < final.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(actual.View()(iCell, j) == doctest::Approx(expected.View()(iCell, j)).epsilon(2e-7).scale(1));
            if (backend == 2 && profile == 0)
            {
                // Actual domain archive: preserve BDF history and VR warm
                // state, then compare the next unequal physical step.
                domain.WriteCheckpoint(checkpoint, C::Json::object(), true);
                C::Domain restored("fluid", mesh, settings, time);
                restored.ReadCheckpoint(checkpoint, C::Json::object());
                CHECK(restored.Time() == domain.Time());
                REQUIRE(domain.Advance(1.5e-4).accepted);
                REQUIRE(restored.Advance(1.5e-4).accepted);
                const auto a = domain.Dofs().CoupledField().Owned().To(R::Device::Host());
                const auto b = restored.Dofs().CoupledField().Owned().To(R::Device::Host());
                const auto ar = a.Read(), br = b.Read();
                for (I iCell = 0; iCell < a.RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        CHECK(ar.View()(iCell, j) == doctest::Approx(br.View()(iCell, j)).epsilon(2e-12).scale(1));
                MPI_Barrier(mpi.comm);
                if (mpi.rank == 0)
                    CHECK(std::filesystem::remove(checkpoint));
                MPI_Barrier(mpi.comm);
            }
        }
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
