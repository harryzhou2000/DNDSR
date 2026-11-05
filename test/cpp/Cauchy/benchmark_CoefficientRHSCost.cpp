// E3 observes the real composed flow5 RHS using a link-only wrapper. Production
// CFV has no profiling branch/timer. GNU-compatible --wrap is required here.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CartesianFixture.hpp"
#include <chrono>
#include <iostream>

namespace C = DNDS::Cauchy;
namespace F = DNDS::CFV;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;

namespace PublicationProbe
{
    thread_local bool active = false;
    thread_local double seconds = 0;
    thread_local I calls = 0, logicalBytes = 0;
}
extern "C" void RealCommit(const F::VariationalReconstructionPlan *, F::ReconstructionWorkspace &) asm("__real__ZNK4DNDS3CFV29VariationalReconstructionPlan20CommitReconstructionERNS0_23ReconstructionWorkspaceE");
extern "C" void WrappedCommit(const F::VariationalReconstructionPlan *, F::ReconstructionWorkspace &) asm("__wrap__ZNK4DNDS3CFV29VariationalReconstructionPlan20CommitReconstructionERNS0_23ReconstructionWorkspaceE");
extern "C" void WrappedCommit(const F::VariationalReconstructionPlan *plan, F::ReconstructionWorkspace &work)
{
    if (!PublicationProbe::active)
    {
        RealCommit(plan, work);
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    RealCommit(plan, work);
    PublicationProbe::seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++PublicationProbe::calls;
    PublicationProbe::logicalBytes += 16 * work.coefficients.Owned().RowCount() * work.coefficients.Width();
}

namespace
{
    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 5, "arguments: grid-width none|WBAP repetitions trials");
        const I width = std::stoll(argv[1]);
        const std::string limiter = argv[2];
        const int repetitions = std::stoi(argv[3]), trials = std::stoi(argv[4]);
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        DNDS_check_throw_info(width >= 4 && width <= 400 && width * width >= mpi.size &&
                                  (limiter == "none" || limiter == "WBAP") &&
                                  repetitions >= 1 && repetitions <= 20 && trials >= 1 && trials <= 4,
                              "coefficient RHS assessment outside declared bounds");
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .001;
        settings.execution.overlap = true;
        settings.implicit.collectiveChecks = false;
        settings.numerics.reconstruction = "variational";
        settings.numerics.reconstructionIterations = 5;
        settings.numerics.reconstructionTolerance = 0;
        settings.numerics.limiter = limiter;
        settings.numerics.positivity = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.cfv.SORInstead = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto mean = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
        {
            auto write = mean.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto state = equation.Gas().Conservative({1 + .01 * std::sin(6.28 * point[0]), .2, -.1, .05, 1});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = state[j];
            }
        }
        const auto generation = mean.Owned().Version();
        auto workspace = fv.BuildWorkspace(mean);
        R::EagerEngine engine(0);
        const auto stream = engine.CreateCPUStream();
        const auto evaluate = [&]
        { fv.Evaluate(mean, residual, *workspace, engine, stream); };
        for (int warm = 0; warm < 5; ++warm)
            evaluate();
        const I expected = 5 + (limiter == "WBAP" ? 3 : 0);
        int provided = 0;
        MPI_Query_thread(&provided);
        for (int trial = 0; trial < trials; ++trial)
            for (int phase = 0; phase < 2; ++phase)
            {
                const bool instrumented = (trial % 2 == 0) == (phase == 0);
                PublicationProbe::calls = PublicationProbe::logicalBytes = 0;
                PublicationProbe::seconds = 0;
                MPI_Barrier(mpi.comm);
                PublicationProbe::active = instrumented;
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < repetitions; ++iteration)
                    evaluate();
                const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                PublicationProbe::active = false;
                DNDS_check_throw_info(mean.Owned().Version() == generation, "composed RHS mutated means");
                if (instrumented)
                    DNDS_check_throw_info(PublicationProbe::calls == expected * repetitions &&
                                              PublicationProbe::logicalBytes == expected * repetitions * 16 * fv.Metrics().NumCell() * 45,
                                          "actual production publication count/bytes differ");
                {
                    auto read = residual.Owned().Read();
                    const auto values = read.View<R::UniformTableRows>();
                    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                        for (int j = 0; j < 5; ++j)
                            DNDS_check_throw_info(std::isfinite(values(iCell, j)), "nonfinite composed RHS");
                }
                double maximum[]{total / repetitions, PublicationProbe::seconds / repetitions, PublicationProbe::seconds / total};
                double minimum = maximum[2];
                MPI_Allreduce(MPI_IN_PLACE, maximum, 3, MPI_DOUBLE, MPI_MAX, mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
                I count[]{PublicationProbe::calls, PublicationProbe::logicalBytes};
                MPI_Allreduce(MPI_IN_PLACE, count, 2, MPI_INT64_T, MPI_MAX, mpi.comm);
                if (mpi.rank == 0)
                    std::cout << "PLAN53_COPY_RHS " << J{{"np", mpi.size}, {"width", width}, {"degree", 3}, {"limiter", limiter}, {"trial", trial}, {"instrumented", instrumented}, {"repetitions", repetitions}, {"provided", provided}, {"rhs_seconds", maximum[0]}, {"publication_seconds", maximum[1]}, {"local_fraction_min", minimum}, {"local_fraction_max", maximum[2]}, {"publication_calls_max", count[0]}, {"publication_logical_bytes_max", count[1]}, {"finite", true}, {"mean_generation_unchanged", true}}.dump() << std::endl;
            }
    }
}

int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    try
    {
        Run(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "coefficient RHS assessment: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
