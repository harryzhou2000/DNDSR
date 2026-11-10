// Isolate the general equation adapter using the same physical flow5 arrays,
// perfect-gas model and composed constant-FV operation. Both paths include
// their normal mean/coefficient halos. No nonlinear solve, PP, VR or sources.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "CartesianFixture.hpp"
#include <iostream>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(argc == 5, "arguments: grid-width viscous(0|1) repetitions trials");
        const I width = std::stoll(argv[1]);
        const bool viscous = std::stoi(argv[2]);
        const int repetitions = std::stoi(argv[3]), trials = std::stoi(argv[4]);
        DNDS_check_throw_info(repetitions > 0 && trials > 0, "invalid adapter comparison controls");
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        settings.numerics.cfv.intOrder = 5;
        settings.numerics.firstOrderConvection = true;
        settings.numerics.flux = "Roe_M1";
        settings.numerics.faceFallback = settings.numerics.positivity = false;
        settings.numerics.riemann.useStencil = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.equation.transport = viscous ? "NewtonianFourier" : "inviscid";
        settings.equation.viscosity = viscous ? .001 : 0;
        settings.execution.checkAccess = settings.execution.overlap = false;
        settings.implicit.collectiveChecks = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5), physicalOutput = fv.Metrics().BuildField(5), equationOutput = fv.Metrics().BuildField(5);
        {
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = state.Owned().Write();
            const auto values = write.View<R::FixedRows<5>>();
            const double pi = std::acos(-1.);
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto p = metric.GetCellBary(iCell);
                const auto mean = equation.Gas().Conservative({1 + .03 * std::sin(2 * pi * p[0]), .2 + .02 * std::cos(2 * pi * p[1]), -.1, .03, 1.1});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = mean[j];
            }
        }
        const auto version = state.Owned().Version();
        auto logical = C::EquationDofs::Coupled(equation.Schema(), state);
        auto physicalLogicalOutput = C::EquationDofs::Coupled(equation.Schema(), physicalOutput);
        auto logicalOutput = C::EquationDofs::Coupled(equation.Schema(), equationOutput);
        auto physicalWork = fv.BuildWorkspace(
            logical, std::nullopt,
            C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
        auto equationWork = fv.BuildWorkspace(logical);
        auto physicalReconstruction = fv.BuildReconstructionState(logical);
        auto equationReconstruction = fv.BuildReconstructionState(logical);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        const auto physical = [&]
        { fv.Evaluate(logical, physicalLogicalOutput, *physicalReconstruction, *physicalWork, engine, stream); };
        const auto general = [&]
        { fv.Evaluate(logical, logicalOutput, *equationReconstruction, *equationWork, engine, stream); };
        for (int warmup = 0; warmup < 3; ++warmup)
            physical(), general();
        double error = 0, magnitude = 0;
        {
            const auto aRead = physicalOutput.Owned().Read(), bRead = equationOutput.Owned().Read();
            const auto a = aRead.View<R::FixedRows<5>>(), b = bRead.View<R::FixedRows<5>>();
            for (I iCell = 0; iCell < a.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    error = std::max(error, std::abs(a(iCell, j) - b(iCell, j)));
                    magnitude = std::max(magnitude, std::abs(a(iCell, j)));
                }
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &magnitude, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::isfinite(error) && error < 1e-11 * std::max(1., magnitude), "physical/equation spatial results differ; do not time unmatched operations");
        const auto measure = [&](auto &&run)
        {
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int repetition = 0; repetition < repetitions; ++repetition)
                run();
            double elapsed = (MPI_Wtime() - start) / repetitions;
            MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return elapsed;
        };
        for (int trial = 0; trial < trials; ++trial)
        {
            double physicalSeconds, equationSeconds;
            if (trial % 2)
                equationSeconds = measure(general), physicalSeconds = measure(physical);
            else
                physicalSeconds = measure(physical), equationSeconds = measure(general);
            DNDS_check_throw_info(state.Owned().Version() == version, "spatial component changed owned means");
            if (mpi.rank == 0)
                std::cout << "EQUATION_ADAPTER " << C::Json{{"cells", width * width}, {"np", mpi.size}, {"viscous", viscous}, {"repetitions", repetitions}, {"trial", trial}, {"physical_seconds", physicalSeconds}, {"equation_seconds", equationSeconds}, {"max_error", error}, {"rhs_magnitude", magnitude}, {"includes_halo", true}, {"includes_nonlinear", false}, {"reconstruction", "constant"}, {"owned_generation_preserved", true}}.dump() << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    return DNDS::MPI::Finalize();
}
