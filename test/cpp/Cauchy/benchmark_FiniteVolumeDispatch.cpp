// Compare the forced generic and automatically resolved fixed-flow5 RHS leaves
// through the same unified EquationDofs finite-volume schedule. Both paths use
// the same perfect-gas arrays and constant-FV operation, including their normal
// mean/coefficient halos. No nonlinear solve, PP, VR or sources.
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
        DNDS_check_throw_info(repetitions > 0 && trials > 0, "invalid finite-volume dispatch comparison controls");
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
        auto state = fv.Metrics().BuildField(5), genericOutput = fv.Metrics().BuildField(5), fixedFlow5Output = fv.Metrics().BuildField(5);
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
        auto genericLogicalOutput = C::EquationDofs::Coupled(equation.Schema(), genericOutput);
        auto fixedFlow5LogicalOutput = C::EquationDofs::Coupled(equation.Schema(), fixedFlow5Output);
        auto genericWork = fv.BuildWorkspace(
            logical, std::nullopt,
            C::FiniteVolume::WorkspaceOptions{C::FiniteVolume::RHSKernelPolicy::Generic});
        auto fixedFlow5Work = fv.BuildWorkspace(logical);
        auto genericReconstruction = fv.BuildReconstructionState(logical);
        auto fixedFlow5Reconstruction = fv.BuildReconstructionState(logical);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        const auto generic = [&]
        { fv.Evaluate(logical, genericLogicalOutput, *genericReconstruction, *genericWork, engine, stream); };
        const auto fixedFlow5 = [&]
        { fv.Evaluate(logical, fixedFlow5LogicalOutput, *fixedFlow5Reconstruction, *fixedFlow5Work, engine, stream); };
        for (int warmup = 0; warmup < 3; ++warmup)
            generic(), fixedFlow5();
        double error = 0, magnitude = 0;
        {
            const auto aRead = genericOutput.Owned().Read(), bRead = fixedFlow5Output.Owned().Read();
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
        DNDS_check_throw_info(std::isfinite(error) && error < 1e-11 * std::max(1., magnitude),
                              "generic/fixed-flow5 spatial results differ; do not time unmatched operations");
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
            double genericSeconds, fixedFlow5Seconds;
            if (trial % 2)
                fixedFlow5Seconds = measure(fixedFlow5), genericSeconds = measure(generic);
            else
                genericSeconds = measure(generic), fixedFlow5Seconds = measure(fixedFlow5);
            DNDS_check_throw_info(state.Owned().Version() == version, "spatial component changed owned means");
            if (mpi.rank == 0)
                std::cout << "FINITE_VOLUME_DISPATCH " << C::Json{{"cells", width * width}, {"np", mpi.size}, {"viscous", viscous}, {"repetitions", repetitions}, {"trial", trial}, {"generic_seconds", genericSeconds}, {"fixed_flow5_seconds", fixedFlow5Seconds}, {"max_error", error}, {"rhs_magnitude", magnitude}, {"includes_halo", true}, {"includes_nonlinear", false}, {"reconstruction", "constant"}, {"owned_generation_preserved", true}}.dump() << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    return DNDS::MPI::Finalize();
}
