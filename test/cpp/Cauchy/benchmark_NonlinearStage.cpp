// E6 actual bounded coupled-equation stage, no physical time advancement.
// Sources and Jacobians use the existing reconstructed quadrature producers.
#include "RHSBenchmarkFixture.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include <chrono>
#include <iostream>

namespace
{
    namespace C = DNDS::Cauchy;
    namespace R = DNDS::Runtime;
    using I = DNDS::index;
    using J = nlohmann::ordered_json;
    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 6, "arguments: width flow5|split-mixture assembled|evaluated repetitions trials");
        const I width = std::stoll(argv[1]);
        const std::string physics = argv[2], storage = argv[3];
        const int repetitions = std::stoi(argv[4]), trials = std::stoi(argv[5]);
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        DNDS_check_throw_info(width >= 4 && width <= 400 && width * width >= mpi.size &&
                                  (physics == "flow5" || physics == "split-mixture") && (storage == "assembled" || storage == "evaluated") &&
                                  repetitions > 0 && repetitions <= 5 && trials > 0 && trials <= 4,
                              "nonlinear assessment outside declared workload bounds");
        Plan53::RHSFixture fixture(mpi, width, physics == "split-mixture", "none");
        auto settings = fixture.settings;
        settings.implicit.linear.method = "sgs";
        settings.implicit.linear.iterations = 5;
        settings.implicit.linear.relativeTolerance = settings.implicit.linear.absoluteTolerance = 0;
        settings.implicit.relativeTolerance = settings.implicit.absoluteTolerance = 0;
        settings.implicit.diagonal = "entries";
        settings.implicit.jacobianAction = storage == "assembled" ? "assembled" : "semiMatrixFree";
        auto state = C::EquationDofOperations::NewVector(fixture.state);
        auto source = C::EquationDofOperations::NewVector(fixture.state);
        C::EquationDofOperations::Copy(state, fixture.state);
        R::EagerEngine engine(0);
        const auto stream = engine.CreateCPUStream();
        auto work = fixture.fv.BuildWorkspace(state, fixture.physical.workers);
        auto reconstruction = fixture.fv.BuildReconstructionState(state);
        C::NonlinearSolver solver(fixture.fv, state, reconstruction, work, engine, stream, settings,
                                  DNDS::Algebra::ComponentSelection(state.ComponentNumber()), fixture.physical.workers);
        constexpr double mass = 1e6;
        solver.EvaluateEquation(state, source, mass, 1, 0);
        const auto sourceValues = Plan53::RHSFixture::Values(source);
        const auto reset = [&]
        {
            C::EquationDofOperations::Copy(state, fixture.state);
            fixture.fv.ResetReconstruction(*reconstruction, *work);
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < fixture.fv.Metrics().NumCell(); ++iCell)
                flow(iCell, 4) += .0001 * flow(iCell, 0);
        };
        const auto solve = [&](C::NonlinearProfile &profile)
        {
            const auto report = solver.Solve(state, source, mass, 1, 0, 0, {1, 0, std::ref(profile)});
            DNDS_check_throw_info(!report.converged && report.iterations == 1 && report.linearIterations == 5 &&
                                      profile.residualEvaluations == 2 && report.initialResidual[4] > 0 && report.residual[4] < report.initialResidual[4],
                                  "nonlinear assessment changed work or failed to reduce its energy defect");
            return report;
        };
        reset();
        C::NonlinearProfile warm;
        solve(warm);
        const auto expected = Plan53::RHSFixture::Values(state);
        int provided = 0;
        MPI_Query_thread(&provided);
        for (int trial = 0; trial < trials; ++trial)
        {
            std::array<double, C::NonlinearProfile::PhaseNumber> phases{};
            double wall = 0, energyRatio = 0;
            for (int repetition = 0; repetition < repetitions; ++repetition)
            {
                reset();
                MPI_Barrier(mpi.comm);
                C::NonlinearProfile profile;
                const auto start = std::chrono::steady_clock::now();
                const auto report = solve(profile);
                wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
                double sum = 0;
                for (int p = 0; p < C::NonlinearProfile::Total; ++p)
                    sum += profile.seconds[p];
                DNDS_check_throw_info(sum <= profile.seconds[C::NonlinearProfile::Total], "nonlinear profile phases overlap");
                for (int p = 0; p < C::NonlinearProfile::PhaseNumber; ++p)
                    phases[p] += profile.seconds[p] / repetitions;
                energyRatio = std::max(energyRatio, report.residual[4] / report.initialResidual[4]);
                DNDS_check_throw_info(Plan53::RHSFixture::Values(state) == expected && Plan53::RHSFixture::Values(source) == sourceValues,
                                      "reset nonlinear trial differs or immutable source was modified");
                fixture.CheckFrozen();
            }
            MPI_Allreduce(MPI_IN_PLACE, phases.data(), int(phases.size()), MPI_DOUBLE, MPI_MAX, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, &wall, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            J values;
            const char *names[]{"entry", "residual", "convergence", "pseudo_time_pp", "jacobian", "linear_solve", "correction", "total"};
            for (int p = 0; p < C::NonlinearProfile::PhaseNumber; ++p)
                values[names[p]] = phases[p];
            if (mpi.rank == 0)
                std::cout << "PLAN53_NONLINEAR " << J{{"np", mpi.size}, {"width", width}, {"physics", physics}, {"storage", storage}, {"trial", trial}, {"repetitions", repetitions}, {"provided", provided}, {"degree", 3}, {"mass", mass}, {"fields", state.PhysicalFieldNumber()}, {"components", state.ComponentNumber()}, {"wall_seconds", wall}, {"phases_max_seconds", values}, {"iterations_per_solve", 1}, {"linear_iterations_per_solve", 5}, {"residual_evaluations_per_solve", 2}, {"energy_ratio_rank0", energyRatio}, {"repeat_full_field_equal", true}, {"source_unchanged", true}, {"reference_unchanged", true}}.dump() << std::endl;
        }
        // Independent storage-mode complement after timing: same reset state,
        // source, exact work and full fields, rather than residual-norm-only.
        settings.implicit.jacobianAction = storage == "assembled" ? "semiMatrixFree" : "assembled";
        auto alternate = C::EquationDofOperations::NewVector(fixture.state);
        C::EquationDofOperations::Copy(alternate, fixture.state);
        {
            auto write = alternate.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < fixture.fv.Metrics().NumCell(); ++iCell)
                flow(iCell, 4) += .0001 * flow(iCell, 0);
        }
        auto alternativeWork = fixture.fv.BuildWorkspace(alternate, fixture.physical.workers);
        auto alternativeReconstruction = fixture.fv.BuildReconstructionState(alternate);
        C::NonlinearSolver other(fixture.fv, alternate, alternativeReconstruction, alternativeWork, engine, stream, settings,
                                 DNDS::Algebra::ComponentSelection(alternate.ComponentNumber()), fixture.physical.workers);
        C::NonlinearProfile unused;
        const auto report = other.Solve(alternate, source, mass, 1, 0, 0, {1, 0, std::ref(unused)});
        const auto actual = Plan53::RHSFixture::Values(alternate);
        DNDS_check_throw_info(actual.size() == expected.size(), "storage-mode oracle field sizes differ");
        bool valid = report.iterations == 1 && report.linearIterations == 5;
        double error = 0;
        for (I i = 0; i < I(actual.size()); ++i)
        {
            error = std::max(error, std::abs(actual[i] - expected[i]));
            valid = valid && std::abs(actual[i] - expected[i]) <= 2e-12 * std::max({1., std::abs(actual[i]), std::abs(expected[i])});
        }
        int all = valid;
        MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(all, "assembled/evaluated nonlinear complete-field oracle failed");
        if (mpi.rank == 0)
            std::cout << "PLAN53_NONLINEAR_VERIFIED " << J{{"full_field", true}, {"max_absolute", error}}.dump() << std::endl;
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
        std::cerr << "nonlinear assessment: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
