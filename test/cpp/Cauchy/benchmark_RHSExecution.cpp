#include "RHSBenchmarkFixture.hpp"
#include "RHSExecutionBenchmark.hpp"
#include <chrono>
#include <iostream>

namespace
{
    using J = nlohmann::ordered_json;
    using I = DNDS::index;
    namespace R = DNDS::Runtime;
    namespace C = DNDS::Cauchy;
    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 8, "arguments: width flow5|split-mixture none|WBAP production|caller|compute-worker|mpi-worker direct|packed repetitions trials");
        const I width = std::stoll(argv[1]);
        const std::string profile = argv[2], limiter = argv[3], mode = argv[4], transport = argv[5];
        const int repetitions = std::stoi(argv[6]), trials = std::stoi(argv[7]);
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        DNDS_check_throw_info(width >= 2 && width <= 400 &&
                                  (profile == "flow5" || profile == "split-mixture") && (limiter == "none" || limiter == "WBAP") &&
                                  repetitions > 0 && repetitions <= 5 && trials > 0 && trials <= 4,
                              "RHS execution workload outside declared bounds");
        // Declared first: all workspaces/fields/engine lanes are drained and
        // destroyed before their MPI worker and before main-thread Finalize.
        Plan53::RHSExecution execution(mpi, mode, transport);
        Plan53::RHSFixture fixture(mpi, width, profile == "split-mixture", limiter);
        auto &fv = fixture.fv;
        R::EagerEngine engine(execution.HostThreads());
        const auto stream = engine.CreateCPUStream();
        execution.PinCompute(engine, stream);

        // Reference uses the original public Direct bindings and inline CPU
        // kernels. Complete elementwise output comparison, not just a digest.
        execution.OriginalBindings(true);
        auto reference = fv.BuildEquationWorkspace(fixture.state, fixture.physical.workers);
        execution.OriginalBindings(false);
        R::EagerEngine inlineEngine(0);
        const auto inlineStream = inlineEngine.CreateCPUStream();
        auto expected = C::EquationDofOperations::NewVector(fixture.state);
        fv.ResetReconstruction(*reference);
        for (int repetition = 0; repetition < repetitions; ++repetition)
            fv.Evaluate(fixture.state, expected, *reference, inlineEngine, inlineStream);
        const auto oracle = Plan53::RHSFixture::Values(expected);
        auto work = fv.BuildEquationWorkspace(fixture.state, fixture.physical.workers);
        for (int warm = 0; warm < 2; ++warm)
            fv.Evaluate(fixture.state, fixture.residual, *work, engine, stream);
        int provided = 0;
        MPI_Query_thread(&provided);
        const auto placements = execution.Placement();
        const I core = I(fv.Metrics().CellWorkPartition().Core().size());
        const auto compare = [&]
        {
            fixture.CheckFrozen();
            const auto values = Plan53::RHSFixture::Values(fixture.residual);
            DNDS_check_throw_info(values == oracle && fv.LastReconstructionIterations(*work) == 5,
                                  "complete RHS differs from original Direct/inline fixed-work oracle");
        };
        for (int trial = 0; trial < trials; ++trial)
            for (int phase = 0; phase < 2; ++phase)
            {
                const bool instrumented = (trial % 2 == 0) == (phase == 0);
                // Both phases start from the same zero coefficients. Reset is
                // outside timing; exactly five sweeps occur inside every RHS.
                fv.ResetReconstruction(*work);
                MPI_Barrier(mpi.comm);
                execution.BeginSample(instrumented);
                const auto start = std::chrono::steady_clock::now();
                for (int repetition = 0; repetition < repetitions; ++repetition)
                    fv.Evaluate(fixture.state, fixture.residual, *work, engine, stream);
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
                auto trace = execution.EndSample();
                compare(); // Outside every numerical/transport clock.
                double maximum = elapsed;
                MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
                J maxima;
                J timeline;
                if (instrumented)
                {
                    const I fields = fixture.state.PhysicalFieldNumber();
                    const I components = fixture.state.ComponentNumber();
                    const I coefficientStarts = 6 + (limiter == "WBAP" ? 3 : 0);
                    const I logicalWidth = components * (1 + 9 * coefficientStarts);
                    const I starts = fields * (1 + coefficientStarts);
                    DNDS_check_throw_info(mode == "production" || (trace.at("starts").get<I>() == repetitions * starts &&
                                                                   trace.at("staging_calls").get<I>() == (transport == "packed" ? repetitions * 3 * starts : 0) &&
                                                                   trace.at("staging_bytes").get<I>() == (transport == "packed" ? repetitions * 8 *
                                                                                                                                      (fv.Metrics().NumCell() + 2 * (fv.Metrics().NumCellProc() - fv.Metrics().NumCell())) * logicalWidth
                                                                                                                                : 0) &&
                                                                   trace.at("receive_bytes").get<I>() == repetitions * 8 * (fv.Metrics().NumCellProc() - fv.Metrics().NumCell()) * logicalWidth),
                                          "actual transfer starts/staging/payload differ from fixed reconstruction schedule");
                    DNDS_check_throw_info(trace.at("vr_core_calls").get<I>() == repetitions * 5 * fixture.state.PhysicalFieldNumber(),
                                          "reconstruction core call count differs from declared five sweeps/fields");
                    for (auto it = trace.begin(); it != trace.end(); ++it)
                    {
                        if (it.key() == "timeline")
                        {
                            timeline = it.value();
                            continue;
                        }
                        double value = it.value().get<double>() / repetitions;
                        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
                        maxima[it.key()] = value;
                    }
                }
                if (mpi.rank == 0)
                    std::cout << "PLAN53_RHS_EXECUTION " << J{{"np", mpi.size}, {"width", width}, {"profile", profile}, {"limiter", limiter}, {"mode", mode}, {"transport", transport}, {"trial", trial}, {"instrumented", instrumented}, {"provided", provided}, {"repetitions", repetitions}, {"degree", 3}, {"components", fixture.state.ComponentNumber()}, {"fields", fixture.state.PhysicalFieldNumber()}, {"placements", placements}, {"core_rows_rank0", core}, {"rhs_seconds", maximum}, {"trace_max_per_rhs", std::move(maxima)}, {"timeline_rank0", std::move(timeline)}, {"full_field_oracle", true}, {"unchanged_means", true}}.dump() << std::endl;
            }
        engine.Synchronize();
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
        std::cerr << "RHS execution assessment: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
