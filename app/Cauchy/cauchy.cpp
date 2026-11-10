#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/IO/ProgressOutput.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "argparse.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>

namespace
{
    std::string TimeToken(DNDS::real time)
    {
        std::ostringstream stream;
        stream << std::scientific << std::setprecision(std::numeric_limits<DNDS::real>::max_digits10) << time;
        return "-T" + stream.str();
    }

    std::string StepToken(uint64_t step) { return "-S" + std::to_string(step); }

    uint64_t NextTimeOrdinal(DNDS::real time, DNDS::real interval)
    {
        if (!(interval > 0))
            return 0;
        const auto completed = std::floor(time / interval);
        DNDS_check_throw_info(completed >= 0 && completed < DNDS::real(std::numeric_limits<uint64_t>::max() - 1),
                              "Cauchy output-time ordinal overflow");
        return uint64_t(completed) + 1;
    }

    DNDS::real TimeAt(uint64_t ordinal, DNDS::real interval)
    {
        return DNDS::real(ordinal) * interval;
    }

}

/// Keep every MPI-owning solver object inside this scope, before DNDS Finalize.
static int Run(int argc, char **argv, double started)
{
    int status = 0;
    {
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        const DNDS::MPI::SolverFailureScope solverFailures(mpi.comm);
        try
        {
            // Own help handling so argparse never exits before MPI finalization.
            argparse::ArgumentParser arguments("cauchy.exe", "1.0", argparse::default_arguments::none);
            arguments.add_argument("config").default_value(std::string{}).help("Cauchy JSON input; paths are relative to the working directory");
            arguments.add_argument("-h", "--help").flag();
            arguments.add_argument("--check-config").flag().help("Print resolved configuration without constructing a mesh");
            arguments.add_argument("--restart").default_value(std::string{}).help("Restart from a Cauchy checkpoint");
            arguments.add_argument("-k", "--overwrite_key").append().default_value<std::vector<std::string>>({}).help("Resolved-config JSON pointer to override; pair with -v, may repeat");
            arguments.add_argument("-v", "--overwrite_value").append().default_value<std::vector<std::string>>({}).help("JSON value for the corresponding -k (quote JSON strings)");
            DNDS::MPI::LocalStage(mpi, [&]
                                  { arguments.parse_args(argc, argv); });
            if (arguments.get<bool>("--help"))
            {
                if (mpi.rank == 0)
                    std::cout << arguments;
                return 0;
            }
            const auto configPath = arguments.get<std::string>("config");
            const auto restartPath = arguments.get<std::string>("--restart");
            const bool checkConfig = arguments.get<bool>("--check-config");
            DNDS::Cauchy::Json input;
            std::vector<DNDS::Cauchy::ConfigOverride> overrides;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                DNDS_check_throw_info(!configPath.empty(), "provide a Cauchy config file or --help");
                DNDS_check_throw_info(!checkConfig || restartPath.empty(), "--check-config and --restart are mutually exclusive");
                std::ifstream stream(configPath);
                DNDS_check_throw_info(bool(stream),"cannot open Cauchy config");
                stream >> input;
                const auto keys = arguments.get<std::vector<std::string>>("--overwrite_key");
                const auto values = arguments.get<std::vector<std::string>>("--overwrite_value");
                DNDS_check_throw_info(keys.size() == values.size(), "overwrite keys and values must be paired");
                for (DNDS::index i = 0; i < DNDS::index(keys.size()); ++i)
                {
                    DNDS_check_throw_info(!keys[i].empty() && keys[i][0] == '/', "overwrite keys must be nonempty JSON pointers");
                    overrides.push_back({keys[i], DNDS::Cauchy::Json::parse(values[i])});
                } });
            auto config = DNDS::Cauchy::SimulationConfig::ResolveCollective(input, mpi.comm, overrides);
            if (checkConfig)
            {
                if (mpi.rank == 0)
                    std::cout << config.Root().dump(2) << '\n';
                return 0;
            }
            DNDS::Cauchy::Simulation simulation(config, mpi.comm);
            if (!restartPath.empty())
                simulation.ReadCheckpoint(restartPath);
            const auto &output = config.Output();
            const auto &time = config.Time();
            const auto domainCount = DNDS::index(config.Domains().size());
            // Actions may enter domain collectives. Propagate unexpected errors
            // to the fatal boundary; never try a control reduction after a
            // rank-local failure while its peers may still be in that action.
            const auto visitDomains = [&](const auto &action)
            {
                for (DNDS::index iDomain = 0; iDomain < domainCount; ++iDomain)
                    if (simulation.Local(iDomain))
                        action(iDomain, simulation.GetDomain(iDomain));
            };
            const auto domainPrefix = [&](DNDS::index iDomain)
            { return output.prefix + "-D" + std::to_string(iDomain); };
            auto logger = std::make_shared<DNDS::Cauchy::OperationalLogger>(output, mpi.rank);
            std::vector<std::unique_ptr<DNDS::Cauchy::ProgressOutputController>> progress(static_cast<size_t>(domainCount));
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0)
                {
                    const auto parent = std::filesystem::path(output.prefix).parent_path();
                    if (!parent.empty()) std::filesystem::create_directories(parent);
                    std::ofstream resolved(output.prefix + "-resolved.json");
                    resolved << config.Root().dump(2) << '\n';
                    DNDS_check_throw_info(bool(resolved), "cannot write resolved Cauchy configuration");
                } });
            visitDomains([&](DNDS::index iDomain, auto &domain)
                         {
                DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
                                      {
                    progress[iDomain] = std::make_unique<DNDS::Cauchy::ProgressOutputController>(
                        domain.Equation().Schema(), output, domain.Name(), domainPrefix(iDomain) + "-history",
                        simulation.Placement().DomainRank(iDomain) == 0, logger); });
                domain.SetProgressObserver([&, iDomain](const DNDS::Cauchy::NonlinearProgress &entry)
                                           { progress[iDomain]->CaptureIteration(entry); }); });
            const auto flushPending = [&]
            {
                visitDomains([&](DNDS::index iDomain, auto &domain)
                             { DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
                                                     { progress[iDomain]->FlushIterations(); }); });
            };
            std::vector<double> baselines(size_t(domainCount), 0);
            bool reached = false, wallStop = false;
            std::vector<DNDS::Cauchy::PhysicalStepLimit> limits(static_cast<size_t>(domainCount));
            const auto record = [&](DNDS::real dt, int retries, bool useLimits = false, bool printConsole = true, bool forceHistory = false)
            {
                int allReached = 1;
                visitDomains([&](DNDS::index iDomain, auto &domain)
                             {
                const auto values = domain.Diagnostics(config.Domains()[iDomain].settings, output.forceZones);
                auto &baseline = baselines[iDomain];
                if (baseline == 0) baseline = std::max(values[4], DNDS::real(1e-300));
                const auto limit = useLimits ? limits[iDomain] : DNDS::Cauchy::PhysicalStepLimit{};
                const auto fallbacks = domain.LastFaceFallbacks();
                DNDS::index nonlinearIterations = 0, linearIterations = 0;
                for (const auto &report : domain.LastImplicitStages())
                {
                    nonlinearIterations += report.iterations;
                    linearIterations += report.linearIterations;
                }
                // File rotation/open failures on the writer rank must not
                // strand peers entering the next solver collective.
                DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
                                      {
                const DNDS::Cauchy::ProgressLocation location{domain.Epoch().acceptedStep, -1, -1, -1, DNDS::index(retries),
                                                               domain.Time(), dt, MPI_Wtime() - started};
                for (DNDS::index iStage = 0; iStage < DNDS::index(domain.LastImplicitStages().size()); ++iStage)
                {
                    auto stage = location;
                    stage.stage = iStage;
                    progress[iDomain]->StageEnd(stage, domain.LastImplicitStages()[iStage]);
                }
                DNDS::Cauchy::AcceptedStepProgress accepted;
                accepted.diagnostics = values;
                accepted.faceFallbacks = fallbacks;
                accepted.retries = retries;
                accepted.implicitStages = DNDS::index(domain.LastImplicitStages().size());
                accepted.nonlinearIterations = nonlinearIterations;
                accepted.linearIterations = linearIterations;
                accepted.coupledIterations = domain.LastCoupledIterations();
                accepted.coupledResidualTargetRatio = domain.LastCoupledTargetRatio();
                accepted.positivity = domain.LastPositivity();
                accepted.timePPLimitedCells = limit.limitedCells;
                accepted.timePPMinimumAlpha = limit.minimumAlpha;
                accepted.timeGrowthLimited = limit.growthLimited;
                progress[iDomain]->StepEnd(location, accepted, forceHistory, printConsole);
                } );
                const bool outerTarget = time.steadyTolerance <= 0 || values[4] / baseline <= time.steadyTolerance;
                if (time.steady)
                    allReached &= !domain.LastImplicitStages().empty() && std::all_of(domain.LastImplicitStages().begin(), domain.LastImplicitStages().end(), [](const auto &stage)
                                                                               { return stage.converged; }) &&
                           outerTarget;
                else
                    allReached &= time.steadyTolerance > 0 && domain.Epoch().acceptedStep >= uint64_t(time.steadyMinSteps) && outerTarget; });
                MPI_Allreduce(MPI_IN_PLACE, &allReached, 1, MPI_INT, MPI_MIN, mpi.comm);
                return bool(allReached);
            };
            // The initial state remains in the CSV history, but console
            // iteration labels begin with the first completed physical step.
            record(0, 0, false, false, true);
            const DNDS::index visualizationInterval = output.visualizationInterval < 0 ? output.checkpointInterval : output.visualizationInterval;
            const DNDS::real visualizationTimeInterval = output.visualizationTimeInterval < 0 ? output.checkpointTimeInterval : output.visualizationTimeInterval;
            uint64_t nextCheckpointTime = NextTimeOrdinal(simulation.Time(), output.checkpointTimeInterval);
            uint64_t nextVisualizationTime = NextTimeOrdinal(simulation.Time(), visualizationTimeInterval);
            const auto writeCheckpoint = [&](const std::string &token)
            {
                simulation.WriteCheckpoint(output.prefix + (domainCount == 1 ? "-D0" : "") + token + ".h5",
                                           output.checkpointReconstruction);
            };
            const auto writeVisualization = [&](const std::string &token)
            {
                visitDomains([&](DNDS::index iDomain, auto &domain)
                             { domain.WriteVisualization(domainPrefix(iDomain) + token, output.hdfDeflateLevel); });
            };
            for (DNDS::index step = 0; time.tEnd > 0 ? !DNDS::Cauchy::EquationTimeStepper::ReachedTimeTarget(simulation.Time(), time.tEnd) : step < time.steps; ++step)
            {
                int stop = time.wallSeconds > 0 && MPI_Wtime() - started >= time.wallSeconds;
                MPI_Allreduce(MPI_IN_PLACE, &stop, 1, MPI_INT, MPI_MAX, mpi.comm);
                if (stop)
                {
                    wallStop = true;
                    break;
                }
                DNDS::real dt = time.dt;
                if (time.cfl > 0)
                    visitDomains([&](DNDS::index, auto &domain)
                                 { dt = std::min(dt, domain.StableStep(time.cfl)); });
                MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
                if (time.positivity.enabled)
                {
                    visitDomains([&](DNDS::index iDomain, auto &domain)
                                 { limits[iDomain] = domain.LimitPhysicalStep(dt); });
                    for (DNDS::index iDomain = 0; iDomain < domainCount; ++iDomain)
                        if (simulation.Local(iDomain))
                            dt = std::min(dt, limits[iDomain].dt);
                    MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
                }
                std::optional<DNDS::real> exactTarget;
                if (!time.steady)
                {
                    DNDS::real target = std::numeric_limits<DNDS::real>::infinity();
                    if (time.tEnd > 0)
                        target = std::min(target, time.tEnd);
                    if (output.checkpointTimeInterval > 0)
                        target = std::min(target, TimeAt(nextCheckpointTime, output.checkpointTimeInterval));
                    if (visualizationTimeInterval > 0)
                        target = std::min(target, TimeAt(nextVisualizationTime, visualizationTimeInterval));
                    const DNDS::real remaining = target - simulation.Time();
                    const DNDS::real tolerance = output.endpointSnapRelativeTolerance * dt;
                    if (std::isfinite(target) && remaining > 0 && remaining <= dt + tolerance)
                    {
                        dt = remaining;
                        exactTarget = target;
                    }
                }
                int retries = 0;
                for (;;)
                {
                    visitDomains([&](DNDS::index iDomain, auto &)
                                 { progress[iDomain]->BeginAttempt({simulation.Step() + 1, -1, -1, -1, DNDS::index(retries),
                                                                    time.steady ? simulation.Time() : exactTarget.value_or(simulation.Time() + dt),
                                                                    dt, MPI_Wtime() - started}); });
                    const auto result = simulation.Advance(dt, exactTarget);
                    flushPending();
                    if (result)
                        break;
                    if (retries++ >= time.maxRetries)
                        DNDS::MPI::Abort(mpi.comm, "Cauchy timestep retries exhausted: " + result.reason);
                    logger->Warn("Retry candidate with half dt: " + result.reason);
                    dt *= .5;
                    exactTarget.reset();
                }
                const bool printStep = output.consoleInterval > 0 && simulation.Step() % uint64_t(output.consoleInterval) == 0;
                reached = record(time.steady ? 0 : dt, retries, time.positivity.enabled, printStep) || reached;
                bool checkpointTimeDue = false, visualizationTimeDue = false;
                if (!time.steady && output.checkpointTimeInterval > 0)
                    while (TimeAt(nextCheckpointTime, output.checkpointTimeInterval) <= simulation.Time())
                    {
                        checkpointTimeDue = true;
                        ++nextCheckpointTime;
                    }
                if (!time.steady && visualizationTimeInterval > 0)
                    while (TimeAt(nextVisualizationTime, visualizationTimeInterval) <= simulation.Time())
                    {
                        visualizationTimeDue = true;
                        ++nextVisualizationTime;
                    }
                const bool checkpointStepDue = output.checkpointInterval > 0 && simulation.Step() % uint64_t(output.checkpointInterval) == 0;
                const bool visualizationStepDue = visualizationInterval > 0 && simulation.Step() % uint64_t(visualizationInterval) == 0;
                if (checkpointTimeDue || checkpointStepDue)
                    writeCheckpoint(checkpointTimeDue ? TimeToken(simulation.Time()) : StepToken(simulation.Step()));
                if (visualizationTimeDue || visualizationStepDue)
                    writeVisualization(visualizationTimeDue ? TimeToken(simulation.Time()) : StepToken(simulation.Step()));
                if (reached)
                    break;
            }
            writeCheckpoint("-final");
            writeVisualization("-final");
            visitDomains([&](DNDS::index iDomain, auto &)
                         { progress[iDomain]->Flush(); });
            logger->Flush();
            const bool timeReached = time.tEnd <= 0 || DNDS::Cauchy::EquationTimeStepper::ReachedTimeTarget(simulation.Time(), time.tEnd);
            const bool stepStop = !wallStop && ((!timeReached && time.tEnd > 0) || ((time.steady || time.steadyTolerance > 0) && !reached));
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
            if (mpi.rank == 0)
            {
                DNDS::Cauchy::Json result{{"time", simulation.Time()}, {"steps", simulation.Step()}, {"wallSeconds", MPI_Wtime() - started}, {"status", wallStop ? "wall-capped" : stepStop ? "step-capped"
                                                                                                                                                                               : reached    ? "steady-converged"
                                                                                                                                                                                            : "target-reached"},
                                          {"steadyTargetMet", reached},
                                          {"steadyMode", time.steady},
                                          {"physicalTimeTarget", time.tEnd},
                                          {"steadyTolerance", time.steadyTolerance}};
                std::ofstream summary(output.prefix + "-summary.json");
                summary << result.dump(2) << '\n';
                DNDS_check_throw_info(bool(summary), "cannot write Cauchy summary");
            } });
            if (wallStop || stepStop)
                status = 2;
        }
        catch (const std::exception &exception)
        {
            // Throws are allowed; coordinated tests may catch and recover.
            // This production try includes collectively ordered MPI work,
            // including point-to-point exchanges. A catch can report locally
            // and abort only: no retry, drain, error agreement or finalization.
            DNDS::MPI::Abort(mpi.comm, "Cauchy solver failed", exception.what());
        }
        catch (...)
        {
            DNDS::MPI::Abort(mpi.comm, "Cauchy: unknown fatal exception");
        }
        MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, mpi.comm);
    }
    return status;
}

int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
    {
        // Initialization failed: no MPI clock, solver construction or finalize.
        std::cerr << "Cauchy: MPI initialization failed\n";
        return 1;
    }
    // Include import/partition/setup in the external wall-time budget.
    const int status = Run(argc, argv, MPI_Wtime());
    DNDS::MPI::Finalize();
    return status;
}
