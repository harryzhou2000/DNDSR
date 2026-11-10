#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/StepResult.hpp"
#include "Cauchy/IO/OutputPlan.hpp"
#include "Cauchy/IO/ProgressOutput.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "argparse.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

namespace
{
    DNDS::Cauchy::Json GatherJsonArrays(const DNDS::MPIInfo &mpi, const DNDS::Cauchy::Json &local)
    {
        const auto encoded = local.dump();
        DNDS_check_throw_info(encoded.size() <= size_t(INT_MAX), "Cauchy manifest contribution is too large");
        const int count = int(encoded.size());
        std::vector<int> counts(mpi.rank == 0 ? size_t(mpi.size) : 0);
        MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, mpi.comm);
        std::vector<int> offsets;
        std::vector<char> bytes;
        if (mpi.rank == 0)
        {
            offsets.resize(size_t(mpi.size));
            int total = 0;
            for (int rank = 0; rank < mpi.size; ++rank)
            {
                offsets[rank] = total;
                DNDS_check_throw_info(counts[rank] <= INT_MAX - total, "Cauchy manifest gather is too large");
                total += counts[rank];
            }
            bytes.resize(size_t(total));
        }
        MPI_Gatherv(encoded.data(), count, MPI_CHAR, bytes.data(), counts.data(), offsets.data(), MPI_CHAR, 0, mpi.comm);
        DNDS::Cauchy::Json result = DNDS::Cauchy::Json::array();
        if (mpi.rank == 0)
            for (int rank = 0; rank < mpi.size; ++rank)
            {
                const auto contribution = DNDS::Cauchy::Json::parse(bytes.data() + offsets[rank],
                                                                    bytes.data() + offsets[rank] + counts[rank]);
                for (const auto &entry : contribution)
                    result.push_back(entry);
            }
        return result;
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
            const DNDS::Cauchy::OutputPlan outputPlan(output, domainCount);
            const auto &restoredOutput = simulation.RestoredOutputController();
            const bool continueOutput = restoredOutput && restoredOutput->is_object() &&
                                        restoredOutput->value("prefix", std::string{}) == output.prefix;
            auto outputState = restoredOutput ? outputPlan.RestoreState(*restoredOutput, simulation.Time())
                                              : outputPlan.InitialState(simulation.Time());
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
            { return outputPlan.DomainPrefix(iDomain); };
            auto logger = std::make_shared<DNDS::Cauchy::OperationalLogger>(output, mpi.rank);
            std::vector<std::unique_ptr<DNDS::Cauchy::ProgressOutputController>> progress(static_cast<size_t>(domainCount));
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0)
                {
                    const auto parent = std::filesystem::path(output.prefix).parent_path();
                    if (!parent.empty()) std::filesystem::create_directories(parent);
                    std::ofstream resolved(outputPlan.ResolvedPath());
                    resolved << config.Root().dump(2) << '\n';
                    DNDS_check_throw_info(bool(resolved), "cannot write resolved Cauchy configuration");
                } });
            visitDomains([&](DNDS::index iDomain, auto &domain)
                         {
                DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
                                      {
                    progress[iDomain] = std::make_unique<DNDS::Cauchy::ProgressOutputController>(
                        domain.Equation().Schema(), output, domain.Name(), outputPlan.HistoryBare(iDomain),
                        simulation.Placement().DomainRank(iDomain) == 0, logger,
                        continueOutput ? std::optional(outputState.history.at(size_t(iDomain))) : std::nullopt); });
                domain.SetProgressObserver([&, iDomain](const DNDS::Cauchy::NonlinearProgress &entry)
                                           { progress[iDomain]->CaptureIteration(entry); }); });
            DNDS::Cauchy::Json localDomains = DNDS::Cauchy::Json::array();
            for (DNDS::index iDomain = 0; iDomain < domainCount; ++iDomain)
                if (simulation.Local(iDomain) && simulation.Placement().DomainRank(iDomain) == 0)
                {
                    const auto &domain = simulation.GetDomain(iDomain);
                    localDomains.push_back({{"index", iDomain}, {"name", domain.Name()}, {"history", outputPlan.HistoryBare(iDomain) + ".csv"}, {"components", DNDS::Cauchy::EquationComponentNames(domain.Equation().Schema())}});
                }
            const auto manifestDomains = GatherJsonArrays(mpi, localDomains);
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0)
                {
                    const DNDS::Cauchy::Json manifest{
                        {"schema", "dndsr.cauchy.run-output.v1"},
                        {"resolvedConfig", outputPlan.ResolvedPath()},
                        {"historySchema", "dndsr.cauchy.detailed-history.v1"},
                        {"progressStatus", {{"nonlinearIteration", 0}, {"implicitStageEnd", 1}, {"acceptedStepEnd", 2}}},
                        {"domains", manifestDomains}};
                    std::ofstream file(outputPlan.RunManifestPath());
                    file << manifest.dump(2) << '\n';
                    DNDS_check_throw_info(bool(file), "cannot write Cauchy run manifest");
                } });
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
            if (!continueOutput)
                record(0, 0, false, false, true);
            const DNDS::index visualizationInterval = outputPlan.VisualizationInterval();
            const DNDS::real visualizationTimeInterval = outputPlan.VisualizationTimeInterval();
            uint64_t nextCheckpointTime = outputState.nextCheckpointTime;
            uint64_t nextVisualizationTime = outputState.nextVisualizationTime;
            const auto captureOutputState = [&]
            {
                std::vector<int64_t> positions(size_t(2 * domainCount), 0);
                std::vector<int> roots(size_t(domainCount), 0);
                visitDomains([&](DNDS::index iDomain, auto &)
                             {
                    progress[iDomain]->Flush();
                    if (simulation.Placement().DomainRank(iDomain) == 0)
                    {
                        const auto cursor = progress[iDomain]->HistoryPosition();
                        positions[size_t(2 * iDomain)] = cursor.block;
                        positions[size_t(2 * iDomain + 1)] = cursor.line;
                        roots[size_t(iDomain)] = 1;
                    } });
                MPI_Allreduce(MPI_IN_PLACE, positions.data(), int(positions.size()), MPI_INT64_T, MPI_SUM, mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, roots.data(), int(roots.size()), MPI_INT, MPI_SUM, mpi.comm);
                outputState.nextCheckpointTime = nextCheckpointTime;
                outputState.nextVisualizationTime = nextVisualizationTime;
                for (DNDS::index iDomain = 0; iDomain < domainCount; ++iDomain)
                {
                    DNDS_check_throw_info(roots[size_t(iDomain)] == 1 && positions[size_t(2 * iDomain)] >= 0 &&
                                              positions[size_t(2 * iDomain + 1)] >= 0,
                                          "Cauchy output history cursor has no unique domain root");
                    outputState.history[size_t(iDomain)] = {int(positions[size_t(2 * iDomain)]),
                                                            positions[size_t(2 * iDomain + 1)]};
                }
                return outputState.ToJson();
            };
            const auto writeCheckpoint = [&](const std::string &token)
            {
                simulation.WriteCheckpoint(outputPlan.CheckpointPath(token), output.checkpointReconstruction,
                                           captureOutputState());
            };
            const auto writeVisualization = [&](const std::string &token)
            {
                visitDomains([&](DNDS::index iDomain, auto &domain)
                             { domain.WriteVisualization(outputPlan.VisualizationPrefix(iDomain, token), output.hdfDeflateLevel); });
            };
            const auto writeSnapshot = [&](const std::string &token, bool checkpoint, bool visualization)
            {
                if (checkpoint)
                    writeCheckpoint(token);
                if (visualization)
                    writeVisualization(token);
                DNDS::MPI::LocalStage(mpi, [&]
                                      {
                    if (mpi.rank != 0)
                        return;
                    DNDS::Cauchy::Json artifacts = DNDS::Cauchy::Json::array();
                    if (checkpoint)
                        artifacts.push_back({{"kind", "checkpoint"}, {"path", outputPlan.CheckpointPath(token)},
                                             {"schema", "dndsr.cauchy.checkpoint.v3"}});
                    if (visualization)
                        for (DNDS::index iDomain = 0; iDomain < domainCount; ++iDomain)
                        {
                            const auto prefix = outputPlan.VisualizationPrefix(iDomain, token);
                            artifacts.push_back({{"kind", "volume"}, {"domain", iDomain}, {"path", prefix + ".vtkhdf"},
                                                 {"schema", "dndsr.cauchy.visualization.v1"}});
                            DNDS::Cauchy::Json shards = DNDS::Cauchy::Json::array();
                            for (int rank = 0; rank < int(config.Domains()[iDomain].placement.ranks.size()); ++rank)
                                shards.push_back(prefix + "_surface_rank" + std::to_string(rank) + ".csv");
                            artifacts.push_back({{"kind", "surface"}, {"domain", iDomain}, {"paths", std::move(shards)},
                                                 {"schema", "dndsr.cauchy.surface-samples.v1"}});
                        }
                    const DNDS::Cauchy::Json manifest{{"schema", "dndsr.cauchy.snapshot.v1"},
                                                       {"step", simulation.Step()}, {"time", simulation.Time()},
                                                       {"token", token}, {"artifacts", std::move(artifacts)}};
                    std::ofstream file(outputPlan.SnapshotManifestPath(token));
                    file << manifest.dump(2) << '\n';
                    DNDS_check_throw_info(bool(file), "cannot write Cauchy snapshot manifest"); });
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
                    const auto adjusted = outputPlan.SnapStep(simulation.Time(), dt, time.tEnd,
                                                              nextCheckpointTime, nextVisualizationTime);
                    dt = adjusted.dt;
                    exactTarget = adjusted.exactTime;
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
                    while (DNDS::Cauchy::OutputPlan::TimeAt(nextCheckpointTime, output.checkpointTimeInterval) <= simulation.Time())
                    {
                        checkpointTimeDue = true;
                        ++nextCheckpointTime;
                    }
                if (!time.steady && visualizationTimeInterval > 0)
                    while (DNDS::Cauchy::OutputPlan::TimeAt(nextVisualizationTime, visualizationTimeInterval) <= simulation.Time())
                    {
                        visualizationTimeDue = true;
                        ++nextVisualizationTime;
                    }
                const bool checkpointStepDue = output.checkpointInterval > 0 && simulation.Step() % uint64_t(output.checkpointInterval) == 0;
                const bool visualizationStepDue = visualizationInterval > 0 && simulation.Step() % uint64_t(visualizationInterval) == 0;
                const bool checkpointDue = checkpointTimeDue || checkpointStepDue;
                const bool visualizationDue = visualizationTimeDue || visualizationStepDue;
                if (checkpointDue || visualizationDue)
                {
                    const auto token = checkpointTimeDue || visualizationTimeDue
                                           ? DNDS::Cauchy::OutputPlan::TimeToken(simulation.Time())
                                           : DNDS::Cauchy::OutputPlan::StepToken(simulation.Step());
                    writeSnapshot(token, checkpointDue, visualizationDue);
                }
                if (reached)
                    break;
            }
            writeSnapshot("-final", true, true);
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
                std::ofstream summary(outputPlan.SummaryPath());
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
