#include "Cauchy/Core/Simulation.hpp"
#include "Cauchy/Core/StepRejected.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/IO/CsvLog.hpp"
#include "argparse.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

/// Keep every MPI-owning solver object inside this scope, before DNDS Finalize.
static int Run(int argc, char **argv, double started)
{
    int status = 0;
    {
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
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
            { return output.prefix + (domainCount == 1 ? "" : "_domain" + std::to_string(iDomain)); };
            const std::vector<std::string> columns{
                "step", "time", "dt", "wall", "mass", "energy", "rho_min", "p_min", "residual_l2", "force_x", "force_y", "force_z",
                "vr_sweeps", "vr_relative_update", "rho_error_l1", "rho_error_l2", "rho_error_linf", "temperature_min", "face_fallbacks", "retries",
                "pp_rec_cells", "pp_beta_min", "pp_stage_cells", "pp_alpha_min", "characteristic_fallbacks", "implicit_stages",
                "nonlinear_iterations", "linear_iterations", "implicit_residual_target_ratio", "coupled_iterations", "coupled_residual_target_ratio",
                "implicit_final_vr_sweeps", "implicit_final_vr_update", "time_pp_cells", "time_pp_alpha", "time_growth_limited"};
            std::vector<std::unique_ptr<DNDS::CsvLog>> histories(static_cast<size_t>(domainCount));
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0)
                {
                    const auto parent = std::filesystem::path(output.prefix).parent_path();
                    if (!parent.empty()) std::filesystem::create_directories(parent);
                    std::ofstream resolved(output.prefix + "_resolved.json");
                    resolved << config.Root().dump(2) << '\n';
                    DNDS_check_throw_info(bool(resolved), "cannot write resolved Cauchy configuration");
                } });
            visitDomains([&](DNDS::index iDomain, auto &domain)
                         { DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
                                                 {
                if (simulation.Placement().DomainRank(iDomain) == 0)
                    histories[iDomain] = std::make_unique<DNDS::CsvLog>(columns, domainPrefix(iDomain) + "_history", ".csv", output.historySegmentRows); }); });
            std::vector<double> baselines(size_t(domainCount), 0);
            bool reached = false, wallStop = false;
            std::vector<DNDS::Cauchy::PhysicalStepLimit> limits(static_cast<size_t>(domainCount));
            const auto record = [&](DNDS::real dt, int retries, bool useLimits = false)
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
                DNDS::real implicitRatio = 0;
                int implicitFinalVRSweeps = 0;
                DNDS::real implicitFinalVRUpdate = 0;
                for (const auto &report : domain.LastImplicitStages())
                {
                    nonlinearIterations += report.iterations;
                    linearIterations += report.linearIterations;
                    implicitFinalVRSweeps = std::max(implicitFinalVRSweeps, report.reconstructionIterations);
                    implicitFinalVRUpdate = std::max(implicitFinalVRUpdate, report.reconstructionRelativeUpdate);
                    implicitRatio = std::max(implicitRatio, report.residualTargetRatio);
                }
                // File rotation/open failures on the writer rank must not
                // strand peers entering the next solver collective.
                DNDS::MPI::LocalStage(DNDS::MPIInfo(domain.Discretization().Metrics().Communicator()), [&]
                                      {
                if (simulation.Placement().DomainRank(iDomain) == 0)
                {
                    DNDS::tLogSimpleDIValueMap row;
                    DNDS::index column = 0;
                    const auto append = [&](auto value) { row[columns.at(column++)] = value; };
                    append(domain.Epoch().acceptedStep); append(domain.Time()); append(dt); append(MPI_Wtime() - started);
                    for (auto value : values) append(value);
                    append(fallbacks); append(retries);
                    for (auto value : domain.LastPositivity()) append(value);
                    append(domain.LastImplicitStages().size()); append(nonlinearIterations); append(linearIterations); append(implicitRatio);
                    append(domain.LastCoupledIterations()); append(domain.LastCoupledTargetRatio());
                    append(implicitFinalVRSweeps); append(implicitFinalVRUpdate);
                    append(limit.limitedCells); append(limit.minimumAlpha); append(limit.growthLimited);
                    DNDS_check_throw_info(column == DNDS::index(columns.size()), "Cauchy history field count differs from its header");
                    histories[iDomain]->WriteLine(row, 17);
                    std::cout << std::setprecision(12) << "Cauchy " << (domainCount == 1 ? "" : domain.Name() + " ") << "step " << domain.Epoch().acceptedStep << " time " << domain.Time() << " dt " << dt
                              << " residual " << values[4] << " rho_min " << values[2] << " p_min " << values[3] << " VR " << values[8] << ':' << values[9]
                              << " rho L1 " << values[10] << " face fallbacks " << fallbacks << " implicit " << nonlinearIterations << '/' << linearIterations
                              << " target ratio " << implicitRatio << " stage VR " << implicitFinalVRSweeps << ':' << implicitFinalVRUpdate << " wall " << MPI_Wtime() - started << std::endl;
                } });
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
            record(0, 0);
            for (DNDS::index step = 0; time.finalTime > 0 ? !DNDS::Cauchy::EquationTimeStepper::ReachedTimeTarget(simulation.Time(), time.finalTime) : step < time.steps; ++step)
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
                if (time.finalTime > 0)
                    dt = std::min(dt, time.finalTime - simulation.Time());
                if (time.positivity.enabled)
                {
                    visitDomains([&](DNDS::index iDomain, auto &domain)
                                 { limits[iDomain] = domain.LimitPhysicalStep(dt); });
                    for (DNDS::index iDomain = 0; iDomain < domainCount; ++iDomain)
                        if (simulation.Local(iDomain))
                            dt = std::min(dt, limits[iDomain].dt);
                    MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
                }
                int retries = 0;
                for (;;)
                {
                    try
                    {
                        simulation.Advance(dt);
                        break;
                    }
                    catch (const DNDS::Cauchy::StepRejected &failure)
                    {
                        if (retries++ >= time.maxRetries)
                            throw;
                        if (mpi.rank == 0)
                            std::cout << "Retry candidate with half dt: " << failure.what() << std::endl;
                        dt *= .5;
                    }
                }
                if (time.steady || simulation.Step() % uint64_t(output.historyInterval) == 0)
                    reached = record(time.steady ? 0 : dt, retries, time.positivity.enabled);
                if (output.interval && simulation.Step() % uint64_t(output.interval) == 0)
                {
                    simulation.WriteCheckpoint(output.prefix + "_" + std::to_string(simulation.Step()) + ".h5", output.checkpointReconstruction);
                    simulation.WriteVisualization(output.prefix + "_" + std::to_string(simulation.Step()), output.hdfDeflateLevel);
                }
                if (reached)
                    break;
            }
            reached = record(0, 0) || reached;
            simulation.WriteCheckpoint(output.prefix + "_final.h5", output.checkpointReconstruction);
            simulation.WriteVisualization(output.prefix + "_final", output.hdfDeflateLevel);
            const bool stepStop = (time.steady || time.steadyTolerance > 0) && !reached && !wallStop;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
            if (mpi.rank == 0)
            {
                DNDS::Cauchy::Json result{{"time", simulation.Time()}, {"steps", simulation.Step()}, {"wallSeconds", MPI_Wtime() - started}, {"status", wallStop ? "wall-capped" : stepStop ? "step-capped"
                                                                                                                                                                               : reached    ? "steady-converged"
                                                                                                                                                                                            : "target-reached"},
                                          {"steadyTargetMet", reached},
                                          {"steadyMode", time.steady},
                                          {"physicalTimeTarget", time.finalTime},
                                          {"steadyTolerance", time.steadyTolerance}};
                std::ofstream summary(output.prefix + "_summary.json");
                summary << result.dump(2) << '\n';
                DNDS_check_throw_info(bool(summary), "cannot write Cauchy summary");
            } });
            if (wallStop || stepStop)
                status = 2;
        }
        catch (const std::exception &exception)
        {
            // No agreement/finalization here: another rank may still be inside
            // a solver collective. Unexpected errors terminate the MPI job.
            DNDS::MPI::Abort(mpi.comm, std::string("Cauchy: ") + exception.what());
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
    DNDS::MPI::Init_thread(&argc, &argv);
    // Include import/partition/setup in the external wall-time budget.
    const int status = Run(argc, argv, MPI_Wtime());
    DNDS::MPI::Finalize();
    return status;
}
