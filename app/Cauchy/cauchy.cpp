#include "Cauchy/Core/Simulation.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    // The external process budget includes mesh import/partition/setup. Start
    // here so a costly setup cannot consume the reserve for final evidence.
    const double started = MPI_Wtime();
    int status = 0;
    {
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        try
        {
            DNDS_check_throw_info(argc == 2 || argc == 3 || argc == 4, "usage: cauchy.exe config.json [--check-config | --restart checkpoint.h5]");
            DNDS::Cauchy::Json input;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                std::ifstream stream(argv[1]);
                DNDS_check_throw_info(bool(stream),"cannot open Cauchy config");
                stream>>input; });
            auto config = DNDS::Cauchy::SimulationConfig::ResolveCollective(input, mpi.comm);
            if (argc == 3)
            {
                DNDS_check_throw_info(std::string(argv[2]) == "--check-config", "expected --check-config");
                if (mpi.rank == 0)
                    std::cout << config.Root().dump(2) << '\n';
                MPI_Finalize();
                return 0;
            }
            DNDS_check_throw_info(config.Domains().size() == 1, "Cauchy physical executable currently requires one domain");
            DNDS_check_throw_info(config.Domains()[0].placement.ranks.size() == size_t(mpi.size),
                                  "Cauchy physical executable currently requires its domain to include every simulation rank");
            DNDS::Cauchy::Simulation simulation(config, mpi.comm);
            auto &domain = simulation.GetDomain(0);
            if (argc == 4)
            {
                DNDS_check_throw_info(std::string(argv[2]) == "--restart", "expected --restart checkpoint.h5");
                simulation.ReadCheckpoint(argv[3]);
            }
            const auto &output = config.Output();
            const auto &time = config.Time();
            std::ofstream history;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                if (mpi.rank == 0)
                {
                    const auto parent = std::filesystem::path(output.prefix).parent_path();
                    if (!parent.empty()) std::filesystem::create_directories(parent);
                    history.open(output.prefix + "_history.csv");
                    DNDS_check_throw_info(bool(history), "cannot open Cauchy history");
                    history << "step,time,dt,wall,mass,energy,rho_min,p_min,residual_l2,force_x,force_y,force_z,vr_sweeps,vr_relative_update,rho_error_l1,rho_error_l2,rho_error_linf,temperature_min,face_fallbacks,retries,pp_rec_cells,pp_beta_min,pp_stage_cells,pp_alpha_min,characteristic_fallbacks,implicit_stages,nonlinear_iterations,linear_iterations,implicit_residual_target_ratio,coupled_iterations,coupled_residual_target_ratio,implicit_final_vr_sweeps,implicit_final_vr_update\n" << std::setprecision(17);
                    std::ofstream resolved(output.prefix + "_resolved.json"); resolved << config.Root().dump(2) << '\n';
                } });
            double baseline = 0;
            bool reached = false, wallStop = false;
            const auto record = [&](DNDS::real dt, int retries)
            {
                const auto values = domain.Diagnostics(config.Domains()[0].settings, output.forceZones);
                if (baseline == 0)
                    baseline = std::max(values[4], DNDS::real(1e-300));
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
                if (mpi.rank == 0)
                {
                    history << domain.Epoch().acceptedStep << ',' << domain.Time() << ',' << dt << ',' << MPI_Wtime() - started;
                    for (auto value : values)
                        history << ',' << value;
                    history << ',' << fallbacks << ',' << retries;
                    for (auto value : domain.LastPositivity())
                        history << ',' << value;
                    history << ',' << domain.LastImplicitStages().size() << ',' << nonlinearIterations << ',' << linearIterations << ',' << implicitRatio
                            << ',' << domain.LastCoupledIterations() << ',' << domain.LastCoupledTargetRatio()
                            << ',' << implicitFinalVRSweeps << ',' << implicitFinalVRUpdate;
                    history << '\n';
                    history.flush();
                    std::cout << std::setprecision(12) << "Cauchy step " << domain.Epoch().acceptedStep << " time " << domain.Time() << " dt " << dt
                              << " residual " << values[4] << " rho_min " << values[2] << " p_min " << values[3] << " VR " << values[8] << ':' << values[9]
                              << " rho L1 " << values[10] << " face fallbacks " << fallbacks << " implicit " << nonlinearIterations << '/' << linearIterations
                              << " target ratio " << implicitRatio << " stage VR " << implicitFinalVRSweeps << ':' << implicitFinalVRUpdate << " wall " << MPI_Wtime() - started << std::endl;
                }
                const bool outerTarget = time.steadyTolerance <= 0 || values[4] / baseline <= time.steadyTolerance;
                if (time.steady)
                    return !domain.LastImplicitStages().empty() && std::all_of(domain.LastImplicitStages().begin(), domain.LastImplicitStages().end(), [](const auto &stage)
                                                                               { return stage.converged; }) &&
                           outerTarget;
                return time.steadyTolerance > 0 && domain.Epoch().acceptedStep >= uint64_t(time.steadyMinSteps) && outerTarget;
            };
            record(0, 0);
            for (DNDS::index step = 0; time.finalTime > 0 ? !DNDS::ODE::FieldTimeStepper::ReachedTimeTarget(simulation.Time(), time.finalTime) : step < time.steps; ++step)
            {
                int stop = time.wallSeconds > 0 && MPI_Wtime() - started >= time.wallSeconds;
                MPI_Allreduce(MPI_IN_PLACE, &stop, 1, MPI_INT, MPI_MAX, mpi.comm);
                if (stop)
                {
                    wallStop = true;
                    break;
                }
                DNDS::real dt = time.cfl > 0 ? std::min(time.dt, domain.StableStep(time.cfl)) : time.dt;
                if (time.finalTime > 0)
                    dt = std::min(dt, time.finalTime - simulation.Time());
                int retries = 0;
                for (;;)
                {
                    try
                    {
                        simulation.Advance(dt);
                        break;
                    }
                    catch (const std::exception &failure)
                    {
                        if (retries++ >= time.maxRetries)
                            throw;
                        if (mpi.rank == 0)
                            std::cout << "Retry candidate with half dt: " << failure.what() << std::endl;
                        dt *= .5;
                    }
                }
                if (time.steady || domain.Epoch().acceptedStep % uint64_t(output.historyInterval) == 0)
                    reached = record(time.steady ? 0 : dt, retries);
                if (output.interval && domain.Epoch().acceptedStep % uint64_t(output.interval) == 0)
                {
                    domain.WriteCheckpoint(output.prefix + "_" + std::to_string(domain.Epoch().acceptedStep) + ".h5", config.Root(), output.checkpointReconstruction);
                    domain.WriteVisualization(output.prefix + "_" + std::to_string(domain.Epoch().acceptedStep), output.hdfDeflateLevel);
                }
                if (reached)
                    break;
            }
            reached = record(0, 0) || reached;
            domain.WriteCheckpoint(output.prefix + "_final.h5", config.Root(), output.checkpointReconstruction);
            domain.WriteVisualization(output.prefix + "_final", output.hdfDeflateLevel);
            const bool stepStop = (time.steady || time.steadyTolerance > 0) && !reached && !wallStop;
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
            }
            if (wallStop || stepStop)
                status = 2;
        }
        catch (const std::exception &exception)
        {
            std::cerr << "Cauchy rank " << mpi.rank << ": " << exception.what() << '\n';
            status = 1;
        }
        MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, mpi.comm);
    }
    MPI_Finalize();
    return status;
}
