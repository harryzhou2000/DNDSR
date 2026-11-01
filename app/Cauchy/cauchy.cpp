#include "Cauchy/Core/Simulation.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int status = 0;
    {
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        try
        {
            DNDS_check_throw_info(argc == 2 || argc == 4, "usage: cauchy.exe config.json [--restart checkpoint.h5]");
            DNDS::Cauchy::Json input;
            DNDS::MPI::LocalStage(mpi, [&]
                                  {
                std::ifstream stream(argv[1]);
                DNDS_check_throw_info(bool(stream),"cannot open Cauchy config");
                stream>>input; });
            auto config = DNDS::Cauchy::SimulationConfig::ResolveCollective(input, mpi.comm);
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
            for (DNDS::index step = 0; step < config.Time().steps; ++step)
            {
                simulation.Advance(config.Time().dt);
                const auto fallbacks = domain.LastFaceFallbacks();
                if (mpi.rank == 0)
                    std::cout << "Cauchy step " << domain.Epoch().acceptedStep << " time " << domain.Time() << " face fallbacks " << fallbacks << '\n';
                if (output.interval && domain.Epoch().acceptedStep % uint64_t(output.interval) == 0)
                    domain.WriteCheckpoint(output.prefix + "_" + std::to_string(domain.Epoch().acceptedStep) + ".h5", config.Root());
            }
            domain.WriteCheckpoint(output.prefix + "_final.h5", config.Root());
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
