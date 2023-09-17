#include "Euler/EulerSolver.hpp"

int main(int argc, char *argv[])
{
    // std::cout << "====================\n";
    // std::cout << argc << std::endl;
    // for (int i = 0; i < argc; i++)
    //     std::cout << argv[i] << "\n";
    MPI_Init(&argc, &argv);
    // std::cout << "====================\n";
    // std::cout << argc << std::endl;
    // for (int i = 0; i < argc; i++)
    //     std::cout << argv[i] << "\n";
    // return 0;

    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        auto strategy = DNDS::MPI::CommStrategy::Instance().GetArrayStrategy();
        DNDS::Euler::EulerSolver<DNDS::Euler::EulerModel::NS> solver(mpi);
        solver.ConfigureFromJson("../cases/euler_default_config.json", false);
        solver.ConfigureFromJson("../cases/euler_default_config.json", true, "../cases/euler_config.json");
        solver.ReadMeshAndInitialize();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
}