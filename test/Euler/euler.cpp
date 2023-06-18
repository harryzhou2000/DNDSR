#include "Euler/EulerSolver.hpp"


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        DNDS::Euler::EulerSolver<DNDS::Euler::EulerModel::NS> solver(mpi);
        solver.ConfigureFromJson("../cases/euler_default_config.json", false);
        solver.ConfigureFromJson("../cases/euler_default_config.json", true, "../cases/euler_config.json");
        solver.ReadMeshAndInitialize();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
}