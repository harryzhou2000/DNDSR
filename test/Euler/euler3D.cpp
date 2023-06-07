#include "Euler/EulerSolver.hpp"


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        DNDS::Euler::EulerSolver<DNDS::Euler::EulerModel::NS_3D> solver(mpi);
        solver.ConfigureFromJson("../cases/euler3D_config.json");
        solver.ReadMeshAndInitialize();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
}