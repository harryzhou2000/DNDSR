#include "Euler/EulerSolver.hpp"


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        auto strategy = DNDS::MPI::CommStrategy::Instance().GetArrayStrategy();
        DNDS::Euler::EulerSolver<DNDS::Euler::EulerModel::NS_SA> solver(mpi);
        solver.ConfigureFromJson("../cases/eulerSA_default_config.json", false);
        solver.ConfigureFromJson("../cases/eulerSA_default_config.json", true, "../cases/eulerSA_config.json");
        solver.ReadMeshAndInitialize();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
}