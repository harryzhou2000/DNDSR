#include "Euler/EulerSolver.hpp"


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    std::string defaultConfJson = "../cases/eulerSA_default_config.json";
    std::string confJson = "../cases/eulerSA_config.json";
    if (argc > 1 && std::stoi(argv[1]) == 1)
    {
        defaultConfJson = "./eulerSA_default_config.json";
        confJson = "./eulerSA_config.json";
    }
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        auto strategy = DNDS::MPI::CommStrategy::Instance().GetArrayStrategy();
        DNDS::Euler::EulerSolver<DNDS::Euler::EulerModel::NS_SA> solver(mpi);
        solver.ConfigureFromJson(defaultConfJson, false);
        solver.ConfigureFromJson(defaultConfJson, true, confJson);
        solver.ReadMeshAndInitialize();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
}