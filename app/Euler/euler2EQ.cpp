#include "Euler/EulerSolver.hpp"


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    std::string defaultConfJson = "../cases/euler2EQ_default_config.json";
    std::string confJson = "../cases/euler2EQ_config.json";
    if(argc > 1 && std::stoi(argv[1]) == 1)
    {
        defaultConfJson = "./euler2EQ_default_config.json";
        confJson = "./euler2EQ_config.json";
    }
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        auto strategy = DNDS::MPI::CommStrategy::Instance().GetArrayStrategy();
        DNDS::Euler::EulerSolver<DNDS::Euler::EulerModel::NS_2EQ> solver(mpi);
        solver.ConfigureFromJson(defaultConfJson, false);
        solver.ConfigureFromJson(defaultConfJson, true, confJson);
        solver.ReadMeshAndInitialize();
        solver.RunImplicitEuler();
    }
    MPI_Finalize();
}