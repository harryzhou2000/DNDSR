#include "Geom/Mesh.hpp"
#include "stdio.h"
#include "unistd.h"

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log
*/

void testCGNS()
{
    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();
    char buf[512];
    std::cout << getcwd(buf, 512) << std::endl;
    auto mesh = std::make_shared<Geom::UnstructuredMesh>(mpi, 3);
    auto reader = Geom::UnstructuredMeshSerialRW(mesh, 0);
    // "../data/mesh/SC20714_MixedA.cgns"
    reader.ReadFromCGNSSerial("../data/mesh/Ball.cgns");
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }
    testCGNS();

    MPI_Finalize();

    return 0;
}