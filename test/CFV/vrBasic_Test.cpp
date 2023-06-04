#include  "CFV/VariationalReconstruction.hpp"

#include <cstdlib>

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log
*/

void testConstruct()
{
    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();
    // DNDS::Debug::MPIDebugHold(mpi);
    char buf[512];
    // std::cout << getcwd(buf, 512) << std::endl;
    auto mesh = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, 3);
    auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
    // "../data/mesh/FourTris_V1.pw.cgns"
    // "../data/mesh/SC20714_MixedA.cgns"
    // "../data/mesh/UniformDM240_E120.cgns"
    // "../data/mesh/Ball.cgns"
    reader.ReadFromCGNSSerial("../data/mesh/Ball2.cgns");
    reader.BuildCell2Cell();
    reader.MeshPartitionCell2Cell();
    reader.PartitionReorderToMeshCell2Cell();
    reader.BuildSerialOut();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    auto vr = DNDS::CFV::VariationalReconstruction<3>(mpi, mesh);
    vr.ConstructMetrics();
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

    testConstruct();

    MPI_Finalize();

    return 0;
}