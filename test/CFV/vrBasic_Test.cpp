#include "CFV/VariationalReconstruction.hpp"

#include <cstdlib>
#include <omp.h>

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
    auto mesh = std::make_shared<DNDS::Geom::UnstructuredMesh>(mpi, 2);
    auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
    // "../data/mesh/FourTris_V1.pw.cgns"
    // "../data/mesh/SC20714_MixedA.cgns"
    // "../data/mesh/UniformDM240_E120.cgns"
    // "../data/mesh/Ball.cgns"
    reader.ReadFromCGNSSerial("../data/mesh/SC20714_MixedA.cgns");
    reader.BuildCell2Cell();
    reader.MeshPartitionCell2Cell();
    reader.PartitionReorderToMeshCell2Cell();
    reader.BuildSerialOut();
    mesh->BuildGhostPrimary();
    mesh->AdjGlobal2LocalPrimary();
    mesh->InterpolateFace();
    mesh->AssertOnFaces();

    auto vr = DNDS::CFV::VariationalReconstruction<2>(mpi, mesh);
#ifdef DNDS_USE_OMP
    omp_set_num_threads(DNDS::MPIWorldSize() == 1 ? std::min(omp_get_num_procs(), omp_get_max_threads()) : 1);
#endif
    // omp_set_num_threads(1);
    vr.ConstructMetrics();
    vr.ConstructBaseAndWeight();
    vr.ConstructRecCoeff();
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