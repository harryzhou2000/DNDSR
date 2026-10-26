#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Geom/Mesh/Mesh.hpp"
#include "DNDS/Serializer/SerializerJSON.hpp"
#include "stdio.h"

#include <filesystem>

std::vector<double> argD;

/*
running: with openmpi:
mpirun --mca threads_pthreads_yield_strategy nanosleep --mca threads_pthreads_nanosleep_time 1000000000  --ove
rsubscribe -np 512 test/partitionMeshSerial.exe
*/

void doPartitioning(const std::string &meshName, int dim)
{
    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();
    // DNDS::Debug::MPIDebugHold(mpi);
    char buf[512];
    // std::cout << getcwd(buf, 512) << std::endl;
    DNDS::Geom::NativePeriodicImportOptions periodic;
    periodic.frames.translation[3].map() *= 10;
    auto mesh = DNDS::Geom::ReadNativeMeshFromCGNS(mpi, dim, meshName,
                                                   DNDS::Geom::PartitionOptions{}, DNDS::Geom::FBC_Name_2_ID_Default, periodic);

    /**************************/
    // about serializerPP:
    std::filesystem::path meshPath{meshName};
    auto meshOutName = std::string(meshName) + "_part_" + std::to_string(mpi.size) + ".dir";
    std::filesystem::path meshOutDir{meshOutName};
    std::filesystem::create_directories(meshOutDir);
    std::string meshPartPath = DNDS::getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

    auto serializerJSON = DNDS::make_ssp<DNDS::Serializer::SerializerJSON>();
    serializerJSON->SetUseCodecOnUint8(true);
    DNDS::Serializer::SerializerBaseSSP serializerP = serializerJSON;

    serializerP->OpenFile(meshPartPath, false);
    mesh->WriteNativeArchive(serializerP, "meshPart");
    serializerP->CloseFile();

    MPISerialDo(mpi, [&]()
                { DNDS::log() << "    Rank: " << mpi.rank << " nCell " << mesh->NumCell() << " nCellGhost " << mesh->NumCellGhost() << std::endl; });
    MPISerialDo(mpi, [&]()
                { DNDS::log() << "    Rank: " << mpi.rank << " nNode " << mesh->NumNode() << " nNodeGhost " << mesh->NumNodeGhost() << std::endl; });

    /**************************/
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    MPI_Init(&argc, &argv);
    DNDS_assert_info(argc == 3, "need 2 arguments of [mesh file] and [dim]");

    doPartitioning(argv[1], std::stoi(argv[2]));

    MPI_Finalize();

    return 0;
}