#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Geom/Mesh/Mesh.hpp"
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
    // DNDS::Debug::MPIDebugHold(mpi);
    char buf[512];
    // std::cout << getcwd(buf, 512) << std::endl;
    int dim = 2;
    auto mesh = DNDS::Geom::ReadNativeMeshFromCGNS(mpi, dim,
                                                   "../data/mesh/Uniform32_Periodic.cgns", DNDS::Geom::PartitionOptions{},
                                                   DNDS::Geom::FBC_Name_2_ID_Default, DNDS::Geom::NativePeriodicImportOptions{});
    auto reader = DNDS::Geom::UnstructuredMeshSerialRW(mesh, 0);
    DNDS::Geom::PrepareMesh(*mesh, reader, {0, 1, false});

    auto meshBnd = mesh->ExtractBoundary();
    auto readerBnd = DNDS::Geom::UnstructuredMeshSerialRW(meshBnd, 0);
    readerBnd.BuildSerialOut();

    // Four repetitions retain the original mapping-completeness purpose.
    // Owned global expectations come independently from rank-prefix counts.
    for (int pass = 0; pass < 4; ++pass)
        for (auto kind : {DNDS::Geom::EntityKind::Cell, DNDS::Geom::EntityKind::Node,
                          DNDS::Geom::EntityKind::Face, DNDS::Geom::EntityKind::Bnd})
        {
            const auto space = mesh->Entities(kind);
            DNDS::index owned = space.NumOwned(), first = 0;
            MPI_Exscan(&owned, &first, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            if (mpi.rank == 0)
                first = 0;
            for (DNDS::index row = 0; row < space.NumLocal(); ++row)
            {
                const auto global = space.LocalToGlobal(row);
                if (row < owned)
                    DNDS_check_throw_info(global == first + row, "owned mesh global order mismatch");
                const auto local = space.GlobalToLocal(global);
                DNDS_check_throw_info(local && *local == row, "native mesh entity roundtrip incomplete");
            }
        }

    mesh->ApplyGeometryTransform(Eigen::Matrix3d::Identity(), 2);

    reader.BuildSerialOut(); // refresh physical output geometry after transform

    reader.PrintSerialPartPltBinaryDataArray(
        "../data/out/debug",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        0);
    reader.PrintSerialPartPltBinaryDataArray(
        "../data/out/debug",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        1);
    reader.PrintSerialPartVTKDataArray(
        "../data/out/debug",
        "../data/out/debug",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        0);
    reader.PrintSerialPartVTKDataArray(
        "../data/out/debug",
        "../data/out/debug",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        1);

    readerBnd.PrintSerialPartPltBinaryDataArray(
        "../data/out/debugBnd",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        0);
    readerBnd.PrintSerialPartPltBinaryDataArray(
        "../data/out/debugBnd",
        1, 1,
        [](int)
        { return "vCell"; },
        [](int, DNDS::index)
        { return 1; },
        [](int)
        { return "vPoint"; },
        [](int, DNDS::index)
        { return 1; },
        0.0,
        0);
    readerBnd.PrintSerialPartVTKDataArray(
        "../data/out/debugBnd",
        "../data/out/debugBnd",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        0);
    readerBnd.PrintSerialPartVTKDataArray(
        "../data/out/debugBnd",
        "../data/out/debugBnd",
        1, 1, 1, 1,
        [](int)
        { return "aScalar"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVector"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        [](int)
        { return "aScalarP"; },
        [](int, DNDS::index iCell)
        { return iCell; },
        [](int)
        { return "aVectorP"; },
        [](int, DNDS::index iCell, DNDS::rowsize idim)
        { return idim; },
        0.0,
        1);
    // char c;
    // std::cin >> c;
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