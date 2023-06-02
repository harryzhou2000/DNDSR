#pragma once

#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "json.hpp"

namespace DNDS::CFV
{
    struct RecAtr
    {
        real relax = UnInitReal;
        uint8_t NDOF = -1;
        uint8_t NDIFF = -1;
        uint8_t intOrder = 1;
    };

    class VariationalReconstruction
    {
    public:
        MPIInfo mpi;
        Geom::UnstructuredMesh *mesh;

        std::vector<real> volumeLocal;
        std::vector<real> faceArea;

        VariationalReconstruction(MPIInfo nMpi, Geom::UnstructuredMesh *nMesh)
            : mpi(nMpi), mesh(nMesh)
        {
            
        }
    };
}