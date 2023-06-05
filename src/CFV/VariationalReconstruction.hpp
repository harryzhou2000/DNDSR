#pragma once

#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "json.hpp"

namespace DNDS::CFV
{
    struct VRSettings
    {
        using json = nlohmann::json;
        json jsonSetting;

        int maxOrder{3}; // P degree
        int intOrder{5}; // note this is actually reduced somewhat

        VRSettings()
        {
            jsonSetting = json::object();
            WriteIntoJson();
        }

        void WriteIntoJson()
        {
            jsonSetting["maxOrder"] = maxOrder;
            jsonSetting["intOrder"] = intOrder;
        }

        void ParseFromJson()
        {
            maxOrder = jsonSetting["maxOrder"];
            intOrder = jsonSetting["intOrder"];
        }
    };
}

namespace DNDS::CFV
{
    struct RecAtr
    {
        real relax = UnInitReal;
        uint8_t NDOF = -1;
        uint8_t NDIFF = -1;
        uint8_t intOrder = 1;
    };

    using tCoeffPair = DNDS::ArrayPair<DNDS::ParArray<real, NonUniformSize>>;
    using tCoeff = decltype(tCoeffPair::father);
    using tVecsPair = DNDS::ArrayPair<DNDS::ArrayEigenUniMatrixBatch<3, 1>>;
    using tVecs = decltype(tVecsPair::father);
    using tVecPair = Geom::tCoordPair;
    using tVec = Geom::tCoord;

    template <int dim>
    class VariationalReconstruction
    {
    public:
        MPI_int mRank{0};
        MPIInfo mpi;
        VRSettings settings;
        std::shared_ptr<Geom::UnstructuredMesh> mesh;

        std::vector<real> volumeLocal;
        std::vector<real> faceArea;
        std::vector<RecAtr> cellAtr;
        std::vector<RecAtr> faceAtr;
        tCoeffPair cellIntJacobiDet;
        tCoeffPair faceIntJacobiDet;
        tVecsPair faceUnitNorm;
        tVecPair faceMeanNorm;
        tVecPair cellBary;
        tVecPair faceCent;
        tVecPair cellCent;
        tVecsPair cellIntPPhysics;
        tVecsPair faceIntPPhysics;

        VariationalReconstruction(MPIInfo nMpi, std::shared_ptr<Geom::UnstructuredMesh> nMesh)
            : mpi(nMpi), mesh(nMesh)
        {
            DNDS_assert(dim == mesh->dim);
            mRank = mesh->mRank;
        }

        /**
         * @brief make pair with default MPI type, match cell layout
         */
        template <class TArrayPair, class... TOthers>
        void MakePairDefaultOnCell(TArrayPair &aPair)
        {
            DNDS_MAKE_SSP(aPair.father, mpi);
            DNDS_MAKE_SSP(aPair.son, mpi);
            aPair.father->Resize(mesh->NumCell(), others...);
            aPair.son->Resize(mesh->NumCellGhost(), others...);
        }

        /**
         * @brief make pair with default MPI type, match face layout
         */
        template <class TArrayPair, class... TOthers>
        void MakePairDefaultOnFace(TArrayPair &aPair, TOthers... others)
        {
            DNDS_MAKE_SSP(aPair.father, mpi);
            DNDS_MAKE_SSP(aPair.son, mpi);
            aPair.father->Resize(mesh->NumFace(), others...);
            aPair.son->Resize(mesh->NumFaceGhost(), others...);
        }

        void ConstructMetrics();
    };
}