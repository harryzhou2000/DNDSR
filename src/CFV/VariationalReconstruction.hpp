#pragma once

#include "json.hpp"

#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatirx.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"

#include "BaseFunction.hpp"

namespace DNDS::CFV
{
    struct VRSettings
    {
        using json = nlohmann::json;
        json jsonSetting;

        int maxOrder{3}; // P degree
        int intOrder{5}; // note this is actually reduced somewhat
        bool cacheDiffBase = true;

        VRSettings()
        {
            jsonSetting = json::object();
            WriteIntoJson();
        }

        void WriteIntoJson()
        {
            jsonSetting["maxOrder"] = maxOrder;
            jsonSetting["intOrder"] = intOrder;
            jsonSetting["cacheDiffBase"] = cacheDiffBase;
        }

        void ParseFromJson()
        {
            maxOrder = jsonSetting["maxOrder"];
            intOrder = jsonSetting["intOrder"];
            cacheDiffBase = jsonSetting["cacheDiffBase"];
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
    using t3VecsPair = DNDS::ArrayPair<DNDS::ArrayEigenUniMatrixBatch<3, 1>>;
    using t3Vecs = decltype(t3VecsPair::father);
    using t3VecPair = Geom::tCoordPair;
    using t3Vec = Geom::tCoord;
    using t3MatPair = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<3, 3>>;
    using t3Mat = decltype(t3MatPair::father);

    // Corresponds to mean/rec dofs
    using tVVecPair = DNDS::ArrayPair<DNDS::ArrayEigenVector<DynamicSize>>;
    using tVVec = decltype(tVVecPair::father);
    using tMatsPair = DNDS::ArrayPair<DNDS::ArrayEigenUniMatrixBatch<DynamicSize, DynamicSize>>;
    using tMats = decltype(tMatsPair::father);
    using tVecsPair = DNDS::ArrayPair<DNDS::ArrayEigenUniMatrixBatch<DynamicSize, 1>>;
    using tVecs = decltype(tVecsPair::father);
    using tVMatPair = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<DynamicSize, DynamicSize>>;
    using tVMat = decltype(tVMatPair::father);

    template <int nVarsFixed>
    using tURec = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<DynamicSize, nVarsFixed>>;
    template <int nVarsFixed>
    using tUDof = DNDS::ArrayPair<DNDS::ArrayEigenMatrix<nVarsFixed, 1>>;

    template <int dim>
    class VariationalReconstruction
    {
    public:
        MPI_int mRank{0};
        MPIInfo mpi;
        VRSettings settings;
        std::shared_ptr<Geom::UnstructuredMesh> mesh;

        std::vector<real> volumeLocal; // ConstructMetrics()
        std::vector<real> faceArea;    // ConstructMetrics()
        std::vector<RecAtr> cellAtr;   // ConstructMetrics()
        std::vector<RecAtr> faceAtr;   // ConstructMetrics()
        tCoeffPair cellIntJacobiDet;   // ConstructMetrics()
        tCoeffPair faceIntJacobiDet;   // ConstructMetrics()
        t3VecsPair faceUnitNorm;       // ConstructMetrics()
        t3VecPair faceMeanNorm;        // ConstructMetrics()
        t3VecPair cellBary;            // ConstructMetrics()
        t3VecPair faceCent;            // ConstructMetrics()
        t3VecPair cellCent;            // ConstructMetrics()
        t3VecsPair cellIntPPhysics;    // ConstructMetrics()
        t3VecsPair faceIntPPhysics;    // ConstructMetrics()
        t3VecPair cellAlignedHBox;     // ConstructMetrics()
        t3VecPair cellMajorHBox;       // ConstructMetrics() //TODO
        t3MatPair cellMajorCoord;      // ConstructMetrics() //TODO

        t3VecPair faceAlignedScales;     // ConstructBaseAndWeight()
        tVVecPair cellBaseMoment;        // ConstructBaseAndWeight()
        tVVecPair faceWeight;            // ConstructBaseAndWeight()
        tMatsPair cellDiffBaseCache;     // ConstructBaseAndWeight()
        tMatsPair faceDiffBaseCache;     // ConstructBaseAndWeight()
        tVMatPair cellDiffBaseCacheCent; // ConstructBaseAndWeight()//TODO
        tVMatPair faceDiffBaseCacheCent; // ConstructBaseAndWeight()//TODO

        tMatsPair matrixAB;        // ConstructRecCoeff()
        tVecsPair vectorB;         // ConstructRecCoeff()
        tMatsPair matrixAAInvB;    // ConstructRecCoeff()
        tMatsPair vectorAInvB;     // ConstructRecCoeff()
        tVMatPair matrixSecondary; // ConstructRecCoeff()//TODO

        VariationalReconstruction(MPIInfo nMpi, std::shared_ptr<Geom::UnstructuredMesh> nMesh)
            : mpi(nMpi), mesh(nMesh)
        {
            DNDS_assert(dim == mesh->dim);
            mRank = mesh->mRank;
        }

        void ConstructMetrics();

        // TODO: add customizable bnd type processing
        void ConstructBaseAndWeight();

        void ConstructRecCoeff();

        template <int nVarsFixed, class TFBoundary>
        void DoReconstructionIter(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            TFBoundary &&FBoundary)
        {
        }

        /**
         * @brief make pair with default MPI type, match cell layout
         */
        template <class TArrayPair, class... TOthers>
        void MakePairDefaultOnCell(TArrayPair &aPair, TOthers... others)
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

        Geom::Elem::Quadrature GetFaceQuad(index iFace)
        {
            auto e = mesh->GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, faceAtr[iFace].intOrder};
        }

        Geom::Elem::Quadrature GetCellQuad(index iCell)
        {
            auto e = mesh->GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, cellAtr[iCell].intOrder};
        }

        /**
         * @brief flag = 0 means use moment data, or else use no moment (as 0)
         */
        template <class TOut>
        void FDiffBaseValue(TOut &DiBj,
                            const Geom::tPoint &pPhy, // conventional input above
                            index iCell, index iFace, int iG, int flag = 0)
        {
            using namespace Geom;
            auto simpleScale = cellAlignedHBox[iCell];
            auto pCen = cellCent[iCell];
            tPoint pPhysicsCScaled = (pPhy - pCen).array() / simpleScale.array();
            for (int idiff = 0; idiff < DiBj.rows(); idiff++)
                for (int ibase = 0; ibase < DiBj.cols(); ibase++)
                {
                    int px = diffOperatorOrderList2D[ibase][0];
                    int py = diffOperatorOrderList2D[ibase][1];
                    int pz = diffOperatorOrderList2D[ibase][2];
                    int ndx = diffOperatorOrderList2D[idiff][0];
                    int ndy = diffOperatorOrderList2D[idiff][1];
                    int ndz = diffOperatorOrderList2D[idiff][2];
                    if constexpr (dim == 2)
                    {
                        DiBj(idiff, ibase) =
                            FPolynomial3D(px, py, pz, ndx, ndy, ndz,
                                          pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2)) /
                            (std::pow(simpleScale(0), ndx) * std::pow(simpleScale(1), ndy) * std::pow(1, ndz));
                    }
                    else
                    {
                        DiBj(idiff, ibase) =
                            FPolynomial3D(px, py, pz, ndx, ndy, ndz,
                                          pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2)) /
                            (std::pow(simpleScale(0), ndx) * std::pow(simpleScale(1), ndy) * std::pow(simpleScale(2), ndz));
                    }
                }
            if (flag == 0)
            {
                auto baseMoment = cellBaseMoment[iCell];
                DiBj(0, Eigen::all) -= baseMoment.transpose();
            }
        }

        bool CellIsFaceBack(index iCell, index iFace)
        {
            DNDS_assert(mesh->face2cell(iFace, 0) == iCell || mesh->face2cell(iFace, 1) == iCell);
            return mesh->face2cell(iFace, 0) == iCell;
        }

        index CellFaceOther(index iCell, index iFace)
        {
            return CellIsFaceBack(iCell, iFace)
                       ? mesh->face2cell(iFace, 1)
                       : mesh->face2cell(iFace, 0);
        }

        /**
         * @brief if if2c < 0, then calculated, if maxDiff == 255, then seen as all diffs
         * if iFace < 0, then seen as cell int points
         */
        template <class TList>
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>
        GetIntPointDiffBaseValue(
            index iCell, index iFace, index if2c, index iG,
            TList &&diffList = Eigen::all,
            uint8_t maxDiff = UINT8_MAX)
        {
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;

            if (iFace >= 0)
            {
                if (settings.cacheDiffBase)
                {
                    // auto gFace = this->GetFaceQuad(iFace);
                    return faceDiffBaseCache(iFace, iG + (faceDiffBaseCache.RowSize(iFace) / 2) * if2c)(
                        std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                }
                else
                {
                    // Actual computing: //TODO: take care of periodic case
                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                    dbv.resize(std::min(maxDiff, faceAtr[iFace].NDIFF), cellAtr[iCell].NDOF);
                    FDiffBaseValue(dbv, faceIntPPhysics(iFace, iG), iCell, iFace, iG, 0);
                    return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                }
            }
            else
            {
                if (settings.cacheDiffBase)
                {
                    return cellDiffBaseCache(iCell, iG)(
                        std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                }
                else
                {
                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                    dbv.resize(std::min(maxDiff, cellAtr[iCell].NDIFF), cellAtr[iCell].NDOF);
                    FDiffBaseValue(dbv, cellIntPPhysics(iCell, iG), iCell, -1, iG, 0);
                    return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                }
            }
        }

        template <class TDiffI, class TDiffJ>
        auto FFaceFunctional(
            TDiffI &&DiffI, TDiffJ &&DiffJ,
            index iFace, index iG)
        {
            Eigen::Vector<real, Eigen::Dynamic> wgd = faceWeight[iFace].array().square();
            DNDS_assert(DiffI.rows() == DiffJ.rows());
            int cnDiffs = DiffI.rows();

            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> Conj;
            Conj.resize(DiffI.cols(), DiffJ.cols());
            Conj.setZero();

            //* PJH - rotation invariant scheme
            auto faceLV = faceAlignedScales[iFace];
            real faceL = std::sqrt(faceLV.array().square().mean());

            if constexpr (dim == 2)
            {
                DNDS_assert(cnDiffs == 10 || cnDiffs == 6 || cnDiffs == 3 || cnDiffs == 1);
                for (int i = 0; i < DiffI.cols(); i++)
                    for (int j = 0; j < DiffJ.cols(); j++)
                    {
                        switch (cnDiffs)
                        {
                        case 10:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<2, 3>(
                                    DiffI({6, 7, 8, 9}, {i}),
                                    DiffJ({6, 7, 8, 9}, {j})) *
                                wgd(3) * std::pow(faceL, 3 * 2);
                        case 6:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<2, 2>(
                                    DiffI({3, 4, 5}, {i}),
                                    DiffJ({3, 4, 5}, {j})) *
                                wgd(2) * std::pow(faceL, 2 * 2);
                        case 3:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<2, 1>(
                                    DiffI({1, 2}, {i}),
                                    DiffJ({1, 2}, {j})) *
                                wgd(1) * std::pow(faceL, 1 * 2);
                        case 1:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<2, 0>(
                                    DiffI({0}, {i}),
                                    DiffJ({0}, {j})) * //! i, j needed in {}!
                                wgd(0);
                            break;
                        }
                    }
            }
            else
            {
                DNDS_assert(cnDiffs == 20 || cnDiffs == 10 || cnDiffs == 4 || cnDiffs == 1);
                for (int i = 0; i < DiffI.cols(); i++)
                    for (int j = 0; j < DiffJ.cols(); j++)
                    {
                        switch (cnDiffs)
                        {
                        case 20:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<3, 3>(
                                    DiffI({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, {i}),
                                    DiffJ({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, {j})) *
                                wgd(3) * std::pow(faceL, 3 * 2);
                        case 10:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<3, 2>(
                                    DiffI({4, 5, 6, 7, 8, 9}, {i}),
                                    DiffJ({4, 5, 6, 7, 8, 9}, {j})) *
                                wgd(2) * std::pow(faceL, 2 * 2);
                        case 4:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<3, 1>(
                                    DiffI({1, 2, 3}, {i}),
                                    DiffJ({1, 2, 3}, {j})) *
                                wgd(1) * std::pow(faceL, 1 * 2);
                        case 1:
                            Conj(i, j) +=
                                NormSymDiffOrderTensorV<3, 0>(
                                    DiffI({0}, {i}),
                                    DiffJ({0}, {j})) *
                                wgd(0);
                            break;
                        }
                    }
            }
            return Conj;
        }
    };
}