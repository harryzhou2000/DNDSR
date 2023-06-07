#pragma once

#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatirx.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"

#include "BaseFunction.hpp"
#include "Limiters.hpp"

#define JSON_ASSERT DNDS_assert
#include "json.hpp"

namespace DNDS::CFV
{
    struct VRSettings
    {
        using json = nlohmann::json;
        json jsonSetting;

        int maxOrder{3}; // P degree
        int intOrder{5}; // note this is actually reduced somewhat
        bool cacheDiffBase = true;

        real jacobiRelax = 1.0;
        bool SORInstead = true;

        real smoothThreshold = 0.01;
        real WBAP_nStd = 10;
        bool normWBAP = false;

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
            jsonSetting["jacobiRelax"] = jacobiRelax;
            jsonSetting["SORInstead"] = SORInstead;

            jsonSetting["smoothThreshold"] = smoothThreshold;
            jsonSetting["WBAP_nStd"] = WBAP_nStd;
            jsonSetting["normWBAP"] = normWBAP;
        }

        void ParseFromJson()
        {
            maxOrder = jsonSetting["maxOrder"];
            intOrder = jsonSetting["intOrder"];
            cacheDiffBase = jsonSetting["cacheDiffBase"];
            jacobiRelax = jsonSetting["jacobiRelax"];
            SORInstead = jsonSetting["SORInstead"];

            smoothThreshold = jsonSetting["smoothThreshold"];
            WBAP_nStd = jsonSetting["WBAP_nStd"];
            normWBAP = jsonSetting["normWBAP"];
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

    using tScalarPair = DNDS::ArrayPair<DNDS::ParArray<real, 1>>;
    using tScalar = decltype(tScalarPair::father);

    template <int dim>
    class VariationalReconstruction
    {
    public:
        MPI_int mRank{0};
        MPIInfo mpi;
        VRSettings settings;
        std::shared_ptr<Geom::UnstructuredMesh> mesh;

        real volGlobal{0};             // ConstructMetrics()
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
        tVMatPair cellDiffBaseCacheCent; // ConstructBaseAndWeight()//TODO *test
        tVMatPair faceDiffBaseCacheCent; // ConstructBaseAndWeight()//TODO *test

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

        void ConstructBaseAndWeight(const std::function<real(Geom::t_index)> &id2faceDircWeight = [](Geom::t_index id)
                                    { return 1.0; });

        void ConstructRecCoeff();

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

        Geom::Elem::Quadrature GetFaceQuadO1(index iFace)
        {
            auto e = mesh->GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, 1};
        }

        Geom::Elem::Quadrature GetCellQuad(index iCell)
        {
            auto e = mesh->GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, cellAtr[iCell].intOrder};
        }

        Geom::Elem::Quadrature GetCellQuadO1(index iCell)
        {
            auto e = mesh->GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, 1};
        }

        /**
         * @brief flag = 0 means use moment data, or else use no moment (as 0)
         * if iFace < 0, means anywhere
         * if iFace > 0, iG == -1, means center; iG < -1, then anywhere
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
                    if constexpr (dim == 2)
                    {
                        int px = diffOperatorOrderList2D[ibase][0];
                        int py = diffOperatorOrderList2D[ibase][1];
                        int pz = diffOperatorOrderList2D[ibase][2];
                        int ndx = diffOperatorOrderList2D[idiff][0];
                        int ndy = diffOperatorOrderList2D[idiff][1];
                        int ndz = diffOperatorOrderList2D[idiff][2];
                        DiBj(idiff, ibase) =
                            FPolynomial3D(px, py, pz, ndx, ndy, ndz,
                                          pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2)) /
                            (std::pow(simpleScale(0), ndx) * std::pow(simpleScale(1), ndy) * std::pow(1, ndz));
                    }
                    else
                    {
                        int px = diffOperatorOrderList[ibase][0];
                        int py = diffOperatorOrderList[ibase][1];
                        int pz = diffOperatorOrderList[ibase][2];
                        int ndx = diffOperatorOrderList[idiff][0];
                        int ndy = diffOperatorOrderList[idiff][1];
                        int ndz = diffOperatorOrderList[idiff][2];
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
         * if iFace < 0, then seen as cell int points; if iG < 1, then seen as center
         * @todo : divide GetIntPointDiffBaseValue into different calls
         */
        template <class TList>
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>
        GetIntPointDiffBaseValue(
            index iCell, index iFace, index if2c, index iG,
            TList &&diffList = Eigen::all,
            uint8_t maxDiff = UINT8_MAX)
        {
            if (iFace >= 0)
            {
                if (if2c < 0)
                    if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                if (iG >= 0)
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
                        // auto gFace = this->GetFaceQuad(iFace);
                        int maxNDOF = faceDiffBaseCacheCent[iFace].cols() / 2;
                        return faceDiffBaseCacheCent[iFace](
                            std::forward<TList>(diffList),
                            Eigen::seq(if2c * maxNDOF + 1,
                                       if2c * maxNDOF + maxNDOF - 1));
                    }
                    else
                    {
                        // Actual computing: //TODO: take care of periodic case
                        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                        dbv.resize(std::min(maxDiff, faceAtr[iFace].NDIFF), cellAtr[iCell].NDOF);
                        FDiffBaseValue(dbv, faceCent[iFace], iCell, iFace, -1, 0);
                        return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                    }
                }
            }
            else
            {
                if (iG >= 0)
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
                else
                {
                    if (settings.cacheDiffBase)
                    {
                        return cellDiffBaseCacheCent[iCell](
                            std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                    }
                    else
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                        dbv.resize(std::min(maxDiff, cellAtr[iCell].NDIFF), cellAtr[iCell].NDOF);
                        FDiffBaseValue(dbv, cellCent[iCell], iCell, -1, -1, 0);
                        return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                    }
                }
            }
        }

        /**
         * @brief if if2c < 0, then calculated with iCell hen seen as all diffs
         */
        auto
        GetMatrixSecondary(index iCell, index iFace, index if2c)
        {
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            int maxNDOFM1 = matrixSecondary[iFace].cols() / 2;
            return matrixSecondary[iFace](
                Eigen::all, Eigen::seq(
                                if2c * maxNDOFM1 + 0,
                                if2c * maxNDOFM1 + maxNDOFM1 - 1));
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
            real faceL = (faceLV.array().maxCoeff());
            // faceL = std::sqrt(faceLV.array().square().mean());

            // std::cout << DiffI.transpose() << "\n"
            //           << DiffJ.transpose() << std::endl;
            // std::cout << faceL << std::endl;
            // std::cout << wgd.transpose() << std::endl;
            // std::abort();
            // std::cout << "old len " << std::sqrt(faceLV.array().square().mean()) << std::endl;

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
            // std::cout << DiffI << std::endl << std::endl;
            // std::cout << Conj << std::endl;
            // std::abort();
            return Conj;
        }

        template <int nVarsFixed = 1>
        void BuildUDof(tUDof<nVarsFixed> &u, int nVars)
        {
            DNDS_MAKE_SSP(u.father, mpi);
            DNDS_MAKE_SSP(u.son, mpi);
            u.father->Resize(mesh->NumCell(), nVars, 1);
            u.son->Resize(mesh->NumCellGhost(), nVars, 1);
            u.TransAttach();
            u.trans.BorrowGGIndexing(mesh->cell2node.trans);
            u.trans.createMPITypes();
            u.trans.initPersistentPull();
            u.trans.initPersistentPush();

            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
                u[iCell].setZero();
        }

        template <int nVarsFixed>
        void BuildURec(tURec<nVarsFixed> &u, int nVars)
        {
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            DNDS_MAKE_SSP(u.father, mpi);
            DNDS_MAKE_SSP(u.son, mpi);
            u.father->Resize(mesh->NumCell(), maxNDOF - 1, nVars);
            u.son->Resize(mesh->NumCellGhost(), maxNDOF - 1, nVars);
            u.TransAttach();
            u.trans.BorrowGGIndexing(mesh->cell2node.trans);
            u.trans.createMPITypes();
            u.trans.initPersistentPull();
            u.trans.initPersistentPush();

            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
                u[iCell].setZero();
        }

        void BuildScalar(tScalarPair &u)
        {
            DNDS_MAKE_SSP(u.father, mpi);
            DNDS_MAKE_SSP(u.son, mpi);
            u.father->Resize(mesh->NumCell());
            u.son->Resize(mesh->NumCellGhost());
            u.TransAttach();
            u.trans.BorrowGGIndexing(mesh->cell2node.trans);
            u.trans.createMPITypes();
            u.trans.initPersistentPull();
            u.trans.initPersistentPush();
        }

        /**
         * @brief
         * @param FBoundary Vec F(const Vec &uL, const tPoint &unitNorm, const tPoint &p, t_index faceID),
         * with Vec == Eigen::Vector<real, nVarsFixed>
         */
        template <int nVarsFixed, class TFBoundary>
        void DoReconstructionIter(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            TFBoundary &&FBoundary,
            bool putIntoNew = false)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                real relax = cellAtr[iCell].relax;
                if (settings.SORInstead)
                    uRec[iCell] = uRec[iCell] * (1 - relax);
                else
                    uRecNew[iCell] = uRec[iCell] * (1 - relax);

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = matrixAAInvB[iCell];
                auto vectorAInvBRow = vectorAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);
                    if (iCellOther != UnInitIndex)
                    {
                        if (settings.SORInstead)
                            uRec[iCell] +=
                                relax *
                                (matrixAAInvBRow[ic2f + 1] * uRec[iCellOther] +
                                 vectorAInvBRow[ic2f] * (u[iCellOther] - u[iCell]).transpose());
                        else
                            uRecNew[iCell] +=
                                relax *
                                (matrixAAInvBRow[ic2f + 1] * uRec[iCellOther] +
                                 vectorAInvBRow[ic2f] * (u[iCellOther] - u[iCell]).transpose());
                    }
                    else
                    {
                        auto faceID = mesh->GetFaceZone(iFace);
                        DNDS_assert(FaceIDIsExternalBC(faceID));

                        int nVars = u[iCell].size();

                        Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> BCC;
                        BCC.setZero(uRec[iCell].rows(), uRec[iCell].cols());

                        auto qFace = this->GetFaceQuad(iFace);
                        qFace.IntegrationSimple(
                            BCC,
                            [&](auto &vInc, int iG)
                            {
                                Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
                                    this->GetIntPointDiffBaseValue(
                                        iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
                                Eigen::Vector<real, nVarsFixed> uBL =
                                    (dbv *
                                     uRec[iCell])
                                        .transpose();
                                uBL += u[iCell]; //! need fixing?
                                Eigen::Vector<real, nVarsFixed> uBV =
                                    FBoundary(
                                        uBL,
                                        u[iCell],
                                        faceUnitNorm(iFace, iG),
                                        faceIntPPhysics(iFace, iG), faceID);
                                Eigen::RowVector<real, nVarsFixed> uIncBV = (uBV - u[iCell]).transpose();
                                vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iG) * faceIntJacobiDet(iFace, iG);
                                // std::cout << faceWeight[iFace].transpose() << std::endl;
                            });
                        // BCC *= 0;
                        if (settings.SORInstead)
                            uRec[iCell] +=
                                relax * matrixAAInvBRow[0] * BCC;
                        else
                            uRecNew[iCell] +=
                                relax * matrixAAInvBRow[0] * BCC;
                    }
                }
            }

            if (putIntoNew && settings.SORInstead)
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    uRecNew[iCell].swap(uRec[iCell]);
            if ((!putIntoNew) && (!settings.SORInstead))
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    uRec[iCell].swap(uRecNew[iCell]);
        }

        template <size_t nVarsSee, class TUREC, class TUDOF>
        void DoCalculateSmoothIndicator(
            tScalarPair &si, TUREC &uRec, TUDOF &u,
            const std::array<int, nVarsSee> &varsSee)
        {
            using namespace Geom;
            static const int maxNDiff = dim == 2 ? 10 : 20;

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                // int NRecDOF = cellAtr[iCell].NDOF - 1; // ! not good ! TODO

                auto c2f = mesh->cell2face[iCell];
                Eigen::Matrix<real, nVarsSee, 2> IJIISIsum;
                IJIISIsum.setZero();
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    auto gFace = this->GetFaceQuadO1(iFace);
                    decltype(IJIISIsum) IJIISI;
                    IJIISI.setZero();
                    gFace.IntegrationSimple(
                        IJIISI,
                        [&](auto &finc, int ig)
                        {
                            int nDiff = faceAtr[iFace].NDIFF;
                            tPoint unitNorm = faceMeanNorm[iFace];

                            Eigen::Matrix<real, Eigen::Dynamic, nVarsSee, 0, maxNDiff>
                                uRecVal(nDiff, nVarsSee), uRecValL(nDiff, nVarsSee), uRecValR(nDiff, nVarsSee), uRecValJump(nDiff, nVarsSee);
                            uRecVal.setZero(), uRecValJump.setZero();
                            uRecValL = this->GetIntPointDiffBaseValue(iCell, iFace, -1, -1, Eigen::all) *
                                       uRec[iCell](Eigen::all, varsSee);
                            uRecValL(0, Eigen::all) += u[iCell](varsSee).transpose();

                            if (iCellOther != UnInitIndex)
                            {
                                uRecValR = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, -1, Eigen::all) *
                                           uRec[iCellOther](Eigen::all, varsSee);
                                uRecValR(0, Eigen::all) += u[iCellOther](varsSee).transpose();
                                uRecVal = (uRecValL + uRecValR) * 0.5;
                                uRecValJump = (uRecValL - uRecValR) * 0.5;
                            }

                            Eigen::Matrix<real, nVarsSee, nVarsSee> IJI, ISI;
                            IJI = FFaceFunctional(uRecValJump, uRecValJump, iFace, -1);
                            ISI = FFaceFunctional(uRecVal, uRecVal, iFace, -1);

                            finc(Eigen::all, 0) = IJI.diagonal();
                            finc(Eigen::all, 1) = ISI.diagonal();

                            finc *= faceArea[iFace]; // don't forget this
                        });
                    IJIISIsum += IJIISI;
                }
                Eigen::Vector<real, nVarsSee> smoothIndicator =
                    (IJIISIsum(Eigen::all, 0).array() /
                     (IJIISIsum(Eigen::all, 1).array() + verySmallReal))
                        .matrix();
                real sImax = smoothIndicator.array().abs().maxCoeff();
                si(iCell, 0) = std::sqrt(sImax) * sqr(settings.maxOrder);
            }
        }

        /**
         * @brief FM(uLeft,uRight,norm) gives vsize * vsize mat of Left Eigen Vectors
         *
         */
        template <class TEval, typename TFM, typename TFMI, class TUREC, class TUDOF>
        void DoLimiterWBAP_C(
            const TEval &eval,
            TUDOF &u,
            TUREC &uRec,
            TUREC &uRecNew,
            TUREC &uRecBuf,
            tScalarPair &si,
            bool ifAll,
            TFM &&FM, TFMI &&FMI,
            bool putIntoNew = false)
        {
            using namespace Geom;

            static const int maxRecDOFBatch = dim == 2 ? 4 : 10;
            static const int maxRecDOF = dim == 2 ? 9 : 19;
            static const int maxNDiff = dim == 2 ? 10 : 20;
            static const int nVars_Fixed = TEval::nVars_Fixed;

            static const int maxNeighbour = 6;

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if ((!ifAll) &&
                    si(iCell, 0) < settings.smoothThreshold)
                {
                    uRecNew[iCell] = uRec[iCell]; //! no lim need to copy !!!!
                    continue;
                }
                index NRecDOF = cellAtr[iCell].NDOF - 1;
                auto c2f = mesh->cell2face[iCell];
                std::vector<Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOF>> uFaces(c2f.size());
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    // * safty initialization
                    index iFace = c2f[ic2f];
                    index iCellOther = this->CellFaceOther(iCell, iFace);
                    if (iCellOther != UnInitIndex)
                    {
                        uFaces[ic2f].resizeLike(uRec[iCellOther]);
                    }
                }

                int cPOrder = settings.maxOrder;
                for (; cPOrder >= 1; cPOrder--)
                {
                    int LimStart, LimEnd; // End is inclusive
                    if constexpr (dim == 2)
                        switch (cPOrder)
                        {
                        case 3:
                            LimStart = 5;
                            LimEnd = 8;
                            break;
                        case 2:
                            LimStart = 2;
                            LimEnd = 4;
                            break;
                        case 1:
                            LimStart = 0;
                            LimEnd = 1;
                            break;
                        default:
                            LimStart = -200;
                            LimEnd = -100;
                            DNDS_assert(false);
                        }
                    else
                        switch (cPOrder)
                        {
                        case 3:
                            LimStart = 18;
                            LimEnd = 9;
                            break;
                        case 2:
                            LimStart = 3;
                            LimEnd = 8;
                            break;
                        case 1:
                            LimStart = 0;
                            LimEnd = 2;
                            break;
                        default:
                            LimStart = -200;
                            LimEnd = -100;
                            DNDS_assert(false);
                        }

                    std::vector<Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>>
                        uOthers;
                    Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                        uC = uRec[iCell](
                            Eigen::seq(
                                LimStart,
                                LimEnd),
                            Eigen::all);
                    uOthers.reserve(maxNeighbour);
                    uOthers.push_back(uC); // using uC centered
                    // InsertCheck(mpi, "HereAAC");
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        auto f2c = mesh->face2cell[iFace];
                        index iCellOther = this->CellFaceOther(iCell, iFace);
                        index iCellAtFace = f2c[0] == iCell ? 0 : 1;

                        if (iCellOther != UnInitIndex)
                        {
                            index NRecDOFOther = cellAtr[iCellOther].NDOF - 1;
                            index NRecDOFLim = std::min(NRecDOFOther, NRecDOF);
                            if (NRecDOFLim < (LimEnd + 1))
                                continue; // reserved for p-adaption
                            // if (!(ifUseLimiter[iCell] & 0x0000000FU))
                            //     continue;

                            tPoint unitNorm = faceMeanNorm[iFace].stableNormalized();

                            const auto &matrixSecondary =
                                this->GetMatrixSecondary(iCell, iFace, -1);

                            const auto &matrixSecondaryOther =
                                this->GetMatrixSecondary(iCellOther, iFace, -1);

                            // std::cout << "A"<<std::endl;
                            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOF>
                                uOtherOther = uRec[iCellOther](Eigen::seq(0, NRecDOFLim - 1), Eigen::all);

                            if (LimEnd < uOtherOther.rows() - 1) // successive SR
                                uOtherOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all) =
                                    matrixSecondaryOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::seq(LimEnd + 1, NRecDOFLim - 1)) *
                                    uFaces[ic2f](Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all);

                            // std::cout << "B" << std::endl;
                            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                                uOtherIn =
                                    matrixSecondary(Eigen::seq(LimStart, LimEnd), Eigen::all) * uOtherOther;

                            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                                uThisIn =
                                    uC.matrix();

                            // 2 char space :
                            auto uR = iCellAtFace ? u[iCell] : u[iCellOther];
                            auto uL = iCellAtFace ? u[iCellOther] : u[iCell];
                            auto M = FM(uL, uR, unitNorm);

                            uOtherIn = (M * uOtherIn.transpose()).transpose();
                            uThisIn = (M * uThisIn.transpose()).transpose();

                            Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                                uLimOutArray;

                            real n = settings.WBAP_nStd;
                            if (settings.normWBAP)
                                FWBAP_L2_Biway_Polynomial2D(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                            else
                                FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);

                            // to phys space
                            auto MI = FMI(uL, uR, unitNorm);
                            uLimOutArray = (MI * uLimOutArray.matrix().transpose()).transpose().array();

                            uFaces[ic2f](Eigen::seq(LimStart, LimEnd), Eigen::all) = uLimOutArray.matrix();
                            uOthers.push_back(uLimOutArray);
                        }
                        else
                        {
                        }
                    }
                    Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                        uLimOutArray;

                    real n = settings.WBAP_nStd;
                    if (settings.normWBAP)
                        FWBAP_L2_Multiway_Polynomial2D(uOthers, uOthers.size(), uLimOutArray, n);

                    else
                        FWBAP_L2_Multiway(uOthers, uOthers.size(), uLimOutArray, n);

                    uRecNew[iCell](
                        Eigen::seq(
                            LimStart,
                            LimEnd),
                        Eigen::all) = uLimOutArray.matrix();
                }
            }
            uRecNew.trans.startPersistentPull();
            uRecNew.trans.waitPersistentPull();
            if (!putIntoNew)
            {
                for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                    uRec[iCell] = uRecNew[iCell];
            }
        }
    };
}