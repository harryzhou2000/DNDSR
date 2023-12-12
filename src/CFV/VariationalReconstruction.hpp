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

#include "VRSettings.hpp"

#include "Eigen/Dense"
#include "fmt/core.h"

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
    using tVVecPair = ::DNDS::ArrayPair<DNDS::ArrayEigenVector<DynamicSize>>;
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

    /**
     * @brief
     * The VR class that provides any information needed in high-order CFV
     *
     * @details
     * VR holds a primitive mesh and any needed derived information about geometry and
     * general reconstruction coefficients
     *
     * The (differentiable) input of VR is merely: geometry (mesh),
     * and the weight inputs if using distributed weight;
     * output are all values derived using construct* method
     *
     */
    template <int dim>
    class VariationalReconstruction
    {
    public:
        MPI_int mRank{0};
        MPIInfo mpi;
        VRSettings settings = VRSettings{dim};
        std::shared_ptr<Geom::UnstructuredMesh> mesh;

    private:
        real volGlobal{0};             /// @brief constructed using ConstructMetrics()
        std::vector<real> volumeLocal; /// @brief constructed using ConstructMetrics()
        std::vector<real> faceArea;    /// @brief constructed using ConstructMetrics()
        std::vector<RecAtr> cellAtr;   /// @brief constructed using ConstructMetrics()
        std::vector<RecAtr> faceAtr;   /// @brief constructed using ConstructMetrics()
        tCoeffPair cellIntJacobiDet;   /// @brief constructed using ConstructMetrics()
        tCoeffPair faceIntJacobiDet;   /// @brief constructed using ConstructMetrics()
        t3VecsPair faceUnitNorm;       /// @brief constructed using ConstructMetrics()
        t3VecPair faceMeanNorm;        /// @brief constructed using ConstructMetrics()
        t3VecPair cellBary;            /// @brief constructed using ConstructMetrics()
        t3VecPair faceCent;            /// @brief constructed using ConstructMetrics()
        t3VecPair cellCent;            /// @brief constructed using ConstructMetrics()
        t3VecsPair cellIntPPhysics;    /// @brief constructed using ConstructMetrics()
        t3VecsPair faceIntPPhysics;    /// @brief constructed using ConstructMetrics()
        t3VecPair cellAlignedHBox;     /// @brief constructed using ConstructMetrics()
        t3VecPair cellMajorHBox;       /// @brief constructed using ConstructMetrics()
        t3MatPair cellMajorCoord;      /// @brief constructed using ConstructMetrics()
        t3MatPair cellInertia;         /// @brief constructed using ConstructMetrics()

        t3VecPair faceAlignedScales;     /// @brief constructed using ConstructBaseAndWeight()
        t3MatPair faceMajorCoordScale;   /// @brief constructed using ConstructBaseAndWeight() //TODO
        tVVecPair cellBaseMoment;        /// @brief constructed using ConstructBaseAndWeight()
        tVVecPair faceWeight;            /// @brief constructed using ConstructBaseAndWeight()
        tMatsPair cellDiffBaseCache;     /// @brief constructed using ConstructBaseAndWeight()
        tMatsPair faceDiffBaseCache;     /// @brief constructed using ConstructBaseAndWeight()
        tVMatPair cellDiffBaseCacheCent; /// @brief constructed using ConstructBaseAndWeight() //TODO *test
        tVMatPair faceDiffBaseCacheCent; /// @brief constructed using ConstructBaseAndWeight() //TODO *test

        tMatsPair matrixAB;        /// @brief constructed using ConstructRecCoeff()
        tVecsPair vectorB;         /// @brief constructed using ConstructRecCoeff()
        tMatsPair matrixAAInvB;    /// @brief constructed using ConstructRecCoeff()
        tMatsPair vectorAInvB;     /// @brief constructed using ConstructRecCoeff()
        tVMatPair matrixSecondary; /// @brief constructed using ConstructRecCoeff()
        tVMatPair matrixAHalf_GG;  /// @brief constructed using ConstructRecCoeff()

    public:
        VariationalReconstruction(MPIInfo nMpi, std::shared_ptr<Geom::UnstructuredMesh> nMesh)
            : mpi(nMpi), mesh(nMesh)
        {
            DNDS_assert(dim == mesh->dim);
            mRank = mesh->mRank;
        }

        void ConstructMetrics();

        using tFGetBoundaryWeight = std::function<real(Geom::t_index, int)>;

        void ConstructBaseAndWeight(const tFGetBoundaryWeight &id2faceDircWeight = [](Geom::t_index id, int iOrder)
                                    { return 1.0; });

        void ConstructRecCoeff();

        /**
         * @brief make pair with default MPI type, match **cell** layout
         *
         * @tparam TArrayPair ArrayPair's type
         * @tparam TOthers A list of additional resizing parameter types
         * @param aPair the pair to be constructed
         * @param others additional resizing parameters
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
         * @brief make pair with default MPI type, match **face** layout
         *
         * @tparam TArrayPair ArrayPair's type
         * @tparam TOthers A list of additional resizing parameter types
         * @param aPair the pair to be constructed
         * @param others additional resizing parameters
         */
        template <class TArrayPair, class... TOthers>
        void MakePairDefaultOnFace(TArrayPair &aPair, TOthers... others)
        {
            DNDS_MAKE_SSP(aPair.father, mpi);
            DNDS_MAKE_SSP(aPair.son, mpi);
            aPair.father->Resize(mesh->NumFace(), others...);
            aPair.son->Resize(mesh->NumFaceGhost(), others...);
        }

        Geom::Elem::Quadrature GetFaceQuad(index iFace) const
        {
            auto e = mesh->GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, faceAtr[iFace].intOrder};
        }

        Geom::Elem::Quadrature GetFaceQuadO1(index iFace) const
        {
            auto e = mesh->GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, 1};
        }

        Geom::Elem::Quadrature GetCellQuad(index iCell) const
        {
            auto e = mesh->GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, cellAtr[iCell].intOrder};
        }

        Geom::Elem::Quadrature GetCellQuadO1(index iCell) const
        {
            auto e = mesh->GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, 1};
        }

        bool CellIsFaceBack(index iCell, index iFace) const
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

        Geom::tPoint GetFaceNorm(index iFace, int iG)
        {
            if (iG >= 0)
                return faceUnitNorm(iFace, iG);
            else
                return faceMeanNorm[iFace];
        }

        auto GetFaceAtr(index iFace)
        {
            return faceAtr.at(iFace);
        }

        auto GetCellAtr(index iCell)
        {
            return cellAtr.at(iCell);
        }

        real GetGlobalVol() { return volGlobal; }

        Geom::tPoint GetFaceNormFromCell(index iFace, index iCell, rowsize if2c, int iG)
        {
            if (!mesh->isPeriodic)
                return GetFaceNorm(iFace, iG);
            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceNorm(iFace, iG);
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                return mesh->periodicInfo.TransVector(GetFaceNorm(iFace, iG), faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                return mesh->periodicInfo.TransVectorBack(GetFaceNorm(iFace, iG), faceID);
            return GetFaceNorm(iFace, iG);
        }

        Geom::tPoint GetFaceQuadraturePPhys(index iFace, int iG)
        {
            if (iG >= 0)
                return faceIntPPhysics(iFace, iG);
            else
                return faceCent[iFace];
        }

        Geom::tPoint GetFaceQuadraturePPhysFromCell(index iFace, index iCell, rowsize if2c, int iG)
        {
            if (!mesh->isPeriodic)
                return GetFaceQuadraturePPhys(iFace, iG);
            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceQuadraturePPhys(iFace, iG);
            if (if2c < 0)
                if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom ::FaceIDIsPeriodicMain(faceID)) // I am donor
            {
                // std::cout << iFace <<" " << iCell << " " <<if2c << std::endl;
                // std::cout << GetFaceQuadraturePPhys(iFace, iG).transpose() << std::endl;
                // std::cout << mesh->periodicInfo.TransCoord(GetFaceQuadraturePPhys(iFace, iG), faceID).transpose() << std::endl;
                // std::abort();
                return mesh->periodicInfo.TransCoord(GetFaceQuadraturePPhys(iFace, iG), faceID);
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) // I am main
                return mesh->periodicInfo.TransCoordBack(GetFaceQuadraturePPhys(iFace, iG), faceID);
            return GetFaceQuadraturePPhys(iFace, iG);
        }

        Geom::tPoint GetOtherCellBaryFromCell(
            index iCell, index iCellOther,
            index iFace)
        {
            if (!mesh->isPeriodic)
                return GetCellBary(iCellOther);

            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetCellBary(iCellOther);
            rowsize if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->periodicInfo.TransCoord(GetCellBary(iCellOther), faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->periodicInfo.TransCoordBack(GetCellBary(iCellOther), faceID);
            return GetCellBary(iCellOther);
        }

        Geom::tGPoint GetOtherCellInertiaFromCell(
            index iCell, index iCellOther,
            index iFace)
        { // structure copy of GetOtherCellBaryFromCell
            if (!mesh->isPeriodic)
                return cellInertia[iCellOther];

            auto faceID = mesh->faceElemInfo[iFace]->zone;
            if (!Geom::FaceIDIsPeriodic(faceID))
                return cellInertia[iCellOther];
            rowsize if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return mesh->periodicInfo.TransMat<3>(cellInertia[iCellOther], faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return mesh->periodicInfo.TransMatBack<3>(cellInertia[iCellOther], faceID);
            return cellInertia[iCellOther];
        }

        Geom::tPoint GetCellQuadraturePPhys(index iCell, int iG)
        {
            if (iG >= 0)
                return cellIntPPhysics(iCell, iG);
            else
                return cellCent[iCell];
        }

        Geom::tPoint GetCellBary(index iCell) { return cellBary[iCell]; }

        real GetCellVol(index iCell) { return volumeLocal[iCell]; }
        real GetFaceArea(index iFace) { return faceArea[iFace]; }

        real GetCellJacobiDet(index iCell, rowsize iG) { return cellIntJacobiDet(iCell, iG); }
        real GetFaceJacobiDet(index iFace, rowsize iG) { return faceIntJacobiDet(iFace, iG); }

        auto GetCellRecMatA(index iCell)
        {
            return matrixAB(iCell, 0);
        }

        /**
         * @brief flag = 0 means use moment data, or else use no moment (as 0)
         * pPhy must be relative to cell
         * if iFace < 0, means anywhere
         * if iFace > 0, iG == -1, means center; iG < -1, then anywhere
         */
        template <class TOut>
        void FDiffBaseValue(TOut &DiBj,
                            const Geom::tPoint &pPhy, // conventional input above
                            index iCell, index iFace, int iG, int flag = 0)
        {
            using namespace Geom;

            auto pCen = cellCent[iCell];
            tPoint pPhysicsC = pPhy - pCen;

            if (!settings.baseSettings.localOrientation)
            {
                tPoint simpleScale = cellAlignedHBox[iCell];
                if (!settings.baseSettings.anisotropicLengths)
                {
                    if constexpr (dim == 2)
                        simpleScale({0, 1}).setConstant(simpleScale({0, 1}).array().maxCoeff());
                    else
                        simpleScale.setConstant(simpleScale.array().maxCoeff());
                }
                tPoint pPhysicsCScaled = pPhysicsC.array() / simpleScale.array();
                if constexpr (dim == 2)
                    FPolynomialFill2D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), DiBj.rows(), DiBj.cols());
                else
                    FPolynomialFill3D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), DiBj.rows(), DiBj.cols());
            }
            else
            {
                tPoint simpleScale = cellMajorHBox[iCell];
                if (!settings.baseSettings.anisotropicLengths)
                {
                    if constexpr (dim == 2)
                        simpleScale({0, 1}).setConstant(simpleScale({0, 1}).array().maxCoeff());
                    else
                        simpleScale.setConstant(simpleScale.array().maxCoeff());
                }
                // std::cout << simpleScale.transpose() << std::endl;
                tPoint pPhysicsCMajor = cellMajorCoord[iCell].transpose() * pPhysicsC;
                tPoint pPhysicsCScaled = pPhysicsCMajor.array() / simpleScale.array();
                if constexpr (dim == 2)
                    FPolynomialFill2D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), DiBj.rows(), DiBj.cols());
                else
                    FPolynomialFill3D(DiBj, pPhysicsCScaled(0), pPhysicsCScaled(1), pPhysicsCScaled(2), simpleScale(0), simpleScale(1), simpleScale(2), DiBj.rows(), DiBj.cols());
                tGPoint dXijdxi = cellMajorCoord[iCell];
                ConvertDiffsLinMap<dim>(DiBj, dXijdxi);
            }

            if (flag == 0)
            {
                auto baseMoment = cellBaseMoment[iCell];
                DiBj(0, Eigen::all) -= baseMoment.transpose();
            }
        }

        /**
         * @brief if if2c < 0, then calculated, if maxDiff == 255, then seen as all diffs
         * if iFace < 0, then seen as cell int points; if iG < 1, then seen as center
         * @todo : divide GetIntPointDiffBaseValue into different calls
         * @warning maxDiff is max(diffList) + 1 not len(difflist)
         * @todo:  //TODO add support for rotational periodic boundary!
         */
        template <class TList>
        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>
        GetIntPointDiffBaseValue(
            index iCell, index iFace, rowsize if2c, index iG,
            TList &&diffList = Eigen::all,
            uint8_t maxDiff = UINT8_MAX)
        {
            if (iFace >= 0)
            {
                maxDiff = std::min(maxDiff, faceAtr[iFace].NDIFF);
                if (if2c < 0)
                    if2c = CellIsFaceBack(iCell, iFace) ? 0 : 1;
                if (settings.cacheDiffBase && maxDiff <= settings.cacheDiffBaseSize)
                {
                    // auto gFace = this->GetFaceQuad(iFace);

                    if (iG >= 0)
                    {
                        return faceDiffBaseCache(iFace, iG + (faceDiffBaseCache.RowSize(iFace) / 2) * if2c)(
                            std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                    }
                    else
                    {
                        int maxNDOF = faceDiffBaseCacheCent[iFace].cols() / 2;
                        return faceDiffBaseCacheCent[iFace](
                            std::forward<TList>(diffList),
                            Eigen::seq(if2c * maxNDOF + 1,
                                       if2c * maxNDOF + maxNDOF - 1));
                    }
                }
                else
                {
                    // Actual computing:
                    ///@todo //!!!!TODO: take care of periodic case
                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                    dbv.resize(maxDiff, cellAtr[iCell].NDOF);
                    FDiffBaseValue(dbv, GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG), iCell, iFace, iG, 0);
                    return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                }
            }
            else
            {
                maxDiff = std::min(maxDiff, cellAtr[iCell].NDIFF);
                if (settings.cacheDiffBase && maxDiff <= settings.cacheDiffBaseSize)
                {
                    if (iG >= 0)
                    {
                        return cellDiffBaseCache(iCell, iG)(
                            std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                    }
                    else
                    {
                        return cellDiffBaseCacheCent[iCell](
                            std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
                    }
                }
                else
                {
                    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                    dbv.resize(maxDiff, cellAtr[iCell].NDOF);
                    FDiffBaseValue(dbv, GetCellQuadraturePPhys(iCell, iG), iCell, -1, iG, 0);
                    return dbv(std::forward<TList>(diffList), Eigen::seq(Eigen::fix<1>, Eigen::last));
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

        template <class TDiffIDerived, class TDiffJDerived>
        auto FFaceFunctional(
            const Eigen::MatrixBase<TDiffIDerived> &DiffI, const Eigen::MatrixBase<TDiffJDerived> &DiffJ,
            index iFace, index iG, index iCellL, index iCellR)
        {
            using namespace Geom;
            Eigen::Vector<real, Eigen::Dynamic> wgd = faceWeight[iFace].array().square();
            DNDS_assert(DiffI.rows() == DiffJ.rows());
            int cnDiffs = DiffI.rows();

            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> Conj;
            Conj.resize(DiffI.cols(), DiffJ.cols());
            Conj.setZero();

            //* PJH - rotation invariant scheme
            tPoint faceLV;
            switch (settings.functionalSettings.scaleType)
            {
            case VRSettings::FunctionalSettings::BaryDiff:
                faceLV = faceAlignedScales[iFace];
                break;
            case VRSettings::FunctionalSettings::MeanAACBB:
                faceLV = faceAlignedScales[iFace];
                break;
            default:
                DNDS_assert(false);
            }

            // real faceL = (faceLV.array().maxCoeff());
            // * warning component_3 of the scale vector in 2D is forced to 1! not 0!
            real faceL = 0;
            if (settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::MeanAACBB)
                faceL = std::sqrt(faceLV(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).array().square().mean());
            if (settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::BaryDiff)
                faceL = faceLV(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).norm();
            real faceLOrig = faceL;
            faceL *= settings.functionalSettings.scaleMultiplier;

            // std::cout << DiffI.transpose() << "\n"
            //           << DiffJ.transpose() << std::endl;
            // std::cout << faceL << std::endl;
            // std::cout << wgd.transpose() << std::endl;
            // std::abort();
            // std::cout << "old len " << std::sqrt(faceLV.array().square().mean()) << std::endl;

            if (!settings.functionalSettings.useAnisotropicFunctional)
            {
#ifdef USE_ECCENTRIC_COMB_POW_2
#define __POWV 2
#else
#define __POWV 1
#endif
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
                                    NormSymDiffOrderTensorV<2, 3, __POWV>(
                                        DiffI({6, 7, 8, 9}, {i}),
                                        DiffJ({6, 7, 8, 9}, {j})) *
                                    wgd(3) * std::pow(faceL, 3 * 2);
                            case 6:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<2, 2, __POWV>(
                                        DiffI({3, 4, 5}, {i}),
                                        DiffJ({3, 4, 5}, {j})) *
                                    wgd(2) * std::pow(faceL, 2 * 2);
                            case 3:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<2, 1, __POWV>(
                                        DiffI({1, 2}, {i}),
                                        DiffJ({1, 2}, {j})) *
                                    wgd(1) * std::pow(faceL, 1 * 2);
                            case 1:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<2, 0, __POWV>(
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
                                    NormSymDiffOrderTensorV<3, 3, __POWV>(
                                        DiffI({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, {i}),
                                        DiffJ({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, {j})) *
                                    wgd(3) * std::pow(faceL, 3 * 2);
                            case 10:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<3, 2, __POWV>(
                                        DiffI({4, 5, 6, 7, 8, 9}, {i}),
                                        DiffJ({4, 5, 6, 7, 8, 9}, {j})) *
                                    wgd(2) * std::pow(faceL, 2 * 2);
                            case 4:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<3, 1, __POWV>(
                                        DiffI({1, 2, 3}, {i}),
                                        DiffJ({1, 2, 3}, {j})) *
                                    wgd(1) * std::pow(faceL, 1 * 2);
                            case 1:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<3, 0, __POWV>(
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
            }
            else
            {
                using TMatCopy = Eigen::Matrix<
                    real,
                    Eigen::MatrixBase<TDiffIDerived>::RowsAtCompileTime,
                    Eigen::MatrixBase<TDiffIDerived>::ColsAtCompileTime>;
                TMatCopy DiffI_Norm = DiffI;
                TMatCopy DiffJ_Norm = DiffJ;
                tGPoint coordTrans = faceMajorCoordScale[iFace].transpose() *
                                     settings.functionalSettings.scaleMultiplier;
                {
                    // tPoint norm = this->GetFaceNorm(iFace, -1);
                    // real normScale = (coordTrans * norm).norm();
                    // coordTrans(0, Eigen::all) = norm.transpose() * faceL;
                    // coordTrans({1, 2}, Eigen::all).setZero();
                }
                {
                    // coordTrans = Geom::NormBuildLocalBaseV<3>(norm).transpose() * faceL;
                }
                {
                    // coordTrans *=  (2 * std::sqrt(3));
                    // tPoint lengths = coordTrans.rowwise().norm();
                    // // tPoint lengthsNew = lengths.array() * faceL / (faceL + lengths.array());
                    // tPoint lengthsNew = lengths.array().min(faceL);
                    // coordTrans.array().colwise() *= lengthsNew.array() / (lengths.array() + verySmallReal);

                    // // if (lengths(0) > 1e2 * lengths(1) || lengths(1) > 1e2 * lengths(0))
                    // // {
                    // //     std::cout << "FACE " << iFace << std::endl;
                    // //     std::cout << lengths << std::endl;
                    // //     std::cout << faceL << std::endl;
                    // // }
                }
                {
                    // tPoint norm = this->GetFaceNorm(iFace, -1);
                    // auto getCellFaceMajorPNorm = [&](index iCell) -> tPoint
                    // {
                    //     tGPoint cellMajorCoordTrans = this->cellMajorCoord[iCell].transpose().rowwise().normalized();
                    //     tPoint normCos = (cellMajorCoordTrans * norm).array().abs();
                    //     tPoint pNorm;
                    //     if (normCos(0) < normCos(1))
                    //         pNorm = cellMajorCoordTrans(1, Eigen::all).transpose();
                    //     else
                    //         pNorm = cellMajorCoordTrans(0, Eigen::all).transpose();
                    //     return pNorm;
                    // };
                    // tPoint pNormL = getCellFaceMajorPNorm(iCellL);
                    // tPoint pNormR = getCellFaceMajorPNorm(iCellR);
                    // auto getProjection = [](tPoint n) -> tGPoint
                    // {
                    //     tGPoint U = Geom::NormBuildLocalBaseV<3>(n).transpose();
                    //     return U.transpose() * Eigen::Vector3d{1, 0, 0}.asDiagonal() * U;
                    // };
                    // ConvertDiffsLinMap<dim>(DiffI_Norm, getProjection(pNormL));
                    // ConvertDiffsLinMap<dim>(DiffJ_Norm, getProjection(pNormR));

                    // coordTrans(0, Eigen::all) = norm.transpose() * faceL;
                    // coordTrans({1, 2}, Eigen::all).setZero();
                }
                {
                    tPoint norm = this->GetFaceNorm(iFace, -1);
                    real areaL = this->GetFaceArea(iFace);
                    if constexpr (dim == 3)
                        areaL = std::sqrt(areaL);
                    // real tw = 1. / std::max({1.0, areaL / (faceLOrig + 0.001 * areaL)});
                    real tw = 1. / std::min({1.0, faceLOrig / (areaL + 0.001 * faceLOrig)});
                    // std::cout << tw << std::endl;

                    coordTrans = Geom::NormBuildLocalBaseV<3>(norm).transpose() * faceL;
                    coordTrans({1, 2}, Eigen::all) *= tw;
                }

                // std::cout << "face " << iFace << std::endl;
                // std::cout << coordTrans << std::endl;
                ConvertDiffsLinMap<dim>(DiffI_Norm, coordTrans);
                ConvertDiffsLinMap<dim>(DiffJ_Norm, coordTrans);

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
                                    NormSymDiffOrderTensorV<2, 3, __POWV>(
                                        DiffI_Norm({6, 7, 8, 9}, {i}),
                                        DiffJ_Norm({6, 7, 8, 9}, {j})) *
                                    wgd(3);
                            case 6:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<2, 2, __POWV>(
                                        DiffI_Norm({3, 4, 5}, {i}),
                                        DiffJ_Norm({3, 4, 5}, {j})) *
                                    wgd(2);
                            case 3:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<2, 1, __POWV>(
                                        DiffI_Norm({1, 2}, {i}),
                                        DiffJ_Norm({1, 2}, {j})) *
                                    wgd(1);
                            case 1:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<2, 0, __POWV>(
                                        DiffI_Norm({0}, {i}),
                                        DiffJ_Norm({0}, {j})) * //! i, j needed in {}!
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
                                    NormSymDiffOrderTensorV<3, 3, __POWV>(
                                        DiffI_Norm({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, {i}),
                                        DiffJ_Norm({10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, {j})) *
                                    wgd(3);
                            case 10:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<3, 2, __POWV>(
                                        DiffI_Norm({4, 5, 6, 7, 8, 9}, {i}),
                                        DiffJ_Norm({4, 5, 6, 7, 8, 9}, {j})) *
                                    wgd(2);
                            case 4:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<3, 1, __POWV>(
                                        DiffI_Norm({1, 2, 3}, {i}),
                                        DiffJ_Norm({1, 2, 3}, {j})) *
                                    wgd(1);
                            case 1:
                                Conj(i, j) +=
                                    NormSymDiffOrderTensorV<3, 0, __POWV>(
                                        DiffI_Norm({0}, {i}),
                                        DiffJ_Norm({0}, {j})) *
                                    wgd(0);
                                break;
                            }
                        }
                }
#ifdef __POWV
#undef __POWV
#endif
            }
            return Conj;
        }

        real GetGreenGauss1WeightOnCell(index iCell)
        {
            if (settings.functionalSettings.greenGaussSpacial == 0)
            {
                real AR = GetCellAR(iCell);
                real v = std::max(0.0, std::log(AR));
                return settings.functionalSettings.greenGauss1Weight *
                       std::pow(std::tanh(v / 4), 3);
            }
            else
            {
                return settings.functionalSettings.greenGauss1Weight;
            }
        }

        real GetCellAR(index iCell)
        {
            static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);
            auto lens = this->cellMajorHBox[iCell](Seq012);
            return (lens.maxCoeff() + verySmallReal) / (lens.minCoeff() + verySmallReal);
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

        template <int nVarsFixed>
        using TFBoundary = std::function<Eigen::Vector<real, nVarsFixed>(
            const Eigen::Vector<real, nVarsFixed> &, // UBL
            const Eigen::Vector<real, nVarsFixed> &, // UMEAN
            index, index,                            // iCell, iFace
            const Geom::tPoint &,                    // Norm
            const Geom::tPoint &,                    // pPhy
            Geom::t_index fType                      // fCode
            )>;

        template <int nVarsFixed>
        using TFBoundaryDiff = std::function<Eigen::Vector<real, nVarsFixed>(
            const Eigen::Vector<real, nVarsFixed> &, // UBL
            const Eigen::Vector<real, nVarsFixed> &, // dUMEAN
            const Eigen::Vector<real, nVarsFixed> &, // UMEAN
            index, index,                            // iCell, iFace
            const Geom::tPoint &,                    // Norm
            const Geom::tPoint &,                    // pPhy
            Geom::t_index fType                      // fCode
            )>;

        template <int nVarsFixed>
        void DoReconstruction2nd(
            tURec<nVarsFixed> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method);

        /**
         * @brief do reconstruction iteration
         * if recordInc, value in the output array is actually defined as :
         * $$
         * -(A_{i}^{-1}B_{ij}ur_j +  A_{i}^{-1}b_{i}) + ur_i
         * $$
         * which is the RHS of Block-Jacobi preconditioned system
         *
         * @param FBoundary Vec F(const Vec &uL, const tPoint &unitNorm, const tPoint &p, t_index faceID),
         * with Vec == Eigen::Vector<real, nVarsFixed>
         * @warning mind that uRec could be overwritten
         */
        template <int nVarsFixed>
        void DoReconstructionIter(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            bool putIntoNew = false,
            bool recordInc = false);
        /***********************************************************/

        /**
         * @brief puts into uRecNew with Mat * uRecDiff; uses the Block-jacobi preconditioned reconstruction system as Mat:
         * $$
         * ur_i = A_{i}^{-1}B_{ij}ur_j +  A_{i}^{-1}b_{i}
         * $$
         * @param FBoundary Vec F(const Vec &uL, const Vec& dUL, const tPoint &unitNorm, const tPoint &p, t_index faceID),
         * with Vec == Eigen::Vector<real, nVarsFixed>
         * uRecDiff should be untouched
         */
        template <int nVarsFixed>
        void DoReconstructionIterDiff(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecDiff,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff);
        /***********************************************************/

        /**
         * @brief do a SOR iteration from uRecNew, with uRecInc as the RHSterm of Block-Jacobi preconditioned system
         */
        template <int nVarsFixed>
        void DoReconstructionIterSOR(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecInc,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,
            bool reverse = false);

        /***********************************************************/

        /***********************************************************/

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
                    // if (iCellOther != UnInitIndex)
                    // {
                    //     uRec[iCell].setConstant(1);
                    //     uRec[iCellOther].setConstant(0);
                    //     u[iCell].setConstant(1);
                    //     u[iCellOther].setConstant(0);
                    // }
                    IJIISI.setZero();
                    gFace.IntegrationSimple(
                        IJIISI,
                        [&](auto &finc, int ig)
                        {
                            int nDiff = faceAtr[iFace].NDIFF;
                            // int nDiff = 1;
                            tPoint unitNorm = faceMeanNorm[iFace];

                            Eigen::Matrix<real, Eigen::Dynamic, nVarsSee, Eigen::DontAlign, maxNDiff, nVarsSee>
                                uRecVal(nDiff, nVarsSee), uRecValL(nDiff, nVarsSee), uRecValR(nDiff, nVarsSee), uRecValJump(nDiff, nVarsSee);
                            uRecVal.setZero(), uRecValJump.setZero();
                            uRecValL = this->GetIntPointDiffBaseValue(iCell, iFace, -1, -1, Eigen::seq(0, nDiff - 1)) *
                                       uRec[iCell](Eigen::all, varsSee);
                            uRecValL(0, Eigen::all) += u[iCell](varsSee).transpose();

                            if (iCellOther != UnInitIndex)
                            {
                                uRecValR = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, -1, Eigen::seq(0, nDiff - 1)) *
                                           uRec[iCellOther](Eigen::all, varsSee);
                                uRecValR(0, Eigen::all) += u[iCellOther](varsSee).transpose();
                                uRecVal = (uRecValL + uRecValR) * 0.5;
                                uRecValJump = (uRecValL - uRecValR) * 0.5;
                            }

                            Eigen::Matrix<real, nVarsSee, nVarsSee> IJI, ISI;
                            IJI = FFaceFunctional(uRecValJump, uRecValJump, iFace, -1, iCell, iCellOther);
                            ISI = FFaceFunctional(uRecVal, uRecVal, iFace, -1, iCell, iCellOther);

                            finc(Eigen::all, 0) = IJI.diagonal();
                            finc(Eigen::all, 1) = ISI.diagonal();

                            finc *= GetFaceArea(iFace); // don't forget this

                            // if (iCell == 12517)
                            // {
                            //     std::cout << "   === Face:   ";
                            //     std::cout << uRecValL << std::endl;
                            //     std::cout << uRecValR << std::endl;
                            //     std::cout << IJI << std::endl;
                            //     std::cout << ISI << std::endl;
                            //     std::cout << uRec[iCell] << std::endl;
                            //     std::cout << uRec[iCellOther] << std::endl;
                            //     std::cout << this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, ig, Eigen::all) << std::endl;
                            // }
                        });
                    IJIISIsum += IJIISI;
                    // if (iCell == 12517)
                    // {
                    //     std::cout << "iFace " << iFace << " iCellOther " << iCellOther << std::endl;
                    //     std::cout << IJIISI << std::endl;
                    // }
                }
                Eigen::Vector<real, nVarsSee> smoothIndicator =
                    (IJIISIsum(Eigen::all, 0).array() /
                     (IJIISIsum(Eigen::all, 1).array() + verySmallReal))
                        .matrix();
                real sImax = smoothIndicator.array().abs().maxCoeff();
                si(iCell, 0) = std::sqrt(sImax) * sqr(settings.maxOrder);
                // if (iCell == 12517)
                // {
                //     std::cout << "SUM:\n";
                //     std::cout << IJIISIsum << std::endl;
                //     std::abort();
                // }
            }
        }

        template <size_t nVarsSee, class TUREC, class TUDOF, class TFPost>
        void DoCalculateSmoothIndicatorV1(
            tScalarPair &si, TUREC &uRec, TUDOF &u,
            const std::array<int, nVarsSee> &varsSee,
            TFPost &&FPost)
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
                    // if (iCellOther != UnInitIndex)
                    // {
                    //     uRec[iCell].setConstant(1);
                    //     uRec[iCellOther].setConstant(0);
                    //     u[iCell].setConstant(1);
                    //     u[iCellOther].setConstant(0);
                    // }
                    IJIISI.setZero();
                    gFace.IntegrationSimple(
                        IJIISI,
                        [&](auto &finc, int ig)
                        {
                            tPoint unitNorm = faceMeanNorm[iFace];

                            Eigen::Matrix<real, 1, nVarsSee>
                                uRecVal(1, nVarsSee), uRecValL(1, nVarsSee), uRecValR(1, nVarsSee), uRecValJump(1, nVarsSee);
                            uRecVal.setZero(), uRecValJump.setZero();
                            uRecValL = this->GetIntPointDiffBaseValue(iCell, iFace, -1, -1, std::array<int, 1>{0}, 1) *
                                       uRec[iCell](Eigen::all, varsSee);
                            uRecValL(0, Eigen::all) += u[iCell](varsSee).transpose();
                            FPost(uRecValL);

                            if (iCellOther != UnInitIndex)
                            {
                                uRecValR = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, -1, std::array<int, 1>{0}, 1) *
                                           uRec[iCellOther](Eigen::all, varsSee);
                                uRecValR(0, Eigen::all) += u[iCellOther](varsSee).transpose();
                                FPost(uRecValR);
                                uRecVal = (uRecValL + uRecValR) * 0.5;
                                uRecValJump = (uRecValL - uRecValR) * 0.5;
                            }

                            Eigen::Matrix<real, nVarsSee, nVarsSee> IJI, ISI;
                            IJI = FFaceFunctional(uRecValJump, uRecValJump, iFace, -1, iCell, iCellOther);
                            ISI = FFaceFunctional(uRecVal, uRecVal, iFace, -1, iCell, iCellOther);

                            finc(Eigen::all, 0) = IJI.diagonal();
                            finc(Eigen::all, 1) = ISI.diagonal();

                            finc *= GetFaceArea(iFace); // don't forget this
                        });
                    IJIISIsum += IJIISI;
                }
                Eigen::Vector<real, nVarsSee> smoothIndicator =
                    (IJIISIsum(Eigen::all, 0).array() /
                     (IJIISIsum(Eigen::all, 1).array() + verySmallReal))
                        .matrix();
                real sImax = smoothIndicator.array().abs().maxCoeff();
                si(iCell, 0) = std::sqrt(sImax) * sqr(settings.maxOrder);
                // if (iCell == 12517)
                // {
                //     std::cout << "SUM:\n";
                //     std::cout << IJIISIsum << std::endl;
                //     std::abort();
                // }
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

            static const int maxNeighbour = 7;

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
                    // * safety initialization
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
                            LimStart = 5, LimEnd = 8;
                            break;
                        case 2:
                            LimStart = 2, LimEnd = 4;
                            break;
                        case 1:
                            LimStart = 0, LimEnd = 1;
                            break;
                        default:
                            LimStart = -200, LimEnd = -100;
                            DNDS_assert(false);
                        }
                    else
                        switch (cPOrder)
                        {
                        case 3:
                            LimStart = 9, LimEnd = 18;
                            break;
                        case 2:
                            LimStart = 3, LimEnd = 8;
                            break;
                        case 1:
                            LimStart = 0, LimEnd = 2;
                            break;
                        default:
                            LimStart = -200, LimEnd = -100;
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
                    // DNDS_MPI_InsertCheck(mpi, "HereAAC");
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

                            tPoint unitNorm = faceMeanNorm[iFace];

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

                            // 2 eig space :
                            auto uR = iCellAtFace ? u[iCell] : u[iCellOther];
                            auto uL = iCellAtFace ? u[iCellOther] : u[iCell];
                            auto M = FM(uL, uR, unitNorm);

                            uOtherIn = (M * uOtherIn.transpose()).transpose();
                            uThisIn = (M * uThisIn.transpose()).transpose();

                            Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                                uLimOutArray;

                            real n = settings.WBAP_nStd;
                            switch (settings.limiterBiwayAlter)
                            {
                            case 0:
                                FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            case 1:
                                FMINMOD_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            case 2:
                                FWBAP_L2_Biway_PolynomialNorm<dim, nVars_Fixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            case 3:
                                FMEMM_Biway_PolynomialNorm<dim, nVars_Fixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            default:
                                DNDS_assert_info(false, "no such limiterBiwayAlter code!");
                            }

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

        /**
         * @brief FM(uLeft,uRight,norm) gives vsize * vsize mat of Left Eigen Vectors
         *
         */
        template <class TEval, typename TFM, typename TFMI, class TUREC, class TUDOF>
        void DoLimiterWBAP_3(
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

            static const int maxNeighbour = 7;

            int cPOrder = settings.maxOrder;
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                uRecNew[iCell] = uRec[iCell];
            for (; cPOrder >= 1; cPOrder--)
            {
                int LimStart, LimEnd; // End is inclusive
                if constexpr (dim == 2)
                    switch (cPOrder)
                    {
                    case 3:
                        LimStart = 5, LimEnd = 8;
                        break;
                    case 2:
                        LimStart = 2, LimEnd = 4;
                        break;
                    case 1:
                        LimStart = 0, LimEnd = 1;
                        break;
                    default:
                        LimStart = -200, LimEnd = -100;
                        DNDS_assert(false);
                    }
                else
                    switch (cPOrder)
                    {
                    case 3:
                        LimStart = 9, LimEnd = 18;
                        break;
                    case 2:
                        LimStart = 3, LimEnd = 8;
                        break;
                    case 1:
                        LimStart = 0, LimEnd = 2;
                        break;
                    default:
                        LimStart = -200, LimEnd = -100;
                        DNDS_assert(false);
                    }
                for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                    uRecBuf[iCell] = uRecNew[iCell];
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    if ((!ifAll) &&
                        si(iCell, 0) < settings.smoothThreshold)
                    {
                        // uRecNew[iCell] = uRecBuf[iCell]; //! no copy for 3wbap!
                        continue;
                    }
                    index NRecDOF = cellAtr[iCell].NDOF - 1;
                    auto c2f = mesh->cell2face[iCell];
                    // std::vector<Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOF>> uFaces(c2f.size());
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        // * safety initialization
                        index iFace = c2f[ic2f];
                        index iCellOther = this->CellFaceOther(iCell, iFace);
                        if (iCellOther != UnInitIndex)
                        {
                            // uFaces[ic2f].resizeLike(uRec[iCellOther]);
                        }
                    }

                    std::vector<Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>>
                        uOthers;
                    Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                        uC = uRecBuf[iCell](
                            Eigen::seq(
                                LimStart,
                                LimEnd),
                            Eigen::all);
                    uOthers.reserve(maxNeighbour);
                    uOthers.push_back(uC); // using uC centered
                    // DNDS_MPI_InsertCheck(mpi, "HereAAC");
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

                            tPoint unitNorm = faceMeanNorm[iFace];

                            const auto &matrixSecondary =
                                this->GetMatrixSecondary(iCell, iFace, -1);

                            const auto &matrixSecondaryOther =
                                this->GetMatrixSecondary(iCellOther, iFace, -1);

                            // std::cout << "A"<<std::endl;
                            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOF>
                                uOtherOther = uRecBuf[iCellOther](Eigen::seq(0, NRecDOFLim - 1), Eigen::all);

                            // if (LimEnd < uOtherOther.rows() - 1) // successive SR
                            //     uOtherOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all) =
                            //         matrixSecondaryOther(Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::seq(LimEnd + 1, NRecDOFLim - 1)) *
                            //         uFaces[ic2f](Eigen::seq(LimEnd + 1, NRecDOFLim - 1), Eigen::all);

                            // std::cout << "B" << std::endl;
                            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                                uOtherIn =
                                    matrixSecondary(Eigen::seq(LimStart, LimEnd), Eigen::all) * uOtherOther;

                            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                                uThisIn =
                                    uC.matrix();

                            // 2 eig space :
                            auto uR = iCellAtFace ? u[iCell] : u[iCellOther];
                            auto uL = iCellAtFace ? u[iCellOther] : u[iCell];
                            auto M = FM(uL, uR, unitNorm);

                            uOtherIn = (M * uOtherIn.transpose()).transpose();
                            uThisIn = (M * uThisIn.transpose()).transpose();

                            Eigen::Array<real, Eigen::Dynamic, nVars_Fixed, 0, maxRecDOFBatch>
                                uLimOutArray;

                            real n = settings.WBAP_nStd;

                            switch (settings.limiterBiwayAlter)
                            {
                            case 0:
                                FWBAP_L2_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            case 1:
                                FMINMOD_Biway(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            case 2:
                                FWBAP_L2_Biway_PolynomialNorm<dim, nVars_Fixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            case 3:
                                FMEMM_Biway_PolynomialNorm<dim, nVars_Fixed>(uThisIn.array(), uOtherIn.array(), uLimOutArray, 1);
                                break;
                            default:
                                DNDS_assert_info(false, "no such limiterBiwayAlter code!");
                            }

                            // to phys space
                            auto MI = FMI(uL, uR, unitNorm);
                            uLimOutArray = (MI * uLimOutArray.matrix().transpose()).transpose().array();

                            // uFaces[ic2f](Eigen::seq(LimStart, LimEnd), Eigen::all) = uLimOutArray.matrix();
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
                uRecNew.trans.startPersistentPull();
                uRecNew.trans.waitPersistentPull();
            }
            if (!putIntoNew)
            {
                for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // mind the edge
                    uRec[iCell] = uRecNew[iCell];
            }
        }
    };
}
