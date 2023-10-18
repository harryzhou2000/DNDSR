
#include "VariationalReconstruction.hpp"
#include "omp.h"
#include "DNDS/HardEigen.hpp"

namespace DNDS::CFV
{
    template <int dim>
    void
    VariationalReconstruction<dim>::
        ConstructMetrics()
    {
        using namespace Geom;
        using namespace Geom::Elem;

        /***************************************/
        // get volumes
        real sumVolume{0};
        volumeLocal.resize(mesh->NumCellProc());
        cellAtr.resize(mesh->NumCellProc());
        this->MakePairDefaultOnCell(cellIntJacobiDet);
        this->MakePairDefaultOnCell(cellBary);
        this->MakePairDefaultOnCell(cellCent);
        this->MakePairDefaultOnCell(cellIntPPhysics);
        this->MakePairDefaultOnCell(cellAlignedHBox);
        this->MakePairDefaultOnCell(cellMajorHBox);
        this->MakePairDefaultOnCell(cellMajorCoord, 3, 3);
        this->MakePairDefaultOnCell(cellInertia, 3, 3);
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            cellAtr[iCell].intOrder = settings.intOrder;
            auto eCell = mesh->GetCellElement(iCell);
            auto qCell = Quadrature{eCell, cellAtr[iCell].intOrder};
            auto qCellO1 = Quadrature{eCell, 1};
            DNDS_assert(qCellO1.GetNumPoints() == 1);
            cellIntJacobiDet.ResizeRow(iCell, qCell.GetNumPoints());
            cellIntPPhysics.ResizeRow(iCell, qCell.GetNumPoints());

            tSmallCoords coordsCell;
            mesh->GetCoordsOnCell(iCell, coordsCell);
            //****** Get Int Point Det and Vol
            real v{0};
            qCell.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coordsCell, DiNj);
                    real JDet;
                    if constexpr (dim == 2)
                        JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                    else
                        JDet = J.determinant();
                    vInc = 1 * JDet;
                    cellIntJacobiDet(iCell, iG) = JDet;
                });
            volumeLocal[iCell] = v;
            if (iCell < mesh->NumCell()) // non-ghost
#ifdef DNDS_USE_OMP
#pragma omp critical
#endif
                sumVolume += v;
            //****** Get Int Point PPhy and Bary
            tPoint b{0, 0, 0};
            qCell.Integration(
                b,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    vInc = pPhy * this->GetCellJacobiDet(iCell, iG);
                    cellIntPPhysics(iCell, iG) = pPhy;
                });
            cellBary[iCell] = b / v;
            //****** Get Center
            SummationNoOp noOp;
            qCellO1.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    cellCent[iCell] = pPhy;
                });

            //****** Get HBox aligned
            tSmallCoords coordsCellC = coordsCell.colwise() - this->GetCellBary(iCell);
            DNDS_assert(coordsCellC.cols() == coordsCell.cols());
            tPoint hBox = coordsCellC.array().abs().rowwise().maxCoeff();
            if constexpr (dim == 2)
                hBox(2) = 1;
            cellAlignedHBox[iCell] = hBox;

            //****** Get Major Axis
            tJacobi inertia;
            inertia.setZero();
            qCell.Integration(
                inertia,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    tPoint pPhyC = (pPhy - cellBary[iCell]);
                    vInc = (pPhyC * pPhyC.transpose()) * cellIntJacobiDet(iCell, iG);
                });
            inertia /= this->GetCellVol(iCell);
            real inerNorm = inertia.norm();
            real inerCond = 1;
            if constexpr (dim == 2)
                inerCond = HardEigen::Eigen2x2RealSymEigenDecompositionGetCond(inertia({0, 1}, {0, 1}));
            else
                inerCond = HardEigen::Eigen3x3RealSymEigenDecompositionGetCond(inertia);

            if (inerCond < 1 + smallReal)
            {
                inertia(0, 0) += inerNorm * smallReal * 10;
                inertia(1, 1) += inerNorm * smallReal;
            }

            cellInertia[iCell] = inertia;
            tJacobi decRet;
            decRet.setIdentity();
            if constexpr (dim == 3)
                decRet = HardEigen::Eigen3x3RealSymEigenDecompositionNormalized(inertia);
            else
                decRet({0, 1}, {0, 1}) = HardEigen::Eigen2x2RealSymEigenDecompositionNormalized(inertia({0, 1}, {0, 1}));
            cellMajorCoord[iCell] = decRet;
            tSmallCoords coordsCellM = cellMajorCoord[iCell].transpose() * coordsCellC;
            tPoint hBoxM = coordsCellM.array().abs().rowwise().maxCoeff();
            if constexpr (dim == 2)
                hBoxM(2) = 1;
            cellMajorHBox[iCell] = hBoxM;
        }
        real sumVolumeAll{0};
        MPI::Allreduce(&sumVolume, &sumVolumeAll, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        if (mpi.rank == mRank)
            std::cout
                << "VariationalReconstruction<dim>::ConstructMetrics() === Sum Volume is ["
                << std::setprecision(10) << sumVolumeAll << "] " << std::endl;
        volGlobal = sumVolumeAll;
        cellIntJacobiDet.CompressBoth();
        cellIntPPhysics.CompressBoth();
        /***************************************/
        // get areas
        faceArea.resize(mesh->NumFaceProc());
        faceAtr.resize(mesh->NumFaceProc());
        this->MakePairDefaultOnFace(faceIntJacobiDet);
        this->MakePairDefaultOnFace(faceMeanNorm);
        this->MakePairDefaultOnFace(faceUnitNorm);
        this->MakePairDefaultOnFace(faceIntPPhysics);
        this->MakePairDefaultOnFace(faceCent);

#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            faceAtr[iFace].intOrder = settings.intOrder;
            auto eFace = mesh->GetFaceElement(iFace);
            auto qFace = Quadrature{eFace, faceAtr[iFace].intOrder};
            auto qFaceO1 = Quadrature{eFace, 1};
            DNDS_assert(qFaceO1.GetNumPoints() == 1);
            faceIntJacobiDet.ResizeRow(iFace, qFace.GetNumPoints());
            faceIntPPhysics.ResizeRow(iFace, qFace.GetNumPoints());
            faceUnitNorm.ResizeRow(iFace, qFace.GetNumPoints());

            tSmallCoords coords;
            mesh->GetCoordsOnFace(iFace, coords);

            //****** Get Int Point Det and Vol
            real v{0};
            qFace.Integration(
                v,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
                    real JDet;
                    if constexpr (dim == 2)
                        JDet = J(Eigen::all, 0).stableNorm();
                    else
                        JDet = J(Eigen::all, 0).cross(J(Eigen::all, 1)).stableNorm();
                    vInc = 1 * JDet;
                    faceIntJacobiDet(iFace, iG) = JDet;
                });
            faceArea[iFace] = v;

            //****** Get Int Point Norm/pPhy and Mean Norm
            tPoint n{0, 0, 0};
            qFace.Integration(
                n,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tJacobi J = Elem::ShapeJacobianCoordD01Nj(coords, DiNj);
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                    tPoint np;
                    real JDet;
                    if constexpr (dim == 2)
                        np = FacialJacobianToNormVec<2>(J);
                    else
                        np = FacialJacobianToNormVec<3>(J);
                    np.stableNormalize();
                    faceUnitNorm(iFace, iG) = np;
                    faceIntPPhysics(iFace, iG) = pPhy;
                    vInc = np * faceIntJacobiDet(iFace, iG);
                });
            faceMeanNorm[iFace] = n / v;

            //****** Get Center
            SummationNoOp noOp;
            qFaceO1.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coords, DiNj);
                    faceCent[iFace] = pPhy;
                });
            // std::cout << " ================================ " << std::endl;
            // std::cout << coords << std::endl;
            // std::cout << faceCent[iFace].transpose() << std::endl;
            // std::cout << (faceCent[iFace] - cellCent[mesh->face2cell(iFace, 0)]).transpose() << std::endl;
            // std::cout << faceMeanNorm[iFace].transpose() << std::endl;
            // std::cout << "=\n";
            // tSmallCoords coordsCell;
            // mesh->GetCoords(mesh->cell2node[mesh->face2cell(iFace, 0)], coordsCell);
            // std::cout << coordsCell << std::endl;

            /// ! if the faces and f2c are created right, and not distorting too much
            DNDS_assert_info(
                (this->GetFaceQuadraturePPhysFromCell(iFace, mesh->face2cell(iFace, 0), -1, -1) -
                 this->GetCellQuadraturePPhys(mesh->face2cell(iFace, 0), -1))
                        .dot(faceMeanNorm[iFace]) > 0,
                "face mean norm is not the same side as faceCenter - cellCenter");
        }
        faceUnitNorm.CompressBoth();
        faceIntPPhysics.CompressBoth();
        faceIntJacobiDet.CompressBoth();
    }

    template void
    VariationalReconstruction<2>::
        ConstructMetrics();
    template void
    VariationalReconstruction<3>::
        ConstructMetrics();

    template <int dim>
    void
    VariationalReconstruction<dim>::
        ConstructBaseAndWeight(const std::function<real(Geom::t_index)> &id2faceDircWeight)
    {
        using namespace Geom;
        using namespace Geom::Elem;
        int maxNDOF = GetNDof<dim>(settings.maxOrder);
        // for polynomial: ndiff = ndof
        int maxNDIFF = maxNDOF;
        /******************************/
        // *cell's moment and cache
        this->MakePairDefaultOnCell(cellBaseMoment, maxNDOF);
        if (settings.cacheDiffBase)
        {
            this->MakePairDefaultOnCell(
                cellDiffBaseCache,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF);
            this->MakePairDefaultOnCell(
                cellDiffBaseCacheCent,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF);
        }
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            cellAtr[iCell].NDIFF = maxNDIFF;
            cellAtr[iCell].NDOF = maxNDOF;
            cellAtr[iCell].relax = settings.jacobiRelax;
            auto qCell = this->GetCellQuad(iCell);
            auto qCellO1 = this->GetCellQuadO1(iCell);
            if (settings.cacheDiffBase)
            {
                cellDiffBaseCache.ResizeRow(iCell, qCell.GetNumPoints());
            }
            // std::cout << "hare" << std::endl;

            Eigen::RowVector<real, Eigen::Dynamic> m;
            m.setZero(cellAtr[iCell].NDOF);
            qCell.Integration(
                m,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    Eigen::RowVector<real, Eigen::Dynamic> vv;
                    vv.resizeLike(m);
                    this->FDiffBaseValue(vv, this->GetCellQuadraturePPhys(iCell, iG), iCell, -1, iG, 1);
                    vInc = vv * this->GetCellJacobiDet(iCell, iG);
                });
            // std::cout << m << std::endl;
            cellBaseMoment[iCell] = m.transpose() / this->GetCellVol(iCell);
            SummationNoOp noOp;
            qCell.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    if (settings.cacheDiffBase)
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                        dbv.resize(
                            std::min(cellAtr[iCell].NDIFF, settings.cacheDiffBaseSize),
                            cellAtr[iCell].NDOF);
                        this->FDiffBaseValue(dbv, this->GetCellQuadraturePPhys(iCell, iG), iCell, -1, iG, 0);
                        cellDiffBaseCache(iCell, iG) = dbv;
                    }
                });
            qCellO1.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    if (settings.cacheDiffBase)
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                        dbv.resize(
                            std::min(cellAtr[iCell].NDIFF, settings.cacheDiffBaseSize),
                            cellAtr[iCell].NDOF);
                        this->FDiffBaseValue(dbv, this->GetCellQuadraturePPhys(iCell, -1), iCell, -1, -1, 0);
                        cellDiffBaseCacheCent[iCell] = dbv;
                    }
                });
        }

        /******************************/
        // *face's weight and cache
        this->MakePairDefaultOnFace(faceWeight, settings.maxOrder + 1);
        this->MakePairDefaultOnFace(faceAlignedScales);
        this->MakePairDefaultOnFace(faceMajorCoordScale, 3, 3);
        if (settings.cacheDiffBase)
        {
            this->MakePairDefaultOnFace(
                faceDiffBaseCache,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF);
            this->MakePairDefaultOnFace(
                faceDiffBaseCacheCent,
                std::min(maxNDIFF, static_cast<int>(settings.cacheDiffBaseSize)), maxNDOF * 2);
        }
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            faceAtr[iFace].NDIFF = maxNDIFF;
            faceAtr[iFace].NDOF = 0;
            auto qFace = this->GetFaceQuad(iFace);
            auto qFaceO1 = this->GetFaceQuadO1(iFace);
            if (settings.cacheDiffBase)
                faceDiffBaseCache.ResizeRow(iFace, 2 * qFace.GetNumPoints());

            // * get bdv cache
            SummationNoOp noOp;
            for (int if2c = 0; if2c < 2; if2c++)
            {
                index iCell = mesh->face2cell(iFace, if2c);
                if (iCell == UnInitIndex)
                    continue;
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                    DNDS_assert(if2c == 0);
                else if (FaceIDIsPeriodic(mesh->GetFaceZone(iFace)))
                {
                    // TODO: handle the case with periodic
                    // DNDS_assert(false); //! do nothing?
                }
                qFace.Integration(
                    noOp,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        if (settings.cacheDiffBase)
                        {
                            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                            dbv.resize(
                                std::min(faceAtr[iFace].NDIFF, settings.cacheDiffBaseSize),
                                cellAtr[iCell].NDOF);
                            this->FDiffBaseValue(dbv, this->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG), iCell, iFace, iG, 0);
                            faceDiffBaseCache(iFace, if2c * qFace.GetNumPoints() + iG) = dbv;
                        }
                    });
                qFaceO1.Integration(
                    noOp,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        if (settings.cacheDiffBase)
                        {
                            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                            dbv.resize(
                                std::min(faceAtr[iFace].NDIFF, settings.cacheDiffBaseSize),
                                cellAtr[iCell].NDOF);
                            this->FDiffBaseValue(dbv, this->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1), iCell, iFace, -1, 0);
                            int maxNDOF = GetNDof<dim>(settings.maxOrder);
                            faceDiffBaseCacheCent[iFace](
                                Eigen::all,
                                Eigen::seq(if2c * maxNDOF,
                                           if2c * maxNDOF + maxNDOF - 1)) = dbv;
                        }
                    });
            }

            // *get face (derivatives) scale: cell average AlignedHBox mode
            //! note: if use cell-2-cell distance, note periodic, use wrapped call here
            tPoint faceScale{0, 0, 0};
            int nF2C{0};
            for (int if2c = 0; if2c < 2; if2c++)
            {
                index iCell = mesh->face2cell(iFace, if2c);
                if (iCell == UnInitIndex)
                    continue;
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                    DNDS_assert(if2c == 0);
                else if (FaceIDIsPeriodic(mesh->GetFaceZone(iFace)))
                {
                    // TODO: handle the case with periodic
                    // DNDS_assert(false); // !do nothing for now
                }
                faceScale += cellAlignedHBox[iCell];
                nF2C++;
            }
            DNDS_assert(nF2C > 0);
            faceScale /= nF2C;
            faceAlignedScales[iFace] = faceScale;
            Geom::tPoint faceBaryDiffV;
            if (mesh->face2cell(iFace, 1) != UnInitIndex)
            {
                faceBaryDiffV =
                    this->GetOtherCellBaryFromCell(mesh->face2cell(iFace, 0),
                                                   mesh->face2cell(iFace, 1), iFace) -
                    this->GetCellBary(mesh->face2cell(iFace, 0));
            }
            else
            {
                Geom::tPoint vCB = this->GetFaceQuadraturePPhysFromCell(
                                       iFace,
                                       mesh->face2cell(iFace, 0), 0, -1) -
                                   this->GetCellBary(mesh->face2cell(iFace, 0));
                auto uNorm = this->GetFaceNormFromCell(
                    iFace,
                    mesh->face2cell(iFace, 0), 0, -1);
                faceBaryDiffV =
                    2 * vCB.dot(uNorm) *
                    this->GetFaceNorm(iFace, -1);
            }
            Geom::tPoint faceBaryDiffL, faceBaryDiffR;
            faceBaryDiffL =
                this->GetCellBary(mesh->face2cell(iFace, 0)) -
                this->GetFaceQuadraturePPhysFromCell(iFace, mesh->face2cell(iFace, 0), 0, -1);
            faceBaryDiffR = faceBaryDiffV + faceBaryDiffL;
            // std::cout << faceBaryDiffV.transpose() << std::endl;
            // std::cout << faceBaryDiffV.norm() << std::endl;
            real volL, volR;
            volL = volR = std::pow(GetCellVol(mesh->face2cell(iFace, 0)) + verySmallReal, settings.functionalSettings.inertiaWeightPower);
            Geom::tGPoint cellInertiaL = cellInertia[mesh->face2cell(iFace, 0)] * volL;
            Geom::tGPoint cellInertiaR = cellInertiaL;
            if (mesh->face2cell(iFace, 1) != UnInitIndex)
            {
                volR = std::pow(GetCellVol(mesh->face2cell(iFace, 1)) + verySmallReal, settings.functionalSettings.inertiaWeightPower);
                cellInertiaR = this->GetOtherCellInertiaFromCell(
                                   mesh->face2cell(iFace, 0),
                                   mesh->face2cell(iFace, 1), iFace) *
                               volR;
            }
            Geom::tGPoint faceInertiaC =
                (cellInertiaL + cellInertiaR) / (volL + volR);

            Geom::tGPoint faceCoord;
            faceCoord.setIdentity();
            if constexpr (dim == 3)
                faceCoord = HardEigen::Eigen3x3RealSymEigenDecomposition(faceInertiaC);
            else
                faceCoord({0, 1}, {0, 1}) = HardEigen::Eigen2x2RealSymEigenDecomposition(faceInertiaC({0, 1}, {0, 1}));
            faceMajorCoordScale[iFace] = faceCoord;
            if (settings.functionalSettings.anisotropicType == VRSettings::FunctionalSettings::InertiaCoordBB)
            {
                Geom::tGPoint faceCoordNorm = faceCoord.colwise().normalized();
                tSmallCoords coords, coordsL, coordsR;
                mesh->GetCoordsOnFace(iFace, coords);
                coords.colwise() -= faceCent[iFace];
                coordsL = coords.colwise() - faceBaryDiffL;
                coordsR = coords.colwise() - faceBaryDiffR;
                coordsL = faceCoordNorm.transpose() * coordsL;
                coordsR = faceCoordNorm.transpose() * coordsR;
                Geom::tPoint faceCoordBB =
                    coordsL.array().abs().rowwise().maxCoeff().max(
                        coordsR.array().abs().rowwise().maxCoeff());

                faceMajorCoordScale[iFace] = faceCoordNorm * faceCoordBB.asDiagonal();
            }
            // std::cout << "Face scale: ";
            // std::cout << faceCent[iFace] << std::endl;
            // std::cout << faceInertiaC << std::endl;
            // std::cout
            //     << faceCoord << "\n"
            //     << std::endl;
            if (settings.functionalSettings.scaleType == VRSettings::FunctionalSettings::BaryDiff)
            {
                faceAlignedScales[iFace] = faceBaryDiffV;
                if constexpr (dim == 2)
                    faceAlignedScales[iFace](2) = 1;
            }

            // *get geom weight ic2f
            real wg = 1;
            if (settings.functionalSettings.geomWeightScheme == VRSettings::FunctionalSettings::HQM_SD)
                wg = std::pow(std::pow(this->GetFaceArea(iFace), 1. / real(dim - 1)) / faceBaryDiffV.norm(), settings.functionalSettings.geomWeightPower * 0.5);
            if (settings.functionalSettings.geomWeightScheme == VRSettings::FunctionalSettings::SD_Power)
                wg = std::pow(this->GetFaceArea(iFace), settings.functionalSettings.geomWeightPower1 * 0.5) *
                     std::pow(faceBaryDiffV.norm(), settings.functionalSettings.geomWeightPower2 * 0.5);

            // *get dir weight
            Eigen::Vector<real, Eigen::Dynamic> wd;
            wd.resize(settings.maxOrder + 1);
            switch (settings.functionalSettings.dirWeightScheme)
            {
            case VRSettings::FunctionalSettings::Factorial:
                for (int p = 0; p < wd.size(); p++)
                    wd[p] = 1. / factorials[p];
                break;
            case VRSettings::FunctionalSettings::HQM_OPT:
                switch (settings.maxOrder)
                {
                case 1:
                    wd[0] = wd[1] = 1;
                    break;
                case 2:
                    wd[0] = 1, wd[1] = 0.4643, wd[2] = 0.1559;
                    break;
                case 3:
                    wd[0] = 1, wd[1] = .5295, wd[2] = wd[3] = .2117;
                    break;
                default:
                    DNDS_assert(false);
                    break;
                }
                break;
            case VRSettings::FunctionalSettings::ManualDirWeight:
                switch (settings.maxOrder)
                {
                case 3:
                    wd[3] = settings.functionalSettings.manualDirWeights(3);
                case 2:
                    wd[2] = settings.functionalSettings.manualDirWeights(2);
                case 1:
                    wd[1] = settings.functionalSettings.manualDirWeights(1);
                    wd[0] = settings.functionalSettings.manualDirWeights(0);
                    break;
                default:
                    DNDS_assert(false);
                    break;
                }
                break;
            default:
                DNDS_assert(false);
                break;
            }

            if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                wd(Eigen::seq(1, Eigen::last)).setZero(), wd *= id2faceDircWeight(mesh->GetFaceZone(iFace)); // customizable
            faceWeight[iFace] = wd * wg;
        }

        if (settings.cacheDiffBase)
        {
            faceDiffBaseCache.CompressBoth();
            cellDiffBaseCache.CompressBoth();

            // faceDiffBaseCache.trans.pullOnce(); //!err: need adding comm preparation first
            // faceDiffBaseCacheCent.trans.pullOnce(); //!err: need adding comm preparation first
        }

        // faceWeight.trans.pullOnce(); //!err: need adding comm preparation first
        // faceAlignedScales.trans.pullOnce(); //!err: need adding comm preparation first
        // faceMajorCoordScale.trans.pullOnce(); //!err: need adding comm preparation first
    }

    template void
    VariationalReconstruction<2>::
        ConstructBaseAndWeight(const std::function<real(Geom::t_index)> &id2faceDircWeight);
    template void
    VariationalReconstruction<3>::
        ConstructBaseAndWeight(const std::function<real(Geom::t_index)> &id2faceDircWeight);

    template <int dim>
    void
    VariationalReconstruction<dim>::ConstructRecCoeff()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        int maxNDOF = GetNDof<dim>(settings.maxOrder);
        this->MakePairDefaultOnCell(matrixAB, maxNDOF - 1, maxNDOF - 1);
        this->MakePairDefaultOnCell(matrixAAInvB, maxNDOF - 1, maxNDOF - 1);
        this->MakePairDefaultOnCell(vectorB, maxNDOF - 1, 1);
        this->MakePairDefaultOnCell(vectorAInvB, maxNDOF - 1, 1);
        real maxCond = 0.0;
#ifdef DNDS_USE_OMP
#pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++) // only non-ghost
        {
            matrixAB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell) + 1);
            matrixAAInvB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell) + 1);
            vectorB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell));
            vectorAInvB.ResizeRow(iCell, mesh->cell2face.RowSize(iCell));

            //*get A
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> A;
            A.setZero(cellAtr[iCell].NDOF - 1, cellAtr[iCell].NDOF - 1);
            for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
            {
                index iFace = mesh->cell2face(iCell, ic2f);
                auto qFace = this->GetFaceQuad(iFace);
                qFace.Integration(
                    A,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        decltype(A) DiffI = this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, Eigen::all);
                        vInc = this->FFaceFunctional(DiffI, DiffI, iFace, iG, iCell, iCell);
                        vInc *= this->GetFaceJacobiDet(iFace, iG);
                        // if (iCell == 71)
                        // {
                        // std::cout << "DI\n"
                        //           << DiffI << std::endl;
                        // }
                    });
                // std::cout << faceAlignedScales[iFace] << std::endl;
                // std::cout << "face "<< faceWeight[iFace].transpose() << std::endl;
            }
            decltype(A) AInv;
            real aCond = HardEigen::EigenLeastSquareInverse_Filtered(A, AInv);
            matrixAB(iCell, 0) = A;
            matrixAAInvB(iCell, 0) = AInv;

            maxCond = std::max(aCond, maxCond);

            // if (iCell == 71)
            // {
            // if (std::abs(A(0, 0) - 0.2083333333) > 1e-5)
            // {
            //     std::cout << "=================" << std::endl;
            //     std::cout << A << std::endl;
            //     std::cout << cellCent[iCell] << std::endl;
            // }
            // std::abort();
            // }
            //*get B
            for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
            {
                index iFace = mesh->cell2face(iCell, ic2f);
                auto qFace = this->GetFaceQuad(iFace);
                index iCellOther = CellFaceOther(iCell, iFace);
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                {
                    DNDS_assert(iCellOther == UnInitIndex);
                    continue;
                    // if is periodic, already handled correctly
                }
                Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> B;
                B.setZero(cellAtr[iCell].NDOF - 1, cellAtr[iCellOther].NDOF - 1);
                qFace.Integration(
                    B,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        decltype(B) DiffI = this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, Eigen::all);
                        decltype(B) DiffJ = this->GetIntPointDiffBaseValue(iCellOther, iFace, -1, iG, Eigen::all);
                        vInc = this->FFaceFunctional(DiffI, DiffJ, iFace, iG, iCell, iCellOther);
                        vInc *= this->GetFaceJacobiDet(iFace, iG);
                    });
                matrixAB(iCell, 1 + ic2f) = B;
                matrixAAInvB(iCell, 1 + ic2f) = AInv * B;
            }

            //*get b
            for (int ic2f = 0; ic2f < mesh->cell2face.RowSize(iCell); ic2f++)
            {
                index iFace = mesh->cell2face(iCell, ic2f);
                auto qFace = this->GetFaceQuad(iFace);
                Eigen::Matrix<real, Eigen::Dynamic, 1> b;
                b.setZero(cellAtr[iCell].NDOF - 1, 1);
                qFace.Integration(
                    b,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        Eigen::RowVector<real, Eigen::Dynamic> DiffI =
                            this->GetIntPointDiffBaseValue(iCell, iFace, -1, iG, std::array<int, 1>{0}, 1);
                        vInc = this->FFaceFunctional(DiffI, Eigen::MatrixXd::Ones(1, 1), iFace, iG, iCell, iCell);
                        vInc *= this->GetFaceJacobiDet(iFace, iG);
                    });
                vectorB(iCell, ic2f) = b;
                vectorAInvB(iCell, ic2f) = AInv * b;
            }
            DNDS_assert(AInv.allFinite());
            // std::cout << "=============" << std::endl;
            // std::cout << AInv << std::endl;
            // std::abort();
        }
        matrixAB.CompressBoth();
        matrixAAInvB.CompressBoth();
        vectorAInvB.CompressBoth();
        vectorB.CompressBoth();

        // Get Secondary matrices
        this->MakePairDefaultOnFace(matrixSecondary, maxNDOF - 1, (maxNDOF - 1) * 2);
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            index iCellL = mesh->face2cell(iFace, 0);
            index iCellR = mesh->face2cell(iFace, 1);
            if (iCellR == UnInitIndex) // only for faces with two
                continue;
            // get dbv from 1st derivative to last
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> DiffI = this->GetIntPointDiffBaseValue(iCellL, iFace, 0, -1, Eigen::seq(1, Eigen::last));
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> DiffJ = this->GetIntPointDiffBaseValue(iCellR, iFace, 1, -1, Eigen::seq(1, Eigen::last));
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> M2_L2R, M2_R2L;
            HardEigen::EigenLeastSquareSolve(DiffI, DiffJ, M2_R2L);
            HardEigen::EigenLeastSquareSolve(DiffJ, DiffI, M2_L2R);
            // TODO: cleanse M2s' lower triangle
            int maxNDOFM1 = matrixSecondary[iFace].cols() / 2;
            matrixSecondary[iFace](
                Eigen::all, Eigen::seq(
                                0 * maxNDOFM1 + 0,
                                0 * maxNDOFM1 + maxNDOFM1 - 1)) = M2_R2L;
            matrixSecondary[iFace](
                Eigen::all, Eigen::seq(
                                1 * maxNDOFM1 + 0,
                                1 * maxNDOFM1 + maxNDOFM1 - 1)) = M2_L2R;
            // std::cout << "DiffI\n"
            //           << DiffI << std::endl;
            // std::cout << "DiffJ\n"
            //           << DiffJ << std::endl;

            // std::cout << "M2_R2L\n";
            // std::cout << this->GetMatrixSecondary(-1, iFace, 0) << std::endl;
            // std::cout << "M2_L2R\n";
            // std::cout << this->GetMatrixSecondary(-1, iFace, 1) << std::endl;
            // std::abort();
        }
        real maxCondAll = 0;
        MPI::Allreduce(&maxCond, &maxCondAll, 1, DNDS_MPI_REAL, MPI_MAX, mesh->getMPI().comm);
        if (mesh->getMPI().rank == 0)
            log() << std::scientific << std::setprecision(3)
                  << "VariationalReconstruction<dim>::ConstructRecCoeff() === A cond Max: [ " << maxCondAll << "] " << std::endl;
    }

    template void
    VariationalReconstruction<2>::ConstructRecCoeff();

    template void
    VariationalReconstruction<3>::ConstructRecCoeff();
}