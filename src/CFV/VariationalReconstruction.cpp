
#include "VariationalReconstruction.hpp"
#include "omp.h"

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
#ifdef DNDS_USE_OMP
        omp_set_num_threads(MPIWorldSize() == 1 ? omp_get_num_procs() : 1);
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
            mesh->GetCoords(mesh->cell2node[iCell], coordsCell);
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
                sumVolume += v;
            //****** Get Int Point PPhy and Bary
            tPoint b{0, 0, 0};
            qCell.Integration(
                b,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    tPoint pPhy = Elem::PPhysicsCoordD01Nj(coordsCell, DiNj);
                    vInc = pPhy * cellIntJacobiDet(iCell, iG);
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
            tSmallCoords coordsCellC = coordsCell.colwise() - cellBary[iCell];
            DNDS_assert(coordsCellC.cols() == coordsCell.cols());
            tPoint hBox = coordsCellC.array().abs().rowwise().maxCoeff();
            cellAlignedHBox[iCell] = hBox;
        }
        real sumVolumeAll{0};
        MPI_Allreduce(&sumVolume, &sumVolumeAll, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        if (mpi.rank == mRank)
            std::cout
                << "VariationalReconstruction<dim>::ConstructMetrics() === Sum Volume is ["
                << std::setprecision(10) << sumVolumeAll << "] " << std::endl;
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
        omp_set_num_threads(MPIWorldSize() == 1 ? omp_get_num_procs() : 1);
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
            mesh->GetCoords(mesh->face2node[iFace], coords);

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
                (faceCent[iFace] - cellCent[mesh->face2cell(iFace, 0)]).dot(faceMeanNorm[iFace]) > 0,
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
    VariationalReconstruction<dim>::ConstructBaseAndWeight()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        int maxNDOF = -1;
        switch (settings.maxOrder)
        {
        case 0:
            maxNDOF = PolynomialNDOF<dim, 0>();
            break;
        case 1:
            maxNDOF = PolynomialNDOF<dim, 1>();
            break;
        case 2:
            maxNDOF = PolynomialNDOF<dim, 2>();
            break;
        case 3:
            maxNDOF = PolynomialNDOF<dim, 3>();
            break;
        default:
        {
            DNDS_assert_info(false, "maxNDOF invalid");
        }
        }
        // for polynomial: ndiff = ndof
        int maxNDIFF = maxNDOF;
        /******************************/
        // *cell's moment and cache
        this->MakePairDefaultOnCell(cellBaseMoment, maxNDOF);
        if (settings.cacheDiffBase)
        {
            this->MakePairDefaultOnCell(cellDiffBaseCache, maxNDIFF, maxNDOF);
        }
#ifdef DNDS_USE_OMP
        omp_set_num_threads(MPIWorldSize() == 1 ? omp_get_num_procs() : 1);
#pragma omp parallel for
#endif
        for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
        {
            cellAtr[iCell].NDIFF = maxNDIFF;
            cellAtr[iCell].NDOF = maxNDOF;
            auto eCell = mesh->GetFaceElement(iCell);
            auto qCell = Quadrature{eCell, cellAtr[iCell].intOrder};
            if (settings.cacheDiffBase)
            {
                cellDiffBaseCache.ResizeRow(iCell, qCell.GetNumPoints());
            }

            Eigen::RowVector<real, Eigen::Dynamic> m;
            m.setZero(cellAtr[iCell].NDOF);
            qCell.Integration(
                m,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    Eigen::RowVector<real, Eigen::Dynamic> vv;
                    vv.resizeLike(m);
                    this->FDiffBaseValue(vv, cellIntPPhysics(iCell, iG), iCell, -1, iG, 1);
                    vInc = vv * cellIntJacobiDet(iCell, iG);
                });
            cellBaseMoment[iCell] = m.transpose();
            SummationNoOp noOp;
            qCell.Integration(
                noOp,
                [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                {
                    if (settings.cacheDiffBase)
                    {
                        Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                        dbv.resize(cellAtr[iCell].NDIFF, cellAtr[iCell].NDOF);
                        this->FDiffBaseValue(dbv, cellIntPPhysics(iCell, iG), iCell, -1, iG, 0);
                        cellDiffBaseCache(iCell, iG) = dbv;
                    }
                });
        }

        /******************************/
        // *face's weight and cache
        this->MakePairDefaultOnFace(faceWeight, settings.maxOrder + 1);
        this->MakePairDefaultOnFace(faceAlignedScales);
        if (settings.cacheDiffBase)
        {
            this->MakePairDefaultOnFace(faceDiffBaseCache, maxNDIFF, maxNDOF);
        }
#ifdef DNDS_USE_OMP
        omp_set_num_threads(MPIWorldSize() == 1 ? omp_get_num_procs() : 1);
#pragma omp parallel for
#endif
        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            faceAtr[iFace].NDIFF = maxNDIFF;
            auto qFace = this->GetFaceQuad(iFace);
            if (settings.cacheDiffBase)
                faceDiffBaseCache.ResizeRow(iFace, 2 * qFace.GetNumPoints());

            // * get bdv cache
            SummationNoOp noOp;
            for (int ic2f = 0; ic2f < 2; ic2f++)
            {
                index iCell = mesh->face2cell(iFace, ic2f);
                if (iCell == UnInitIndex)
                    continue;
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                    DNDS_assert(ic2f == 0);
                else if (FaceIDIsPeriodic(mesh->GetFaceZone(iFace)))
                {
                    // TODO: handle the case with periodic
                    DNDS_assert(false);
                }
                qFace.Integration(
                    noOp,
                    [&](auto &vInc, int iG, const tPoint &pParam, const Elem::tD01Nj &DiNj)
                    {
                        if (settings.cacheDiffBase)
                        {
                            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> dbv;
                            dbv.resize(faceAtr[iFace].NDIFF, cellAtr[iCell].NDOF);
                            this->FDiffBaseValue(dbv, faceIntPPhysics(iFace, iG), iCell, iFace, iG, 0);
                            faceDiffBaseCache(iFace, ic2f * qFace.GetNumPoints() + iG) = dbv;
                        }
                    });
            }

            // *get face (derivatives) scale: cell average AlignedHBox mode
            tPoint faceScale{0, 0, 0};
            int nF2C{0};
            for (int ic2f = 0; ic2f < 2; ic2f++)
            {
                index iCell = mesh->face2cell(iFace, ic2f);
                if (iCell == UnInitIndex)
                    continue;
                if (FaceIDIsExternalBC(mesh->GetFaceZone(iFace)))
                    DNDS_assert(ic2f == 0);
                else if (FaceIDIsPeriodic(mesh->GetFaceZone(iFace)))
                {
                    // TODO: handle the case with periodic
                    DNDS_assert(false);
                }
                faceScale += cellAlignedHBox[iCell];
                nF2C++;
            }
            DNDS_assert(nF2C > 0);
            faceScale /= nF2C;
            faceAlignedScales[iFace] = faceScale;

            // *get geom weight
            real wg = 1;

            // *get dir weight
            Eigen::Vector<real, Eigen::Dynamic> wd;
            wd.resize(settings.maxOrder + 1);
            for (int p = 0; p < wd.size(); p++)
                wd[p] = 1. / factorials[p];
            faceWeight[iFace] = wd * wg;
        }
    }
    template void
    VariationalReconstruction<2>::
        ConstructBaseAndWeight();
    template void
    VariationalReconstruction<3>::
        ConstructBaseAndWeight();

    
}