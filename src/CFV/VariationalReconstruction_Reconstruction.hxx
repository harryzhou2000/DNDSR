#pragma once

#include "VariationalReconstruction.hpp"

namespace DNDS
{
    namespace CFV
    {

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstruction2nd(
            tURec<nVarsFixed> &uRec,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            int method)
        {
            using namespace Geom;
            using namespace Geom::Elem;

            if (method == 1)
            {
                // simple gauss rule
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {

                    auto c2f = mesh->cell2face[iCell];
                    Eigen::Matrix<real, nVarsFixed, dim> grad;
                    grad.setZero();

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        index iCellOther = CellFaceOther(iCell, iFace);

                        if (iCellOther != UnInitIndex)
                        {
                            grad += (u[iCellOther] - u[iCell]) * 0.5 *
                                    this->GetFaceNorm(iFace, -1)(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).transpose() *
                                    this->GetFaceArea(iFace) * (CellIsFaceBack(iCell, iFace) ? 1. : -1.);
                        }
                        else
                        {
                            auto faceID = mesh->GetFaceZone(iFace);
                            DNDS_assert(FaceIDIsExternalBC(faceID));

                            int nVars = u[iCell].size();

                            Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
                                this->GetIntPointDiffBaseValue(
                                    iCell, iFace, -1, -1, std::array<int, 1>{0}, 1);
                            Eigen::Vector<real, nVarsFixed> uBL =
                                (dbv *
                                 uRec[iCell])
                                    .transpose();
                            uBL += u[iCell]; //! need fixing?
                            Eigen::Vector<real, nVarsFixed> uBV =
                                FBoundary(
                                    uBL,
                                    u[iCell],
                                    this->GetFaceNorm(iFace, -1),
                                    this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1), faceID);
                            grad += (uBV - u[iCell]) * 0.5 *
                                    this->GetFaceNorm(iFace, -1)(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>)).transpose() *
                                    this->GetFaceArea(iFace);
                        }
                    }

                    grad /= GetCellVol(iCell);
                    Eigen::Matrix<real, dim, dim> d1bv;
                    if constexpr (dim == 2)
                        d1bv =
                            this->GetIntPointDiffBaseValue(
                                iCell, -1, -1, -1, std::array<int, 2>{1, 2}, 3);
                    if constexpr (dim == 3)
                        d1bv =
                            this->GetIntPointDiffBaseValue(
                                iCell, -1, -1, -1, std::array<int, 3>{1, 2, 3}, 4);
                    Eigen::Matrix3d m;
                    auto lud1bv = d1bv.partialPivLu();
                    if (lud1bv.rcond() > 1e9)
                        std::cout << "Large Cond " << lud1bv.rcond() << std::endl;
                    uRec[iCell] = lud1bv.solve(grad.transpose());
                    // std::cout << " g " << std::endl;
                    // std::cout << grad << std::endl;
                    // std::cout << uRec[iCell] << std::endl;
                    // std::cout << d1bv << std::endl;
                    // std::cout << lud1bv.inverse() << std::endl;
                    // std::abort();
                }
            }
            else if (method == 2)
            {
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {

                    auto c2f = mesh->cell2face[iCell];
                    Eigen::Matrix<real, nVarsFixed, dim> grad;
                    grad.setZero();
                    Eigen::Matrix<real, Eigen::Dynamic, dim> dcs;
                    Eigen::Matrix<real, Eigen::Dynamic, nVarsFixed> dus;
                    dcs.resize(c2f.size(), dim);
                    dus.resize(c2f.size(), nVarsFixed);

                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        index iCellOther = CellFaceOther(iCell, iFace);

                        if (iCellOther != UnInitIndex)
                        {
                            dcs(ic2f, Eigen::all) = (GetCellBary(iCellOther) - GetCellBary(iCell))(Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>))
                                                        .transpose();
                            dus(ic2f, Eigen::all) = (u[iCellOther] - u[iCell]).transpose();
                        }
                        else
                        {
                            auto faceID = mesh->GetFaceZone(iFace);
                            DNDS_assert(FaceIDIsExternalBC(faceID));

                            int nVars = u[iCell].size();

                            Eigen::Matrix<real, 1, Eigen::Dynamic> dbv =
                                this->GetIntPointDiffBaseValue(
                                    iCell, iFace, -1, -1, std::array<int, 1>{0}, 1);
                            Eigen::Vector<real, nVarsFixed> uBL =
                                (dbv *
                                 uRec[iCell])
                                    .transpose();
                            uBL += u[iCell]; //! need fixing?
                            Eigen::Vector<real, nVarsFixed> uBV =
                                FBoundary(
                                    uBL,
                                    u[iCell],
                                    this->GetFaceNorm(iFace, -1),
                                    this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, -1), faceID);
                            dus(ic2f, Eigen::all) = (uBV - u[iCell]).transpose();

                            dcs(ic2f, Eigen::all) =
                                ((GetCellBary(iCell) - GetFaceQuadraturePPhys(iFace, -1)).dot(GetFaceNorm(iFace, -1)) * (-2.) * GetFaceNorm(iFace, -1))(
                                    Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>))
                                    .transpose();
                        }
                    }
                    // Eigen::MatrixXd m;
                    // m.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve()
                    auto svd = dcs.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
                    grad = svd.solve(dus).transpose();

                    Eigen::Matrix<real, dim, dim> d1bv;
                    if constexpr (dim == 2)
                        d1bv =
                            this->GetIntPointDiffBaseValue(
                                iCell, -1, -1, -1, std::array<int, 2>{1, 2}, 3);
                    if constexpr (dim == 3)
                        d1bv =
                            this->GetIntPointDiffBaseValue(
                                iCell, -1, -1, -1, std::array<int, 3>{1, 2, 3}, 4);
                    Eigen::Matrix3d m;
                    auto lud1bv = d1bv.partialPivLu();
                    if (lud1bv.rcond() > 1e9)
                        std::cout << "Large Cond " << lud1bv.rcond() << std::endl;
                    uRec[iCell] = lud1bv.solve(grad.transpose());
                }
            }
            else
            {
                DNDS_assert_info(false, "NO SUCH 2nd rec METHOD");
            }
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstructionIter(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundary<nVarsFixed> &FBoundary,
            bool putIntoNew,
            bool recordInc)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            if (recordInc)
                DNDS_assert_info(putIntoNew, "the -RHS must be put into uRecNew");
            if (settings.maxOrder == 1 && settings.subs2ndOrder != 0)
            {
                if (recordInc)
                    this->DoReconstruction2nd(uRecNew, u, (FBoundary), settings.subs2ndOrder);
                else if (putIntoNew)
                    this->DoReconstruction2nd(uRecNew, u, (FBoundary), settings.subs2ndOrder);
                else
                    this->DoReconstruction2nd(uRec, u, (FBoundary), settings.subs2ndOrder);
                return;
            }
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                real relax = cellAtr[iCell].relax;

                if (recordInc)
                    uRecNew[iCell] = uRec[iCell];
                else if (settings.SORInstead)
                    uRec[iCell] = uRec[iCell] * ((recordInc ? 0 : 1) - relax);
                else
                    uRecNew[iCell] = uRec[iCell] * ((recordInc ? 0 : 1) - relax);

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = matrixAAInvB[iCell];
                auto vectorAInvBRow = vectorAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);
                    if (iCellOther != UnInitIndex)
                    {
                        if (recordInc)
                            uRecNew[iCell] -=
                                (matrixAAInvBRow[ic2f + 1] * uRec[iCellOther] +
                                 vectorAInvBRow[ic2f] * (u[iCellOther] - u[iCell]).transpose());
                        else if (settings.SORInstead)
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
                                        this->GetFaceNorm(iFace, iG),
                                        this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                                Eigen::RowVector<real, nVarsFixed> uIncBV = (uBV - u[iCell]).transpose();
                                vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iG) * this->GetFaceJacobiDet(iFace, iG);
                                // std::cout << faceWeight[iFace].transpose() << std::endl;
                            });
                        // BCC *= 0;
                        if (recordInc)
                            uRecNew[iCell] -=
                                matrixAAInvBRow[0] * BCC;
                        else if (settings.SORInstead && !recordInc)
                            uRec[iCell] +=
                                relax * matrixAAInvBRow[0] * BCC;
                        else
                            uRecNew[iCell] +=
                                relax * matrixAAInvBRow[0] * BCC;
                    }
                }
            }

            if (!recordInc)
            {
                if (putIntoNew && settings.SORInstead)
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                        uRecNew[iCell].swap(uRec[iCell]);
                if ((!putIntoNew) && (!settings.SORInstead))
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                        uRec[iCell].swap(uRecNew[iCell]);
            }
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::DoReconstructionIterDiff(
            tURec<nVarsFixed> &uRec,
            tURec<nVarsFixed> &uRecDiff,
            tURec<nVarsFixed> &uRecNew,
            tUDof<nVarsFixed> &u,
            const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            if (settings.maxOrder == 1 && settings.subs2ndOrder != 0)
            {
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    uRecNew[iCell].setZero();
                return;
            }
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                uRecNew[iCell] = uRecDiff[iCell];

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = matrixAAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);
                    if (iCellOther != UnInitIndex)
                    {

                        uRecNew[iCell] -= matrixAAInvBRow[ic2f + 1] * uRecDiff[iCellOther]; // mind the sign
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
                                Eigen::Vector<real, nVarsFixed> uBLDiff =
                                    (dbv *
                                     uRecDiff[iCell])
                                        .transpose();
                                Eigen::Vector<real, nVarsFixed>
                                    uBV =
                                        FBoundaryDiff(
                                            uBL,
                                            uBLDiff,
                                            u[iCell],
                                            this->GetFaceNorm(iFace, iG),
                                            this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                                Eigen::RowVector<real, nVarsFixed> uIncBV = uBV.transpose();
                                vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iG) * this->GetFaceJacobiDet(iFace, iG);
                                // std::cout << faceWeight[iFace].transpose() << std::endl;
                            });
                        // BCC *= 0;
                        uRecNew[iCell] -= matrixAAInvBRow[0] * BCC; // mind the sign
                    }
                }
            }
        }

        template <int dim>
        template <int nVarsFixed>
        void VariationalReconstruction<dim>::
            DoReconstructionIterSOR(
                tURec<nVarsFixed> &uRec,
                tURec<nVarsFixed> &uRecInc,
                tURec<nVarsFixed> &uRecNew,
                tUDof<nVarsFixed> &u,
                const TFBoundaryDiff<nVarsFixed> &FBoundaryDiff,
                bool reverse)
        {
            using namespace Geom;
            using namespace Geom::Elem;
            int maxNDOF = GetNDof<dim>(settings.maxOrder);
            for (index iScan = 0; iScan < mesh->NumCell(); iScan++)
            {
                index iCell = iScan;
                if (reverse)
                    iCell = mesh->NumCell() - 1 - iCell;
                real relax = cellAtr[iCell].relax;

                uRecNew[iCell] = (1 - relax) * uRecNew[iCell] + uRecInc[iCell];

                auto c2f = mesh->cell2face[iCell];
                auto matrixAAInvBRow = matrixAAInvB[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iFace = c2f[ic2f];
                    index iCellOther = CellFaceOther(iCell, iFace);

                    if (iCellOther != UnInitIndex)
                    {
                        uRecNew[iCell] += relax * matrixAAInvBRow[ic2f + 1] * uRecNew[iCellOther]; // mind the sign
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
                                Eigen::Vector<real, nVarsFixed> uBLDiff =
                                    (dbv *
                                     uRecNew[iCell])
                                        .transpose();
                                Eigen::Vector<real, nVarsFixed>
                                    uBV =
                                        FBoundaryDiff(
                                            uBL,
                                            uBLDiff,
                                            u[iCell],
                                            this->GetFaceNorm(iFace, iG),
                                            this->GetFaceQuadraturePPhysFromCell(iFace, iCell, -1, iG), faceID);
                                Eigen::RowVector<real, nVarsFixed> uIncBV = uBV.transpose();
                                vInc = this->FFaceFunctional(dbv, uIncBV, iFace, iG) * this->GetFaceJacobiDet(iFace, iG);
                                // std::cout << faceWeight[iFace].transpose() << std::endl;
                            });
                        // BCC *= 0;
                        uRecNew[iCell] += relax * matrixAAInvBRow[0] * BCC; // mind the sign
                    }
                }
            }
        }
    }
}