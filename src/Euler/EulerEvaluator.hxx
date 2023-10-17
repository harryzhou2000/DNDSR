#pragma once
#include "EulerEvaluator.hpp"
#include "DNDS/HardEigen.hpp"

namespace DNDS::Euler
{

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSMatrixInit(
        ArrayDOFV<nVars_Fixed> &JDiag,
        ArrayDOFV<nVars_Fixed> &JSource,
        std::vector<real> &dTau, real dt, real alphaDiag,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        int jacobianCode,
        real t)
    {
        // TODO: for code0: flux jacobian with lambdaFace, and source jacobian with integration, only diagpart dealt with
        DNDS_assert(jacobianCode == 0);
        int cnvars = nVars;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto c2f = mesh->cell2face[iCell];

            // LUSGS diag part
            real fpDivisor = 1.0 / dTau[iCell] + 1.0 / dt;
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                fpDivisor += (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) * lambdaFace[iFace] / vfv->GetCellVol(iCell);
            }
            JDiag[iCell].setConstant(fpDivisor);

            // std::cout << fpDivisor << std::endl;

            // jacobian diag

            if (!settings.ignoreSourceTerm)
                JDiag[iCell] += alphaDiag * JSource[iCell];

            // jacobianCellInv[iCell] = jacobianCell[iCell].partialPivLu().inverse();

            // std::cout << "jacobian Diag\n"
            //           << jacobianCell[iCell] << std::endl;
            // std::cout << dTau[iCell] << "\n";
        }
        // exit(-1);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::LUSGSMatrixVec(
        real alphaDiag,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayDOFV<nVars_Fixed> &uInc,
        ArrayDOFV<nVars_Fixed> &JDiag,
        ArrayDOFV<nVars_Fixed> &AuInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->mpi, "LUSGSMatrixVec 1");
        int cnvars = nVars;
        for (index iScan = 0; iScan < mesh->NumCell(); iScan++)
        {
            index iCell = iScan;
            // iCell = (*vfv->SOR_iScan2iCell)[iCell];//TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // norhs
            auto uINCi = uInc[iCell];

            if (uINCi.hasNaN())
            {
                std::cout << uINCi << std::endl;
                DNDS_assert(false);
            }

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {

                    if (true)
                    {
                        auto uINCj = uInc[iCellOther];
                        auto uj = u[iCellOther];
                        TU fInc;
                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            // fInc = fluxJacobian0_Right(
                            //            u[iCellOther],
                            //            unitNorm,
                            //            BoundaryType::Inner) *
                            //        uInc[iCellOther]; //! always inner here
                            fInc = fluxJacobian0_Right_Times_du(
                                uj,
                                unitNorm,
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);
                        if (uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite()))
                        {
                            std::cout
                                << fInc.transpose() << std::endl
                                << uInc[iCellOther].transpose() << std::endl;
                            DNDS_assert(!(uIncNewBuf.hasNaN() || (!uIncNewBuf.allFinite())));
                        }
                    }
                }
            }
            // uIncNewBuf /= fpDivisor;
            // uIncNew[iCell] = uIncNewBuf;
            auto AuIncI = AuInc[iCell];
            AuIncI = JDiag[iCell].array() * uInc[iCell].array() - uIncNewBuf.array();

            if (AuIncI.hasNaN())
            {
                std::cout << AuIncI.transpose() << std::endl
                          << uINCi.transpose() << std::endl
                          << u[iCell].transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!AuInc[iCell].hasNaN());
            }
        }
        InsertCheck(u.father->mpi, "LUSGSMatrixVec -1");
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSForward(
        real alphaDiag,
        ArrayDOFV<nVars_Fixed> &rhs,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayDOFV<nVars_Fixed> &uInc,
        ArrayDOFV<nVars_Fixed> &JDiag,
        ArrayDOFV<nVars_Fixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->mpi, "UpdateLUSGSForward 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = iScan; // TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            // std::cout << rhs[iCell](0) << std::endl;
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {

                    index iScanOther = iCellOther; // TODO: add rb-sor
                    if (iScanOther < iScan)
                    {
                        TU fInc;
                        auto uINCj = uInc[iCellOther];

                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);

                        if ((!uIncNewBuf.allFinite()))
                        {
                            std::cout << RHSI.transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uINCj.transpose() << std::endl;
                            DNDS_assert(false);
                        }
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
            uIncNewI.array() = JDiag[iCell].array().inverse() * uIncNewBuf.array();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!uIncNew[iCell].hasNaN());
            }
        }
        InsertCheck(u.father->mpi, "UpdateLUSGSForward -1");
        // exit(-1);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateLUSGSBackward(
        real alphaDiag,
        ArrayDOFV<nVars_Fixed> &rhs,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayDOFV<nVars_Fixed> &uInc,
        ArrayDOFV<nVars_Fixed> &JDiag,
        ArrayDOFV<nVars_Fixed> &uIncNew)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->mpi, "UpdateLUSGSBackward 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        for (index iScan = nCellDist - 1; iScan >= 0; iScan--)
        {
            index iCell = iScan;
            iCell = iScan;

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(cnvars);
            uIncNewBuf.setZero(); // backward
            auto RHSI = rhs[iCell];

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = iCellOther;
                    if (iScanOther > iScan) // backward
                    {
                        TU fInc;

                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                Geom::BC_ID_INTERNAL, uInc[iCellOther], lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);
                    }
                }
            }
            auto uIncNewI = uIncNew[iCell];
            uIncNewI.array() += JDiag[iCell].array().inverse() * uIncNewBuf.array();
        }
        InsertCheck(u.father->mpi, "UpdateLUSGSBackward -1");
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateSGS(
        real alphaDiag,
        ArrayDOFV<nVars_Fixed> &rhs,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayDOFV<nVars_Fixed> &uInc,
        ArrayDOFV<nVars_Fixed> &JDiag,
        bool forward, TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->mpi, "UpdateSGS 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = forward ? iScan : nCellDist - 1 - iScan; // TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            // std::cout << rhs[iCell](0) << std::endl;
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = iCellOther; // TODO: add rb-sor
                    if (true)
                    {
                        TU fInc;
                        auto uINCj = uInc[iCellOther];

                        {
                            TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                            (iCellAtFace ? -1 : 1); // faces out

                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);

                        if ((!uIncNewBuf.allFinite()))
                        {
                            std::cout << RHSI.transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uINCj.transpose() << std::endl;
                            DNDS_assert(false);
                        }
                    }
                }
            }
            auto uIncNewI = uInc[iCell];
            TU uIncOld = uIncNewI;

            uIncNewI.array() = JDiag[iCell].array().inverse() * uIncNewBuf.array();
            sumInc.array() += (uIncNewI - uIncOld).array().abs();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!uInc[iCell].hasNaN());
            }
        }
        TU sumIncAll(cnvars);
        MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->mpi.comm);
        InsertCheck(u.father->mpi, "UpdateSGS -1");
        // exit(-1);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateSGSWithRec(
        real alphaDiag,
        ArrayDOFV<nVars_Fixed> &rhs,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        ArrayDOFV<nVars_Fixed> &uInc,
        ArrayRECV<nVars_Fixed> &uRecInc,
        ArrayDOFV<nVars_Fixed> &JDiag,
        bool forward, TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->mpi, "UpdateSGS 1");
        int cnvars = nVars;
        index nCellDist = mesh->NumCell();
        sumInc.setZero(cnvars);
        for (index iScan = 0; iScan < nCellDist; iScan++)
        {
            index iCell = iScan;
            iCell = forward ? iScan : nCellDist - 1 - iScan; // TODO: add rb-sor

            auto c2f = mesh->cell2face[iCell];
            TU uIncNewBuf(nVars);
            auto RHSI = rhs[iCell];
            // std::cout << rhs[iCell](0) << std::endl;
            uIncNewBuf = RHSI;

            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iFace = c2f[ic2f];
                auto f2c = mesh->face2cell[iFace];
                index iCellOther = f2c[0] == iCell ? f2c[1] : f2c[0];
                index iCellAtFace = f2c[0] == iCell ? 0 : 1;
                if (iCellOther != UnInitIndex)
                {
                    index iScanOther = iCellOther; // TODO: add rb-sor
                    if (true)
                    {
                        TU fInc, fIncS;
                        auto uINCj = uInc[iCellOther];
                        TVec unitNorm = vfv->GetFaceNormFromCell(iFace, iCellOther, iCellAtFace, -1)(Seq012) *
                                        (iCellAtFace ? -1 : 1); // faces out
                        {
                            fInc = fluxJacobian0_Right_Times_du(
                                u[iCellOther],
                                unitNorm,
                                Geom::BC_ID_INTERNAL, uINCj, lambdaFace[iFace], lambdaFaceC[iFace]); //! always inner here
                        }
                        {
                            TU uRecSLInc =
                                (vfv->GetIntPointDiffBaseValue(iCell, iFace, iCellAtFace, -1, std::array<int, 1>{0}, 1) *
                                 uRecInc[iCell])
                                    .transpose();
                            TU uRecSRInc =
                                (vfv->GetIntPointDiffBaseValue(iCellOther, iFace, 1 - iCellAtFace, -1, std::array<int, 1>{0}, 1) *
                                 uRecInc[iCellOther])
                                    .transpose();
                            TU fIncSL = fluxJacobianC_Right_Times_du(u[iCell], unitNorm, Geom::BC_ID_INTERNAL, uRecSLInc);
                            TU fIncSR = fluxJacobianC_Right_Times_du(u[iCellOther], unitNorm, Geom::BC_ID_INTERNAL, uRecSRInc);
                            fIncS = fIncSL + fIncSR + lambdaFaceC[iFace] * (uRecSLInc - uRecSRInc);
                        }

                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fInc);
                        uIncNewBuf -= (0.5 * alphaDiag) * vfv->GetFaceArea(iFace) / vfv->GetCellVol(iCell) *
                                      (fIncS);

                        if ((!uIncNewBuf.allFinite()))
                        {
                            std::cout << RHSI.transpose() << std::endl
                                      << fInc.transpose() << std::endl
                                      << uINCj.transpose() << std::endl;
                            DNDS_assert(false);
                        }
                    }
                }
            }
            auto uIncNewI = uInc[iCell];
            TU uIncOld = uIncNewI;

            uIncNewI.array() = JDiag[iCell].array().inverse() * uIncNewBuf.array();
            sumInc.array() += (uIncNewI - uIncOld).array().abs();

            if (uIncNewI.hasNaN())
            {
                std::cout << uIncNewI.transpose() << std::endl
                          << uIncNewBuf.transpose() << std::endl
                          << JDiag[iCell] << std::endl
                          << iCell << std::endl;
                DNDS_assert(!uInc[iCell].hasNaN());
            }
        }
        TU sumIncAll(cnvars);
        MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->mpi.comm);
        InsertCheck(u.father->mpi, "UpdateSGS -1");
    }

    template <EulerModel model>
    void EulerEvaluator<model>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        // TODO: make spacial filter jacobian
        return; // ! nofix shortcut
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateResidual(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise)
    {
        res.resize(nVars);
        if (P < 3)
        {
            TU resc;
            resc.setZero(nVars);

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (rhs[iCell].hasNaN() || (!rhs[iCell].allFinite()))
                {
                    std::cout << rhs[iCell] << std::endl;
                    DNDS_assert(false);
                }
                if (volWise)
                    resc += rhs[iCell].array().abs().pow(P).matrix() * vfv->GetCellVol(iCell);
                else
                    resc += rhs[iCell].array().abs().pow(P).matrix();
            }
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->mpi.comm);
            res = res.array().pow(1.0 / P).matrix();
            // std::cout << res << std::endl;
        }
        else
        {
            TU resc;
            resc.resizeLike(rhs[0]);
            resc.setZero();
            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                resc = resc.array().max(rhs[iCell].array().abs()).matrix();
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, rhs.father->mpi.comm);
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateURecBeta(
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        ArrayRECV<1> &uRecBeta, index &nLim, real &betaMin)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        index nLimLocal = 0;
        real minBetaLocal = 1;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            auto gCell = vfv->GetCellQuad(iCell);
            int nPoint = gCell.GetNumPoints();
            auto c2f = mesh->cell2face[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                auto gFace = vfv->GetFaceQuad(c2f[ic2f]);
                nPoint += gFace.GetNumPoints();
            }
            /***********/
            Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> quadBase;
            quadBase.resize(nPoint, vfv->GetCellAtr(iCell).NDOF - 1);
            for (int iG = 0; iG < gCell.GetNumPoints(); iG++)
                quadBase(iG, Eigen::all) = vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, 0, 1);
            nPoint = gCell.GetNumPoints();
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                auto gFace = vfv->GetFaceQuad(c2f[ic2f]);
                for (int iG = 0; iG < gFace.GetNumPoints(); iG++)
                    quadBase(nPoint + iG, Eigen::all) = vfv->GetIntPointDiffBaseValue(iCell, c2f[ic2f], -1, iG, 0, 1);
                nPoint += gFace.GetNumPoints();
            }
            /***********/

            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed> recInc = quadBase * uRec[iCell];
            Eigen::Vector<real, Eigen::Dynamic> rhoS = recInc(Eigen::all, 0) + u[iCell](0);
            real rhoMin = rhoS.minCoeff();
            real theta1 = std::min(
                1.,
                ((rhoS.array() - rhoEps) / (rhoS.array() - rhoMin + verySmallReal)).minCoeff());
            // recInc *= theta1;
            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed> recVRhoG = recInc.rowwise() + u[iCell].transpose();

            real gamma = settings.idealGasProperty.gamma;
            Eigen::Vector<real, Eigen::Dynamic> pS =
                (gamma - 1) *
                (recVRhoG(Eigen::all, I4) -
                 0.5 * (recVRhoG.rowwise()(Seq123).squaredNorm()).array() / recVRhoG(Eigen::all, 0).array());
            real thetaP = 1.0;
            for (int iG = 0; iG < pS.size(); iG++)
            {
                if (pS < pEps)
                {
                    real thetaThis = IdealGasGetCompressionRatioPressure<dim>(
                        u[iCell], recInc(iG, Eigen::all).transpose(), pEps / (gamma - 1));
                    thetaP = std::min(thetaP, thetaThis);
                }
            }

            // uRecBeta[iCell](0) = theta1 * thetaP;
            uRecBeta[iCell](0) = std::min(theta1, thetaP);
            if (uRecBeta[iCell](0) < 1)
                nLimLocal++, betaMin = std::min(uRecBeta[iCell](0), minBetaLocal);
        }
        MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u->father.getMPI().comm);
        MPI::Allreduce(&minBetaLocal, &betaMin, 1, DNDS_MPI_REAL, MPI_MIN, u->father.getMPI().comm);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateCellRHSAlpha(
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        ArrayRECV<1> &uRecBeta,
        ArrayDOFV<nVars_Fixed> &res,
        std::vector<real> &dTau,
        ArrayRECV<1> &cellRHSAlpha, index &nLim, real &alphaMin)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        index nLimLocal = 0;
        real alphaMinLocal = 1;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            real alphaRho = 1;
            TU inc = res[iCell] * dTau[iCell];
            DNDS_assert(u[iCell](0) >= rhoEps);
            if (inc(0) < 0)
                alphaRho = std::min(1.0, (u[iCell](0) - rhoEps) / (-inc(0)));

            // inc *= alphaRho;

            TU uNew = u[iCell] + inc;
            real pNew = (uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0)) * (gamma - 1);

            real alphaP = 1;
            if (pNew < pEps)
            {
                // todo: use high order accurate
                real alphaC = IdealGasGetCompressionRatioPressure<dim>(
                    u[iCell], inc, pEps / (gamma - 1));
                alphaP = std::min(alphaP, alphaC);
            }
            // cellRHSAlpha[iCell](0) = alphaRho * alphaP;
            cellRHSAlpha[iCell](0) = std::min(alphaRho, alphaP);
            if (cellRHSAlpha[iCell](0) < 1)
                nLimLocal++, alphaMinLocal = std::min(alphaMinLocal, cellRHSAlpha[iCell](0));
        }
        MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u->father.getMPI().comm);
        MPI::Allreduce(&alphaMinLocal, &alphaMin, 1, DNDS_MPI_REAL, MPI_MIN, u->father.getMPI().comm);
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            if (cellRHSAlpha[iCell](0) < 1)
                cellRHSAlpha[iCell](0) = alphaMin;
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateCellRHSAlphaExpansion(
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        ArrayRECV<1> &uRecBeta,
        ArrayDOFV<nVars_Fixed> &res,
        std::vector<real> &dTau,
        ArrayRECV<1> &cellRHSAlpha, index &nLim, real &alphaMin)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        auto cellIsHalfAlpha = [&](index iCell) -> bool // iCell should be internal
        {
            bool ret = false;
            if (cellRHSAlpha[iCell](0) == 1.0)
            {
                auto c2f = mesh->cell2face(iCell);
                for (int ic2f = 0; ic2f < f2c.size(); ic2f++)
                {
                    index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f]);
                    if (cellRHSAlpha[iCellOther](0) != 1.0)
                        ret = true;
                }
            }
            return ret;
        };

        auto cellAdjAlphaMin = [&](index iCell) -> real // iCell should be internal
        {
            real ret = 1;
            auto c2f = mesh->cell2face(iCell);
            for (int ic2f = 0; ic2f < f2c.size(); ic2f++)
            {
                index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f]);
                ret = std::min(ret, cellRHSAlpha[iCellOther](0));
            }
            return ret;
        };

        std::vector<index> InterCells;

        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            if (cellIsHalfAlpha(iCell))
                InterCells.emplace_back(iCell);

        index nLimLocal = 0;
        index nLimAdd = 0;
        for (index iCell : InterCells)
        {
            real gamma = settings.idealGasProperty.gamma;
            TU inc = res[iCell] * dTau[iCell];

            TU uNew = u[iCell] + inc;
            real pNew = (uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0)) * (gamma - 1);

            if (pNew < pEps || uNew(0) < rhoEps)
            {
                cellRHSAlpha[iCell](0) = cellAdjAlphaMin(iCell);
                DNDS_assert(cellRHSAlpha[iCell](0) == alphaMin);
            }
        }
        MPI::Allreduce(&nLimLocal, &nLimAdd, 1, DNDS_MPI_INDEX, MPI_SUM, u->father.getMPI().comm);
        nLim += nLimAdd;
    }
}
