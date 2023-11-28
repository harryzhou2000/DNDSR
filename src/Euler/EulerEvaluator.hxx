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
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixVec 1");
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
        DNDS_MPI_InsertCheck(u.father->getMPI(), "LUSGSMatrixVec -1");
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
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSForward 1");
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
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSForward -1");
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
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSBackward 1");
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
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateLUSGSBackward -1");
    }

    template <EulerModel model>
    void EulerEvaluator<model>::UpdateSGS(
        real alphaDiag,
        ArrayDOFV<nVars_Fixed> &rhs,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayDOFV<nVars_Fixed> &uInc,
        ArrayDOFV<nVars_Fixed> &uIncNew,
        ArrayDOFV<nVars_Fixed> &JDiag,
        bool forward, TU &sumInc)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
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
                    index iScanOther = forward ? iCellOther : nCellDist - 1 - iCellOther; // TODO: add rb-sor
                    if (iCell != iCellOther)
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
            // if (iScan == 100)
        }
        TU sumIncAll(cnvars);
        // std::abort();
        MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS -1");
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
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS 1");
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
                    index iScanOther = forward ? iCellOther : nCellDist - 1 - iCellOther; // TODO: add rb-sor
                    if (iCell != iCellOther)
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
        MPI::Allreduce(sumInc.data(), sumIncAll.data(), sumInc.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
        DNDS_MPI_InsertCheck(u.father->getMPI(), "UpdateSGS -1");
    }

    template <EulerModel model>
    void EulerEvaluator<model>::FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        // TODO: make spacial filter jacobian
        return; // ! nofix shortcut
    }

    template <EulerModel model>
    void EulerEvaluator<model>::TimeAverageAddition(ArrayDOFV<nVars_Fixed> &w, ArrayDOFV<nVars_Fixed> &wAveraged, real dt, real &tCur)
    {
        wAveraged *= (tCur / (tCur + dt + verySmallReal));
        wAveraged.addTo(w, (dt + verySmallReal) / (tCur + dt + verySmallReal));
        tCur += dt + verySmallReal;
    }

    template <EulerModel model>
    void EulerEvaluator<model>::MeanValueCons2Prim(ArrayDOFV<nVars_Fixed> &u, ArrayDOFV<nVars_Fixed> &w)
    {
        for (index iCell = 0; iCell < u.Size(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            TU out;
            Gas::IdealGasThermalConservative2Primitive(u[iCell], out, gamma);
            w[iCell] = out;
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::MeanValuePrim2Cons(ArrayDOFV<nVars_Fixed> &w, ArrayDOFV<nVars_Fixed> &u)
    {
        for (index iCell = 0; iCell < w.Size(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            TU out;
            Gas::IdealGasThermalPrimitive2Conservative(w[iCell], out, gamma);
            u[iCell] = out;
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateNorm(Eigen::Vector<real, -1> &res, ArrayDOFV<nVars_Fixed> &rhs, index P, bool volWise)
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
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_SUM, rhs.father->getMPI().comm);
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
            MPI::Allreduce(resc.data(), res.data(), res.size(), DNDS_MPI_REAL, MPI_MAX, rhs.father->getMPI().comm);
        }
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateURecBeta(
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        ArrayDOFV<1> &uRecBeta, index &nLim, real &betaMin)
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
            Eigen::Vector<real, Eigen::Dynamic> rhoS = recInc(Eigen::all, 0).array() + u[iCell](0);
            real rhoMin = rhoS.minCoeff();
            real theta1 = 1;
            DNDS_assert(u[iCell](0) >= rhoEps);
            if (rhoMin < rhoEps)
                theta1 = std::min(1.0, (u[iCell](0) - rhoEps) / (u[iCell](0) - rhoMin + verySmallReal));
#ifdef USE_NS_SA_NUT_REDUCED_ORDER
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                Eigen::Vector<real, Eigen::Dynamic> v1S = recInc(Eigen::all, I4 + 1).array() + u[iCell](I4 + 1);
                real v1Min = v1S.minCoeff();
                if (v1Min < v1Eps)
                    theta1 = std::min(theta1,
                                      (u[iCell](I4 + 1) - v1Eps) / (u[iCell](I4 + 1) - v1Min + verySmallReal))
                        // * 0 // to gain fully reduced order
                        ;
            }
#endif
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                Eigen::Vector<real, Eigen::Dynamic> v1S = recInc(Eigen::all, I4 + 1).array() + u[iCell](I4 + 1);
                real v1Min = v1S.minCoeff();
                if (v1Min < v1Eps)
                    theta1 = std::min(theta1,
                                      (u[iCell](I4 + 1) - v1Eps) / (u[iCell](I4 + 1) - v1Min + verySmallReal));
                Eigen::Vector<real, Eigen::Dynamic> v2S = recInc(Eigen::all, I4 + 2).array() + u[iCell](I4 + 2);
                real v2Min = v2S.minCoeff();
                if (v2Min < v2Eps)
                    theta1 = std::min(theta1,
                                      (u[iCell](I4 + 2) - v2Eps) / (u[iCell](I4 + 2) - v2Min + verySmallReal));
            }

            recInc *= theta1;
            Eigen::Matrix<real, Eigen::Dynamic, nVars_Fixed>
                recVRhoG = recInc.rowwise() + u[iCell].transpose();

            real gamma = settings.idealGasProperty.gamma;
            Eigen::Vector<real, Eigen::Dynamic> ek =
                0.5 * (recVRhoG(Eigen::all, Seq123).array().square().rowwise().sum()) / recVRhoG(Eigen::all, 0).array();
            Eigen::Vector<real, Eigen::Dynamic> pS =
                (gamma - 1) *
                (recVRhoG(Eigen::all, I4) -
                 ek);
            real thetaP = 1.0;
            real pCent = (gamma - 1) * (u[iCell](I4) - 0.5 * u[iCell](Seq123).squaredNorm() / u[iCell](0));
            if (pCent <= 2 * pEps)
                thetaP = 0;
            else
                for (int iG = 0; iG < pS.size(); iG++)
                {
                    if (pS(iG) < 2 * pEps)
                    {
                        real thetaThis = Gas::IdealGasGetCompressionRatioPressure<dim, nVars_Fixed>(
                            u[iCell], recInc(iG, Eigen::all).transpose(), 1 * pEps / (gamma - 1));
                        thetaP = std::min(thetaP, thetaThis);
                    }
                }

            uRecBeta[iCell](0) = theta1 * thetaP;
            // if (uRecBeta[iCell](0) < 1)
            //     uRecBeta[iCell](0) *= 1 - 1e-8;
            // uRecBeta[iCell](0) = std::min(theta1, thetaP);
            if (uRecBeta[iCell](0) < 1)
            {
                nLimLocal++;
                // uRecBeta[iCell](0) *= uRecBeta[iCell](0) < 0.99 ? 0. : 0.99; //! for safety
                uRecBeta[iCell](0) *= std::pow(uRecBeta[iCell](0), 11) * 0.99;
                minBetaLocal = std::min(uRecBeta[iCell](0), minBetaLocal);
            }
            if (uRecBeta[iCell](0) < 0)
            {
                std::cout << fmt::format("theta1 {}, thetaP {}", theta1, thetaP) << std::endl;
                DNDS_assert(false);
            }

            // validation:
            recInc = quadBase * uRec[iCell] * uRecBeta[iCell](0);
            recVRhoG = recInc.rowwise() + u[iCell].transpose();
            ek =
                0.5 * (recVRhoG(Eigen::all, Seq123).array().square().rowwise().sum()) / recVRhoG(Eigen::all, 0).array();
            pS =
                (gamma - 1) *
                (recVRhoG(Eigen::all, I4) -
                 ek);
            for (int iG = 0; iG < pS.size(); iG++)
            {
                if (pS(iG) < pEps)
                {
                    // std::cout << std::scientific;
                    // std::cout << pS.transpose() << std::endl;
                    // std::cout << fmt::format("{} {} {}", theta1, thetaP, uRecBeta[iCell](0)) << std::endl;
                    // std::cout << u[iCell] << std::endl;
                    // std::cout << recInc.transpose() << std::endl;
                    // DNDS_assert(false);
                }
            }
        }
        MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
        MPI::Allreduce(&minBetaLocal, &betaMin, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
    }

    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateCellRHSAlpha(
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        ArrayDOFV<1> &uRecBeta,
        ArrayDOFV<nVars_Fixed> &res,
        ArrayDOFV<1> &cellRHSAlpha, index &nLim, real &alphaMin,
        int flag)
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
            TU inc = res[iCell];
            DNDS_assert(u[iCell](0) >= rhoEps);
            if (inc(0) < 0) // not < rhoEps!!!
                alphaRho = std::min(1.0, (u[iCell](0) - rhoEps) / (-inc(0) - smallReal * inc(0)));
            DNDS_assert(alphaRho >= 0 && alphaRho <= 1);
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                // ** ! currently do not mass - fix for SA
                // static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                // if (inc(I4 + 1) < 0)
                //     alphaRho = std::min(alphaRho,
                //                         (u[iCell](I4 + 1) - v1Eps) / (-inc(I4 + 1) - smallReal * inc(I4 + 1)));
                // // use exp down:
                // if (inc(I4 + 1) + u[iCell](I4 + 1) < v1Eps)
                // {
                //     DNDS_assert(inc(I4 + 1) < 0);
                //     real declineV = inc(I4 + 1) / (u[iCell](I4 + 1) + 1e-6);
                //     real newu5 = u[iCell](I4 + 1) * std::exp(declineV);
                //     alphaRho = std::min(alphaRho,
                //                         (u[iCell](I4 + 1) - newu5) / (-inc(I4 + 1) - smallReal * inc(I4 + 1)));
                // }
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                if (inc(I4 + 1) < 0)
                    alphaRho = std::min(alphaRho,
                                        (u[iCell](I4 + 1) - v1Eps) / (-inc(I4 + 1) - smallReal * inc(I4 + 1)));
                if (inc(I4 + 2) < 0)
                    alphaRho = std::min(alphaRho,
                                        (u[iCell](I4 + 2) - v2Eps) / (-inc(I4 + 2) - smallReal * inc(I4 + 2)));
            }

            inc *= alphaRho;

            TU uNew = u[iCell] + inc;
            real pNew = (uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0)) * (gamma - 1);

            real alphaP = 1;
            if (pNew < pEps)
            {
                // todo: use high order accurate
                real alphaC = Gas::IdealGasGetCompressionRatioPressure<dim, nVars_Fixed>(
                    u[iCell], inc, pEps / (gamma - 1));
                alphaP = std::min(alphaP, alphaC);
            }
            cellRHSAlpha[iCell](0) = alphaRho * alphaP;
            // cellRHSAlpha[iCell](0) = std::min(alphaRho, alphaP);
            if (cellRHSAlpha[iCell](0) < 1)
                nLimLocal++,
                    cellRHSAlpha[iCell] *= (0.9),
                    alphaMinLocal = std::min(alphaMinLocal, cellRHSAlpha[iCell](0)); //! for safety
        }
        MPI::Allreduce(&nLimLocal, &nLim, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
        MPI::Allreduce(&alphaMinLocal, &alphaMin, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);
        if (flag == 0)
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
        ArrayDOFV<1> &uRecBeta,
        ArrayDOFV<nVars_Fixed> &res,
        ArrayDOFV<1> &cellRHSAlpha, index &nLim, real alphaMin)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        real rhoEps = smallReal * settings.refUPrim(0);
        real pEps = smallReal * settings.refUPrim(I4);

        auto cellIsHalfAlpha = [&](index iCell) -> bool // iCell should be internal
        {
            bool ret = false;
            if (cellRHSAlpha[iCell](0) == 1.0)
            {
                auto c2f = mesh->cell2face[iCell];
                for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                {
                    index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f]);
                    if (iCellOther != UnInitIndex)
                        if (cellRHSAlpha[iCellOther](0) != 1.0)
                            ret = true;
                }
            }
            return ret;
        };

        auto cellAdjAlphaMin = [&](index iCell) -> real // iCell should be internal
        {
            real ret = 1;
            auto c2f = mesh->cell2face[iCell];
            for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
            {
                index iCellOther = vfv->CellFaceOther(iCell, c2f[ic2f]);
                if (iCellOther != UnInitIndex)
                    ret = std::min(ret, cellRHSAlpha[iCellOther](0));
            }
            return ret;
        };

        // std::vector<index> InterCells;

        // for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        //     if (cellIsHalfAlpha(iCell))
        //         InterCells.emplace_back(iCell);

        index nLimLocal = 0;
        index nLimAdd = 0;
        // for (index iCell : InterCells)
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            real gamma = settings.idealGasProperty.gamma;
            TU inc = res[iCell];

            TU uNew = u[iCell] + inc;
            real pNew = (uNew(I4) - 0.5 * uNew(Seq123).squaredNorm() / uNew(0)) * (gamma - 1);

            if (pNew < pEps || uNew(0) < rhoEps)
            {
                // cellRHSAlpha[iCell](0) = cellAdjAlphaMin(iCell);
                // DNDS_assert(cellRHSAlpha[iCell](0) == alphaMin);
                cellRHSAlpha[iCell](0) = alphaMin;
            }
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                if (uNew(I4 + 1) < v1Eps)
                    cellRHSAlpha[iCell](0) = alphaMin;
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                static real v1Eps = smallReal * settings.refUPrim(I4 + 1);
                static real v2Eps = smallReal * settings.refUPrim(I4 + 2);
                if (uNew(I4 + 1) < v1Eps)
                    cellRHSAlpha[iCell](0) = alphaMin;
                if (uNew(I4 + 2) < v2Eps)
                    cellRHSAlpha[iCell](0) = alphaMin;
            }
        }
        MPI::Allreduce(&nLimLocal, &nLimAdd, 1, DNDS_MPI_INDEX, MPI_SUM, u.father->getMPI().comm);
        nLim += nLimAdd;
    }
}
