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
            resc.resizeLike(rhs[0]);
            resc.setZero();

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

}
