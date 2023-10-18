#pragma once
#include "EulerEvaluator.hpp"

namespace DNDS::Euler
{
    /**
     * @details
     * about RHS:
     * with topology fixed, RHS is dependent on:
     * flux:
     *      dofs:  u_L u_R, urec_L, urec_R,
     *      geoms: dbv_l, dbv_r, detJacobian_f, uNorm_f,
     */
#define IF_NOT_NOREC (1)
    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateRHS(
        ArrayDOFV<nVars_Fixed> &rhs,
        ArrayDOFV<nVars_Fixed> &JSource,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        ArrayDOFV<1> &uRecBeta,
        ArrayDOFV<1> &cellRHSAlpha,
        bool onlyOnHalfAlpha,
        real t)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->getMPI(), "EvaluateRHS 1");
        int cnvars = nVars;
        auto rsType = settings.rsType;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            rhs[iCell].setZero();
        }
        TU fluxWallSumLocal;
        fluxWallSumLocal.setZero(cnvars);
        fluxWallSum.setZero(cnvars);
        nFaceReducedOrder = 0;

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

        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            auto gFace = vfv->GetFaceQuad(iFace);
#ifdef USE_FLUX_BALANCE_TERM
            Eigen::Matrix<real, nVars_Fixed, 3, Eigen::ColMajor> fluxEs(cnvars, 3);
#else
            Eigen::Matrix<real, nVars_Fixed, 1, Eigen::ColMajor> fluxEs(cnvars, 1);
#endif
            if (onlyOnHalfAlpha)
            {
                // bool lIsHalfAlpha = cellIsHalfAlpha(f2c[0]);
                // bool rIsHalfAlpha =
                //     (f2c[1] != UnInitIndex && f2c[1] < mesh->NumCell()) // must be owned cell!
                //         ? cellIsHalfAlpha(f2c[0])
                //         : false;
                // if (!(lIsHalfAlpha || rIsHalfAlpha))
                //     continue;
            }

            fluxEs.setZero();

            // auto f2n = mesh->face2node[iFace];
            // Geom::tSmallCoords coords;
            // mesh->LoadCoords(f2n, coords);

            Geom::Elem::SummationNoOp noOp;
            bool faceOrderReducedL = false;
            bool faceOrderReducedR = false;
            auto faceBndID = mesh->GetFaceZone(iFace);
            auto faceBCType = pBCHandler->GetTypeFromID(faceBndID);

#ifdef USE_TOTAL_REDUCED_ORDER_CELL
            gFace.IntegrationSimple(
                noOp,
                [&](auto &finc, int iG)
                {
                    if (!faceOrderReducedL)
                    {
                        TU ULxy =
                            (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 1>{0}, 1) *
                             uRec[f2c[0]])
                                .transpose() *
                            IF_NOT_NOREC * uRecBeta[f2c[0]](0);
                        ULxy = CompressRecPart(u[f2c[0]], ULxy, faceOrderReducedL);
                    }

                    if (f2c[1] != UnInitIndex)
                    {
                        if (!faceOrderReducedR)
                        {
                            TU URxy =
                                (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 1>{0}, 1) *
                                 uRec[f2c[1]])
                                    .transpose() *
                                IF_NOT_NOREC * uRecBeta[f2c[1]](0);
                            URxy = CompressRecPart(u[f2c[1]], URxy, faceOrderReducedR);
                        }
                    }
                });
#endif

            gFace.IntegrationSimple(
                fluxEs,
                [&](decltype(fluxEs) &finc, int iG)
                {
                    finc.resizeLike(fluxEs);
                    int nDiff = vfv->GetFaceAtr(iFace).NDIFF;
                    TVec unitNorm = vfv->GetFaceNorm(iFace, iG)(Seq012);
                    TMat normBase = Geom::NormBuildLocalBaseV<dim>(unitNorm);
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);

                    TU ULxy = u[f2c[0]];
                    bool pointOrderReducedL = false;
                    bool pointOrderReducedR = false;
                    if (!faceOrderReducedL)
                    {

                        ULxy = CompressRecPart(
                            ULxy,
                            (vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 1>{0}, 1) *
                             uRec[f2c[0]])
                                    .transpose() *
                                IF_NOT_NOREC * uRecBeta[f2c[0]](0),
                            pointOrderReducedL);
                    }
                    this->UFromCell2Face(ULxy, iFace, f2c[0], 0);

                    TU ULMeanXy = u[f2c[0]];
                    this->UFromCell2Face(ULMeanXy, iFace, f2c[0], 0);
                    TU URMeanXy;

                    TU URxy;
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradULxy, GradURxy;
                    GradULxy.resize(Eigen::NoChange, cnvars);
                    GradURxy.resize(Eigen::NoChange, cnvars);
                    GradULxy.setZero(), GradURxy.setZero();

                    if constexpr (gDim == 2)
                        GradULxy({0, 1}, Eigen::all) =
                            vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 2>{1, 2}, 3) *
                            uRec[f2c[0]] * IF_NOT_NOREC * uRecBeta[f2c[0]](0); // 2d here
                    else
                        GradULxy({0, 1, 2}, Eigen::all) =
                            vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 3>{1, 2, 3}, 4) *
                            uRec[f2c[0]] * IF_NOT_NOREC * uRecBeta[f2c[0]](0); // 3d here
                    this->DiffUFromCell2Face(GradULxy, iFace, f2c[0], 0);

#endif
                    real minVol = vfv->GetCellVol(f2c[0]);
                    // InsertCheck(u.father->getMPI(), "RHS inner 2");

                    if (f2c[1] != UnInitIndex)
                    {
                        URxy = u[f2c[1]];
                        this->UFromCell2Face(URxy, iFace, f2c[1], 1);
                        if (!faceOrderReducedR)
                        {
                            URxy = CompressRecPart(
                                URxy,
                                (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 1>{0}, 1) *
                                 uRec[f2c[1]])
                                        .transpose() *
                                    IF_NOT_NOREC * uRecBeta[f2c[1]](0),
                                pointOrderReducedR);
                        }

                        URMeanXy = u[f2c[1]];
                        this->UFromCell2Face(URMeanXy, iFace, f2c[1], 1);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        if constexpr (gDim == 2)
                            GradURxy({0, 1}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 2>{1, 2}, 3) *
                                uRec[f2c[1]] * IF_NOT_NOREC * uRecBeta[f2c[1]](0); // 2d here
                        else
                            GradURxy({0, 1, 2}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 3>{1, 2, 3}, 4) *
                                uRec[f2c[1]] * IF_NOT_NOREC * uRecBeta[f2c[1]](0); // 3d here
                        this->DiffUFromCell2Face(GradURxy, iFace, f2c[1], 1);

#endif

                        minVol = std::min(minVol, vfv->GetCellVol(f2c[1]));
                    }
                    else if (true)
                    {
                        URxy = generateBoundaryValue(
                            ULxy, ULMeanXy, f2c[0], iFace,
                            unitNorm,
                            normBase,
                            vfv->GetFaceQuadraturePPhys(iFace, -1)(Seq012),
                            t,
                            mesh->GetFaceZone(iFace), true);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        GradURxy = GradULxy; //! generated boundary value couldn't use any periodic conversion?
#endif
                        URMeanXy = generateBoundaryValue(
                            ULMeanXy, ULMeanXy, f2c[0], iFace,
                            unitNorm,
                            normBase,
                            vfv->GetFaceQuadraturePPhys(iFace, -1)(Seq012),
                            t,
                            mesh->GetFaceZone(iFace), false);
                    }
                    PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterB);
                    // UR = URxy;
                    // UL = ULxy;
                    // UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});
                    // UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
                    real distGRP = minVol / vfv->GetFaceArea(iFace) * 2;
#ifdef USE_DISABLE_DIST_GRP_FIX_AT_WALL
                    distGRP +=
                        (faceBCType == EulerBCType::BCWall ||
                         faceBCType == EulerBCType::BCWallInvis)
                            ? veryLargeReal
                            : 0.0;
#endif
                    // real distGRP = (vfv->cellBaries[f2c[0]] -
                    //                 (f2c[1] != FACE_2_VOL_EMPTY
                    //                      ? vfv->cellBaries[f2c[1]]
                    //                      : 2 * vfv->faceCenters[iFace] - vfv->cellBaries[f2c[0]]))
                    //                    .norm();
                    // InsertCheck(u.father->getMPI(), "RHS inner 1");
                    TU UMeanXy = 0.5 * (ULxy + URxy);

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradUMeanXy = (GradURxy + GradULxy) * 0.5 +
                                         (1.0 / distGRP) *
                                             (unitNorm * (URxy - ULxy).transpose());

#else
                    TDiffU GradUMeanXy;
#endif

                    TU FLFix, FRFix;
                    FLFix.setZero(cnvars), FRFix.setZero(cnvars);
                    if (!GradUMeanXy.allFinite())
                    {
                        std::cout << GradURxy << std::endl;
                        std::cout << GradULxy << std::endl;
                        std::cout << distGRP << std::endl;
                        std::cout << f2c[0] << " " << f2c[1] << " " << mesh->NumCell() << " " << mesh->NumCellProc() << std::endl;
                        std::cout << uRec[f2c[0]] << std::endl;
                        std::cout << "-----------------------------------\n";
                        if (f2c[1] != UnInitIndex)
                            std::cout << uRec[f2c[1]] << std::endl;
                        std::cout << "-----------------------------------\n";
                        std::cout << u[f2c[0]].transpose() << std::endl;
                        if (f2c[1] != UnInitIndex)
                            std::cout << u[f2c[1]].transpose() << std::endl;
                        DNDS_assert(false);
                    }
                    TU fincC = fluxFace(
                        ULxy,
                        URxy,
                        ULMeanXy,
                        URMeanXy,
                        GradUMeanXy,
                        unitNorm,
                        normBase,
                        FLFix, FRFix,
                        mesh->GetFaceZone(iFace),
                        rsType,
                        iFace, iG);

                    finc(Eigen::all, 0) = fincC;
#ifdef USE_FLUX_BALANCE_TERM
                    finc(Eigen::all, 1) = FLFix;
                    finc(Eigen::all, 2) = FRFix;
#endif

                    finc *= vfv->GetFaceJacobiDet(iFace, iG); // don't forget this

                    if (pointOrderReducedL)
                        nFaceReducedOrder++, faceOrderReducedL = false;
                    if (pointOrderReducedR)
                        nFaceReducedOrder++, faceOrderReducedR = false;
                    if (faceOrderReducedL)
                        nFaceReducedOrder++;
                    if (faceOrderReducedR)
                        nFaceReducedOrder++;

                    // if (iFace == 0)
                    // {
                    //     std::cout << finc.transpose() << std::endl;
                    //     DNDS_assert(false);
                    // }
                });

            // if (f2c[0] == 10756)
            // {
            //     std::cout << std::setprecision(16)
            //               << fluxEs(Eigen::all, 0).transpose() << std::endl;
            //     // exit(-1);
            // }

            TU fluxIncL = fluxEs(Eigen::all, 0);
            TU fluxIncR = -fluxEs(Eigen::all, 0);
#ifdef USE_FLUX_BALANCE_TERM
            fluxIncL -= fluxEs(Eigen::all, 1);
            fluxIncR += fluxEs(Eigen::all, 2);
#endif
            this->UFromFace2Cell(fluxIncL, iFace, f2c[0], 0);
            if (f2c[1] != UnInitIndex)
                this->UFromFace2Cell(fluxIncR, iFace, f2c[1], 1); // periodic back to cell
            real alphaFace = cellRHSAlpha[f2c[0]](0);
            if (f2c[1] != UnInitIndex)
                alphaFace = std::min(alphaFace, cellRHSAlpha[f2c[1]](0));

            rhs[f2c[0]] += fluxIncL / vfv->GetCellVol(f2c[0]);
            if (f2c[1] != UnInitIndex)
                rhs[f2c[1]] += fluxIncR / vfv->GetCellVol(f2c[1]);

            if (faceBCType == EulerBCType::BCWall || // TODO: update to general
                (faceBCType == EulerBCType::BCWallInvis && settings.idealGasProperty.muGas < 1e-99))
            {
                fluxWallSumLocal -= fluxEs(Eigen::all, 0);
            }
            if (f2c[1] == UnInitIndex)
            {
                DNDS_assert(mesh->face2bnd.find(iFace) != mesh->face2bnd.end());
                fluxBnd.at(mesh->face2bnd[iFace]) = fluxEs(Eigen::all, 0) / vfv->GetFaceArea(iFace);
            }
        }

        InsertCheck(u.father->getMPI(), "EvaluateRHS After Flux");

        if (!settings.ignoreSourceTerm)
        {
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) // force zero source jacobian
                JSource[iCell].setZero();

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                if (onlyOnHalfAlpha)
                {
                    //     if (!cellIsHalfAlpha(iCell))
                    //         continue;
                }
                auto gCell = vfv->GetCellQuad(iCell);

                Eigen::Vector<real, nvarsFixedMultiply<nVars_Fixed, 2>()> sourceV(cnvars * 2); // now includes sourcejacobian diag
                sourceV.setZero();

                Geom::Elem::SummationNoOp noOp;
                bool cellOrderReduced = false;

#ifdef USE_TOTAL_REDUCED_ORDER_CELL
                gCell.IntegrationSimple(
                    noOp,
                    [&](decltype(noOp) &finc, int ig)
                    {
                        if (!cellOrderReduced)
                        {
                            TU ULxy =
                                (vfv->GetIntPointDiffBaseValue(iCell, -1, -1, ig, std::array<int, 1>{0}, 1) *
                                 uRec[iCell])
                                    .transpose() *
                                IF_NOT_NOREC * uRecBeta[iCell](0);
                            ULxy = CompressRecPart(u[iCell], ULxy, cellOrderReduced);
                        }
                    });
#endif

                gCell.IntegrationSimple(
                    sourceV,
                    [&](decltype(sourceV) &finc, int iG)
                    {
                        TDiffU GradU;
                        GradU.resize(Eigen::NoChange, cnvars);
                        GradU.setZero();
                        PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterB);
                        if constexpr (gDim == 2)
                            GradU({0, 1}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 2>{1, 2}, 3) *
                                uRec[iCell] * IF_NOT_NOREC * uRecBeta[iCell](0); // 2d specific
                        else
                            GradU({0, 1, 2}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 3>{1, 2, 3}, 4) *
                                uRec[iCell] * IF_NOT_NOREC * uRecBeta[iCell](0); // 3d specific

                        bool pointOrderReduced;
                        TU ULxy = u[iCell];
                        if (!cellOrderReduced)
                        {
                            // ULxy += cellDiBjGaussBatchElemVR.m(ig).row(0).rightCols(uRec[iCell].rows()) *
                            //         uRec[iCell] * IF_NOT_NOREC;
                            ULxy = CompressRecPart(
                                ULxy,
                                (vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 1>{0}, 1) *
                                 uRec[iCell])
                                        .transpose() *
                                    IF_NOT_NOREC * uRecBeta[iCell](0),
                                pointOrderReduced);
                        }
                        PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterB);

                        // bool compressed = false;
                        // ULxy = CompressRecPart(u[iCell], ULxy, compressed); //! do not forget the mean value

                        finc.resizeLike(sourceV);
                        if constexpr (nVars_Fixed > 0)
                        {
                            finc(Eigen::seq(Eigen::fix<0>, Eigen::fix<nVars_Fixed - 1>)) =
                                source(
                                    ULxy,
                                    GradU,
                                    iCell, iG);
                            finc(Eigen::seq(Eigen::fix<nVars_Fixed>, Eigen::fix<2 * nVars_Fixed - 1>)) =
                                sourceJacobianDiag(
                                    ULxy,
                                    GradU,
                                    iCell, iG);
                        }
                        else
                        {
                            finc(Eigen::seq(0, cnvars - 1)) =
                                source(
                                    ULxy,
                                    GradU,
                                    iCell, iG);
                            finc(Eigen::seq(cnvars, 2 * cnvars - 1)) =
                                sourceJacobianDiag(
                                    ULxy,
                                    GradU,
                                    iCell, iG);
                        }

                        finc *= vfv->GetCellJacobiDet(iCell, iG); // don't forget this
                        if (finc.hasNaN() || (!finc.allFinite()))
                        {
                            std::cout << finc.transpose() << std::endl;
                            std::cout << ULxy.transpose() << std::endl;
                            std::cout << GradU << std::endl;
                            DNDS_assert(false);
                        }
                    });
                sourceV *= cellRHSAlpha[iCell](0);
                if constexpr (nVars_Fixed > 0)
                {
                    rhs[iCell] += sourceV(Eigen::seq(Eigen::fix<0>, Eigen::fix<nVars_Fixed - 1>)) / vfv->GetCellVol(iCell);
                    JSource[iCell] = sourceV(Eigen::seq(Eigen::fix<nVars_Fixed>, Eigen::fix<2 * nVars_Fixed - 1>)) / vfv->GetCellVol(iCell);
                }
                else
                {
                    rhs[iCell] += sourceV(Eigen::seq(0, cnvars - 1)) / vfv->GetCellVol(iCell);
                    JSource[iCell] = sourceV(Eigen::seq(cnvars, 2 * cnvars - 1)) / vfv->GetCellVol(iCell);
                }
                // if (iCell == 18195)
                // {
                //     std::cout << rhs[iCell].transpose() << std::endl;
                // }
            }
        }
        // quick aux: reduce the wall flux sum
        MPI::Allreduce(fluxWallSumLocal.data(), fluxWallSum.data(), fluxWallSum.size(), DNDS_MPI_REAL, MPI_SUM, u.father->getMPI().comm);

        InsertCheck(u.father->getMPI(), "EvaluateRHS -1");
    }
}