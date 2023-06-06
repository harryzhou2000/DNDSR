#include "EulerEvaluator.hpp"

namespace DNDS::Euler
{
#define IF_NOT_NOREC (1)
    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateRHS(
        ArrayDOFV<nVars_Fixed> &rhs,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec, real t)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->mpi, "EvaluateRHS 1");
        int cnvars = nVars;
        typename Setting::RiemannSolverType rsType = settings.rsType;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            rhs[iCell].setZero();
        }
        TU fluxWallSumLocal;
        fluxWallSumLocal.setZero(cnvars);
        fluxWallSum.setZero(cnvars);
        nFaceReducedOrder = 0;

        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            auto gFace = vfv->GetFaceQuad(iFace);
#ifdef USE_FLUX_BALANCE_TERM
            Eigen::Matrix<real, nVars_Fixed, 3, Eigen::ColMajor> fluxEs(cnvars, 3);
#else
            Eigen::Matrix<real, nVars_Fixed, 1, Eigen::ColMajor> fluxEs(cnvars, 1);
#endif

            fluxEs.setZero();

            // auto f2n = mesh->face2node[iFace];
            // Geom::tSmallCoords coords;
            // mesh->LoadCoords(f2n, coords);

            Geom::Elem::SummationNoOp noOp;
            bool faceOrderReducedL = false;
            bool faceOrderReducedR = false;

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
                            IF_NOT_NOREC;
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
                                IF_NOT_NOREC;
                            URxy = CompressRecPart(u[f2c[1]], URxy, faceOrderReducedR);
                        }
                    }
                });
#endif

            gFace.IntegrationSimple(
                fluxEs,
                [&](decltype(fluxEs) &finc, int iG)
                {
                    int nDiff = vfv->faceAtr[iFace].NDIFF;
                    TVec unitNorm = vfv->faceUnitNorm(iFace, iG)(Seq012);
                    TMat normBase = Geom::NormBuildLocalBaseV(unitNorm);
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
                                IF_NOT_NOREC,
                            pointOrderReducedL);
                    }

                    TU ULMeanXy = u[f2c[0]];
                    TU URMeanXy;

                    TU URxy;
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradULxy, GradURxy;
                    GradULxy.resize(Eigen::NoChange, cnvars);
                    GradURxy.resize(Eigen::NoChange, cnvars);
                    GradULxy.setZero(), GradURxy.setZero();

                    if constexpr (gDim == 2)
                        GradULxy({0, 1}, Eigen::all) =
                            vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 2>{1, 2}, 2) *
                            uRec[f2c[0]] * IF_NOT_NOREC; // 2d here
                    else
                        GradULxy({0, 1, 2}, Eigen::all) =
                            vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, iG, std::array<int, 3>{1, 2, 3}, 3) *
                            uRec[f2c[0]] * IF_NOT_NOREC; // 3d here

#endif
                    real minVol = vfv->volumeLocal[f2c[0]];
                    // InsertCheck(u.father->mpi, "RHS inner 2");

                    if (f2c[1] != UnInitIndex)
                    {
                        URxy = u[f2c[1]];
                        if (!faceOrderReducedR)
                        {
                            URxy = CompressRecPart(
                                URxy,
                                (vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 1>{0}, 1) *
                                 uRec[f2c[1]])
                                        .transpose() *
                                    IF_NOT_NOREC,
                                pointOrderReducedR);
                        }

                        URMeanXy = u[f2c[1]];
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        if constexpr (gDim == 2)
                            GradURxy({0, 1}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 2>{1, 2}, 2) *
                                uRec[f2c[1]] * IF_NOT_NOREC; // 2d here
                        else
                            GradURxy({0, 1, 2}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, iG, std::array<int, 3>{1, 2, 3}, 3) *
                                uRec[f2c[1]] * IF_NOT_NOREC; // 3d here

#endif

                        minVol = std::min(minVol, vfv->volumeLocal[f2c[1]]);
                    }
                    else if (true)
                    {
                        URxy = generateBoundaryValue(
                            ULxy,
                            unitNorm,
                            normBase,
                            vfv->faceCent[iFace](Seq012),
                            t,
                            mesh->GetFaceZone(iFace), true);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                        GradURxy = GradULxy;
#endif
                        URMeanXy = generateBoundaryValue(
                            ULMeanXy,
                            unitNorm,
                            normBase,
                            vfv->faceCent[iFace](Seq012),
                            t,
                            mesh->GetFaceZone(iFace), false);
                    }
                    PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterB);
                    // UR = URxy;
                    // UL = ULxy;
                    // UR({1, 2, 3}) = normBase.transpose() * UR({1, 2, 3});
                    // UL({1, 2, 3}) = normBase.transpose() * UL({1, 2, 3});
                    real distGRP = minVol / vfv->faceArea[iFace] * 2;
#ifdef USE_DISABLE_DIST_GRP_FIX_AT_WALL
                    distGRP +=
                        (mesh->GetFaceZone(iFace) == Geom::BC_ID_DEFAULT_WALL || mesh->GetFaceZone(iFace) == Geom::BC_ID_DEFAULT_WALL_INVIS
                             ? veryLargeReal
                             : 0.0);
#endif
                    // real distGRP = (vfv->cellBaries[f2c[0]] -
                    //                 (f2c[1] != FACE_2_VOL_EMPTY
                    //                      ? vfv->cellBaries[f2c[1]]
                    //                      : 2 * vfv->faceCenters[iFace] - vfv->cellBaries[f2c[0]]))
                    //                    .norm();
                    // InsertCheck(u.father->mpi, "RHS inner 1");
                    TU UMeanXy = 0.5 * (ULxy + URxy);

#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
                    TDiffU GradUMeanXy = (GradURxy + GradULxy) * 0.5 +
                                         (1.0 / distGRP) *
                                             (unitNorm * (URxy - ULxy).transpose());

#else
                    TDiffU GradUMeanXy;
#endif

                    TU FLFix, FRFix;
                    FLFix.setZero(), FRFix.setZero();

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

                    finc *= vfv->faceIntJacobiDet(iFace, iG); // don't forget this

                    if (pointOrderReducedL)
                        nFaceReducedOrder++, faceOrderReducedL = false;
                    if (pointOrderReducedR)
                        nFaceReducedOrder++, faceOrderReducedR = false;
                    if (faceOrderReducedL)
                        nFaceReducedOrder++;
                    if (faceOrderReducedR)
                        nFaceReducedOrder++;
                });

            // if (f2c[0] == 10756)
            // {
            //     std::cout << std::setprecision(16)
            //               << fluxEs(Eigen::all, 0).transpose() << std::endl;
            //     // exit(-1);
            // }

            rhs[f2c[0]] += fluxEs(Eigen::all, 0) / vfv->volumeLocal[f2c[0]];
            if (f2c[1] != UnInitIndex)
                rhs[f2c[1]] -= fluxEs(Eigen::all, 0) / vfv->volumeLocal[f2c[1]];
#ifdef USE_FLUX_BALANCE_TERM
            rhs[f2c[0]] -= fluxEs(Eigen::all, 1) / vfv->volumeLocal[f2c[0]];
            if (f2c[1] != UnInitIndex)
                rhs[f2c[1]] += fluxEs(Eigen::all, 2) / vfv->volumeLocal[f2c[1]];
#endif

            if (mesh->GetFaceZone(iFace) == Geom::BC_ID_DEFAULT_WALL ||
                mesh->GetFaceZone(iFace) == Geom::BC_ID_DEFAULT_WALL_INVIS)
            {
                fluxWallSumLocal -= fluxEs(Eigen::all, 0);
            }
        }
        // quick aux: reduce the wall flux sum
        MPI_Allreduce(fluxWallSumLocal.data(), fluxWallSum.data(), fluxWallSum.size(), DNDS_MPI_REAL, MPI_SUM, u.father->mpi.comm);

        InsertCheck(u.father->mpi, "EvaluateRHS After Flux");

        if (!settings.ignoreSourceTerm)
        {
            for (index iCell = 0; iCell < jacobianCellSourceDiag.size(); iCell++) // force zero source jacobian
                jacobianCellSourceDiag[iCell].setZero();

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                auto gCell = vfv->GetCellQuad(iCell);

                Eigen::Vector<real, nvarsFixedMultipy<nVars_Fixed, 2>()> sourceV(cnvars * 2); // now includes sourcejacobian diag
                sourceV.setZero();

                Geom::Elem::SummationNoOp noOp;
                bool cellOrderReduced = false;

#ifdef USE_TOTAL_REDUCED_ORDER_CELL
                eCell.IntegrationSimple(
                    noOp,
                    [&](decltype(noOp) &finc, int ig)
                    {
                        if (!cellOrderReduced)
                        {
                            TU ULxy =
                                (vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 1>{0}, 1) *
                                 uRec[iCell])
                                    .transpose() *
                                IF_NOT_NOREC;
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
                                vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 2>{1, 2}, 2) *
                                uRec[iCell] * IF_NOT_NOREC; // 2d specific
                        else
                            GradU({0, 1, 2}, Eigen::all) =
                                vfv->GetIntPointDiffBaseValue(iCell, -1, -1, iG, std::array<int, 3>{1, 2, 3}, 3) *
                                uRec[iCell] * IF_NOT_NOREC; // 3d specific

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
                                    IF_NOT_NOREC,
                                pointOrderReduced);
                        }
                        PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterB);

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

                        finc *= vfv->cellIntJacobiDet(iCell, iG); // don't forget this
                        if (finc.hasNaN() || (!finc.allFinite()))
                        {
                            std::cout << finc.transpose() << std::endl;
                            std::cout << ULxy.transpose() << std::endl;
                            std::cout << GradU << std::endl;
                            DNDS_assert(false);
                        }
                    });
                if constexpr (nVars_Fixed > 0)
                {
                    rhs[iCell] += sourceV(Eigen::seq(Eigen::fix<0>, Eigen::fix<nVars_Fixed - 1>)) / vfv->volumeLocal[iCell];
                    jacobianCellSourceDiag[iCell] = sourceV(Eigen::seq(Eigen::fix<nVars_Fixed>, Eigen::fix<2 * nVars_Fixed - 1>)) / vfv->volumeLocal[iCell];
                }
                else
                {
                    rhs[iCell] += sourceV(Eigen::seq(0, cnvars - 1)) / vfv->volumeLocal[iCell];
                    jacobianCellSourceDiag[iCell] = sourceV(Eigen::seq(cnvars, 2 * cnvars - 1)) / vfv->volumeLocal[iCell];
                }
                // if (iCell == 18195)
                // {
                //     std::cout << rhs[iCell].transpose() << std::endl;
                // }
            }
        }
        InsertCheck(u.father->mpi, "EvaluateRHS -1");
    }

    template void EulerEvaluator<NS>::EvaluateRHS(
        ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec, real t);
    template void EulerEvaluator<NS_SA>::EvaluateRHS(
        ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec, real t);
    template void EulerEvaluator<NS_2D>::EvaluateRHS(
        ArrayDOFV<nVars_Fixed> &rhs, ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec, real t);
}