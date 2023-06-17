#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    template <EulerModel model>
    void EulerSolver<model>::RunImplicitEuler()
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(mpi, "Implicit 1 nvars " + std::to_string(nVars));

        std::shared_ptr<ODE::ImplicitDualTimeStep<decltype(u)>> ode;

        if (config.steadyQuit)
        {
            if (mpi.rank == 0)
                log() << "Using steady!" << std::endl;
            config.odeCode = 1; // To bdf;
            config.nTimeStep = 1;
        }
        switch (config.odeCode)
        {
        case 0: // sdirk4
            if (mpi.rank == 0)
                log() << "=== ODE: SDIRK4 " << std::endl;
            ode = std::make_shared<ODE::ImplicitSDIRK4DualTimeStep<decltype(u)>>(
                mesh->NumCell(),
                [&](decltype(u) &data)
                {
                    vfv->BuildUDof(data, nVars);
                });
            break;
        case 1: // BDF2
            if (mpi.rank == 0)
                log() << "=== ODE: BDF2 " << std::endl;
            ode = std::make_shared<ODE::ImplicitBDFDualTimeStep<decltype(u)>>(
                mesh->NumCell(),
                [&](decltype(u) &data)
                {
                    vfv->BuildUDof(data, nVars);
                },
                2);
            break;
        }

        Linear::GMRES_LeftPreconditioned<decltype(u)> gmres(
            config.nGmresSpace,
            [&](decltype(u) &data)
            {
                vfv->BuildUDof(data, nVars);
            });

        Linear::GMRES_LeftPreconditioned<decltype(uRec)> gmresRec(
            5,
            [&](decltype(uRec) &data)
            {
                vfv->BuildURec(data, nVars);
            });

        EulerEvaluator<model> eval(mesh, vfv);
        eval.settings.jsonSettings = config.eulerSettings;
        eval.settings.ParseFromJson(nVars);

        eval.InitializeUDOF(u);

        /************* Files **************/
        if (mpi.rank == 0)
        {
            std::ofstream logConfig(config.outLogName + "_" + output_stamp + ".config.json");
            gSetting["___Compile_Time_Defines"] = DNDS_Defines_state;
            gSetting["___Runtime_PartitionNumber"] = mpi.size;
            logConfig << std::setw(4) << gSetting;
            logConfig.close();
        }

        std::ofstream logErr(config.outLogName + "_" + output_stamp + ".log");
        /************* Files **************/

        double tstart = MPI_Wtime();
        double trec{0}, tcomm{0}, trhs{0}, tLim{0};
        int stepCount = 0;
        Eigen::Vector<real, -1> resBaseC;
        Eigen::Vector<real, -1> resBaseCInternal;
        resBaseC.resize(nVars);
        resBaseCInternal.resize(nVars);
        resBaseC.setConstant(config.res_base);

        real tSimu = 0.0;
        real nextTout = config.tDataOut;
        int nextStepOut = config.nDataOut;
        int nextStepOutC = config.nDataOutC;
        PerformanceTimer::Instance().clearAllTimer();

        // *** Loop variables
        real CFLNow = config.CFL;
        bool ifOutT = false;
        real curDtMin;
        real curDtImplicit = config.dtImplicit;
        int step;
        bool gradIsZero = true;

        InsertCheck(mpi, "Implicit 2 nvars " + std::to_string(nVars));
        /*******************************************************/
        /*                   DEFINE LAMBDAS                    */
        /*******************************************************/
        auto frhs = [&](ArrayDOFV<nVars_Fixed> &crhs, ArrayDOFV<nVars_Fixed> &cx, int iter, real ct)
        {
            eval.FixUMaxFilter(cx);
            // cx.trans.startPersistentPull();
            // cx.trans.waitPersistentPull();

            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uOld[iCell].m() = uRec[iCell].m();

            InsertCheck(mpi, " Lambda RHS: StartRec");
            int nRec = (gradIsZero ? 5 : 1) * config.nInternalRecStep;
            real recIncBase = 0;
            double tstartA = MPI_Wtime();
            auto FBoundary = [&](const TU &UL, const TU &UMean, const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                auto normBase = Geom::NormBuildLocalBaseV(normOutV);
                bool compressed = false;
                TU ULfixed = eval.CompressRecPart(
                    UMean,
                    UL - UMean,
                    compressed);
                return eval.generateBoundaryValue(ULfixed, normOutV, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true);
            };
            auto FBoundaryDiff = [&](const TU &UL, const TU &dU, const TU &UMean, const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                auto normBase = Geom::NormBuildLocalBaseV(normOutV);
                bool compressed = false;
                TU ULfixed = eval.CompressRecPart(
                    UMean,
                    UL - UMean,
                    compressed);
                TU ULfixedPlus = eval.CompressRecPart(
                    UMean,
                    UL - UMean + dU,
                    compressed);
                return eval.generateBoundaryValue(ULfixedPlus, normOutV, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true) -
                       eval.generateBoundaryValue(ULfixed, normOutV, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true);
            };
            for (int iRec = 1; iRec <= 300 * 0; iRec++)
            {

                uRecNew1 = uRec;

                vfv->DoReconstructionIter(
                    uRec, uRecNew, cx,
                    // FBoundary
                    [&](const TU &UL, const TU &UMean, const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
                    {
                        TVec normOutV = normOut(Seq012);
                        auto normBase = Geom::NormBuildLocalBaseV(normOutV);
                        bool compressed = false;
                        TU ULfixed = eval.CompressRecPart(
                            UMean,
                            UL - UMean,
                            compressed);
                        return eval.generateBoundaryValue(ULfixed, normOutV, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true);
                    },
                    false);

                uRec.trans.startPersistentPull();
                uRec.trans.waitPersistentPull();

                uRecNew1 -= uRec;
                real recInc = uRecNew1.norm2();
                if (iRec == 1)
                    recIncBase = recInc;

                if (recInc < recIncBase * 1e-6)
                {
                    if (mpi.rank == 0)
                        std::cout << iRec << " Rec inc: " << recIncBase << " -> " << recInc << std::endl;
                    break;
                }
            }

            for (int iRec = 1; iRec <= 1; iRec++)
            {
                uRecNew1 = uRec; // old value
                vfv->DoReconstructionIter(
                    uRec, uRecNew, cx,
                    FBoundary,
                    true, true);
                uRecNew.trans.startPersistentPull();
                uRecNew.trans.waitPersistentPull();
                uRec.setConstant(0.0);

                gmresRec.solve(
                    [&](decltype(uRec) &x, decltype(uRec) &Ax)
                    {
                        vfv->DoReconstructionIterDiff(uRecNew1, x, Ax, cx, FBoundaryDiff);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                    },
                    [&](decltype(uRec) &x, decltype(uRec) &MLx)
                    {
                        MLx = x; // initial value; for the input is mostly a good estimation
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        vfv->DoReconstructionIterSOR(uRecNew1, x, MLx, cx, FBoundaryDiff);
                    },
                    uRecNew, uRec, 20,
                    [&](uint32_t i, real res, real resB) -> bool
                    {
                        if (i > 0)
                        {
                            if (mpi.rank == 0)
                            {
                                log() << std::scientific;
                                log() << "GMRES for Rec: " << i << " " << resB << " -> " << res << std::endl;
                            }
                        }
                        return res < resB * 1e-6;
                    });
                uRecNew1.addTo(uRec, -1);
                uRec = uRecNew1;
            }
            trec += MPI_Wtime() - tstartA;
            gradIsZero = false;
            double tstartH = MPI_Wtime();

            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uRec[iCell].m() -= uOld[iCell].m();

            InsertCheck(mpi, " Lambda RHS: StartLim");
            if (config.useLimiter)
            {
                // vfv->ReconstructionWBAPLimitFacial(
                //     cx, uRec, uRecNew, uF0, uF1, ifUseLimiter,

                auto fML = [&](const auto &UL, const auto &UR, const auto &n) -> auto
                {
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234)*0.5;
                    auto normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasLeftEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Eigen::all, Seq123) *= normBase.transpose();

                    Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;
                    PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterA);
                    return ret;
                    // return real(1);
                };
                auto fMR = [&](const auto &UL, const auto &UR, const auto &n) -> auto
                {
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234)*0.5;
                    auto normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasRightEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Seq123, Eigen::all) = normBase * M(Seq123, Eigen::all);

                    Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;

                    PerformanceTimer::Instance().EndTimer(PerformanceTimer::LimiterA);
                    return ret;
                    // return real(1);
                };
                if (config.smoothIndicatorProcedure == 0)
                    vfv->DoCalculateSmoothIndicator(
                        ifUseLimiter, (uRec), (u),
                        std::array<int, 2>{0, I4});
                else if (config.smoothIndicatorProcedure == 1)
                {
                    if constexpr (dim == 2)
                        vfv->DoCalculateSmoothIndicatorV1(
                            ifUseLimiter, (uRec), (u),
                            std::array<int, 4>{0, 1, 2, 3},
                            [&](auto &v)
                            {
                                TU prim;
                                TU cons;
                                cons.setZero();
                                cons(Seq01234) = v.transpose();
                                Gas::IdealGasThermalConservative2Primitive<dim>(cons, prim, eval.settings.idealGasProperty.gamma);
                                v.setConstant(prim(I4));
                            });
                    else
                        vfv->DoCalculateSmoothIndicatorV1(
                            ifUseLimiter, (uRec), (u),
                            std::array<int, 5>{0, 1, 2, 3, 4},
                            [&](auto &v)
                            {
                                TU prim;
                                TU cons;
                                cons.setZero();
                                cons(Seq01234) = v.transpose();
                                Gas::IdealGasThermalConservative2Primitive<dim>(cons, prim, eval.settings.idealGasProperty.gamma);
                                v.setConstant(prim(I4));
                            });
                }
                else
                {
                    DNDS_assert(false);
                }
                if (config.limiterProcedure == 1)
                    vfv->DoLimiterWBAP_C(
                        eval,
                        (cx),
                        (uRec),
                        (uRecNew),
                        (uRecNew1),
                        ifUseLimiter,
                        iter < config.nPartialLimiterStartLocal && step < config.nPartialLimiterStart,
                        fML, fMR, true);
                else
                {
                    DNDS_assert(false);
                }
                // uRecNew.trans.startPersistentPull();
                // uRecNew.trans.waitPersistentPull();
            }
            tLim += MPI_Wtime() - tstartH;

            // uRec.trans.startPersistentPull(); //! this also need to update!
            // uRec.trans.waitPersistentPull();

            // }

            InsertCheck(mpi, " Lambda RHS: StartEval");
            double tstartE = MPI_Wtime();
            eval.setPassiveDiscardSource(iter <= 0);
            if (config.useLimiter)
                eval.EvaluateRHS(crhs, cx, uRecNew, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, cx, uRec, tSimu + ct * curDtImplicit);
            if (getNVars(model) > (I4 + 1) && iter <= config.nFreezePassiveInner)
            {
                for (int i = 0; i < crhs.Size(); i++)
                    crhs[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
            trhs += MPI_Wtime() - tstartE;

            InsertCheck(mpi, " Lambda RHS: End");
        };

        auto fdtau = [&](std::vector<real> &dTau, real alphaDiag)
        {
            eval.FixUMaxFilter(u);
            u.trans.startPersistentPull(); //! this also need to update!
            u.trans.waitPersistentPull();
            // uRec.trans.startPersistentPull();
            // uRec.trans.waitPersistentPull();

            eval.EvaluateDt(dTau, u, CFLNow, curDtMin, 1e100, config.useLocalDt);
            for (auto &i : dTau)
                i /= alphaDiag;
        };

        auto fsolve = [&](ArrayDOFV<nVars_Fixed> &cx, ArrayDOFV<nVars_Fixed> &crhs, std::vector<real> &dTau,
                          real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &cxInc, int iter)
        {
            cxInc.setConstant(0.0);

            if (config.useLimiter) // uses urec value
                eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                     cx, uRecNew,
                                     config.jacobianTypeCode,
                                     tSimu);
            else
                eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                     cx, uRec,
                                     config.jacobianTypeCode,
                                     tSimu);

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                crhs[iCell] = eval.CompressInc(cx[iCell], crhs[iCell] * dTau[iCell], crhs[iCell]) / dTau[iCell];
            }

            if (config.gmresCode == 0 || config.gmresCode == 2)
            {
                // //! LUSGS

                eval.UpdateLUSGSForward(alphaDiag, crhs, cx, cxInc, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                eval.UpdateLUSGSBackward(alphaDiag, crhs, cx, cxInc, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    cxInc[iCell] = eval.CompressInc(cx[iCell], cxInc[iCell], crhs[iCell]);
            }

            if (config.gmresCode != 0)
            {
                // !  GMRES
                // !  for gmres solver: A * uinc = rhsinc, rhsinc is average value insdead of cumulated on vol
                gmres.solve(
                    [&](decltype(u) &x, decltype(u) &Ax)
                    {
                        eval.LUSGSMatrixVec(alphaDiag, cx, x, Ax);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                    },
                    [&](decltype(u) &x, decltype(u) &MLx)
                    {
                        // x as rhs, and MLx as uinc

                        if (config.jacobianTypeCode == 0)
                        {
                            eval.UpdateLUSGSForward(alphaDiag, x, cx, MLx, MLx);
                            MLx.trans.startPersistentPull();
                            MLx.trans.waitPersistentPull();
                            eval.UpdateLUSGSBackward(alphaDiag, x, cx, MLx, MLx);
                            MLx.trans.startPersistentPull();
                            MLx.trans.waitPersistentPull();
                        }
                    },
                    crhs, cxInc, config.nGmresIter,
                    [&](uint32_t i, real res, real resB) -> bool
                    {
                        if (i > 0)
                        {
                            if (mpi.rank == 0)
                            {
                                // log() << std::scientific;
                                // log() << "GMRES: " << i << " " << resB << " -> " << res << std::endl;
                            }
                        }
                        return false;
                    });
                for (index iCell = 0; iCell < cxInc.Size(); iCell++)
                    cxInc[iCell] = eval.CompressInc(cx[iCell], cxInc[iCell], crhs[iCell]); // manually add fixing for gmres results
            }
            // !freeze something
            if (getNVars(model) > I4 + 1 && iter <= config.nFreezePassiveInner)
            {
                for (int i = 0; i < crhs.Size(); i++)
                    cxInc[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
        };

        auto fstop = [&](int iter, ArrayDOFV<nVars_Fixed> &cxinc, int iStep) -> bool
        {
            Eigen::Vector<real, -1> res(nVars);
            eval.EvaluateResidual(res, cxinc, 1, config.useVolWiseResidual);
            // if (iter == 1 && iStep == 1) // * using 1st rk step for reference
            if (iter == 1)
                resBaseCInternal = res;
            else
                resBaseCInternal = resBaseCInternal.array().max(res.array()); //! using max !
            Eigen::Vector<real, -1> resRel = (res.array() / resBaseCInternal.array()).matrix();
            bool ifStop = resRel(0) < config.rhsThresholdInternal; // ! using only rho's residual
            if (iter % config.nConsoleCheckInternal == 0 || iter > config.nTimeStepInternal || ifStop)
            {
                double telapsed = MPI_Wtime() - tstart;
                if (mpi.rank == 0)
                {
                    tcomm = PerformanceTimer::Instance().getTimer(PerformanceTimer::Comm);
                    auto fmt = log().flags();
                    log() << std::setprecision(3) << std::scientific
                          << "\t Internal === Step [" << iStep << ", " << iter << "]   "
                          << "res \033[91m[" << resRel.transpose() << "]\033[39m   "
                          << "t,dTaumin,CFL,nFix \033[92m["
                          << tSimu << ", " << curDtMin << ", " << CFLNow << ", " << eval.nFaceReducedOrder << "]\033[39m   "
                          << std::setprecision(3) << std::fixed
                          << "Time [" << telapsed << "]   recTime ["
                          << trec << "]   rhsTime ["
                          << trhs << "]   commTime ["
                          << tcomm << "]  limTime ["
                          << tLim << "]  limtimeA ["
                          << PerformanceTimer::Instance().getTimer(PerformanceTimer::LimiterA) << "]  limtimeB ["
                          << PerformanceTimer::Instance().getTimer(PerformanceTimer::LimiterB) << "]  ";
                    if (config.consoleOutputMode == 1)
                    {
                        log() << std::setprecision(4) << std::setw(10) << std::scientific
                              << "Wall Flux \033[93m[" << eval.fluxWallSum.transpose() << "]\033[39m";
                    }
                    log() << std::endl;
                    log().setf(fmt);
                    std::string delimC = " ";
                    logErr
                        << std::left
                        << step << delimC
                        << std::left
                        << iter << delimC
                        << std::left
                        << std::setprecision(9) << std::scientific
                        << res.transpose() << delimC
                        << tSimu << delimC
                        << curDtMin << delimC
                        << real(eval.nFaceReducedOrder) << delimC
                        << eval.fluxWallSum.transpose() << std::endl;
                }
                tstart = MPI_Wtime();
                trec = tcomm = trhs = tLim = 0.;
                PerformanceTimer::Instance().clearAllTimer();
            }

            if (iter % config.nDataOutInternal == 0)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.outPltName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter), ode, eval);
                nextStepOut += config.nDataOut;
            }
            if (iter % config.nDataOutCInternal == 0)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.outPltName + "_" + output_stamp + "_" + "C", ode, eval);
                nextStepOutC += config.nDataOutC;
            }
            if (iter >= config.nCFLRampStart && iter <= config.nCFLRampLength + config.nCFLRampStart)
            {
                real inter = real(iter - config.nCFLRampStart) / config.nCFLRampLength;
                real logCFL = std::log(config.CFL) + (std::log(config.CFLRampEnd / config.CFL) * inter);
                CFLNow = std::exp(logCFL);
            }
            // return resRel.maxCoeff() < config.rhsThresholdInternal;
            return ifStop;
        };

        // fmainloop gets the time-variant residual norm,
        // handles the output / log nested loops,
        // integrates physical time tsimu
        // and finally decides if break time loop
        auto fmainloop = [&]() -> bool
        {
            tSimu += curDtImplicit;
            if (ifOutT)
                tSimu = nextTout;
            Eigen::Vector<real, -1> res(nVars);
            eval.EvaluateResidual(res, ode->getLatestRHS(), 1, config.useVolWiseResidual);
            if (stepCount == 0 && resBaseC.norm() == 0)
                resBaseC = res;

            if (step % config.nConsoleCheck == 0)
            {
                double telapsed = MPI_Wtime() - tstart;
                if (mpi.rank == 0)
                {
                    tcomm = PerformanceTimer::Instance().getTimer(PerformanceTimer::Comm);
                    auto fmt = log().flags();
                    log() << std::setprecision(3) << std::scientific
                          << "=== Step [" << step << "]   "
                          << "res \033[91m[" << (res.array() / resBaseC.array()).transpose() << "]\033[39m   "
                          << "t,dt(min) \033[92m[" << tSimu << ", " << curDtMin << "]\033[39m   "
                          << std::setprecision(3) << std::fixed
                          << "Time [" << telapsed << "]   recTime [" << trec << "]   rhsTime [" << trhs << "]   commTime [" << tcomm << "]  limTime [" << tLim << "]  " << std::endl;
                    log().setf(fmt);
                    std::string delimC = " ";
                    logErr
                        << std::left
                        << step << delimC
                        << std::left
                        << -1 << delimC
                        << std::left
                        << std::setprecision(9) << std::scientific
                        << res.transpose() << delimC
                        << tSimu << delimC
                        << curDtMin << delimC
                        << eval.fluxWallSum.transpose() << std::endl;
                }
                tstart = MPI_Wtime();
                trec = tcomm = trhs = tLim = 0.;
                PerformanceTimer::Instance().clearAllTimer();
            }
            if (step == nextStepOut)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.outPltName + "_" + output_stamp + "_" + std::to_string(step), ode, eval);
                nextStepOut += config.nDataOut;
            }
            if (step == nextStepOutC)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.outPltName + "_" + output_stamp + "_" + "C", ode, eval);
                nextStepOutC += config.nDataOutC;
            }
            if (ifOutT)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout), ode, eval);
                nextTout += config.tDataOut;
                if (nextTout > config.tEnd)
                    nextTout = config.tEnd;
            }
            if (eval.settings.specialBuiltinInitializer == 2 && (step % config.nConsoleCheck == 0)) // IV problem special: reduction on solution
            {
                real xymin = 5 + tSimu - 2;
                real xymax = 5 + tSimu + 2;
                real xyc = 5 + tSimu;
                real sumErrRho = 0.0;
                real sumErrRhoSum = 0.0 / 0.0;
                real sumVol = 0.0;
                real sumVolSum = 0.0 / 0.0;
                for (index iCell = 0; iCell < u.father->Size(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    real chi = 5;
                    real gamma = eval.settings.idealGasProperty.gamma;
                    auto c2n = mesh->cell2node[iCell];
                    auto gCell = vfv->GetCellQuad(iCell);
                    TU um;
                    um.setZero();
                    gCell.IntegrationSimple(
                        um,
                        [&](TU &inc, int ig)
                        {
                            // std::cout << coords<< std::endl << std::endl;
                            // std::cout << DiNj << std::endl;
                            Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);
                            real r = std::sqrt(sqr(pPhysics(0) - xyc) + sqr(pPhysics(1) - xyc));
                            real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
                            real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xyc);
                            real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - xyc);
                            real T = dT + 1;
                            real ux = dux + 1;
                            real uy = duy + 1;
                            real S = 1;
                            real rho = std::pow(T / S, 1 / (gamma - 1));
                            real p = T * rho;

                            real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                            // std::cout << T << " " << rho << std::endl;
                            inc.setZero();
                            inc(0) = rho;
                            inc(1) = rho * ux;
                            inc(2) = rho * uy;
                            inc(dim + 1) = E;

                            inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                        });
                    auto cP = vfv->GetCellBary(iCell);

                    if (cP(0) > xymin && cP(0) < xymax && cP(1) > xymin && cP(1) < xymax)
                    {
                        um /= vfv->GetCellVol(iCell); // mean value
                        real errRhoMean = u[iCell](0) - um(0);
                        sumErrRho += std::abs(errRhoMean) * vfv->GetCellVol(iCell);
                        sumVol += vfv->GetCellVol(iCell);
                    }
                }
                MPI_Allreduce(&sumErrRho, &sumErrRhoSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI_Allreduce(&sumVol, &sumVolSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                if (mpi.rank == 0)
                {
                    log() << "=== Mean Error IV: [" << std::scientific << std::setprecision(5) << sumErrRhoSum << ", " << sumErrRhoSum / sumVolSum << "]" << std::endl;
                }
            }
            if (config.zeroGrads)
                uRec.setConstant(0.0), gradIsZero = true;

            stepCount++;

            return tSimu >= config.tEnd;
        };

        /**********************************/
        /*           MAIN LOOP            */
        /**********************************/

        for (step = 1; step <= config.nTimeStep; step++)
        {
            InsertCheck(mpi, "Implicit Step");
            ifOutT = false;
            curDtImplicit = config.dtImplicit; //* could add CFL driven dt here
            if (tSimu + curDtImplicit > nextTout)
            {
                ifOutT = true;
                curDtImplicit = (nextTout - tSimu);
            }
            CFLNow = config.CFL;
            ode->Step(
                u, uInc,
                frhs,
                fdtau,
                fsolve,
                config.nTimeStepInternal,
                fstop,
                curDtImplicit + verySmallReal);

            if (fmainloop())
                break;
        }

        // u.trans.waitPersistentPull();
        logErr.close();
    }
}