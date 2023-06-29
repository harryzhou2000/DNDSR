#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    template <EulerModel model>
    void EulerSolver<model>::RunImplicitEuler()
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(mpi, "Implicit 1 nvars " + std::to_string(nVars));

        /************* Files **************/
        if (mpi.rank == 0)
        {
            std::ofstream logConfig(config.dataIOControl.outLogName + "_" + output_stamp + ".config.json");
            gSetting["___Compile_Time_Defines"] = DNDS_Defines_state;
            gSetting["___Runtime_PartitionNumber"] = mpi.size;
            logConfig << std::setw(4) << gSetting;
            logConfig.close();
        }

        std::ofstream logErr(config.dataIOControl.outLogName + "_" + output_stamp + ".log");
        /************* Files **************/

        std::shared_ptr<ODE::ImplicitDualTimeStep<decltype(u)>> ode;

        if (config.timeMarchControl.steadyQuit)
        {
            if (mpi.rank == 0)
                log() << "Using steady!" << std::endl;
            config.timeMarchControl.odeCode = 1; // To bdf;
            config.timeMarchControl.nTimeStep = 1;
        }
        switch (config.timeMarchControl.odeCode)
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
        case 2: // SSPRK4
            if (mpi.rank == 0)
                log() << "=== ODE: SSPRK4 " << std::endl;
            ode = std::make_shared<ODE::ExplicitSSPRK3TimeStepAsImplicitDualTimeStep<decltype(u)>>(
                mesh->NumCell(),
                [&](decltype(u) &data)
                {
                    vfv->BuildUDof(data, nVars);
                },
                false); // TODO: add local stepping options
            break;
        default:
            DNDS_assert_info(false, "no such ode code");
        }

        using tGMRES_u = Linear::GMRES_LeftPreconditioned<decltype(u)>;
        using tGMRES_uRec = Linear::GMRES_LeftPreconditioned<decltype(uRec)>;
        std::unique_ptr<tGMRES_u> gmres;
        std::unique_ptr<tGMRES_uRec> gmresRec;

        if (config.linearSolverControl.gmresCode == 1 ||
            config.linearSolverControl.gmresCode == 2)
            gmres = std::make_unique<tGMRES_u>(
                config.linearSolverControl.nGmresSpace,
                [&](decltype(u) &data)
                {
                    vfv->BuildUDof(data, nVars);
                });

        if (config.implicitReconstructionControl.recLinearScheme == 1)
            gmresRec = std::make_unique<tGMRES_uRec>(
                config.implicitReconstructionControl.nGmresSpace,
                [&](decltype(uRec) &data)
                {
                    vfv->BuildURec(data, nVars);
                });

        EulerEvaluator<model> eval(mesh, vfv);
        eval.settings.jsonSettings = config.eulerSettings;
        eval.settings.ReadWriteJSON(eval.settings.jsonSettings, nVars, true);

        eval.InitializeUDOF(u);

        double tstart = MPI_Wtime();
        double trec{0}, tcomm{0}, trhs{0}, tLim{0};
        int stepCount = 0;
        Eigen::Vector<real, -1> resBaseC;
        Eigen::Vector<real, -1> resBaseCInternal;
        resBaseC.resize(nVars);
        resBaseCInternal.resize(nVars);
        resBaseC.setConstant(config.convergenceControl.res_base);

        real tSimu = 0.0;
        real nextTout = config.outputControl.tDataOut;
        int nextStepOut = config.outputControl.nDataOut;
        int nextStepOutC = config.outputControl.nDataOutC;
        PerformanceTimer::Instance().clearAllTimer();

        // *** Loop variables
        real CFLNow = config.implicitCFLControl.CFL;
        bool ifOutT = false;
        real curDtMin;
        real curDtImplicit = config.timeMarchControl.dtImplicit;
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
            int nRec = (gradIsZero ? 1 : 1) * config.implicitReconstructionControl.nInternalRecStep;
            real recIncBase = 0;
            double tstartA = MPI_Wtime();
            auto FBoundary = [&](const TU &UL, const TU &UMean, const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
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
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
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
            if (config.implicitReconstructionControl.recLinearScheme == 0)
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    if (nRec > 1)
                        uRecNew1 = uRec;

                    vfv->DoReconstructionIter(
                        uRec, uRecNew, cx,
                        // FBoundary
                        [&](const TU &UL, const TU &UMean, const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
                        {
                            TVec normOutV = normOut(Seq012);
                            Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
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

                    if (nRec > 1)
                    {
                        uRecNew1 -= uRec;
                        real recInc = uRecNew1.norm2();
                        if (iRec == 1)
                            recIncBase = recInc;

                        if (iRec % config.implicitReconstructionControl.nRecConsolCheck == 0)
                        {
                            if (mpi.rank == 0)
                                std::cout << iRec << " Rec inc: " << recIncBase << " -> " << recInc << std::endl;
                        }
                        if (recInc < recIncBase * config.implicitReconstructionControl.recThreshold)
                            break;
                    }
                }
            else if (config.implicitReconstructionControl.recLinearScheme == 1)
            {
                int nGMRESrestartAll{0};
                real gmresResidualB = 0;
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    vfv->DoReconstructionIter(
                        uRec, uRecNew, cx,
                        FBoundary,
                        true, true);
                    uRecNew.trans.startPersistentPull();
                    uRecNew.trans.waitPersistentPull();
                    // uRec.setConstant(0.0);
                    uRecNew1 = uRecNew;
                    if (iRec == 1)
                        gmresResidualB = uRecNew1.norm2();

                    bool gmresConverge =
                        gmresRec->solve(
                            [&](decltype(uRec) &x, decltype(uRec) &Ax)
                            {
                                vfv->DoReconstructionIterDiff(uRec, x, Ax, cx, FBoundaryDiff);
                                Ax.trans.startPersistentPull();
                                Ax.trans.waitPersistentPull();
                            },
                            [&](decltype(uRec) &x, decltype(uRec) &MLx)
                            {
                                MLx = x; // initial value; for the input is mostly a good estimation
                                // MLx no need to comm
                                vfv->DoReconstructionIterSOR(uRec, x, MLx, cx, FBoundaryDiff, false);
                            },
                            uRecNew, uRecNew1, config.implicitReconstructionControl.nGmresIter,
                            [&](uint32_t i, real res, real resB) -> bool
                            {
                                if (i > 0)
                                {
                                    nGMRESrestartAll++;
                                    if (mpi.rank == 0 &&
                                        (nGMRESrestartAll % config.implicitReconstructionControl.nRecConsolCheck == 0))
                                    {
                                        auto fmt = log().flags();
                                        log() << std::scientific;
                                        log() << "GMRES for Rec: " << iRec << " Restart " << i << " " << resB << " -> " << res << std::endl;
                                        log().setf(fmt);
                                    }
                                }
                                return res < gmresResidualB * config.implicitReconstructionControl.recThreshold;
                            });
                    uRec.addTo(uRecNew1, -1);
                    if (gmresConverge)
                        break;
                }
            }
            else
                DNDS_assert_info(false, "no such recLinearScheme");
            trec += MPI_Wtime() - tstartA;
            gradIsZero = false;
            double tstartH = MPI_Wtime();

            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uRec[iCell].m() -= uOld[iCell].m();

            InsertCheck(mpi, " Lambda RHS: StartLim");
            if (config.limiterControl.useLimiter)
            {
                // vfv->ReconstructionWBAPLimitFacial(
                //     cx, uRec, uRecNew, uF0, uF1, ifUseLimiter,

                auto fML = [&](const auto &UL, const auto &UR, const auto &n) -> auto
                {
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234)*0.5;
                    Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasLeftEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Eigen::all, Seq123) *= normBase.transpose();

                    Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;
                    PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterA);
                    return ret;
                    // return real(1);
                };
                auto fMR = [&](const auto &UL, const auto &UR, const auto &n) -> auto
                {
                    PerformanceTimer::Instance().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234)*0.5;
                    Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasRightEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Seq123, Eigen::all) = normBase * M(Seq123, Eigen::all);

                    Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;

                    PerformanceTimer::Instance().StopTimer(PerformanceTimer::LimiterA);
                    return ret;
                    // return real(1);
                };
                if (config.limiterControl.smoothIndicatorProcedure == 0)
                    vfv->DoCalculateSmoothIndicator(
                        ifUseLimiter, (uRec), (u),
                        std::array<int, 2>{0, I4});
                else if (config.limiterControl.smoothIndicatorProcedure == 1)
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
                if (config.limiterControl.limiterProcedure == 1)
                    vfv->DoLimiterWBAP_C(
                        eval,
                        (cx),
                        (uRec),
                        (uRecNew),
                        (uRecNew1),
                        ifUseLimiter,
                        iter < config.limiterControl.nPartialLimiterStartLocal && step < config.limiterControl.nPartialLimiterStart,
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
            if (config.limiterControl.useLimiter)
                eval.EvaluateRHS(crhs, cx, uRecNew, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, cx, uRec, tSimu + ct * curDtImplicit);
            if (getNVars(model) > (I4 + 1) && iter <= config.others.nFreezePassiveInner)
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

            eval.EvaluateDt(dTau, u, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            for (auto &i : dTau)
                i /= alphaDiag;
        };

        auto fsolve = [&](ArrayDOFV<nVars_Fixed> &cx, ArrayDOFV<nVars_Fixed> &crhs, std::vector<real> &dTau,
                          real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &cxInc, int iter)
        {
            cxInc.setConstant(0.0);

            if (config.limiterControl.useLimiter) // uses urec value
                eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                     cx, uRecNew,
                                     0,
                                     tSimu);
            else
                eval.LUSGSMatrixInit(dTau, dt, alphaDiag,
                                     cx, uRec,
                                     0,
                                     tSimu);

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                crhs[iCell] = eval.CompressInc(cx[iCell], crhs[iCell] * dTau[iCell], crhs[iCell]) / dTau[iCell];
            }

            if (config.linearSolverControl.gmresCode == 0 || config.linearSolverControl.gmresCode == 2)
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

            if (config.linearSolverControl.gmresCode != 0)
            {
                // !  GMRES
                // !  for gmres solver: A * uinc = rhsinc, rhsinc is average value insdead of cumulated on vol
                gmres->solve(
                    [&](decltype(u) &x, decltype(u) &Ax)
                    {
                        eval.LUSGSMatrixVec(alphaDiag, cx, x, Ax);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                    },
                    [&](decltype(u) &x, decltype(u) &MLx)
                    {
                        // x as rhs, and MLx as uinc
                        eval.UpdateLUSGSForward(alphaDiag, x, cx, MLx, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        eval.UpdateLUSGSBackward(alphaDiag, x, cx, MLx, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                    },
                    crhs, cxInc, config.linearSolverControl.nGmresIter,
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
            if (getNVars(model) > I4 + 1 && iter <= config.others.nFreezePassiveInner)
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
            eval.EvaluateResidual(res, cxinc, 1, config.convergenceControl.useVolWiseResidual);
            // if (iter == 1 && iStep == 1) // * using 1st rk step for reference
            if (iter == 1)
                resBaseCInternal = res;
            else
                resBaseCInternal = resBaseCInternal.array().max(res.array()); //! using max !
            Eigen::Vector<real, -1> resRel = (res.array() / resBaseCInternal.array()).matrix();
            bool ifStop = resRel(0) < config.convergenceControl.rhsThresholdInternal; // ! using only rho's residual
            if (iter % config.outputControl.nConsoleCheckInternal == 0 || iter > config.convergenceControl.nTimeStepInternal || ifStop)
            {
                double telapsed = MPI_Wtime() - tstart;
                tcomm = PerformanceTimer::Instance().getTimerCollective(PerformanceTimer::Comm, mpi);
                real tLimiterA = PerformanceTimer::Instance().getTimerCollective(PerformanceTimer::LimiterA, mpi);
                real tLimiterB = PerformanceTimer::Instance().getTimerCollective(PerformanceTimer::LimiterB, mpi);
                if (mpi.rank == 0)
                {
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
                          << tLim << "]  limtimeA [" << tLimiterA << "]  limtimeB ["
                          << tLimiterB << "]  ";
                    if (config.outputControl.consoleOutputMode == 1)
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

            if (iter % config.outputControl.nDataOutInternal == 0)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter), ode, eval);
                nextStepOut += config.outputControl.nDataOut;
            }
            if (iter % config.outputControl.nDataOutCInternal == 0)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C", ode, eval);
                nextStepOutC += config.outputControl.nDataOutC;
            }
            if (iter >= config.implicitCFLControl.nCFLRampStart && iter <= config.implicitCFLControl.nCFLRampLength + config.implicitCFLControl.nCFLRampStart)
            {
                real inter = real(iter - config.implicitCFLControl.nCFLRampStart) / config.implicitCFLControl.nCFLRampLength;
                real logCFL = std::log(config.implicitCFLControl.CFL) + (std::log(config.implicitCFLControl.CFLRampEnd / config.implicitCFLControl.CFL) * inter);
                CFLNow = std::exp(logCFL);
            }
            // return resRel.maxCoeff() < config.convergenceControl.rhsThresholdInternal;
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
            eval.EvaluateResidual(res, ode->getLatestRHS(), 1, config.convergenceControl.useVolWiseResidual);
            if (stepCount == 0 && resBaseC.norm() == 0)
                resBaseC = res;

            if (step % config.outputControl.nConsoleCheck == 0)
            {
                double telapsed = MPI_Wtime() - tstart;
                tcomm = PerformanceTimer::Instance().getTimerCollective(PerformanceTimer::Comm, mpi);
                if (mpi.rank == 0)
                {
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
                PrintData(config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step), ode, eval);
                nextStepOut += config.outputControl.nDataOut;
            }
            if (step == nextStepOutC)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C", ode, eval);
                nextStepOutC += config.outputControl.nDataOutC;
            }
            if (ifOutT)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout), ode, eval);
                nextTout += config.outputControl.tDataOut;
                if (nextTout > config.timeMarchControl.tEnd)
                    nextTout = config.timeMarchControl.tEnd;
            }
            if (eval.settings.specialBuiltinInitializer == 2 && (step % config.outputControl.nConsoleCheck == 0)) // IV problem special: reduction on solution
            {
                real xymin = 5 + tSimu - 2;
                real xymax = 5 + tSimu + 2;
                real xyc = 5 + tSimu;
                real sumErrRho = 0.0;
                real sumErrRhoSum = std::nan("1");
                real sumVol = 0.0;
                real sumVolSum = std::nan("1");
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
                MPI::Allreduce(&sumErrRho, &sumErrRhoSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                MPI::Allreduce(&sumVol, &sumVolSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                if (mpi.rank == 0)
                {
                    log() << "=== Mean Error IV: [" << std::scientific << std::setprecision(5) << sumErrRhoSum << ", " << sumErrRhoSum / sumVolSum << "]" << std::endl;
                }
            }
            if (config.implicitReconstructionControl.zeroGrads)
                uRec.setConstant(0.0), gradIsZero = true;

            stepCount++;

            return tSimu >= config.timeMarchControl.tEnd;
        };

        /**********************************/
        /*           MAIN LOOP            */
        /**********************************/

        for (step = 1; step <= config.timeMarchControl.nTimeStep; step++)
        {
            InsertCheck(mpi, "Implicit Step");
            ifOutT = false;
            curDtImplicit = config.timeMarchControl.dtImplicit; //* could add CFL driven dt here
            if (tSimu + curDtImplicit > nextTout)
            {
                ifOutT = true;
                curDtImplicit = (nextTout - tSimu);
            }
            CFLNow = config.implicitCFLControl.CFL;
            ode->Step(
                u, uInc,
                frhs,
                fdtau,
                fsolve,
                config.convergenceControl.nTimeStepInternal,
                fstop,
                curDtImplicit + verySmallReal);

            if (fmainloop())
                break;
        }

        // u.trans.waitPersistentPull();
        logErr.close();
    }
}