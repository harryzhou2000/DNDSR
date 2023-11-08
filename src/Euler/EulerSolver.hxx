#include "EulerSolver.hpp"

namespace DNDS::Euler
{
    template <EulerModel model>
    void EulerSolver<model>::RunImplicitEuler()
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(mpi, "Implicit 1 nvars " + std::to_string(nVars));

        /************* Files **************/
        std::ofstream logErr(config.dataIOControl.outLogName + "_" + output_stamp + ".log");
        /************* Files **************/

        std::shared_ptr<ODE::ImplicitDualTimeStep<decltype(u)>> ode;

        auto hashCoord = mesh->coords.hash();
        if (mpi.rank == 0)
        {
            log() << "Mesh coord hash is: [" << std::hex << hashCoord << std::dec << "]" << std::endl;
        }

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
                },
                1); // 1 for esdirk
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
        case 2: // SSPRK
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
        case 401: // H3S
            if (mpi.rank == 0)
                log() << "=== ODE: Hermite3 (simple jacobian) " << std::endl;
            ode = std::make_shared<ODE::ImplicitHermite3SimpleJacobianDualStep<decltype(u)>>(
                mesh->NumCell(),
                [&](decltype(u) &data)
                {
                    vfv->BuildUDof(data, nVars);
                },
                config.timeMarchControl.odeSetting1 == 0 ? 0.51 : config.timeMarchControl.odeSetting1,
                std::round(config.timeMarchControl.odeSetting2));
            break;
        default:
            DNDS_assert_info(false, "no such ode code");
        }
        if (config.timeMarchControl.useImplicitPP)
        {
            DNDS_assert(config.timeMarchControl.odeCode == 1);
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

        // fmt::print("pEval is {}", (void*)(pEval.get()));
        EulerEvaluator<model> &eval = *pEval;

        eval.InitializeUDOF(u);
        if (config.timeMarchControl.useRestart)
        {
            DNDS_assert(config.restartState.iStep >= 1);
            ReadRestart(config.restartState.lastRestartFile);
        }

        /*******************************************************/
        /*                 SOLVER MAJOR START                  */
        /*******************************************************/

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
        int nextStepRestart = config.outputControl.nRestartOut;
        int nextStepRestartC = config.outputControl.nRestartOutC;
        PerformanceTimer::Instance().clearAllTimer();

        // *** Loop variables
        real CFLNow = config.implicitCFLControl.CFL;
        bool ifOutT = false;
        real curDtMin;
        real curDtImplicit = config.timeMarchControl.dtImplicit;
        int step;
        bool gradIsZero = true;

        index nLimBeta = 0;
        index nLimAlpha = 0;
        real minAlpha = 1;
        real minBeta = 1;
        index nLimInc = 0;
        real alphaMinInc = 1;

        InsertCheck(mpi, "Implicit 2 nvars " + std::to_string(nVars));
        /*******************************************************/
        /*                   DEFINE LAMBDAS                    */
        /*******************************************************/

        auto frhs =
            [&](
                ArrayDOFV<nVars_Fixed> &crhs,
                ArrayDOFV<nVars_Fixed> &cx,
                std::vector<real> &dTau,
                int iter, real ct, int uPos)
        {
            cx.trans.startPersistentPull();
            cx.trans.waitPersistentPull(); // for hermite3
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            auto &uRecIncC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRecInc1 : uRecInc;
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            // if (mpi.rank == 0)
            //     std::cout << uRecC.father.get() << std::endl;

            eval.FixUMaxFilter(cx);
            // cx.trans.startPersistentPull();
            // cx.trans.waitPersistentPull();

            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uOld[iCell].m() = uRec[iCell].m();

            InsertCheck(mpi, " Lambda RHS: StartRec");
            int nRec = (gradIsZero ? config.implicitReconstructionControl.nRecMultiplyForZeroedGrad : 1) *
                       config.implicitReconstructionControl.nInternalRecStep;
            real recIncBase = 0;
            double tstartA = MPI_Wtime();
            typename TVFV::element_type::template TFBoundary<nVars_Fixed>
                FBoundary = [&](const TU &UL, const TU &UMean, index iCell, index iFace,
                                const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
                bool compressed = false;
                TU ULfixed = eval.CompressRecPart(
                    UMean,
                    UL - UMean,
                    compressed);
                return eval.generateBoundaryValue(ULfixed, UMean, iCell, iFace, normOutV, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true);
            };
            typename TVFV::element_type::template TFBoundaryDiff<nVars_Fixed>
                FBoundaryDiff = [&](const TU &UL, const TU &dU, const TU &UMean, index iCell, index iFace,
                                    const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
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
                return eval.generateBoundaryValue(ULfixedPlus, UMean, iCell, iFace, normOutV, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true) -
                       eval.generateBoundaryValue(ULfixed, UMean, iCell, iFace, normOutV, normBase, pPhy(Seq012), tSimu + ct * curDtImplicit, bType, true);
            };
            if (config.implicitReconstructionControl.storeRecInc)
                uRecOld = uRecC;
            if (config.implicitReconstructionControl.recLinearScheme == 0)
            {
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    if (nRec > 1)
                        uRecNew1 = uRecC;

                    vfv->DoReconstructionIter(
                        uRecC, uRecNew, cx,
                        FBoundary,
                        false);

                    uRecC.trans.startPersistentPull();
                    uRecC.trans.waitPersistentPull();

                    if (nRec > 1)
                    {
                        uRecNew1 -= uRecC;
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
            }
            else if (config.implicitReconstructionControl.recLinearScheme == 1)
            {
                int nGMRESrestartAll{0};
                real gmresResidualB = 0;
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    vfv->DoReconstructionIter(
                        uRecC, uRecNew, cx,
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
                            [&](decltype(uRecC) &x, decltype(uRecC) &Ax)
                            {
                                vfv->DoReconstructionIterDiff(uRec, x, Ax, cx, FBoundaryDiff);
                                Ax.trans.startPersistentPull();
                                Ax.trans.waitPersistentPull();
                            },
                            [&](decltype(uRecC) &x, decltype(uRecC) &MLx)
                            {
                                MLx = x; // initial value; for the input is mostly a good estimation
                                // MLx no need to comm
                                vfv->DoReconstructionIterSOR(uRecC, x, MLx, cx, FBoundaryDiff, false);
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
                    uRecC.addTo(uRecNew1, -1);
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
            //     uRecC[iCell].m() -= uOld[iCell].m();

            InsertCheck(mpi, " Lambda RHS: StartLim");
            if (config.limiterControl.useLimiter)
            {
                // vfv->ReconstructionWBAPLimitFacial(
                //     cx, uRecC, uRecNew, uF0, uF1, ifUseLimiter,

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
                    // return ret;
                    return real(1);
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
                    // return ret;
                    return real(1);
                };
                if (config.limiterControl.smoothIndicatorProcedure == 0)
                    vfv->DoCalculateSmoothIndicator(
                        ifUseLimiter, (uRecC), (u),
                        std::array<int, 2>{0, I4});
                else if (config.limiterControl.smoothIndicatorProcedure == 1)
                {
                    if constexpr (dim == 2)
                        vfv->DoCalculateSmoothIndicatorV1(
                            ifUseLimiter, (uRecC), (u),
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
                            ifUseLimiter, (uRecC), (u),
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
                        (uRecC),
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
            if (config.implicitReconstructionControl.storeRecInc)
            {
                uRecIncC = uRecC;
                uRecIncC -= uRecOld; //! uRecIncC now stores uRecIncrement
            }
            if (config.implicitReconstructionControl.storeRecInc && config.implicitReconstructionControl.dampRecIncDTau)
            {
                std::vector<real> damper = dTau;
                for (auto &v : damper)
                    v = v / (curDtImplicit + v); //! warning: teleported value here
                uRecIncC *= damper;
                uRecC = uRecOld;
                uRecC += uRecIncC;
            }

            // uRecC.trans.startPersistentPull(); //! this also need to update!
            // uRecC.trans.waitPersistentPull();

            // }

            InsertCheck(mpi, " Lambda RHS: StartEval");
            double tstartE = MPI_Wtime();
            eval.setPassiveDiscardSource(iter <= 0);

            if (iter == 1)
                alphaPPC.setConstant(1.0); // make RHS un-disturbed
            alphaPP_tmp.setConstant(1.0);  // make RHS un-disturbed
            if (config.limiterControl.usePPRecLimiter)
            {
                nLimBeta = 0;
                minBeta = 1;
                if (config.limiterControl.useLimiter)
                    eval.EvaluateURecBeta(u, uRecNew, betaPPC, nLimBeta, minBeta);
                else
                    eval.EvaluateURecBeta(u, uRecC, betaPPC, nLimBeta, minBeta);
                if (nLimBeta)
                    if (mpi.rank == 0 && config.outputControl.consoleOutputEveryFix != 0)
                    {
                        log() << std::scientific << std::setprecision(3)
                              << "PPRecLimiter: nLimBeta [" << nLimBeta << "]"
                              << " minBeta[" << minBeta << "]" << std::endl;
                    }
                betaPPC.trans.startPersistentPull();
                betaPPC.trans.waitPersistentPull();
            }

            if (config.limiterControl.useLimiter)
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecNew, betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit);
            crhs.trans.startPersistentPull();
            crhs.trans.waitPersistentPull();
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

        auto fdtau = [&](ArrayDOFV<nVars_Fixed> &cx, std::vector<real> &dTau, real alphaDiag, int uPos)
        {
            eval.FixUMaxFilter(cx);
            cx.trans.startPersistentPull(); //! this also need to update!
            cx.trans.waitPersistentPull();
            // uRec.trans.startPersistentPull();
            // uRec.trans.waitPersistentPull();
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            eval.EvaluateDt(dTau, cx, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            // for (auto &i: dTau)
            //     i = std::min({i, curDtMin * 1000, curDtImplicit * 100});
            for (auto &i : dTau)
                i /= alphaDiag;
        };

        auto fsolve = [&](ArrayDOFV<nVars_Fixed> &cx, ArrayDOFV<nVars_Fixed> &crhs, std::vector<real> &dTau,
                          real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &cxInc, int iter, int uPos)
        {
            rhsTemp = crhs;
            eval.CentralSmoothResidual(rhsTemp, crhs, uTemp);

            cxInc.setConstant(0.0);
            auto &JDC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JD1 : JD;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &uRecIncC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRecInc1 : uRecInc;
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;

            {
                // ! experimental: bad now
                rhsTemp = crhs;
                rhsTemp *= dTau;
                index nLimFRes = 0;
                real alphaMinFRes = 1;
                eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, rhsTemp, alphaPP_tmp, nLimFRes, alphaMinFRes, 1);
                if (nLimFRes)
                    if (mpi.rank == 0)
                    {
                        std::cout << std::scientific << std::setw(3);
                        std::cout << "PPFResLimiter: nLimFRes[" << nLimFRes << "] minAlpha [" << alphaMinFRes << "]" << std::endl;
                    }

                crhs *= alphaPP_tmp;
            }

            typename TVFV::element_type::template TFBoundary<nVars_Fixed>
                FBoundary = [&](const TU &UL, const TU &UMean, index iCell, index iFace,
                                const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TU UR = UL;
                UR.setZero();
                return UR;
            };

            if (config.limiterControl.useLimiter) // uses urec value
                eval.LUSGSMatrixInit(JDC, JSourceC,
                                     dTau, dt, alphaDiag,
                                     cx, uRecNew,
                                     0,
                                     tSimu);
            else
                eval.LUSGSMatrixInit(JDC, JSourceC,
                                     dTau, dt, alphaDiag,
                                     cx, uRec,
                                     0,
                                     tSimu);

            // for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            // {
            //     crhs[iCell] = eval.CompressInc(cx[iCell], crhs[iCell] * dTau[iCell]) / dTau[iCell];
            // }

            TU sgsRes(nVars), sgsRes0(nVars);

            if (config.linearSolverControl.gmresCode == 0 || config.linearSolverControl.gmresCode == 2)
            {
                // //! LUSGS

                if (config.linearSolverControl.initWithLastURecInc)
                {
                    DNDS_assert(config.implicitReconstructionControl.storeRecInc);
                    eval.UpdateSGSWithRec(alphaDiag, crhs, cx, uRecC, cxInc, uRecIncC, JDC, true, sgsRes);
                    // for (index iCell = 0; iCell < uRecIncC.Size(); iCell++)
                    //     std::cout << "-------\n"
                    //               << uRecIncC[iCell] << std::endl;
                    cxInc.trans.startPersistentPull();
                    cxInc.trans.waitPersistentPull();
                    eval.UpdateSGSWithRec(alphaDiag, crhs, cx, uRecC, cxInc, uRecIncC, JDC, false, sgsRes);
                    cxInc.trans.startPersistentPull();
                    cxInc.trans.waitPersistentPull();
                }
                else
                {
                    bool useJacobi = config.linearSolverControl.jacobiCode == 0;
                    eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, true, sgsRes);
                    if (useJacobi)
                        cxInc = uTemp;
                    cxInc.trans.startPersistentPull();
                    cxInc.trans.waitPersistentPull();
                    eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, false, sgsRes);
                    if (useJacobi)
                        cxInc = uTemp;
                    cxInc.trans.startPersistentPull();
                    cxInc.trans.waitPersistentPull();
                    // eval.UpdateLUSGSForward(alphaDiag, crhs, cx, cxInc, JDC, cxInc);
                    // cxInc.trans.startPersistentPull();
                    // cxInc.trans.waitPersistentPull();
                    // eval.UpdateLUSGSBackward(alphaDiag, crhs, cx, cxInc, JDC, cxInc);
                    // cxInc.trans.startPersistentPull();
                    // cxInc.trans.waitPersistentPull();
                }

                if (config.linearSolverControl.sgsWithRec != 0)
                    uRecNew.setConstant(0.0);
                for (int iterSGS = 1; iterSGS <= config.linearSolverControl.sgsIter; iterSGS++)
                {
                    if (config.linearSolverControl.sgsWithRec != 0)
                    {
                        vfv->DoReconstructionIter(
                            uRecNew, uRecNew1, cxInc,
                            FBoundary,
                            false);
                        uRecNew.trans.startPersistentPull();
                        uRecNew.trans.waitPersistentPull();
                        eval.UpdateSGSWithRec(alphaDiag, crhs, cx, uRecC, cxInc, uRecNew, JDC, true, sgsRes);
                        cxInc.trans.startPersistentPull();
                        cxInc.trans.waitPersistentPull();
                        eval.UpdateSGSWithRec(alphaDiag, crhs, cx, uRecC, cxInc, uRecNew, JDC, false, sgsRes);
                        cxInc.trans.startPersistentPull();
                        cxInc.trans.waitPersistentPull();
                    }
                    else
                    {
                        bool useJacobi = config.linearSolverControl.jacobiCode == 0;
                        eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, true, sgsRes);
                        if (useJacobi)
                            cxInc = uTemp;
                        cxInc.trans.startPersistentPull();
                        cxInc.trans.waitPersistentPull();
                        eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, false, sgsRes);
                        if (useJacobi)
                            cxInc = uTemp;
                        cxInc.trans.startPersistentPull();
                        cxInc.trans.waitPersistentPull();
                    }
                    if (iterSGS == 1)
                        sgsRes0 = sgsRes;

                    if (mpi.rank == 0 && iterSGS % config.linearSolverControl.nSgsConsoleCheck == 0)
                        log() << std::scientific << "SGS1 " << std::to_string(iterSGS)
                              << " [" << sgsRes0.transpose() << "] ->"
                              << " [" << sgsRes.transpose() << "] " << std::endl;
                }
            }

            if (config.linearSolverControl.gmresCode != 0)
            {
                // !  GMRES
                // !  for gmres solver: A * uinc = rhsinc, rhsinc is average value insdead of cumulated on vol
                gmres->solve(
                    [&](decltype(u) &x, decltype(u) &Ax)
                    {
                        eval.LUSGSMatrixVec(alphaDiag, cx, x, JDC, Ax);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                    },
                    [&](decltype(u) &x, decltype(u) &MLx)
                    {
                        // x as rhs, and MLx as uinc
                        eval.UpdateLUSGSForward(alphaDiag, x, cx, MLx, JDC, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        eval.UpdateLUSGSBackward(alphaDiag, x, cx, MLx, JDC, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        for (int i = 0; i < config.linearSolverControl.sgsIter; i++)
                        {
                            bool useJacobi = config.linearSolverControl.jacobiCode == 0;
                            eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, true, sgsRes);
                            if (useJacobi)
                                cxInc = uTemp;
                            cxInc.trans.startPersistentPull();
                            cxInc.trans.waitPersistentPull();
                            eval.UpdateSGS(alphaDiag, crhs, cx, cxInc, useJacobi ? uTemp : cxInc, JDC, false, sgsRes);
                            if (useJacobi)
                                cxInc = uTemp;
                            cxInc.trans.startPersistentPull();
                            cxInc.trans.waitPersistentPull();
                        }
                    },
                    crhs, cxInc, config.linearSolverControl.nGmresIter,
                    [&](uint32_t i, real res, real resB) -> bool
                    {
                        if (i > 0 && i % config.linearSolverControl.nGmresConsoleCheck == 0)
                        {
                            if (mpi.rank == 0)
                            {
                                log() << std::scientific;
                                log() << "GMRES: " << i << " " << resB << " -> " << res << std::endl;
                            }
                        }
                        return false;
                    });
            }
            // eval.FixIncrement(cx, cxInc);
            // !freeze something
            if (getNVars(model) > I4 + 1 && iter <= config.others.nFreezePassiveInner)
            {
                for (int i = 0; i < crhs.Size(); i++)
                    cxInc[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
        };

        auto fsolveNest = [&](
                              ArrayDOFV<nVars_Fixed> &cx,
                              ArrayDOFV<nVars_Fixed> &cx1,
                              ArrayDOFV<nVars_Fixed> &crhs,
                              std::vector<real> &dTau,
                              const std::vector<real> &Coefs, // coefs are dU * c[0] + dt * c[1] * (I/(dt * c[2]) - JMid) * (I/(dt * c[3]) - J)
                              real dt, real alphaDiag, ArrayDOFV<nVars_Fixed> &cxInc, int iter, int uPos)
        {
            crhs.trans.startPersistentPull();
            crhs.trans.waitPersistentPull();
            rhsTemp = crhs;
            eval.CentralSmoothResidual(rhsTemp, crhs, uTemp);

            cxInc.setConstant(0.0);
            auto &JDC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JD1 : JD;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            // TODO: use "update spectral radius" procedure? or force update in fsolve
            eval.EvaluateDt(dTau, cx1, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            for (auto &v : dTau)
                v *= Coefs[2];
            eval.LUSGSMatrixInit(JD1, JSource1,
                                 dTau, dt * Coefs[2], alphaDiag,
                                 cx1, uRec,
                                 0,
                                 tSimu);
            eval.EvaluateDt(dTau, cx, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            for (auto &v : dTau)
                v *= Coefs[3] * veryLargeReal;
            eval.LUSGSMatrixInit(JD, JSource,
                                 dTau, dt * Coefs[3], alphaDiag,
                                 cx, uRec,
                                 0,
                                 tSimu);

            // for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            // {
            //     crhs[iCell] = eval.CompressInc(cx[iCell], crhs[iCell] * dTau[iCell]) / dTau[iCell];
            // }

            if (config.linearSolverControl.gmresCode == 0 || config.linearSolverControl.gmresCode == 2)
            {
                // //! LUSGS

                eval.UpdateLUSGSForward(alphaDiag, crhs, cx1, cxInc, JD1, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                eval.UpdateLUSGSBackward(alphaDiag, crhs, cx1, cxInc, JD1, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                uTemp = cxInc;
                eval.UpdateLUSGSForward(alphaDiag, uTemp, cx, cxInc, JD, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                eval.UpdateLUSGSBackward(alphaDiag, uTemp, cx, cxInc, JD, cxInc);
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
                cxInc *= 1. / (dt * Coefs[1]);
            }

            if (config.linearSolverControl.gmresCode != 0)
            {
                // !  GMRES
                // !  for gmres solver: A * uinc = rhsinc, rhsinc is average value insdead of cumulated on vol
                gmres->solve(
                    [&](decltype(u) &x, decltype(u) &Ax)
                    {
                        eval.LUSGSMatrixVec(alphaDiag, cx, x, JD, Ax);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                        uTemp = Ax;
                        eval.LUSGSMatrixVec(alphaDiag, cx1, uTemp, JD1, Ax);
                        Ax.trans.startPersistentPull();
                        Ax.trans.waitPersistentPull();
                        Ax *= dt * Coefs[1];
                        Ax.addTo(x, Coefs[0]);
                    },
                    [&](decltype(u) &x, decltype(u) &MLx)
                    {
                        // x as rhs, and MLx as uinc
                        eval.UpdateLUSGSForward(alphaDiag, x, cx1, MLx, JD1, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        eval.UpdateLUSGSBackward(alphaDiag, x, cx1, MLx, JD1, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        uTemp = MLx;
                        eval.UpdateLUSGSForward(alphaDiag, uTemp, cx, MLx, JD, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        eval.UpdateLUSGSBackward(alphaDiag, uTemp, cx, MLx, JD, MLx);
                        MLx.trans.startPersistentPull();
                        MLx.trans.waitPersistentPull();
                        MLx *= 1. / (dt * Coefs[1]);
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
            }
            // eval.FixIncrement(cx, cxInc);
            // !freeze something
            if (getNVars(model) > I4 + 1 && iter <= config.others.nFreezePassiveInner)
            {
                for (int i = 0; i < crhs.Size(); i++)
                    cxInc[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
        };

        auto falphaLimSource = [&](
                                   ArrayDOFV<nVars_Fixed> &v,
                                   int uPos)
        {
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            for (index i = 0; i < v.Size(); i++)
                v[i] *= alphaPPC[i](0);
        };

        auto fresidualIncPP = [&](
                                  ArrayDOFV<nVars_Fixed> &cx,
                                  ArrayDOFV<nVars_Fixed> &xPrev,
                                  ArrayDOFV<nVars_Fixed> &crhs,
                                  ArrayDOFV<nVars_Fixed> &rhsIncPart,
                                  const std::function<void()> &renewRhsIncPart,
                                  real ct,
                                  int uPos)
        {
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            renewRhsIncPart(); // un-fixed now
            // rhsIncPart.trans.startPersistentPull();
            // rhsIncPart.trans.waitPersistentPull(); //seems not needed
            eval.EvaluateCellRHSAlpha(xPrev, uRecC, betaPPC, rhsIncPart, alphaPP_tmp, nLimAlpha, minAlpha, 1);
            alphaPP_tmp.trans.startPersistentPull();
            alphaPP_tmp.trans.waitPersistentPull();
            if (nLimAlpha)
                if (mpi.rank == 0 && config.outputControl.consoleOutputEveryFix != 0)
                {
                    log() << std::scientific << std::setprecision(3)
                          << "PPResidualLimiter: nLimAlpha [" << nLimAlpha << "]"
                          << " minAlpha[" << minAlpha << "]" << std::endl;
                }
            if (config.limiterControl.useLimiter)
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecNew, betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
            // rhs now last-fixed
            crhs.trans.startPersistentPull();
            crhs.trans.waitPersistentPull();

            /********************************************/
            // first fix
            {
                renewRhsIncPart(); // now last-fixed
                // rhsIncPart.trans.startPersistentPull();
                // rhsIncPart.trans.waitPersistentPull(); //seems not needed
                eval.EvaluateCellRHSAlphaExpansion(xPrev, uRecC, betaPPC, rhsIncPart, alphaPP_tmp, nLimAlpha, minAlpha);
                alphaPP_tmp.trans.startPersistentPull();
                alphaPP_tmp.trans.waitPersistentPull();
                if (nLimAlpha)
                    if (mpi.rank == 0 && config.outputControl.consoleOutputEveryFix != 0)
                    {
                        log() << std::scientific << std::setprecision(3)
                              << "PPResidualLimiter - first expand: nLimAlpha [" << nLimAlpha << "]"
                              << " minAlpha[" << minAlpha << "]" << std::endl;
                    }
                alphaPPC = alphaPP_tmp;
                if (config.limiterControl.useLimiter)
                    eval.EvaluateRHS(crhs, JSourceC, cx, uRecNew, betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
                else
                    eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
                crhs.trans.startPersistentPull();
                crhs.trans.waitPersistentPull();
            }
        };

        auto fincrement = [&](
                              ArrayDOFV<nVars_Fixed> &cx,
                              ArrayDOFV<nVars_Fixed> &cxInc,
                              real alpha, int uPos)
        {
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            nLimInc = 0;
            alphaMinInc = 1;
            eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, cxInc, alphaPP_tmp, nLimInc, alphaMinInc, 1);
            if (nLimInc)
                if (mpi.rank == 0 && config.outputControl.consoleOutputEveryFix != 0)
                {
                    std::cout << std::scientific << std::setw(3);
                    std::cout << "PPIncrementLimiter: nIncrementRes[" << nLimInc << "] minAlpha [" << alphaMinInc << "]" << std::endl;
                }

            uTemp = cxInc;
            uTemp *= alphaPP_tmp;
            // cx += uTemp;
            eval.AddFixedIncrement(cx, uTemp, alpha);
            // eval.AddFixedIncrement(cx, cxInc, alpha);
        };

        auto fstop = [&](int iter, ArrayDOFV<nVars_Fixed> &cxinc, int iStep) -> bool
        {
            Eigen::Vector<real, -1> res(nVars);
            eval.EvaluateNorm(res, cxinc, 1, config.convergenceControl.useVolWiseResidual);
            // if (iter == 1 && iStep == 1) // * using 1st rk step for reference
            if (iter == 1)
                resBaseCInternal = res;
            else
                resBaseCInternal = resBaseCInternal.array().max(res.array()); //! using max !
            Eigen::Vector<real, -1> resRel = (res.array() / (resBaseCInternal.array() + verySmallReal)).matrix();
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
                          << "\t Internal === Step [" << step << ", " << iStep << ", " << iter << "]   "
                          << "res \033[91m[" << resRel.transpose() << "]\033[39m   "
                          << "t,dTaumin,CFL,nFix \033[92m["
                          << tSimu << ", " << curDtMin << ", " << CFLNow << ", "
                          << fmt::format("[alphaInc({},{}), betaRec({},{}), alphaRes({},{})]",
                                         nLimInc, alphaMinInc, nLimBeta, minBeta, nLimAlpha, minAlpha)
                          << "]\033[39m   "
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
                        << eval.fluxWallSum.transpose() << delimC
                        << (nLimInc) << delimC << (alphaMinInc) << delimC
                        << (nLimBeta) << delimC << (minBeta) << delimC
                        << (nLimAlpha) << delimC << (minAlpha) << delimC
                        << std::endl;
                }
                tstart = MPI_Wtime();
                trec = tcomm = trhs = tLim = 0.;
                PerformanceTimer::Instance().clearAllTimer();
            }

            if (iter % config.outputControl.nDataOutInternal == 0)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter), ode, eval);
            }
            if (iter % config.outputControl.nDataOutCInternal == 0)
            {
                eval.FixUMaxFilter(u);
                PrintData(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C", ode, eval);
            }
            if (iter % config.outputControl.nRestartOutInternal == 0)
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = iter;
                PrintRestart(config.dataIOControl.outRestartName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter));
            }
            if (iter % config.outputControl.nRestartOutCInternal == 0)
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = iter;
                PrintRestart(config.dataIOControl.outRestartName + "_" + output_stamp + "_" + "C");
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
            eval.EvaluateNorm(res, ode->getLatestRHS(), 1, config.convergenceControl.useVolWiseResidual);
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
                        << real(eval.nFaceReducedOrder) << delimC
                        << eval.fluxWallSum.transpose() << delimC
                        << (nLimInc) << delimC << (alphaMinInc) << delimC
                        << (nLimBeta) << delimC << (minBeta) << delimC
                        << (nLimAlpha) << delimC << (minAlpha) << delimC
                        << std::endl;
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
            if (step == nextStepRestart)
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = -1;
                PrintRestart(config.dataIOControl.outRestartName + "_" + output_stamp + "_" + std::to_string(step));
                nextStepRestart += config.outputControl.nRestartOut;
            }
            if (step == nextStepRestartC)
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = -1;
                PrintRestart(config.dataIOControl.outRestartName + "_" + output_stamp + "_" + "C");
                nextStepRestartC += config.outputControl.nRestartOutC;
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
                real xymin = 5 - 2;
                real xymax = 5 + 2;
                real xyc = 5;

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
                            pPhysics[0] = float_mod(pPhysics[0] - tSimu, 10);
                            pPhysics[1] = float_mod(pPhysics[1] - tSimu, 10);
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
                    cP[0] = float_mod(cP[0] - tSimu, 10);
                    cP[1] = float_mod(cP[1] - tSimu, 10);

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
            if (config.timeMarchControl.useImplicitPP)
                std::dynamic_pointer_cast<ODE::ImplicitBDFDualTimeStep<ArrayDOFV<nVars_Fixed>>>(ode)
                    ->StepPP(
                        u, uInc,
                        frhs,
                        fdtau,
                        fsolve,
                        config.convergenceControl.nTimeStepInternal,
                        fstop, fincrement,
                        curDtImplicit + verySmallReal,
                        falphaLimSource,
                        fresidualIncPP);
            else if (config.timeMarchControl.odeCode == 401 && false)
                std::dynamic_pointer_cast<ODE::ImplicitHermite3SimpleJacobianDualStep<ArrayDOFV<nVars_Fixed>>>(ode)
                    ->StepNested(
                        u, uInc,
                        frhs,
                        fdtau,
                        fsolve, fsolveNest,
                        config.convergenceControl.nTimeStepInternal,
                        fstop, fincrement,
                        curDtImplicit + verySmallReal);
            else
                ode
                    ->Step(
                        u, uInc,
                        frhs,
                        fdtau,
                        fsolve,
                        config.convergenceControl.nTimeStepInternal,
                        fstop, fincrement,
                        curDtImplicit + verySmallReal);

            if (fmainloop())
                break;
        }

        // u.trans.waitPersistentPull();
        logErr.close();
    }
}