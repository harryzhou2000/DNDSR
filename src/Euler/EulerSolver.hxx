#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "EulerSolver.hpp"
#include "DNDS/EigenUtil.hpp"
#include "Solver/ODE.hpp"
#include "Solver/Linear.hpp"
#include "SpecialFields.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif
// #include "fmt/ranges.h"

namespace DNDS::Euler
{
    // /***************/ // IDE mode:
    // static const auto model = NS_SA;
    // template <>
    // void EulerSolver<model>::RunImplicitEuler()
    // /***************/ // IDE mode;
    DNDS_SWITCH_INTELLISENSE(
        // the real definition
        template <EulerModel model>
        ,
        // the intellisense friendly definition
        static const auto model = NS_SA;
        template <>
    )
    void EulerSolver<model>::RunImplicitEuler()
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        using namespace std::literals;

        DNDS_MPI_InsertCheck(mpi, "Implicit 1 nvars " + std::to_string(nVars));
        EulerEvaluator<model> &eval = *pEval;
        auto hashCoord = mesh->coords.hash();
        if (mpi.rank == 0)
        {
            log() << "Mesh coord hash is: [" << std::hex << hashCoord << std::dec << "]" << std::endl
                  << std::scientific;
        }
        if (config.timeMarchControl.partitionMeshOnly)
        {
            if (mpi.rank == 0)
            {
                log() << "Mesh Is not altered; partitioning done" << std::endl;
            }
            return;
        }

        /************* Files **************/
        std::string logErrFileName = config.dataIOControl.getOutLogName() + "_" + output_stamp + ".log";
        std::filesystem::path outFile{logErrFileName};
        std::filesystem::create_directories(outFile.parent_path() / ".");
        std::ofstream logErr(logErrFileName);
        DNDS_assert(logErr);
        /************* Files **************/

        mesh->ObtainLocalFactFillOrdering(*eval.symLU, config.linearSolverControl.directPrecControl);
        mesh->ObtainSymmetricSymbolicFactorization(*eval.symLU, config.linearSolverControl.directPrecControl);
        if (config.linearSolverControl.jacobiCode == 2) // do lu on mean-value jacobian
        {
            DNDS_MAKE_SSP(JLocalLU, eval.symLU, nVars);
        }

        ssp<ODE::ImplicitDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>> ode;
        auto buildDOF = [&](ArrayDOFV<nVarsFixed> &data)
        {
            vfv->BuildUDof(data, nVars);
        };
        auto buildScalar = [&](ArrayDOFV<1> &data)
        {
            vfv->BuildUDof(data, 1);
        };

        if (config.timeMarchControl.steadyQuit)
        {
            if (mpi.rank == 0)
                log() << "Using steady!" << std::endl;
            config.timeMarchControl.odeCode = 1; // To bdf;
            config.timeMarchControl.nTimeStep = 1;
        }
        switch (config.timeMarchControl.odeCode)
        {
        case 0: // esdirk4
            if (mpi.rank == 0)
                log() << "=== ODE: ESDIRK4 " << std::endl;
            ode = std::make_shared<ODE::ImplicitSDIRK4DualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                1); // 1 for esdirk
            break;
        case 101: // sdirk4
            if (mpi.rank == 0)
                log() << "=== ODE: SSP-SDIRK4 " << std::endl;
            ode = std::make_shared<ODE::ImplicitSDIRK4DualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                0);
            break;
        case 1: // BDF2 // Backward Euler
        case 103:
            if (mpi.rank == 0 && config.timeMarchControl.odeCode == 1)
                log() << "=== ODE: BDF2 " << std::endl;
            if (mpi.rank == 0 && config.timeMarchControl.odeCode == 103)
                log() << "=== ODE: Backward Euler " << std::endl;
            ode = std::make_shared<ODE::ImplicitVBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                config.timeMarchControl.odeCode == 1 ? 2 : 1);
            break;
        case 102: // VBDF2
            if (mpi.rank == 0)
                log() << "=== ODE: VBDF2 " << std::endl;
            ode = std::make_shared<ODE::ImplicitVBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                2);
            break;
        case 2: // SSPRK
            if (mpi.rank == 0)
                log() << "=== ODE: SSPRK4 " << std::endl;
            ode = std::make_shared<ODE::ExplicitSSPRK3TimeStepAsImplicitDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                false); // TODO: add local stepping options
            break;
        case 401: // H3S
            if (mpi.rank == 0)
                log() << "=== ODE: Hermite3 (simple jacobian) " << std::endl;
            ode = std::make_shared<ODE::ImplicitHermite3SimpleJacobianDualStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(
                mesh->NumCell(),
                buildDOF, buildScalar,
                config.timeMarchControl.odeSetting1 == 0 ? 0.55 : config.timeMarchControl.odeSetting1,
                std::round(config.timeMarchControl.odeSetting2),                                         // Backward Euler Starter
                0,                                                                                       // method
                config.timeMarchControl.odeSetting3 == 0 ? 0.9146 : config.timeMarchControl.odeSetting3, // thetaM1
                std::round(config.timeMarchControl.odeSetting4)                                          // mask
            );
            break;
        default:
            DNDS_assert_info(false, "no such ode code");
        }
        if (config.timeMarchControl.useImplicitPP)
        {
            DNDS_assert(config.timeMarchControl.odeCode == 1 || config.timeMarchControl.odeCode == 102);
        }

        using tGMRES_u = Linear::GMRES_LeftPreconditioned<decltype(u)>;
        using tGMRES_uRec = Linear::GMRES_LeftPreconditioned<decltype(uRec)>;
        using tPCG_uRec = Linear::PCG_PreconditionedRes<decltype(uRec), Eigen::Array<real, 1, Eigen::Dynamic>>;
        std::unique_ptr<tGMRES_u> gmres;
        std::unique_ptr<tGMRES_uRec> gmresRec;
        std::unique_ptr<tPCG_uRec> pcgRec;

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

        if (config.implicitReconstructionControl.recLinearScheme == 2)
            pcgRec = std::make_unique<tPCG_uRec>(
                [&](decltype(uRec) &data)
                {
                    vfv->BuildURec(data, nVars);
                });

        // fmt::print("pEval is {}", (void*)(pEval.get()));

        eval.InitializeUDOF(u);
        if (config.timeAverageControl.enabled)
            wAveraged.setConstant(0.0);
        if (config.timeMarchControl.useRestart)
        {

            if (!config.restartState.lastRestartFile.empty())
            {
                DNDS_assert(config.restartState.iStep >= 1);
                ReadRestart(config.restartState.lastRestartFile);
            }
            if (!config.restartState.otherRestartFile.empty())
                ReadRestartOtherSolver(config.restartState.otherRestartFile, config.restartState.otherRestartStoreDim);
        }
        OutputPicker outputPicker;
        eval.InitializeOutputPicker(outputPicker, {u, uRec, betaPP, alphaPP});
        tAdditionalCellScalarList addOutList = outputPicker.getSubsetList(config.dataIOControl.outCellScalarNames);

        /*******************************************************/
        /*                 SOLVER MAJOR START                  */
        /*******************************************************/

        double tstart = MPI_Wtime();
        double tstartInternal = tstart;
        std::map<std::string, ScalarStatistics> tInternalStats;
        int stepCount = 0;
        Eigen::VectorFMTSafe<real, -1> resBaseC;
        Eigen::VectorFMTSafe<real, -1> resBaseCInternal;
        resBaseC.resize(nVars);
        resBaseCInternal.resize(nVars);
        resBaseC.setConstant(config.convergenceControl.res_base);

        real tSimu = 0.0;
        real tAverage = 0.0;
        real nextTout = std::min(config.outputControl.tDataOut, config.timeMarchControl.tEnd); // ensures the destination time output
        int nextStepOut = config.outputControl.nDataOut;
        int nextStepOutC = config.outputControl.nDataOutC;
        int nextStepRestart = config.outputControl.nRestartOut;
        int nextStepRestartC = config.outputControl.nRestartOutC;
        int nextStepOutAverage = config.outputControl.nTimeAverageOut;
        int nextStepOutAverageC = config.outputControl.nTimeAverageOutC;
        Timer().clearAllTimer();

        // *** Loop variables
        real CFLNow = config.implicitCFLControl.CFL;
        bool ifOutT = false;
        real curDtMin;
        real curDtImplicit = config.timeMarchControl.dtImplicit;
        std::vector<real> curDtImplicitHistory;
        int step;
        bool gradIsZero = true;

        index nLimBeta = 0;
        index nLimAlpha = 0;
        real minAlpha = 1;
        real minBeta = 1;
        index nLimInc = 0;
        real alphaMinInc = 1;

        int dtIncreaseCounter = 0;

        DNDS_MPI_InsertCheck(mpi, "Implicit 2 nvars " + std::to_string(nVars));

        /*******************************************************/
        /*                   DEFINE LAMBDAS                    */
        /*******************************************************/

        auto frhsOuter =
            [&](
                ArrayDOFV<nVarsFixed> &crhs,
                ArrayDOFV<nVarsFixed> &cx,
                ArrayDOFV<1> &dTau,
                int iter, real ct, int uPos, int reconstructionFlag)
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
            typename TVFV::template TFBoundary<nVarsFixed>
                FBoundary = [&](const TU &UL, const TU &UMean, index iCell, index iFace, int ig,
                                const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TVec normOutV = normOut(Seq012);
                Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(normOutV);
                bool compressed = false;
                TU ULfixed = eval.CompressRecPart(
                    UMean,
                    UL - UMean,
                    compressed);
                return eval.generateBoundaryValue(ULfixed, UMean, iCell, iFace, ig, normOutV, normBase, pPhy, tSimu + ct * curDtImplicit, bType, true, 1);
            };
            typename TVFV::template TFBoundaryDiff<nVarsFixed>
                FBoundaryDiff = [&](const TU &UL, const TU &dU, const TU &UMean, index iCell, index iFace, int ig,
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
                return eval.generateBoundaryValue(ULfixedPlus, UMean, iCell, iFace, ig, normOutV, normBase, pPhy, tSimu + ct * curDtImplicit, bType, true, 1) -
                       eval.generateBoundaryValue(ULfixed, UMean, iCell, iFace, ig, normOutV, normBase, pPhy, tSimu + ct * curDtImplicit, bType, true, 1);
            };

            eval.FixUMaxFilter(cx);
            if ((iter > config.convergenceControl.nAnchorUpdateStart &&
                 (iter - config.convergenceControl.nAnchorUpdateStart - 1) % config.convergenceControl.nAnchorUpdate == 0))
            {
                eval.updateBCAnchors(cx, uRecC);
                eval.updateBCProfiles(cx, uRecC);
                eval.updateBCProfilesPressureRadialEq();
            }
            // cx.trans.startPersistentPull();
            // cx.trans.waitPersistentPull();

            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uOld[iCell].m() = uRec[iCell].m();
            if (!reconstructionFlag)
            {
                betaPPC.setConstant(1.0);
                alphaPP_tmp.setConstant(1.0);
                uRecNew.setConstant(0.0);
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecNew, uRecNew, betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit, TEval::RHS_Ignore_Viscosity);
                // vfv->DoReconstruction2nd(uRecOld, cx, FBoundary, 1, std::vector<int>());
                // eval.EvaluateRHS(crhs, JSourceC, cx, uRecOld, uRecNew, betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit,
                //                  0); // TEval::RHS_Ignore_Viscosity
                return;
            }

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: StartRec");
            int nRec = (gradIsZero ? config.implicitReconstructionControl.nRecMultiplyForZeroedGrad : 1) *
                       config.implicitReconstructionControl.nInternalRecStep;
            if (step <= config.implicitReconstructionControl.zeroRecForSteps ||
                iter <= config.implicitReconstructionControl.zeroRecForStepsInternal)
            {
                nRec = 0;
                uRec.setConstant(0.0);
            }
            real recIncBase = 0;
            Timer().StartTimer(PerformanceTimer::Reconstruction);
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
                                log() << iRec << " Rec inc: " << recIncBase << " -> " << recInc << std::endl;
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
                            [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &Ax)
                            {
                                vfv->DoReconstructionIterDiff(uRec, x, Ax, cx, FBoundaryDiff);
                                Ax.trans.startPersistentPull();
                                Ax.trans.waitPersistentPull();
                            },
                            [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &MLx)
                            {
                                MLx = x; // initial value; for the input is mostly a good estimation
                                // MLx no need to comm
                                vfv->DoReconstructionIterSOR(uRecC, x, MLx, cx, FBoundaryDiff, false);
                            },
                            [&](ArrayRECV<nVarsFixed> &a, ArrayRECV<nVarsFixed> &b) -> real
                            {
                                return a.dot(b); //! need dim balancing here
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
                uRecC.trans.startPersistentPull();
                uRecC.trans.waitPersistentPull();
            }
            else if (config.implicitReconstructionControl.recLinearScheme == 2)
            {
                Eigen::Array<real, 1, Eigen::Dynamic> resB;
                int nPCGIterAll{0};
                if (iter <= 2)       //! consecutive pcg is bad in 0012, using separate pcg
                    pcgRec->reset(); // ! todo: account for inter-solve (need two pcgs!)
                uRecNew = uRecB1;
                vfv->DoReconstructionIter(
                    uRecC, uRecB1, cx,
                    FBoundary, true, true, true);
                if (iter > 2)
                {
                    if (config.implicitReconstructionControl.fpcgResetScheme == 0)
                    {
                        Eigen::RowVectorXd bPrevSqr = uRecNew.dotV(uRecNew);
                        uRecNew.addTo(uRecB1, -1.0);
                        Eigen::RowVectorXd bIncSqr = uRecNew.dotV(uRecNew);
                        real maxPortion = (bIncSqr.array() / (bPrevSqr.array() + smallReal)).sqrt().maxCoeff();
                        if (maxPortion >= config.implicitReconstructionControl.fpcgResetThres)
                        {
                            if (mpi.rank == 0 && config.implicitReconstructionControl.fpcgResetReport > 0)
                                log() << "FPCG force reset at portion " << fmt::format("{:.4g}", maxPortion) << std::endl;
                            pcgRec->reset();
                        }
                    }
                    else if (config.implicitReconstructionControl.fpcgResetScheme == 1)
                        pcgRec->reset();
                    else if (config.implicitReconstructionControl.fpcgResetScheme == 2)
                        ;
                    else
                        DNDS_assert_info(false, "invalid fpcgResetScheme");
                }
                // pcgRec->reset(); // using separate pcg
                for (int iRec = 1; iRec <= nRec; iRec++)
                {
                    bool pcgConverged = pcgRec->solve(
                        [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &Ax)
                        {
                            vfv->DoReconstructionIterDiff(uRec, x, Ax, cx, FBoundaryDiff);
                            Ax.trans.startPersistentPull();
                            Ax.trans.waitPersistentPull();
                            vfv->MatrixAMult(Ax, Ax);
                        },
                        [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &Mx)
                        {
                            vfv->MatrixAMult(x, Mx);
                        },
                        [&](ArrayRECV<nVarsFixed> &x, ArrayRECV<nVarsFixed> &res)
                        {
                            vfv->DoReconstructionIter(
                                x, res, cx,
                                FBoundary, true, true);
                            res.trans.startPersistentPull();
                            res.trans.waitPersistentPull();
                            res *= -1;
                        },
                        [&](ArrayRECV<nVarsFixed> &a, ArrayRECV<nVarsFixed> &b)
                        {
                            return (a.dotV(b)).array();
                        },
                        uRecC, config.implicitReconstructionControl.nGmresIter,
                        [&](uint32_t i, const Eigen::Array<real, 1, Eigen::Dynamic> &res, const Eigen::Array<real, 1, Eigen::Dynamic> &res1) -> bool
                        {
                            if (i == 1 && iRec == 1)
                                resB = res1;
                            if (i > 0)
                            {
                                nPCGIterAll++;
                                if (mpi.rank == 0 &&
                                    (nPCGIterAll % config.implicitReconstructionControl.nRecConsolCheck == 0))
                                {
                                    auto fmt = log().flags();
                                    log() << std::scientific << std::setprecision(3);
                                    log() << "PCG for Rec: " << iRec << " Restart " << i << " [" << resB << "] -> [" << res1 << "]" << std::endl;
                                    log().setf(fmt);
                                }
                            }
                            return (res1 / resB).maxCoeff() < config.implicitReconstructionControl.recThreshold;
                        });
                    if (pcgConverged)
                        break;
                }
                uRecC.trans.startPersistentPull();
                uRecC.trans.waitPersistentPull();
            }
            else
                DNDS_assert_info(false, "no such recLinearScheme");
            if ((model == NS_SA || model == NS_SA_3D) && eval.settings.ransForce2nd)
            {
                std::vector<int> mask;
                mask.resize(1);
                mask[0] = 5;
                vfv->DoReconstruction2nd(uRec, u, FBoundary, 1, mask);
            }
            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                std::vector<int> mask;
                mask.resize(2);
                mask[0] = 5;
                mask[1] = 6;
                vfv->DoReconstruction2nd(uRec, u, FBoundary, 1, mask);
            }
            Timer().StopTimer(PerformanceTimer::Reconstruction);
            if (gradIsZero)
            {
                uRec = uRecC;
                if (config.timeMarchControl.odeCode == 401)
                    uRec1 = uRecC;
                gradIsZero = false;
            }

            Timer().StartTimer(PerformanceTimer::Limiter);
            // for (index iCell = 0; iCell < uOld.size(); iCell++)
            //     uRecC[iCell].m() -= uOld[iCell].m();

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: StartLim");
            if (config.limiterControl.useLimiter)
            {
                // vfv->ReconstructionWBAPLimitFacial(
                //     cx, uRecC, uRecNew, uF0, uF1, ifUseLimiter,

                using tLimitBatch = typename TVFV::template tLimitBatch<nVarsFixed>;

                auto fML = [&](const TU &UL, const TU &UR, const Geom::tPoint &n,
                               const Eigen::Ref<tLimitBatch> &data) -> tLimitBatch
                {
                    Timer().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234) * 0.5;
                    Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasLeftEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Eigen::all, Seq123) *= normBase.transpose();

                    Eigen::Matrix<real, nVarsFixed, nVarsFixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;
                    Timer().StopTimer(PerformanceTimer::LimiterA);
                    return (ret * data.transpose()).transpose();
                    // return real(1);
                };
                auto fMR = [&](const TU &UL, const TU &UR, const Geom::tPoint &n,
                               const Eigen::Ref<tLimitBatch> &data) -> tLimitBatch
                {
                    Timer().StartTimer(PerformanceTimer::LimiterA);
                    Eigen::Vector<real, I4 + 1> UC = (UL + UR)(Seq01234) * 0.5;
                    Eigen::Matrix<real, dim, dim> normBase = Geom::NormBuildLocalBaseV<dim>(n(Seq012));
                    UC(Seq123) = normBase.transpose() * UC(Seq123);

                    auto M = Gas::IdealGas_EulerGasRightEigenVector<dim>(UC, eval.settings.idealGasProperty.gamma);
                    M(Seq123, Eigen::all) = normBase * M(Seq123, Eigen::all);

                    Eigen::Matrix<real, nVarsFixed, nVarsFixed> ret(nVars, nVars);
                    ret.setIdentity();
                    ret(Seq01234, Seq01234) = M;

                    Timer().StopTimer(PerformanceTimer::LimiterA);
                    return (ret * data.transpose()).transpose();
                    // return real(1);
                };
                if (config.limiterControl.smoothIndicatorProcedure == 0)
                    vfv->template DoCalculateSmoothIndicator<nVarsFixed, 2>(
                        ifUseLimiter, (uRecC), (u),
                        std::array<int, 2>{0, I4});
                else if (config.limiterControl.smoothIndicatorProcedure == 1)
                    vfv->template DoCalculateSmoothIndicatorV1<nVarsFixed>(
                        ifUseLimiter, (uRecC), (u),
                        TU::Ones(nVars),
                        [&](Eigen::Matrix<real, 1, nVarsFixed> &v)
                        {
                            TU prim;
                            TU cons;
                            cons = v.transpose();
                            if (cons(0) < verySmallReal)
                            {
                                v.setConstant(-veryLargeReal * v(I4));
                                return;
                            }
                            Gas::IdealGasThermalConservative2Primitive<dim>(cons, prim, eval.settings.idealGasProperty.gamma);
                            v.setConstant(prim(I4));
                            return;
                        });

                else
                {
                    DNDS_assert(false);
                }
                if (config.limiterControl.limiterProcedure == 1)
                    vfv->template DoLimiterWBAP_C<nVarsFixed>(
                        (cx),
                        (uRecC),
                        (uRecNew),
                        (uRecNew1),
                        ifUseLimiter,
                        iter < config.limiterControl.nPartialLimiterStartLocal && step < config.limiterControl.nPartialLimiterStart,
                        fML, fMR, true);
                else if (config.limiterControl.limiterProcedure == 0)
                    vfv->template DoLimiterWBAP_3<nVarsFixed>(
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
            Timer().StopTimer(PerformanceTimer::Limiter);
            if (config.implicitReconstructionControl.storeRecInc)
            {
                uRecIncC = uRecC;
                uRecIncC -= uRecOld; //! uRecIncC now stores uRecIncrement
            }
            if (config.implicitReconstructionControl.storeRecInc && config.implicitReconstructionControl.dampRecIncDTau)
            {
                ArrayDOFV<1> damper;
                ArrayDOFV<1> damper1;
                vfv->BuildUDof(damper, 1);
                vfv->BuildUDof(damper1, 1);
                damper = dTau;
                damper1 = dTau;
                damper1 += curDtImplicit;
                damper /= damper1;
                // for (auto &v : damper)
                //     v = v / (curDtImplicit + v); //! warning: teleported value here

                uRecIncC *= damper;
                uRecC = uRecOld;
                uRecC += uRecIncC;
            }
            if (config.limiterControl.preserveLimited && config.limiterControl.useLimiter)
                uRecC = uRecNew;

            // uRecC.trans.startPersistentPull(); //! this also need to update!
            // uRecC.trans.waitPersistentPull();

            // }

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: StartEval");
            eval.setPassiveDiscardSource(iter <= 0);

            if (iter == 1)
                alphaPPC.setConstant(1.0); // make RHS un-disturbed
            alphaPP_tmp.setConstant(1.0);  // make RHS un-disturbed
            if (config.limiterControl.usePPRecLimiter)
            {
                Timer().StartTimer(PerformanceTimer::Positivity);
                nLimBeta = 0;
                minBeta = 1;
                if (!config.limiterControl.useLimiter)
                    uRecNew = uRecC;
                eval.EvaluateURecBeta(cx, uRecNew, betaPPC, nLimBeta, minBeta,
                                      config.limiterControl.ppRecLimiterCompressToMean
                                          ? TEval::EvaluateURecBeta_COMPRESS_TO_MEAN
                                          : TEval::EvaluateURecBeta_DEFAULT); //*cx instead of u!
                if (nLimBeta)
                    if (mpi.rank == 0 &&
                        (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 3))
                    {
                        log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole)
                              << "PPRecLimiter: nLimBeta [" << nLimBeta << "]"
                              << " minBeta[" << minBeta << "]" << std::endl;
                    }
                uRecNew.trans.startPersistentPull();
                betaPPC.trans.startPersistentPull();
                uRecNew.trans.waitPersistentPull();
                betaPPC.trans.waitPersistentPull();
                Timer().StopTimer(PerformanceTimer::Positivity);
            }

            Timer().StartTimer(PerformanceTimer::RHS);
            if (config.limiterControl.useLimiter || config.limiterControl.usePPRecLimiter) // todo: opt to using limited for uRecUnlim
                eval.EvaluateRHS(crhs, JSourceC, cx, config.limiterControl.useViscousLimited ? uRecNew : uRecC, uRecNew,
                                 betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, uRecC,
                                 betaPPC, alphaPP_tmp, false, tSimu + ct * curDtImplicit);

            crhs.trans.startPersistentPull();
            crhs.trans.waitPersistentPull();
            if (getNVars(model) > (I4 + 1) && iter <= config.others.nFreezePassiveInner)
            {
                for (int i = 0; i < crhs.Size(); i++)
                    crhs[i](Eigen::seq(I4 + 1, Eigen::last)).setZero();
                // if (mpi.rank == 0)
                //     std::cout << "Freezing all passive" << std::endl;
            }
            Timer().StopTimer(PerformanceTimer::RHS);

            DNDS_MPI_InsertCheck(mpi, " Lambda RHS: End");
        };

        auto frhs =
            [&](
                ArrayDOFV<nVarsFixed> &crhs,
                ArrayDOFV<nVarsFixed> &cx,
                ArrayDOFV<1> &dTau,
                int iter, real ct, int uPos)
        {
            return frhsOuter(crhs, cx, dTau, iter, ct, uPos, 1); // reconstructionFlag == 1
        };

        auto fdtau = [&](ArrayDOFV<nVarsFixed> &cx, ArrayDOFV<1> &dTau, real alphaDiag, int uPos)
        {
            eval.FixUMaxFilter(cx);
            cx.trans.startPersistentPull(); //! this also need to update!
            cx.trans.waitPersistentPull();
            // uRec.trans.startPersistentPull();
            // uRec.trans.waitPersistentPull();
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            eval.EvaluateDt(dTau, cx, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            for (int iS = 1; iS <= config.implicitCFLControl.nSmoothDTau; iS++)
            {
                // ArrayDOFV<1> dTauNew = dTau; //TODO: copying is still unusable; consider doing copiers on the level of ArrayDOFV and ArrayRecV
                dTauTmp = dTau;
                dTau.trans.startPersistentPull();
                dTau.trans.waitPersistentPull();
                eval.MinSmoothDTau(dTau, dTauTmp);
                dTau = dTauTmp;
            }

            dTau *= 1. / alphaDiag;
        };

        auto fsolve = [&](ArrayDOFV<nVarsFixed> &cx, ArrayDOFV<nVarsFixed> &crhs, ArrayDOFV<1> &dTau,
                          real dt, real alphaDiag, ArrayDOFV<nVarsFixed> &cxInc, int iter, int uPos)
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
            bool inputIsZero{true}, hasLUDone{false};

            Timer().StartTimer(PerformanceTimer::Positivity);
            if (config.timeMarchControl.rhsFPPMode == 1 || config.timeMarchControl.rhsFPPMode == 11)
            {
                // ! experimental: bad now ?
                rhsTemp = crhs;
                rhsTemp *= dTau;
                rhsTemp *= config.timeMarchControl.rhsFPPScale;
                index nLimFRes = 0;
                real alphaMinFRes = 1;
                eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, rhsTemp, alphaPP_tmp, nLimFRes, alphaMinFRes, config.timeMarchControl.rhsFPPRelax,
                                          2, config.timeMarchControl.rhsFPPMode == 1 ? TEval::EvaluateCellRHSAlpha_DEFAULT : TEval::EvaluateCellRHSAlpha_MIN_IF_NOT_ONE);
                if (nLimFRes)
                    if (mpi.rank == 0)
                    {
                        log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole) << std::setw(3) << TermColor::Red;
                        log() << "PPFResLimiter: nLimFRes[" << nLimFRes << "] minAlpha [" << alphaMinFRes << "]" << TermColor::Reset << std::endl;
                    }

                crhs *= alphaPP_tmp;
            }
            else if (config.timeMarchControl.rhsFPPMode == 2)
            {
                rhsTemp = crhs;
                rhsTemp *= dTau;
                rhsTemp *= config.timeMarchControl.rhsFPPScale;
                index nLimFRes = 0;
                real alphaMinFRes = 1;
                eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, rhsTemp, alphaPP_tmp, nLimFRes, alphaMinFRes, config.timeMarchControl.rhsFPPRelax,
                                          2, TEval::EvaluateCellRHSAlpha_DEFAULT);
                if (nLimFRes)
                    if (mpi.rank == 0)
                    {
                        log() << std::scientific << std::setw(3) << TermColor::Red;
                        log() << "PPFResLimiter: nLimFRes[" << nLimFRes << "] minAlpha [" << alphaMinFRes << "]" << TermColor::Reset << std::endl;
                    }
                alphaPP_tmp.setMaxWith(smallReal); // dTau cannot be zero
                dTauTmp = dTau;
                dTauTmp *= alphaPP_tmp;
            }
            auto &dTauC = config.timeMarchControl.rhsFPPMode == 2 ? dTauTmp : dTau;
            Timer().StopTimer(PerformanceTimer::Positivity);

            typename TVFV::template TFBoundary<nVarsFixed>
                FBoundary = [&](const TU &UL, const TU &UMean, index iCell, index iFace, int iG,
                                const Geom::tPoint &normOut, const Geom::tPoint &pPhy, const Geom::t_index bType) -> TU
            {
                TU UR = UL;
                UR.setZero();
                return UR;
            };

            if (config.limiterControl.useLimiter) // uses urec value
                eval.LUSGSMatrixInit(JDC, JSourceC,
                                     dTauC, dt, alphaDiag,
                                     cx, uRecNew,
                                     0,
                                     tSimu);
            else
                eval.LUSGSMatrixInit(JDC, JSourceC,
                                     dTauC, dt, alphaDiag,
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
                    doPrecondition(alphaDiag, crhs, cx, cxInc, uTemp, JDC, sgsRes, inputIsZero, hasLUDone);
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
                        doPrecondition(alphaDiag, crhs, cx, cxInc, uTemp, JDC, sgsRes, inputIsZero, hasLUDone);
                    }
                    if (iterSGS == 1)
                        sgsRes0 = sgsRes;

                    if (mpi.rank == 0 && iterSGS % config.linearSolverControl.nSgsConsoleCheck == 0)
                        log() << std::scientific << "SGS1 " << std::to_string(iterSGS)
                              << " [" << sgsRes0.transpose() << "] ->"
                              << " [" << sgsRes.transpose() << "] " << std::endl;
                }
            }
            Eigen::VectorXd meanScale;
            if (config.linearSolverControl.gmresScale == 1)
            {
                meanScale = eval.settings.refU;
                meanScale(Seq123).setConstant(std::sqrt(meanScale(0) * meanScale(I4))); //! using consistent rho U scale
                // meanScale(I4) = sqr(meanScale(1)) / (meanScale(0) + verySmallReal);
                // meanScale(0) = 0.01;
                // meanScale(Seq123).setConstant(0.1);
                // meanScale(I4) = 1;
            }
            else if (config.linearSolverControl.gmresScale == 2)
            {
                eval.EvaluateNorm(meanScale, cx, 1, true, true);
                meanScale(Seq123).setConstant(meanScale(Seq123).norm());
                meanScale(I4) = sqr(meanScale(1)) / (meanScale(0) + verySmallReal);
            }
            else
                meanScale.setOnes(nVars);
            // meanScale(0) = 10;
            TU meanScaleInv = (meanScale.array() + verySmallReal).inverse();

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
                        MLx.setConstant(0.0), inputIsZero = true; //! start as zero
                        doPrecondition(alphaDiag, x, cx, MLx, uTemp, JDC, sgsRes, inputIsZero, hasLUDone);
                        for (int i = 0; i < config.linearSolverControl.sgsIter; i++)
                        {
                            doPrecondition(alphaDiag, x, cx, MLx, uTemp, JDC, sgsRes, inputIsZero, hasLUDone);
                        }
                    },
                    [&](decltype(u) &a, decltype(u) &b) -> real
                    {
                        return a.dot(b, meanScaleInv.array(), meanScaleInv.array());
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
                              ArrayDOFV<nVarsFixed> &cx,
                              ArrayDOFV<nVarsFixed> &cx1,
                              ArrayDOFV<nVarsFixed> &crhs,
                              ArrayDOFV<1> &dTau,
                              const std::vector<real> &Coefs, // coefs are dU * c[0] + dt * c[1] * (I/(dt * c[2]) - JMid) * (I/(dt * c[3]) - J)
                              real dt, real alphaDiag, ArrayDOFV<nVarsFixed> &cxInc, int iter, int uPos)
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
            dTau *= Coefs[2];
            eval.LUSGSMatrixInit(JD1, JSource1,
                                 dTau, dt * Coefs[2], alphaDiag,
                                 cx1, uRec,
                                 0,
                                 tSimu);
            eval.EvaluateDt(dTau, cx, uRecC, CFLNow, curDtMin, 1e100, config.implicitCFLControl.useLocalDt);
            dTau *= Coefs[3] * veryLargeReal;
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
                    [&](decltype(u) &a, decltype(u) &b) -> real
                    {
                        return a.dot(b); //! need dim balancing here
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
                                   ArrayDOFV<nVarsFixed> &v,
                                   int uPos)
        {
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            for (index i = 0; i < v.Size(); i++)
                v[i] *= alphaPPC[i](0);
        };

        auto fresidualIncPP = [&](
                                  ArrayDOFV<nVarsFixed> &cx,
                                  ArrayDOFV<nVarsFixed> &xPrev,
                                  ArrayDOFV<nVarsFixed> &crhs,
                                  ArrayDOFV<nVarsFixed> &rhsIncPart,
                                  const std::function<void()> &renewRhsIncPart,
                                  real ct,
                                  int uPos)
        {
            Timer().StartTimer(PerformanceTimer::Positivity);
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            renewRhsIncPart(); // un-fixed now
            // rhsIncPart.trans.startPersistentPull();
            // rhsIncPart.trans.waitPersistentPull(); //seems not needed
            eval.EvaluateCellRHSAlpha(xPrev, uRecC, betaPPC, rhsIncPart, alphaPP_tmp, nLimAlpha, minAlpha, 1.,
                                      2, TEval::EvaluateCellRHSAlpha_MIN_IF_NOT_ONE);
            alphaPP_tmp.trans.startPersistentPull();
            alphaPP_tmp.trans.waitPersistentPull();
            if (nLimAlpha)
                if (mpi.rank == 0 &&
                    (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 4))
                {
                    log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole)
                          << "PPResidualLimiter: nLimAlpha [" << nLimAlpha << "]"
                          << " minAlpha[" << minAlpha << "]" << std::endl;
                }
            alphaPPC = alphaPP_tmp;
            if (config.limiterControl.useLimiter || config.limiterControl.usePPRecLimiter)
                eval.EvaluateRHS(crhs, JSourceC, cx, config.limiterControl.useViscousLimited ? uRecNew : uRecC, uRecNew,
                                 betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
            else
                eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, uRecC,
                                 betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
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
                    if (mpi.rank == 0 &&
                        (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 4))
                    {
                        log() << std::scientific << std::setprecision(config.outputControl.nPrecisionConsole)
                              << "PPResidualLimiter - first expand: nLimAlpha [" << nLimAlpha << "]"
                              << " minAlpha[" << minAlpha << "]" << std::endl;
                    }
                alphaPPC = alphaPP_tmp;
                if (config.limiterControl.useLimiter || config.limiterControl.usePPRecLimiter)
                    eval.EvaluateRHS(crhs, JSourceC, cx, config.limiterControl.useViscousLimited ? uRecNew : uRecC, uRecNew,
                                     betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
                else
                    eval.EvaluateRHS(crhs, JSourceC, cx, uRecC, uRecC,
                                     betaPPC, alphaPPC, false, tSimu + ct * curDtImplicit);
                crhs.trans.startPersistentPull();
                crhs.trans.waitPersistentPull();
            }
            Timer().StopTimer(PerformanceTimer::Positivity);
        };

        auto fincrement = [&](
                              ArrayDOFV<nVarsFixed> &cx,
                              ArrayDOFV<nVarsFixed> &cxInc,
                              real alpha, int uPos)
        {
            Timer().StartTimer(PerformanceTimer::Positivity);
            auto &alphaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? alphaPP1 : alphaPP;
            auto &betaPPC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? betaPP1 : betaPP;
            auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;
            auto &JSourceC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? JSource1 : JSource;
            nLimInc = 0;
            alphaMinInc = 1;
            eval.EvaluateCellRHSAlpha(cx, uRecC, betaPPC, cxInc, alphaPP_tmp, nLimInc, alphaMinInc, config.timeMarchControl.incrementPPRelax,
                                      2, TEval::EvaluateCellRHSAlpha_DEFAULT);
            if (nLimInc)
                if (mpi.rank == 0 &&
                    (config.outputControl.consoleOutputEveryFix == 1 || config.outputControl.consoleOutputEveryFix == 2))
                {
                    log() << std::scientific << std::setw(3) << TermColor::Red;
                    log() << "PPIncrementLimiter: nIncrementRes[" << nLimInc << "] minAlpha [" << alphaMinInc << "]"
                          << TermColor::Reset
                          << std::endl;
                }

            uTemp = cxInc;
            uTemp *= alphaPP_tmp;
            //*directadd
            // cx += uTemp;
            //*fixing add
            if (model == NS_2EQ || model == NS_2EQ_3D)
            {
                if (config.implicitCFLControl.RANSRelax != 1)
                    for (index i = 0; i < uTemp.Size(); i++)
                        uTemp[i]({I4, I4 + 1}) *= config.implicitCFLControl.RANSRelax;
            }
            Timer().StopTimer(PerformanceTimer::Positivity);
            eval.AddFixedIncrement(cx, uTemp, alpha);
            eval.AssertMeanValuePP(cx, true);

            // eval.AddFixedIncrement(cx, cxInc, alpha);
        };

        auto fstop = [&](int iter, ArrayDOFV<nVarsFixed> &cxinc, int iStep) -> bool
        {
            // auto &uRecC = config.timeMarchControl.odeCode == 401 && uPos == 1 ? uRec1 : uRec;

            Eigen::VectorFMTSafe<real, -1> res(nVars);
            eval.EvaluateNorm(res, cxinc, 1, config.convergenceControl.useVolWiseResidual);
            // if (iter == 1 && iStep == 1) // * using 1st rk step for reference
            if (iter == 1)
                resBaseCInternal = res;
            else
                resBaseCInternal = resBaseCInternal.array().max(res.array()); //! using max !
            Eigen::VectorFMTSafe<real, -1> resRel = (res.array() / (resBaseCInternal.array() + verySmallReal)).matrix();
            bool ifStop = resRel(0) < config.convergenceControl.rhsThresholdInternal; // ! using only rho's residual
            if (iter < config.convergenceControl.nTimeStepInternalMin)
                ifStop = false;
            if (iter % config.outputControl.nConsoleCheckInternal == 0 || iter > config.convergenceControl.nTimeStepInternal || ifStop)
            {
                double tWall = MPI_Wtime();
                real telapsed = MPI_Wtime() - tstartInternal;
                bool useCollectiveTimer = config.outputControl.useCollectiveTimer;
                real tcomm = Timer().getTimerColOrLoc(PerformanceTimer::Comm, mpi, useCollectiveTimer);
                real tLimiterA = Timer().getTimerColOrLoc(PerformanceTimer::LimiterA, mpi, useCollectiveTimer);
                real tLimiterB = Timer().getTimerColOrLoc(PerformanceTimer::LimiterB, mpi, useCollectiveTimer);
                real trhs = Timer().getTimerColOrLoc(PerformanceTimer::RHS, mpi, useCollectiveTimer);
                real trec = Timer().getTimerColOrLoc(PerformanceTimer::Reconstruction, mpi, useCollectiveTimer);
                real tLim = Timer().getTimerColOrLoc(PerformanceTimer::Limiter, mpi, useCollectiveTimer);
                real tPP = Timer().getTimerColOrLoc(PerformanceTimer::Positivity, mpi, useCollectiveTimer);
                auto [telapsedM, telapsedS] = tInternalStats["t"].update(telapsed).get();
                auto [tcommM, tcommS] = tInternalStats["c"].update(tcomm).get();
                auto [trhsM, trhsS] = tInternalStats["r"].update(trhs).get();
                auto [trecM, trecS] = tInternalStats["v"].update(trec).get();
                auto [tLimM, tLimS] = tInternalStats["l"].update(tLim).get();
                auto [tPPrM, tPPrS] = tInternalStats["p"].update(tPP).get();

                if (mpi.rank == 0)
                {
                    auto fmt = log().flags();
                    std::string formatStringMain = "";
                    for (auto &s : config.outputControl.consoleMainOutputFormatInternal)
                        formatStringMain += s;
                    log() << fmt::format(formatStringMain +
                                             "  "s +
                                             (config.outputControl.consoleOutputMode == 1
                                                  ? "WallFlux {termYellow}{wallFlux:.6e}{termReset}"s
                                                  : ""s),
                                         DNDS_FMT_ARG(step),
                                         DNDS_FMT_ARG(iStep),
                                         DNDS_FMT_ARG(iter),
                                         fmt::arg("resRel", resRel.transpose()),
                                         fmt::arg("wallFlux", eval.fluxWallSum.transpose()),
                                         DNDS_FMT_ARG(tSimu),
                                         DNDS_FMT_ARG(curDtImplicit),
                                         DNDS_FMT_ARG(curDtMin),
                                         DNDS_FMT_ARG(CFLNow),
                                         DNDS_FMT_ARG(nLimInc),
                                         DNDS_FMT_ARG(alphaMinInc),
                                         DNDS_FMT_ARG(nLimBeta),
                                         DNDS_FMT_ARG(minBeta),
                                         DNDS_FMT_ARG(nLimAlpha),
                                         DNDS_FMT_ARG(minAlpha),
                                         DNDS_FMT_ARG(telapsed), DNDS_FMT_ARG(telapsedM),
                                         DNDS_FMT_ARG(trec), DNDS_FMT_ARG(trecM),
                                         DNDS_FMT_ARG(trhs), DNDS_FMT_ARG(trhsM),
                                         DNDS_FMT_ARG(tcomm), DNDS_FMT_ARG(tcommM),
                                         DNDS_FMT_ARG(tLim), DNDS_FMT_ARG(tLimM),
                                         DNDS_FMT_ARG(tPP), DNDS_FMT_ARG(tPPrM),
                                         DNDS_FMT_ARG(tLimiterA),
                                         DNDS_FMT_ARG(tLimiterB),
                                         DNDS_FMT_ARG(tWall),
                                         fmt::arg("termRed", TermColor::Red),
                                         fmt::arg("termBlue", TermColor::Blue),
                                         fmt::arg("termGreen", TermColor::Green),
                                         fmt::arg("termCyan", TermColor::Cyan),
                                         fmt::arg("termYellow", TermColor::Yellow),
                                         fmt::arg("termBold", TermColor::Bold),
                                         fmt::arg("termReset", TermColor::Reset));
                    log() << std::endl;
                    log().setf(fmt);

                    std::string delimC = " ";
                    logErr
                        << std::left
                        << step << delimC
                        << std::left
                        << iter << delimC
                        << std::left
                        << std::setprecision(config.outputControl.nPrecisionLog) << std::scientific
                        << res.transpose() << delimC
                        << tSimu << delimC
                        << curDtMin << delimC
                        << real(eval.nFaceReducedOrder) << delimC
                        << eval.fluxWallSum.transpose() << delimC
                        << (nLimInc) << delimC << (alphaMinInc) << delimC
                        << (nLimBeta) << delimC << (minBeta) << delimC
                        << (nLimAlpha) << delimC << (minAlpha) << delimC
                        << std::endl;

                    eval.ConsoleOutputBndIntegrations();
                    eval.BndIntegrationLogWriteLine(
                        config.dataIOControl.getOutLogName() + "_" + output_stamp,
                        step, iStep, iter);
                }
                tstartInternal = MPI_Wtime();
                Timer().clearAllTimer();
            }

            if (iter % config.outputControl.nDataOutInternal == 0)
            {
                eval.FixUMaxFilter(u);
                PrintData(
                    config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter),
                    config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step), // internal series
                    [&](index iCell)
                    { return ode->getLatestRHS()[iCell](0); },
                    addOutList,
                    eval, tSimu);
                eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step),
                                     u, uRec);
            }
            if ((iter % config.outputControl.nDataOutCInternal == 0) &&
                !(config.outputControl.lazyCoverDataOutput && (iter % config.outputControl.nDataOutInternal == 0)))
            {
                eval.FixUMaxFilter(u);
                PrintData(
                    config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                    "",
                    [&](index iCell)
                    { return ode->getLatestRHS()[iCell](0); },
                    addOutList,
                    eval, tSimu);
                eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                                     u, uRec);
            }
            if (iter % config.outputControl.nRestartOutInternal == 0)
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = iter;
                PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter));
            }
            if ((iter % config.outputControl.nRestartOutCInternal == 0) &&
                !(config.outputControl.lazyCoverDataOutput && (iter % config.outputControl.nRestartOutInternal == 0)))
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = iter;
                PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + "C");
            }
            if (iter >= config.implicitCFLControl.nCFLRampStart && iter <= config.implicitCFLControl.nCFLRampLength + config.implicitCFLControl.nCFLRampStart)
            {
                real inter = real(iter - config.implicitCFLControl.nCFLRampStart) / config.implicitCFLControl.nCFLRampLength;
                real logCFL = std::log(config.implicitCFLControl.CFL) + (std::log(config.implicitCFLControl.CFLRampEnd / config.implicitCFLControl.CFL) * inter);
                CFLNow = std::exp(logCFL);
            }
            if (ifStop || iter > config.convergenceControl.nTimeStepInternal) //! TODO: reconstruct the framework of ODE-top-level-control
            {
                CFLNow = config.implicitCFLControl.CFL;
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
            Eigen::VectorFMTSafe<real, -1> res(nVars);
            eval.EvaluateNorm(res, ode->getLatestRHS(), 1, config.convergenceControl.useVolWiseResidual);
            if (stepCount == 0 && resBaseC.norm() == 0)
                resBaseC = res;

            if (config.timeAverageControl.enabled)
            {
                eval.MeanValueCons2Prim(u, uTemp); // could use time-step-mean-u instead of latest-u
                eval.TimeAverageAddition(uTemp, wAveraged, curDtImplicit, tAverage);
            }

            if (step % config.outputControl.nConsoleCheck == 0)
            {
                double tWall = MPI_Wtime();
                real telapsed = MPI_Wtime() - tstart;
                bool useCollectiveTimer = config.outputControl.useCollectiveTimer;
                real tcomm = Timer().getTimerColOrLoc(PerformanceTimer::Comm, mpi, useCollectiveTimer);
                real tLimiterA = Timer().getTimerColOrLoc(PerformanceTimer::LimiterA, mpi, useCollectiveTimer);
                real tLimiterB = Timer().getTimerColOrLoc(PerformanceTimer::LimiterB, mpi, useCollectiveTimer);
                real trhs = Timer().getTimerColOrLoc(PerformanceTimer::RHS, mpi, useCollectiveTimer);
                real trec = Timer().getTimerColOrLoc(PerformanceTimer::Reconstruction, mpi, useCollectiveTimer);
                real tLim = Timer().getTimerColOrLoc(PerformanceTimer::Limiter, mpi, useCollectiveTimer);

                tcomm = tInternalStats["c"].update(tcomm).getSum();
                trhs = tInternalStats["r"].update(trhs).getSum();
                trec = tInternalStats["v"].update(trec).getSum();
                tLim = tInternalStats["l"].update(tLim).getSum();
                auto tPPr = tInternalStats["p"].getSum() + Timer().getTimerColOrLoc(PerformanceTimer::PositivityOuter, mpi, useCollectiveTimer);
                if (mpi.rank == 0)
                {
                    auto fmt = log().flags();
                    std::string formatStringMain = "";
                    for (auto &s : config.outputControl.consoleMainOutputFormat)
                        formatStringMain += s;
                    log() << fmt::format(formatStringMain +
                                             "  "s +
                                             (config.outputControl.consoleOutputMode == 1
                                                  ? "WallFlux {termYellow}{wallFlux:.6e}{termReset}"s
                                                  : ""s),
                                         DNDS_FMT_ARG(step),
                                         // DNDS_FMT_ARG(iStep),
                                         // DNDS_FMT_ARG(iter),
                                         fmt::arg("resRel", (res.array() / (resBaseC.array() + verySmallReal)).transpose()),
                                         fmt::arg("wallFlux", eval.fluxWallSum.transpose()),
                                         DNDS_FMT_ARG(tSimu),
                                         DNDS_FMT_ARG(curDtImplicit),
                                         DNDS_FMT_ARG(curDtMin),
                                         DNDS_FMT_ARG(CFLNow),
                                         DNDS_FMT_ARG(nLimInc),
                                         DNDS_FMT_ARG(alphaMinInc),
                                         DNDS_FMT_ARG(nLimBeta),
                                         DNDS_FMT_ARG(minBeta),
                                         DNDS_FMT_ARG(nLimAlpha),
                                         DNDS_FMT_ARG(minAlpha),
                                         DNDS_FMT_ARG(telapsed),
                                         DNDS_FMT_ARG(trec),
                                         DNDS_FMT_ARG(trhs),
                                         DNDS_FMT_ARG(tcomm),
                                         DNDS_FMT_ARG(tLim),
                                         DNDS_FMT_ARG(tPPr),
                                         DNDS_FMT_ARG(tLimiterA),
                                         DNDS_FMT_ARG(tLimiterB),
                                         DNDS_FMT_ARG(tWall),
                                         fmt::arg("termRed", TermColor::Red),
                                         fmt::arg("termBlue", TermColor::Blue),
                                         fmt::arg("termGreen", TermColor::Green),
                                         fmt::arg("termCyan", TermColor::Cyan),
                                         fmt::arg("termYellow", TermColor::Yellow),
                                         fmt::arg("termBold", TermColor::Bold),
                                         fmt::arg("termReset", TermColor::Reset));
                    log() << std::endl;
                    log().setf(fmt);
                    std::string delimC = " ";
                    logErr
                        << std::left
                        << step << delimC
                        << std::left
                        << -1 << delimC
                        << std::left
                        << std::setprecision(config.outputControl.nPrecisionLog) << std::scientific
                        << res.transpose() << delimC
                        << tSimu << delimC
                        << curDtMin << delimC
                        << real(eval.nFaceReducedOrder) << delimC
                        << eval.fluxWallSum.transpose() << delimC
                        << (nLimInc) << delimC << (alphaMinInc) << delimC
                        << (nLimBeta) << delimC << (minBeta) << delimC
                        << (nLimAlpha) << delimC << (minAlpha) << delimC
                        << std::endl;
                    eval.ConsoleOutputBndIntegrations();
                    eval.BndIntegrationLogWriteLine(
                        config.dataIOControl.getOutLogName() + "_" + output_stamp,
                        step, -1, -1);
                }
                tstart = MPI_Wtime();
                Timer().clearAllTimer();
                for (auto &s : tInternalStats)
                    s.second.clear();
            }
            if (step == nextStepOutC)
            {
                if (!(config.outputControl.lazyCoverDataOutput && (step == nextStepOut)))
                {
                    eval.FixUMaxFilter(u);
                    PrintData(
                        config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                        "",
                        [&](index iCell)
                        { return ode->getLatestRHS()[iCell](0); },
                        addOutList,
                        eval, tSimu);
                    eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "C",
                                         u, uRec);
                }
                nextStepOutC += config.outputControl.nDataOutC;
            }
            if (step == nextStepOut)
            {
                eval.FixUMaxFilter(u);
                PrintData(
                    config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step),
                    config.dataIOControl.outPltName + "_" + output_stamp, // physical ts series
                    [&](index iCell)
                    { return ode->getLatestRHS()[iCell](0); },
                    addOutList,
                    eval, tSimu);
                eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + std::to_string(step),
                                     u, uRec);
                nextStepOut += config.outputControl.nDataOut;
            }
            if (step == nextStepOutAverageC)
            {
                if (!(config.outputControl.lazyCoverDataOutput && (step == nextStepOutAverage)))
                {
                    DNDS_assert(config.timeAverageControl.enabled);
                    eval.MeanValuePrim2Cons(wAveraged, uAveraged);
                    eval.FixUMaxFilter(uAveraged);
                    PrintData(
                        config.dataIOControl.outPltName + "_TimeAveraged_" + output_stamp + "_" + "C",
                        "",
                        [&](index iCell)
                        { return ode->getLatestRHS()[iCell](0); },
                        addOutList,
                        eval, tSimu,
                        PrintDataTimeAverage);
                }
                nextStepOutAverageC += config.outputControl.nTimeAverageOutC;
            }
            if (step == nextStepOutAverage)
            {
                DNDS_assert(config.timeAverageControl.enabled);
                eval.MeanValuePrim2Cons(wAveraged, uAveraged);
                eval.FixUMaxFilter(uAveraged);
                PrintData(
                    config.dataIOControl.outPltName + "_TimeAveraged_" + output_stamp + "_" + std::to_string(step),
                    config.dataIOControl.outPltName + "_TimeAveraged_" + output_stamp, // time average series
                    [&](index iCell)
                    { return ode->getLatestRHS()[iCell](0); },
                    addOutList,
                    eval, tSimu,
                    PrintDataTimeAverage);
                nextStepOutAverage += config.outputControl.nTimeAverageOut;
            }
            if (step == nextStepRestartC)
            {
                if (!(config.outputControl.lazyCoverDataOutput && (step == nextStepRestart)))
                {
                    config.restartState.iStep = step;
                    config.restartState.iStepInternal = -1;
                    PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + "C");
                }
                nextStepRestartC += config.outputControl.nRestartOutC;
            }
            if (step == nextStepRestart)
            {
                config.restartState.iStep = step;
                config.restartState.iStepInternal = -1;
                PrintRestart(config.dataIOControl.getOutRestartName() + "_" + output_stamp + "_" + std::to_string(step));
                nextStepRestart += config.outputControl.nRestartOut;
            }
            if (ifOutT)
            {
                eval.FixUMaxFilter(u);
                PrintData(
                    config.dataIOControl.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout),
                    config.dataIOControl.outPltName + "_" + output_stamp, // physical ts series
                    [&](index iCell)
                    { return ode->getLatestRHS()[iCell](0); },
                    addOutList,
                    eval, tSimu);
                eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout),
                                     u, uRec);
                nextTout += config.outputControl.tDataOut;
                if (nextTout >= config.timeMarchControl.tEnd)
                    nextTout = config.timeMarchControl.tEnd;
            }
            if ((eval.settings.specialBuiltinInitializer == 2 ||
                 eval.settings.specialBuiltinInitializer == 203) &&
                (step % config.outputControl.nConsoleCheck == 0)) // IV problem special: reduction on solution
            {
                auto FVal = [&](const Geom::tPoint &p, real t)
                {
                    switch (eval.settings.specialBuiltinInitializer)
                    {
                    case 203:
                        return SpecialFields::IsentropicVortex10(eval, p, t, nVars, 10.0828);
                    default:
                    case 2:
                        return SpecialFields::IsentropicVortex10(eval, p, t, nVars, 5);
                    }
                };
                auto FWeight = [&](const Geom::tPoint &p, real t) -> real
                {
                    // real xyOrig = t;
                    // real xCC = float_mod(p(0) - xyOrig, 10);
                    // real yCC = float_mod(p(1) - xyOrig, 10);
                    // return std::abs(xCC - 5.0) <= 2 && std::abs(yCC - 5.0) <= 2 ? 1.0 : 0.0;
                    return 1.0;
                };
                Eigen::Vector<real, -1> err1, errInf;
                eval.EvaluateRecNorm(
                    err1, u, uRec, 1, true,
                    FVal, FWeight,
                    tSimu);
                eval.EvaluateRecNorm(
                    errInf, u, uRec, 1000, true,
                    FVal, FWeight,
                    tSimu);

                if (mpi.rank == 0)
                {
                    log() << "=== Mean Error IV: [" << std::scientific
                          << std::setprecision(config.outputControl.nPrecisionConsole + 4) << err1(0) << ", "
                          << err1(0) / vfv->GetGlobalVol() << ", "
                          << errInf(0)
                          << "]" << std::endl;
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
        // step 0 extra:
        if (config.outputControl.dataOutAtInit)
        {
            PrintData(
                config.dataIOControl.outPltName + "_" + output_stamp + "_" + "00000",
                "",
                [&](index iCell)
                { return ode->getLatestRHS()[iCell](0); },
                addOutList,
                eval, tSimu);
            eval.PrintBCProfiles(config.dataIOControl.outPltName + "_" + output_stamp + "_" + "00000",
                                 u, uRec);
        }

        for (step = 1; step <= config.timeMarchControl.nTimeStep; step++)
        {
            DNDS_MPI_InsertCheck(mpi, "Implicit Step");
            ifOutT = false;
            real curDtImplicitOld = curDtImplicit;
            curDtImplicit = config.timeMarchControl.dtImplicit;
            CFLNow = config.implicitCFLControl.CFL;
            fdtau(u, dTauTmp, 1., 0);                                                                             // generates a curDtMin / CFLNow value as a CFL=1 dt value
            curDtImplicit = std::min(curDtMin / CFLNow * config.timeMarchControl.dtCFLLimitScale, curDtImplicit); // limits dt by CFL

            if (config.timeMarchControl.useDtPPLimit)
            {
                Timer().StartTimer(PerformanceTimer::PositivityOuter);
                dTauTmp.setConstant(curDtImplicit * config.timeMarchControl.dtPPLimitScale); //? used as damper here, appropriate?
                frhsOuter(rhsTemp, u, dTauTmp, 1, 0.0, 0, 0);                                //* trick: use 0th order reconstruction RHS for dt PP limiting
                uTemp = u;
                rhsTemp *= curDtImplicit * config.timeMarchControl.dtPPLimitScale;
                index nLim = 0;
                real minLim = 1;
                eval.EvaluateCellRHSAlpha(u, uRec, alphaPP, rhsTemp, alphaPP_tmp, nLim, minLim, config.timeMarchControl.dtPPLimitRelax,
                                          1, TEval::EvaluateCellRHSAlpha_DEFAULT); // using compress = 1, only minLim is used as output
                if (nLim)
                    curDtImplicit = std::min(curDtImplicit, minLim * curDtImplicit);
                if (curDtImplicitHistory.size() && curDtImplicit > curDtImplicitHistory.back() * config.timeMarchControl.dtIncreaseLimit)
                {
                    curDtImplicit = curDtImplicitHistory.back() * config.timeMarchControl.dtIncreaseLimit;
                }
                if (mpi.rank == 0 && nLim)
                {
                    log() << "##################################################################" << std::endl;
                    log() << fmt::format("At Step [{:d}] t [{:.8g}] Changing dt to [{}], using PP", step, tSimu, curDtImplicit) << std::endl;
                    log() << "##################################################################" << std::endl;
                }
                Timer().StopTimer(PerformanceTimer::PositivityOuter);
            }
            curDtImplicit = std::max(curDtImplicit, config.timeMarchControl.dtImplicitMin);

            DNDS_assert(curDtImplicit > 0);
            if (tSimu + curDtImplicit >= nextTout - curDtImplicit * smallReal) // limits dt by output nodes
            {
                ifOutT = true;
                curDtImplicit = std::max(0.0, nextTout - tSimu);
            }

            if (config.timeMarchControl.useImplicitPP)
            {
                switch (config.timeMarchControl.odeCode)
                {
                case 1: // BDF2
                    std::dynamic_pointer_cast<ODE::ImplicitBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(ode)
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
                    break;
                case 102: // VBDFPP
                {
                    index nLimAlpha;
                    real minAlpha;
                    auto odeVBDF = std::dynamic_pointer_cast<ODE::ImplicitVBDFDualTimeStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(ode);
                    odeVBDF->LimitDt_StepPPV2(
                        u, [&](ArrayDOFV<nVarsFixed> &u, ArrayDOFV<nVarsFixed> &uInc) -> real
                        {
                            eval.EvaluateCellRHSAlpha(u, uRec, alphaPP, uInc, alphaPP_tmp, nLimAlpha, minAlpha, 1.,
                                                    1, TEval::EvaluateCellRHSAlpha_MIN_IF_NOT_ONE); // using compress == 1, only minAlpha is used
                            return minAlpha; },
                        curDtImplicit, 2); // curDtImplicit modified
                    if (curDtImplicit > curDtImplicitOld)
                    {
                        dtIncreaseCounter++;
                        if (dtIncreaseCounter > config.timeMarchControl.dtIncreaseAfterCount)
                            curDtImplicit = std::min(curDtImplicitOld * config.timeMarchControl.dtIncreaseLimit, curDtImplicit);
                        else
                            curDtImplicit = curDtImplicitOld;
                    }
                    else
                    {
                        dtIncreaseCounter = 0;
                    }

                    if (mpi.rank == 0)
                    {
                        log() << "##################################################################" << std::endl;
                        log() << fmt::format("At Step [{:d}] t [{:.8g}] Changing dt to {}", step, tSimu, curDtImplicit) << std::endl;
                        log() << "##################################################################" << std::endl;
                    }
                    odeVBDF->StepPP(
                        u, uInc,
                        frhs,
                        fdtau,
                        fsolve,
                        config.convergenceControl.nTimeStepInternal,
                        fstop, fincrement,
                        curDtImplicit + verySmallReal,
                        falphaLimSource,
                        fresidualIncPP);
                }
                break;

                default:
                    DNDS_assert_info(false, "unsupported odeCode for PP!");
                    break;
                }
            }
            else if (config.timeMarchControl.odeCode == 401 && false)
                std::dynamic_pointer_cast<ODE::ImplicitHermite3SimpleJacobianDualStep<ArrayDOFV<nVarsFixed>, ArrayDOFV<1>>>(ode)
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
            curDtImplicitHistory.push_back(curDtImplicit);
            if (fmainloop())
                break;
        }
    }

    template <EulerModel model>
    void EulerSolver<model>::doPrecondition(real alphaDiag, TDof &crhs, TDof &cx, TDof &cxInc, TDof &uTemp, JacobianDiagBlock<nVarsFixed> &JDC, TU &sgsRes, bool &inputIsZero, bool &hasLUDone)
    {
        DNDS_assert(pEval);
        auto &eval = *pEval;
        // static int nCall{0};
        // nCall++;
        // if (mpi.rank == 0)
        //     std::cout << "doPrecondition nCall " << nCall << fmt::format(" {} ", hasLUDone) << std::endl;
        if (config.linearSolverControl.jacobiCode <= 1)
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
            inputIsZero = false;
        }
        else if (config.linearSolverControl.jacobiCode == 2)
        {
            DNDS_assert_info(config.linearSolverControl.directPrecControl.useDirectPrec, "need to use config.linearSolverControl.directPrecControl.useDirectPrec first !");
            if (!hasLUDone)
                eval.LUSGSMatrixToJacobianLU(alphaDiag, cx, JDC, *JLocalLU), hasLUDone = true;
            for (int iii = 0; iii < 2; iii++)
            {
                eval.LUSGSMatrixSolveJacobianLU(alphaDiag, crhs, cx, cxInc, uTemp, rhsTemp, JDC, *JLocalLU, sgsRes);
                uTemp.SwapDataFatherSon(cxInc);
                // cxInc = uTemp;
                cxInc.trans.startPersistentPull();
                cxInc.trans.waitPersistentPull();
            }
            inputIsZero = false;
        }
    }
}