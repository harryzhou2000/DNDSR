#pragma once
#include "Gas.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "Solver/ODE.hpp"
#include "Solver/Linear.hpp"
#include "EulerEvaluator.hpp"

#include <iomanip>
#include <functional>

#define JSON_ASSERT DNDS_assert
#include "json.hpp"
#include "EulerBC.hpp"

namespace DNDS::Euler
{

    template <EulerModel model>
    class EulerSolver
    {
        int nVars;

    public:
        typedef EulerEvaluator<model> TEval;
        static const int nVars_Fixed = TEval::nVars_Fixed;

        static const int dim = TEval::dim;
        // static const int gdim = TEval::gdim;
        static const int gDim = TEval::gDim;
        static const int I4 = TEval::I4;

        typedef typename TEval::TU TU;
        typedef typename TEval::TDiffU TDiffU;
        typedef typename TEval::TJacobianU TJacobianU;
        typedef typename TEval::TVec TVec;
        typedef typename TEval::TMat TMat;

    private:
        MPIInfo mpi;
        ssp<Geom::UnstructuredMesh> mesh;
        ssp<CFV::VariationalReconstruction<gDim>> vfv; // ! gDim -> 3 for intellisense
        ssp<Geom::UnstructuredMeshSerialRW> reader;

        ArrayDOFV<nVars_Fixed> u, uInc, uIncRHS, uTemp;
        ArrayRECV<nVars_Fixed> uRec, uRecNew, uRecNew1, uOld;

        int nOUTS = {-1};
        // rho u v w p T M ifUseLimiter RHS
        ssp<ArrayEigenVector<Eigen::Dynamic>> outDist;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerial;
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTrans;

        // std::vector<uint32_t> ifUseLimiter;
        CFV::tScalarPair ifUseLimiter;

        BoundaryHandler<model>
            BCHandler;

    public:
        EulerSolver(const MPIInfo &nmpi) : nVars(getNVars(model)), mpi(nmpi)
        {
            nOUTS = nVars + 4;
        }

        nlohmann::json gSetting;
        std::string output_stamp = "";

        struct Configuration
        {
            int recOrder = 2;
            int nInternalRecStep = 1;
            int nTimeStep = 1000;
            int nConsoleCheck = 10;
            int nConsoleCheckInternal = 1;
            int consoleOutputMode = 0; // 0 for basic, 1 for wall force out
            int nSGSIterationInternal = 0;
            int nDataOut = 10000;
            int nDataOutC = 50;
            int nDataOutInternal = 1;
            int nDataOutCInternal = 1;
            int nTimeStepInternal = 1000;
            real tDataOut = veryLargeReal;
            real tEnd = veryLargeReal;

            real CFL = 0.2;
            real dtImplicit = 1e100;
            real rhsThresholdInternal = 1e-10;

            real meshRotZ = 0;
            std::string meshFile = "data/mesh/NACA0012_WIDE_H3.msh";
            std::string outPltName = "data/out/debugData_";
            std::string outLogName = "data/out/debugData_";
            bool uniqueStamps = true;
            real err_dMax = 0.1;

            real res_base = 0;
            bool useVolWiseResidual = false;

            int nDropVisScale;
            real vDropVisScale;

            int curvilinearOneStep = 500;
            int curvilinearRepeatInterval = 500;
            int curvilinearRepeatNum = 10;

            int curvilinearRestartNstep = 100;
            real curvilinearRange = 0.1;

            bool useLocalDt = true;

            bool useLimiter = true;
            int limiterProcedure = 0; // 0 for V2==3WBAP, 1 for V3==CWBAP

            int nPartialLimiterStart = 2;
            int nPartialLimiterStartLocal = 500;
            int nForceLocalStartStep = -1;
            int nCFLRampStart = 1000;
            int nCFLRampLength = 10000;
            real CFLRampEnd = 10;

            int gmresCode = 0; // 0 for lusgs, 1 for gmres, 2 for lusgs started gmres
            int nGmresSpace = 10;
            int nGmresIter = 2;

            int jacobianTypeCode = 0; // 0 for original LUSGS jacobian, 1 for ad roe, 2 for ad roe ad vis

            int nFreezePassiveInner = 0;

            bool steadyQuit = false;

            int odeCode = 0;

            nlohmann::json eulerSettings;
            nlohmann::json vfvSettings;

        } config;

        void ConfigureFromJson(const std::string &jsonName)
        {

            auto fIn = std::ifstream(jsonName);
            DNDS_assert_info(fIn, "config file not existent");
            gSetting = nlohmann::json::parse(fIn, nullptr, true, true);
            nlohmann::json &gS = gSetting;

#define __gs_to_config(name)                                                  \
    {                                                                         \
        try                                                                   \
        {                                                                     \
            config.name = gS[#name].get<decltype(config.name)>();             \
        }                                                                     \
        catch (...)                                                           \
        {                                                                     \
            DNDS_assert_info(false && "config root not given field:", #name); \
        }                                                                     \
    }
            __gs_to_config(nInternalRecStep);
            __gs_to_config(recOrder);
            __gs_to_config(nTimeStep);
            __gs_to_config(nTimeStepInternal);
            __gs_to_config(nSGSIterationInternal);
            __gs_to_config(nConsoleCheck);
            __gs_to_config(nConsoleCheckInternal);
            __gs_to_config(consoleOutputMode);
            __gs_to_config(nDataOutC);
            __gs_to_config(nDataOut);
            __gs_to_config(nDataOutCInternal);
            __gs_to_config(nDataOutInternal);
            __gs_to_config(tDataOut);
            __gs_to_config(tEnd);
            __gs_to_config(CFL);
            __gs_to_config(dtImplicit);
            __gs_to_config(rhsThresholdInternal);
            __gs_to_config(meshRotZ);
            __gs_to_config(meshFile);
            __gs_to_config(outLogName);
            __gs_to_config(outPltName);
            __gs_to_config(uniqueStamps);
            __gs_to_config(err_dMax);
            __gs_to_config(res_base);
            __gs_to_config(useVolWiseResidual);
            __gs_to_config(useLocalDt);
            __gs_to_config(useLimiter);
            __gs_to_config(limiterProcedure);
            __gs_to_config(nPartialLimiterStart);
            __gs_to_config(nPartialLimiterStartLocal);
            __gs_to_config(nForceLocalStartStep);
            __gs_to_config(nCFLRampStart);
            __gs_to_config(nCFLRampLength);
            __gs_to_config(CFLRampEnd);
            __gs_to_config(gmresCode);
            __gs_to_config(nGmresSpace);
            __gs_to_config(nGmresIter);
            __gs_to_config(jacobianTypeCode);
            __gs_to_config(nFreezePassiveInner);
            __gs_to_config(steadyQuit);
            __gs_to_config(odeCode);
            __gs_to_config(eulerSettings);
            __gs_to_config(vfvSettings);
            DNDS_assert(config.eulerSettings.is_object());
            DNDS_assert(config.vfvSettings.is_object());

            //TODO: BC settings

            if (mpi.rank == 0)
                log() << "JSON: Parse Done ===" << std::endl;
#undef __gs_to_config
        }

        void ReadMeshAndInitialize()
        {
            output_stamp = getTimeStamp(mpi);
            if (!config.uniqueStamps)
                output_stamp = "";
            if (mpi.rank == 0)
                log() << "=== Got Time Stamp: [" << output_stamp << "] ===" << std::endl;
            // Debug::MPIDebugHold(mpi);

            int gDimLocal = gDim; //! or else the linker breaks down here (with clang++ or g++, -g -O0,2; c++ non-optimizer bug?)
            DNDS_MAKE_SSP(mesh, mpi, gDimLocal);

            DNDS_MAKE_SSP(vfv, mpi, mesh);
            vfv->settings.jsonSetting = config.vfvSettings;
            vfv->settings.ParseFromJson();

            DNDS_MAKE_SSP(reader, mesh, 0);
            reader->ReadFromCGNSSerial(config.meshFile); // TODO: add bnd mapping here

            reader->BuildCell2Cell();
            reader->MeshPartitionCell2Cell();
            reader->PartitionReorderToMeshCell2Cell();
            reader->BuildSerialOut();
            mesh->BuildGhostPrimary();
            mesh->AdjGlobal2LocalPrimary();
            mesh->InterpolateFace();
            mesh->AssertOnFaces();

            vfv->ConstructMetrics();
            vfv->ConstructBaseAndWeight(
                [&](Geom::t_index id) -> real
                {
                    auto type = BCHandler.GetTypeFromID(id);
                    if (type == BCFar || type == BCSpecial)
                        return 0;
                    return 1;
                });
            vfv->ConstructRecCoeff();

            vfv->BuildUDof(u, nVars);
            vfv->BuildUDof(uInc, nVars);
            vfv->BuildUDof(uIncRHS, nVars);
            vfv->BuildUDof(uTemp, nVars);

            vfv->BuildURec(uRec, nVars);
            vfv->BuildURec(uRecNew, nVars);
            vfv->BuildURec(uRecNew1, nVars);
            vfv->BuildURec(uOld, nVars);
            vfv->BuildScalar(ifUseLimiter);

            //! serial mesh specific output method

            DNDS_MAKE_SSP(outDist, mpi);
            DNDS_MAKE_SSP(outSerial, mpi);
            outDist->Resize(mesh->NumCell(), nOUTS);
            outDist2SerialTrans.setFatherSon(outDist, outSerial);
            DNDS_assert(reader->mode == Geom::MeshReaderMode::SerialOutput);
            outDist2SerialTrans.BorrowGGIndexing(reader->cell2nodeSerialOutTrans);
            outDist2SerialTrans.createMPITypes();
            outDist2SerialTrans.initPersistentPull();
        }

        void RunImplicitEuler()
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
                for (int iRec = 0; iRec < config.nInternalRecStep; iRec++)
                {
                    double tstartA = MPI_Wtime();
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
                    trec += MPI_Wtime() - tstartA;

                    uRec.trans.startPersistentPull();
                    uRec.trans.waitPersistentPull();
                }
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
                    vfv->DoCalculateSmoothIndicator(
                        ifUseLimiter, (uRec), (u),
                        std::array<int, 2>{0, I4});
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
                    PrintData(config.outPltName + "_" + output_stamp + "_" + std::to_string(step) + "_" + std::to_string(iter) + ".plt", ode, eval);
                    nextStepOut += config.nDataOut;
                }
                if (iter % config.nDataOutCInternal == 0)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + "C" + ".plt", ode, eval);
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
                    PrintData(config.outPltName + "_" + output_stamp + "_" + std::to_string(step) + ".plt", ode, eval);
                    nextStepOut += config.nDataOut;
                }
                if (step == nextStepOutC)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + "C" + ".plt", ode, eval);
                    nextStepOutC += config.nDataOutC;
                }
                if (ifOutT)
                {
                    eval.FixUMaxFilter(u);
                    PrintData(config.outPltName + "_" + output_stamp + "_" + "t_" + std::to_string(nextTout) + ".plt", ode, eval);
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
                        Geom::tPoint pos = vfv->cellBary[iCell];
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
                                Geom::tPoint pPhysics = vfv->cellIntPPhysics(iCell, ig);
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

                                inc *= vfv->cellIntJacobiDet(iCell, ig); // don't forget this
                            });
                        if (vfv->cellBary[iCell](0) > xymin && vfv->cellBary[iCell](0) < xymax && vfv->cellBary[iCell](1) > xymin && vfv->cellBary[iCell](1) < xymax)
                        {
                            um /= vfv->volumeLocal[iCell]; // mean value
                            real errRhoMean = u[iCell](0) - um(0);
                            sumErrRho += std::abs(errRhoMean) * vfv->volumeLocal[iCell];
                            sumVol += vfv->volumeLocal[iCell];
                        }
                    }
                    MPI_Allreduce(&sumErrRho, &sumErrRhoSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                    MPI_Allreduce(&sumVol, &sumVolSum, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
                    if (mpi.rank == 0)
                    {
                        log() << "=== Mean Error IV: [" << std::scientific << std::setprecision(5) << sumErrRhoSum << ", " << sumErrRhoSum / sumVolSum << "]" << std::endl;
                    }
                }

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

        template <typename tODE, typename tEval>
        void PrintData(const std::string &fname, tODE &ode, tEval &eval)
        {

            for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
            {
                DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                // TU recu =
                //     vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 1>{0}, 1) *
                //     uRec[iCell];
                // recu += u[iCell];
                // recu = EulerEvaluator::CompressRecPart(u[iCell], recu);
                TU recu = u[iCell];
                TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                real vsqr = velo.squaredNorm();
                real asqr, p, H;
                Gas::IdealGasThermal(recu(I4), recu(0), vsqr, eval.settings.idealGasProperty.gamma, p, asqr, H);
                // DNDS_assert(asqr > 0);
                real M = std::sqrt(vsqr / asqr);
                real T = p / recu(0) / eval.settings.idealGasProperty.Rgas;

                (*outDist)[iCell][0] = recu(0);
                for (int i = 0; i < dim; i++)
                    (*outDist)[iCell][i + 1] = velo(i);
                (*outDist)[iCell][I4 + 0] = p;
                (*outDist)[iCell][I4 + 1] = T;
                (*outDist)[iCell][I4 + 2] = M;
                // (*outDist)[iCell][7] = (bool)(ifUseLimiter[iCell] & 0x0000000FU);
                (*outDist)[iCell][I4 + 3] = ifUseLimiter[iCell][0] / (vfv->settings.smoothThreshold + verySmallReal);
                // std::cout << iCell << ode.rhsbuf[0][iCell] << std::endl;
                (*outDist)[iCell][I4 + 4] = ode->getLatestRHS()[iCell][0];
                // (*outDist)[iCell][8] = (*vfv->SOR_iCell2iScan)[iCell];//!using SOR rb seq instead

                for (int i = I4 + 1; i < nVars; i++)
                {
                    (*outDist)[iCell][4 + i] = recu(i) / recu(0); // 4 is additional amount offset, not Index of last flow variable (I4)
                }
            }
            outDist2SerialTrans.startPersistentPull();
            outDist2SerialTrans.waitPersistentPull();
            std::vector<std::string> names;
            if constexpr (dim == 2)
                names = {
                    "R", "U", "V", "P", "T", "M", "ifUseLimiter", "RHSr"};
            else
                names = {
                    "R", "U", "V", "W", "P", "T", "M", "ifUseLimiter", "RHSr"};
            for (int i = I4 + 1; i < nVars; i++)
            {
                names.push_back("V" + std::to_string(i - I4));
            }
            reader->PrintSerialPartPltBinaryDataArray(
                fname, nOUTS, //! oprank = 0
                [&](int idata)
                { return names[idata]; },
                [&](int idata, index iv)
                {
                    return (*outSerial)[iv][idata];
                },
                0,
                0); // todo: change this flag for dist output
        }
    };

}