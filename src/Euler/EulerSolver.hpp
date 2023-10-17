#pragma once
#include "Gas.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "Solver/ODE.hpp"
#include "Solver/Linear.hpp"
#include "EulerEvaluator.hpp"
#include "DNDS/JsonUtil.hpp"

#include <iomanip>
#include <functional>

#define JSON_ASSERT DNDS_assert
#include "json.hpp"
#include "EulerBC.hpp"

#include "DNDS/SerializerJSON.hpp"
#include <filesystem>

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
        typedef ssp<CFV::VariationalReconstruction<gDim>> TVFV;

    private:
        MPIInfo mpi;
        ssp<Geom::UnstructuredMesh> mesh, meshBnd;
        TVFV vfv; // ! gDim -> 3 for intellisense
        ssp<Geom::UnstructuredMeshSerialRW> reader, readerBnd;

        ArrayDOFV<nVars_Fixed> u, uInc, uIncRHS, uTemp, rhsTemp;
        ArrayRECV<nVars_Fixed> uRec, uRecNew, uRecNew1, uRecOld, uRec1, uRecInc, uRecInc1;
        ArrayDOFV<nVars_Fixed> JD, JD1, JSource, JSource1;
        ArrayDOFV<1> alphaPP, alphaPP1, betaPP, betaPP1;

        int nOUTS = {-1};
        int nOUTSPoint{-1};
        int nOUTSBnd{-1};
        // rho u v w p T M ifUseLimiter RHS
        ssp<ArrayEigenVector<Eigen::Dynamic>> outDist;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerial;
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTrans;

        ssp<ArrayEigenVector<Eigen::Dynamic>> outDistPoint;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outGhostPoint;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerialPoint;
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTransPoint;
        ArrayPair<ArrayEigenVector<Eigen::Dynamic>> outDistPointPair;

        ssp<ArrayEigenVector<Eigen::Dynamic>> outDistBnd;
        // ssp<ArrayEigenVector<Eigen::Dynamic>> outGhostBnd;
        ssp<ArrayEigenVector<Eigen::Dynamic>> outSerialBnd;
        ArrayTransformerType<ArrayEigenVector<Eigen::Dynamic>>::Type outDist2SerialTransBnd;
        // ArrayPair<ArrayEigenVector<Eigen::Dynamic>> outDistBndPair;

        // std::vector<uint32_t> ifUseLimiter;
        CFV::tScalarPair ifUseLimiter;

        BoundaryHandler<model>
            BCHandler;

    public:
        nlohmann::ordered_json gSetting;
        std::string output_stamp = "";

        struct Configuration
        {

            struct TimeMarchControl
            {
                real dtImplicit = 1e100;
                int nTimeStep = 1000000;
                bool steadyQuit = false;
                bool useRestart = false;
                bool useImplicitPP = false;
                int odeCode = 0;
                real tEnd = veryLargeReal;
                real odeSetting1 = 0;
                real odeSetting2 = 0;
                real odeSetting3 = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    TimeMarchControl,
                    dtImplicit, nTimeStep,
                    steadyQuit, useRestart, useImplicitPP,
                    odeCode, tEnd, odeSetting1, odeSetting2, odeSetting3)
            } timeMarchControl;

            struct ImplicitReconstructionControl
            {

                int nInternalRecStep = 1;
                bool zeroGrads = false;
                int recLinearScheme = 0; // 0 for original SOR, 1 for GMRES
                int nGmresSpace = 5;
                int nGmresIter = 10;
                real recThreshold = 1e-5;
                int nRecConsolCheck = 1;
                int nRecMultiplyForZeroedGrad = 1;
                bool storeRecInc = false;
                bool dampRecIncDTau = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ImplicitReconstructionControl,
                    nInternalRecStep, zeroGrads,
                    recLinearScheme, nGmresSpace, nGmresIter,
                    recThreshold, nRecConsolCheck,
                    nRecMultiplyForZeroedGrad,
                    storeRecInc, dampRecIncDTau)
            } implicitReconstructionControl;

            struct OutputControl
            {
                int nConsoleCheck = 1;
                int nConsoleCheckInternal = 1;
                int consoleOutputMode = 0; // 0 for basic, 1 for wall force out
                int nDataOut = 10000;
                int nDataOutC = 50;
                int nDataOutInternal = 10000;
                int nDataOutCInternal = 1;
                int nRestartOut = 10000;
                int nRestartOutC = 50;
                int nRestartOutInternal = 10000;
                int nRestartOutCInternal = 1;
                real tDataOut = veryLargeReal;

                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    OutputControl,
                    nConsoleCheck,
                    nConsoleCheckInternal,
                    consoleOutputMode,
                    nDataOut,
                    nDataOutC,
                    nDataOutInternal,
                    nDataOutCInternal,
                    nRestartOut,
                    nRestartOutC,
                    nRestartOutInternal,
                    nRestartOutCInternal,
                    tDataOut)
            } outputControl;

            struct ImplicitCFLControl
            {
                real CFL = 10;
                int nForceLocalStartStep = INT_MAX;
                int nCFLRampStart = INT_MAX;
                int nCFLRampLength = INT_MAX;
                real CFLRampEnd = 0;
                bool useLocalDt = true;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ImplicitCFLControl,
                    CFL,
                    nForceLocalStartStep,
                    nCFLRampStart,
                    nCFLRampLength,
                    CFLRampEnd,
                    useLocalDt)
            } implicitCFLControl;

            struct ConvergenceControl
            {
                int nTimeStepInternal = 20;
                real rhsThresholdInternal = 1e-10;
                real res_base = 0;
                bool useVolWiseResidual = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ConvergenceControl,
                    nTimeStepInternal,
                    rhsThresholdInternal,
                    res_base,
                    useVolWiseResidual)
            } convergenceControl;

            struct DataIOControl
            {
                bool uniqueStamps = true;
                real meshRotZ = 0;
                real meshScale = 1.0;
                std::string meshFile = "data/mesh/NACA0012_WIDE_H3.cgns";
                std::string outPltName = "data/out/debugData_";
                std::string outLogName = "data/out/debugData_";
                std::string outRestartName = "data/out/debugData_";

                int outPltMode = 0;   // 0 = serial, 1 = dist plt
                int readMeshMode = 0; // 0 = serial cgns, 1 = dist json
                bool outPltTecplotFormat = true;
                bool outPltVTKFormat = true;
                bool outAtPointData = true;
                bool outAtCellData = true;
                int nASCIIPrecision = 5;
                bool outVolumeData = true;
                bool outBndData = false;

                bool serializerSaveURec = false;

                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    DataIOControl,
                    uniqueStamps,
                    meshRotZ,
                    meshScale,
                    meshFile,
                    outPltName,
                    outLogName,
                    outRestartName,
                    outPltMode,
                    readMeshMode,
                    outPltTecplotFormat,
                    outPltVTKFormat,
                    outAtPointData,
                    outAtCellData,
                    nASCIIPrecision,
                    outVolumeData,
                    outBndData,
                    serializerSaveURec)
            } dataIOControl;

            struct BoundaryDefinition
            {
                Eigen::Vector<real, -1> PeriodicTranslation1;
                Eigen::Vector<real, -1> PeriodicTranslation2;
                Eigen::Vector<real, -1> PeriodicTranslation3;
                BoundaryDefinition()
                {
                    PeriodicTranslation1.resize(3);
                    PeriodicTranslation2.resize(3);
                    PeriodicTranslation3.resize(3);
                    PeriodicTranslation1 << 1, 0, 0;
                    PeriodicTranslation2 << 0, 1, 0;
                    PeriodicTranslation3 << 0, 0, 1;
                }

                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    BoundaryDefinition,
                    PeriodicTranslation1,
                    PeriodicTranslation2,
                    PeriodicTranslation3)
            } boundaryDefinition;

            struct LimiterControl
            {
                bool useLimiter = true;
                bool usePPRecLimiter = true;
                int smoothIndicatorProcedure = 0;
                int limiterProcedure = 0; // 0 for V2==3WBAP, 1 for V3==CWBAP
                int nPartialLimiterStart = 0;
                int nPartialLimiterStartLocal = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    LimiterControl,
                    useLimiter, usePPRecLimiter,
                    smoothIndicatorProcedure, limiterProcedure,
                    nPartialLimiterStart, nPartialLimiterStartLocal)
            } limiterControl;

            struct LinearSolverControl
            {
                int sgsIter = 0;
                int sgsWithRec = 0;
                int gmresCode = 0; // 0 for lusgs, 1 for gmres, 2 for lusgs started gmres
                int nGmresSpace = 10;
                int nGmresIter = 2;
                int nSgsConsoleCheck = 100;
                int nGmresConsoleCheck = 100;
                bool initWithLastURecInc = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    LinearSolverControl,
                    sgsIter, sgsWithRec, gmresCode,
                    nGmresSpace, nGmresIter, initWithLastURecInc)
            } linearSolverControl;

            struct RestartState
            {
                int iStep = -1;
                int iStepInternal = -1;
                int odeCodePrev = -1;
                std::string lastRestartFile = "";
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    RestartState,
                    iStep, iStepInternal, odeCodePrev,
                    lastRestartFile)
            } restartState;

            struct Others
            {
                int nFreezePassiveInner = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    Others,
                    nFreezePassiveInner)
            } others;

            nlohmann::ordered_json eulerSettings = nlohmann::ordered_json::object();
            nlohmann::ordered_json vfvSettings = nlohmann::ordered_json::object();

            void ReadWriteJson(nlohmann::ordered_json &jsonObj, int nVars, bool read)
            {

                __DNDS__json_to_config(timeMarchControl);
                __DNDS__json_to_config(implicitReconstructionControl);
                __DNDS__json_to_config(outputControl);
                __DNDS__json_to_config(implicitCFLControl);
                __DNDS__json_to_config(convergenceControl);
                __DNDS__json_to_config(dataIOControl);
                __DNDS__json_to_config(boundaryDefinition);
                __DNDS__json_to_config(limiterControl);
                __DNDS__json_to_config(linearSolverControl);
                __DNDS__json_to_config(others);
                __DNDS__json_to_config(restartState);

                __DNDS__json_to_config(eulerSettings);
                __DNDS__json_to_config(vfvSettings);
                if (read)
                {
                    DNDS_assert(eulerSettings.is_object());
                    DNDS_assert(vfvSettings.is_object());
                }

                // TODO: BC settings
            }

            Configuration()
            {
            }

            Configuration(int nVars)
            {
                vfvSettings = CFV::VRSettings{gDim};
                typename TEval::Setting().ReadWriteJSON(eulerSettings, nVars, false);
            }

        } config = Configuration{};

    public:
        EulerSolver(const MPIInfo &nmpi) : nVars(getNVars(model)), mpi(nmpi)
        {
            nOUTS = nVars + 4;
            nOUTSPoint = nVars + 2;
            nOUTSBnd = nVars * 2 + 1 + 2 + 3;

            config = Configuration(nVars); //* important to initialize using nVars
        }

        /**
         */
        void ConfigureFromJson(const std::string &jsonName, bool read = false, const std::string &jsonMergeName = "")
        {
            if (read)
            {
                auto fIn = std::ifstream(jsonName);
                DNDS_assert_info(fIn, "config file not existent");
                gSetting = nlohmann::ordered_json::parse(fIn, nullptr, true, true);

                if (read && !jsonMergeName.empty())
                {
                    fIn = std::ifstream(jsonMergeName);
                    DNDS_assert_info(fIn, "config file patch not existent");
                    auto gSettingAdd = nlohmann::ordered_json::parse(fIn, nullptr, true, true);
                    gSetting.merge_patch(gSettingAdd);
                }
                config.ReadWriteJson(gSetting, nVars, read);
                PrintConfig();
                if (mpi.rank == 0)
                    log() << "JSON: read value:" << std::endl
                          << std::setw(4) << gSetting << std::endl;
            }
            else
            {
                gSetting = nlohmann::ordered_json::object();
                config.ReadWriteJson(gSetting, nVars, read);
                if (mpi.rank == 0) // single call for output
                {
                    auto fIn = std::ofstream(jsonName);
                    fIn << std::setw(4) << gSetting;
                }
                MPI_Barrier(mpi.comm); // no go until output done
            }

            if (mpi.rank == 0)
                log() << "JSON: Parse " << (read ? "read" : "write")
                      << " Done ===" << std::endl;
#undef __DNDS__json_to_config
        }

        void ReadMeshAndInitialize()
        {
            output_stamp = getTimeStamp(mpi);
            if (!config.dataIOControl.uniqueStamps)
                output_stamp = "";
            if (mpi.rank == 0)
                log() << "=== Got Time Stamp: [" << output_stamp << "] ===" << std::endl;
            // Debug::MPIDebugHold(mpi);

            int gDimLocal = gDim; //! or else the linker breaks down here (with clang++ or g++, -g -O0,2; c++ non-optimizer bug?)
            DNDS_MAKE_SSP(mesh, mpi, gDimLocal);
            DNDS_MAKE_SSP(meshBnd, mpi, gDimLocal - 1);

            DNDS_MAKE_SSP(vfv, mpi, mesh);
            vfv->settings.ParseFromJson(config.vfvSettings);

            DNDS_MAKE_SSP(reader, mesh, 0);
            DNDS_MAKE_SSP(readerBnd, meshBnd, 0);
            DNDS_assert(config.dataIOControl.readMeshMode == 0 || config.dataIOControl.readMeshMode == 1);
            DNDS_assert(config.dataIOControl.outPltMode == 0 || config.dataIOControl.outPltMode == 1);
            mesh->periodicInfo.translation[1] = config.boundaryDefinition.PeriodicTranslation1;
            mesh->periodicInfo.translation[2] = config.boundaryDefinition.PeriodicTranslation2;
            mesh->periodicInfo.translation[3] = config.boundaryDefinition.PeriodicTranslation3;

            if (config.dataIOControl.readMeshMode == 0)
            {
                reader->ReadFromCGNSSerial(config.dataIOControl.meshFile); // TODO: add bnd mapping here
                reader->Deduplicate1to1Periodic();
                reader->BuildCell2Cell();
                reader->MeshPartitionCell2Cell();
                reader->PartitionReorderToMeshCell2Cell();
                if (config.dataIOControl.outPltMode == 0)
                {
                    reader->BuildSerialOut();
                }
                mesh->BuildGhostPrimary();
                mesh->AdjGlobal2LocalPrimary();
            }
            else
            {
                std::filesystem::path meshPath{config.dataIOControl.meshFile};
                auto meshOutName = std::string(config.dataIOControl.meshFile) + "_part_" + std::to_string(mpi.size) + ".dir";
                std::filesystem::path meshOutDir{meshOutName};
                // std::filesystem::create_directories(meshOutDir); // reading not writing
                std::string meshPartPath = getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

                SerializerJSON serializerJSON;
                serializerJSON.SetUseCodecOnUint8(true);
                SerializerBase *serializer = &serializerJSON;
                serializer->OpenFile(meshPartPath, true);
                mesh->ReadSerialize(serializer, "meshPart");
                serializer->CloseFile();
                if (config.dataIOControl.outPltMode == 0)
                {
                    mesh->AdjLocal2GlobalPrimary();
                    reader->BuildSerialOut();
                    mesh->AdjGlobal2LocalPrimary();
                }
            }
            // std::cout << "here" << std::endl;
            mesh->InterpolateFace();
            mesh->AssertOnFaces();
#ifdef DNDS_USE_OMP
            omp_set_num_threads(DNDS::MPIWorldSize() == 1 ? std::min(omp_get_num_procs(), omp_get_max_threads()) : 1);
#endif
            mesh->ConstructBndMesh(*meshBnd);
            if (config.dataIOControl.outPltMode == 0)
            {
                meshBnd->AdjLocal2GlobalPrimaryForBnd();
                readerBnd->BuildSerialOut();
                meshBnd->AdjGlobal2LocalPrimaryForBnd();
            }

            if (config.dataIOControl.meshRotZ != 0.0)
            {
                real rz = config.dataIOControl.meshRotZ / 180.0 * pi;
                Eigen::Matrix3d Rz{
                    {std::cos(rz), -std::sin(rz), 0},
                    {std::sin(rz), std::cos(rz), 0},
                    {0, 0, 1},
                };
                mesh->TransformCoords(
                    [&](const Geom::tPoint &p)
                    { return Rz * p; });

                for (auto &r : mesh->periodicInfo.rotation)
                    r = Rz * r * Rz.transpose();
                for (auto &p : mesh->periodicInfo.rotationCenter)
                    p = Rz * p;
                for (auto &p : mesh->periodicInfo.translation)
                    p = Rz * p;
                // @todo  //! todo: alter the rotation and translation in  periodicInfo mesh->periodicInfo
            }
            if (config.dataIOControl.meshScale != 1.0)
            {
                auto scale = config.dataIOControl.meshScale;
                mesh->TransformCoords(
                    [&](const Geom::tPoint &p)
                    { return p * scale; });
                for (auto &i : mesh->periodicInfo.translation)
                    i *= scale;
                for (auto &i : mesh->periodicInfo.rotationCenter)
                    i *= scale;
            }
            /// @todo //todo: upgrade to optional
            if (config.dataIOControl.outPltMode == 0)
                reader->coordSerialOutTrans.pullOnce();
            vfv->ConstructMetrics();
            vfv->ConstructBaseAndWeight(
                [&](Geom::t_index id) -> real
                {
                    auto type = BCHandler.GetTypeFromID(id);
                    if (type == BCFar || type == BCSpecial)
                        return 0; // far weight
                    return 1;     // wall weight
                });
            vfv->ConstructRecCoeff();

            vfv->BuildUDof(u, nVars);
            vfv->BuildUDof(uInc, nVars);
            vfv->BuildUDof(uIncRHS, nVars);
            vfv->BuildUDof(uTemp, nVars);
            vfv->BuildUDof(rhsTemp, nVars);

            vfv->BuildURec(uRec, nVars);
            if (config.timeMarchControl.odeCode == 401)
                vfv->BuildURec(uRec1, nVars);
            vfv->BuildURec(uRecNew, nVars);
            vfv->BuildURec(uRecNew1, nVars);
            vfv->BuildURec(uRecOld, nVars);
            vfv->BuildScalar(ifUseLimiter);
            vfv->BuildUDof(betaPP, 1);
            vfv->BuildUDof(alphaPP, 1);
            betaPP.setConstant(1.0);
            alphaPP.setConstant(1.0);
            if (config.timeMarchControl.odeCode == 401)
            {
                vfv->BuildUDof(betaPP1, 1);
                vfv->BuildUDof(alphaPP1, 1);
                betaPP1.setConstant(1.0);
                alphaPP1.setConstant(1.0);
            }
            if (config.implicitReconstructionControl.storeRecInc)
            {
                vfv->BuildURec(uRecInc, nVars);
                if (config.timeMarchControl.odeCode == 401)
                    vfv->BuildURec(uRecInc1, nVars);
            }

            vfv->BuildUDof(JD, nVars);
            vfv->BuildUDof(JSource, nVars);
            if (config.timeMarchControl.odeCode == 401)
                vfv->BuildUDof(JD1, nVars), vfv->BuildUDof(JSource1, nVars);

            DNDS_assert(config.dataIOControl.outAtCellData || config.dataIOControl.outAtPointData);
            DNDS_assert(config.dataIOControl.outPltVTKFormat || config.dataIOControl.outPltTecplotFormat);
            DNDS_MAKE_SSP(outDistBnd, mpi);
            outDistBnd->Resize(mesh->NumBnd(), nOUTSBnd);

            if (config.dataIOControl.outAtCellData)
            {
                DNDS_MAKE_SSP(outDist, mpi);
                outDist->Resize(mesh->NumCell(), nOUTS);
            }
            if (config.dataIOControl.outAtPointData)
            {
                DNDS_MAKE_SSP(outDistPoint, mpi);
                outDistPoint->Resize(mesh->NumNode(), nOUTSPoint);
                DNDS_assert(nOUTSPoint >= nVars);

                outDistPointPair.father = outDistPoint;
                DNDS_MAKE_SSP(outDistPointPair.son, mpi);
                outDistPointPair.TransAttach();
                outDistPointPair.trans.BorrowGGIndexing(mesh->coords.trans);
                outDistPointPair.trans.createMPITypes();
                outDistPointPair.trans.initPersistentPull();
            }

            if (config.dataIOControl.outPltMode == 0)
            {
                //! serial mesh specific output method
                DNDS_MAKE_SSP(outSerialBnd, mpi);
                outDist2SerialTransBnd.setFatherSon(outDistBnd, outSerialBnd);
                DNDS_assert(readerBnd->mode == Geom::MeshReaderMode::SerialOutput);
                outDist2SerialTransBnd.BorrowGGIndexing(readerBnd->cell2nodeSerialOutTrans);
                outDist2SerialTransBnd.createMPITypes();
                outDist2SerialTransBnd.initPersistentPull();

                if (config.dataIOControl.outAtCellData)
                {
                    DNDS_MAKE_SSP(outSerial, mpi);
                    outDist2SerialTrans.setFatherSon(outDist, outSerial);
                    DNDS_assert(reader->mode == Geom::MeshReaderMode::SerialOutput);
                    outDist2SerialTrans.BorrowGGIndexing(reader->cell2nodeSerialOutTrans);
                    outDist2SerialTrans.createMPITypes();
                    outDist2SerialTrans.initPersistentPull();
                }
                if (config.dataIOControl.outAtPointData)
                {
                    DNDS_MAKE_SSP(outSerialPoint, mpi);
                    outDist2SerialTransPoint.setFatherSon(outDistPoint, outSerialPoint);
                    DNDS_assert(reader->mode == Geom::MeshReaderMode::SerialOutput);
                    outDist2SerialTransPoint.BorrowGGIndexing(reader->coordSerialOutTrans);
                    outDist2SerialTransPoint.createMPITypes();
                    outDist2SerialTransPoint.initPersistentPull();
                }
            }
        }

        void RunImplicitEuler();

        void PrintConfig()
        {
            config.ReadWriteJson(gSetting, nVars, false);
            if (mpi.rank == 0)
            {
                std::ofstream logConfig(config.dataIOControl.outLogName + "_" + output_stamp + ".config.json");
                gSetting["___Compile_Time_Defines"] = DNDS_Defines_state;
                gSetting["___Runtime_PartitionNumber"] = mpi.size;
                logConfig << std::setw(4) << gSetting;
                logConfig.close();
            }
        }
        void ReadRestart(std::string fname)
        {
            std::filesystem::path outPath;
            // outPath = {fname + "_p" + std::to_string(mpi.size) + "_restart.dir"};
            outPath = fname;
            // std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%04d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));

            SerializerJSON serializerJSON;
            serializerJSON.SetUseCodecOnUint8(true);
            SerializerBase *serializer = &serializerJSON;
            serializer->OpenFile(fname, true);
            u.ReadSerialize(serializer, "u");
            serializer->CloseFile();
            // config.restartState.lastRestartFile = outPath;
            // PrintConfig();
        }

        void PrintRestart(std::string fname)
        {
            std::filesystem::path outPath;
            outPath = {fname + "_p" + std::to_string(mpi.size) + "_restart.dir"};
            std::filesystem::create_directories(outPath);
            char BUF[512];
            std::sprintf(BUF, "%04d", mpi.rank);
            fname = getStringForcePath(outPath / (std::string(BUF) + ".json"));

            SerializerJSON serializerJSON;
            serializerJSON.SetUseCodecOnUint8(true);
            SerializerBase *serializer = &serializerJSON;
            serializer->OpenFile(fname, false);
            u.WriteSerialize(serializer, "u");
            serializer->CloseFile();
            config.restartState.lastRestartFile = outPath;
            PrintConfig();
        }

        template <typename tODE, typename tEval>
        void PrintData(const std::string &fname, tODE &ode, tEval &eval)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            reader->SetASCIIPrecision(config.dataIOControl.nASCIIPrecision);
            const int cDim = dim;

            if (config.dataIOControl.outVolumeData)
            {
                if (config.dataIOControl.outAtCellData)
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    {
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
                        real M = std::sqrt(std::abs(vsqr / asqr));
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
                        // { // see the cond
                        //     auto A = vfv->GetCellRecMatA(iCell);
                        //     Eigen::MatrixXd AInv = A;
                        //     real aCond = HardEigen::EigenLeastSquareInverse(A, AInv);
                        //     (*outDist)[iCell][I4 + 4] = aCond;
                        // }
                        // (*outDist)[iCell][8] = (*vfv->SOR_iCell2iScan)[iCell];//!using SOR rb seq instead

                        for (int i = I4 + 1; i < nVars; i++)
                        {
                            (*outDist)[iCell][4 + i] = recu(i) / recu(0); // 4 is additional amount offset, not Index of last flow variable (I4)
                        }
                    }

                if (config.dataIOControl.outAtPointData)
                {
                    if (config.limiterControl.useLimiter)
                    {
                        uRecNew.trans.startPersistentPull();
                        uRecNew.trans.waitPersistentPull();
                    }
                    else
                    {
                        uRec.trans.startPersistentPull();
                        uRec.trans.waitPersistentPull();
                    }

                    u.trans.startPersistentPull();
                    u.trans.waitPersistentPull();

                    for (index iN = 0; iN < mesh->NumNodeProc(); iN++)
                        outDistPointPair[iN].setZero();
                    std::vector<int> nN2C(mesh->NumNodeProc(), 0);
                    DNDS_assert(outDistPointPair.father->Size() == mesh->NumNode());
                    DNDS_assert(outDistPointPair.son->Size() == mesh->NumNodeGhost());
                    for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++) //! all cells
                    {
                        for (int ic2n = 0; ic2n < mesh->cell2node.RowSize(iCell); ic2n++)
                        {
                            auto iNode = mesh->cell2node(iCell, ic2n);
                            nN2C.at(iNode)++;
                            auto pPhy = mesh->GetCoordNodeOnCell(iCell, ic2n);

                            Eigen::Matrix<real, 1, Eigen::Dynamic> DiBj;
                            DiBj.resize(1, uRecNew[iCell].rows() + 1);
                            // std::cout << uRecNew[iCell].rows() << std::endl;
                            vfv->FDiffBaseValue(DiBj, pPhy, iCell, -2, -2);

                            TU vRec = (DiBj(Eigen::all, Eigen::seq(1, Eigen::last)) * (config.limiterControl.useLimiter ? uRecNew[iCell] : uRec[iCell])).transpose() + u[iCell];
                            if (iNode < mesh->NumNode())
                                outDistPointPair[iNode](Eigen::seq(0, nVars - 1)) += vRec;
                        }
                    }

                    for (index iN = 0; iN < mesh->NumNode(); iN++)
                    {
                        TU recu = outDistPointPair[iN](Eigen::seq(0, nVars - 1)) / (nN2C.at(iN) + verySmallReal);
                        DNDS_assert(nN2C.at(iN) > 0);

                        TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                        real vsqr = velo.squaredNorm();
                        real asqr, p, H;
                        Gas::IdealGasThermal(recu(I4), recu(0), vsqr, eval.settings.idealGasProperty.gamma, p, asqr, H);
                        // DNDS_assert(asqr > 0);
                        real M = std::sqrt(std::abs(vsqr / asqr));
                        real T = p / recu(0) / eval.settings.idealGasProperty.Rgas;

                        outDistPointPair[iN][0] = recu(0);
                        for (int i = 0; i < dim; i++)
                            outDistPointPair[iN][i + 1] = velo(i);
                        outDistPointPair[iN][I4 + 0] = p;
                        outDistPointPair[iN][I4 + 1] = T;
                        outDistPointPair[iN][I4 + 2] = M;

                        for (int i = I4 + 1; i < nVars; i++)
                        {
                            outDistPointPair[iN][2 + i] = recu(i) / recu(0); // 2 is additional amount offset
                        }
                    }
                    outDistPointPair.trans.startPersistentPull();
                    outDistPointPair.trans.waitPersistentPull();
                }

                int NOUTS_C{0}, NOUTSPoint_C{0};
                if (config.dataIOControl.outAtCellData)
                    NOUTS_C = nOUTS;
                if (config.dataIOControl.outAtPointData)
                    NOUTSPoint_C = nOUTSPoint;

                if (config.dataIOControl.outPltMode == 0)
                {
                    if (config.dataIOControl.outAtCellData)
                    {
                        outDist2SerialTrans.startPersistentPull();
                        outDist2SerialTrans.waitPersistentPull();
                    }
                    if (config.dataIOControl.outAtPointData)
                    {
                        outDist2SerialTransPoint.startPersistentPull();
                        outDist2SerialTransPoint.waitPersistentPull();
                    }
                }

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

                if (config.dataIOControl.outPltTecplotFormat)
                {
                    if (config.dataIOControl.outPltMode == 0)
                    {
                        reader->PrintSerialPartPltBinaryDataArray(
                            fname,
                            NOUTS_C, NOUTSPoint_C,
                            [&](int idata)
                            { return names[idata]; }, // cellNames
                            [&](int idata, index iv)
                            {
                                return (*outSerial)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            { return names[idata] + "_p"; }, // pointNames
                            [&](int idata, index in)
                            { return (*outSerialPoint)[in][idata]; }, // pointData
                            0.0,
                            0);
                    }
                    else if (config.dataIOControl.outPltMode == 1)
                    {

                        reader->PrintSerialPartPltBinaryDataArray(
                            fname,
                            NOUTS_C, NOUTSPoint_C,
                            [&](int idata)
                            { return names[idata]; }, // cellNames
                            [&](int idata, index iv)
                            {
                                return (*outDist)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            { return names[idata < cDim + 4 ? idata : idata + 2] + "_p"; }, // pointNames
                            [&](int idata, index in)
                            { return outDistPointPair[in][idata]; }, // pointData
                            0.0,
                            1);
                    }
                }

                if (config.dataIOControl.outPltVTKFormat)
                {
                    if (config.dataIOControl.outPltMode == 0)
                    {
                        reader->PrintSerialPartVTKDataArray(
                            fname,
                            std::max(NOUTS_C - cDim, 0), std::min(NOUTS_C, 1),
                            std::max(NOUTSPoint_C - cDim, 0), std::min(NOUTSPoint_C, 1), //! vectors number is not cDim but 1
                            [&](int idata)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return names[idata]; // cellNames
                            },
                            [&](int idata, index iv)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return (*outSerial)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            {
                                return "Velo"; // cellVecNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                return (*outSerial)[iv][1 + idim]; // cellVecData
                            },
                            [&](int idata)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                if (idata >= 4)
                                    idata += 2;
                                return names[idata]; // pointNames
                            },
                            [&](int idata, index iv)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return (*outSerialPoint)[iv][idata]; // pointData
                            },
                            [&](int idata)
                            {
                                return "Velo"; // pointVecNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                idata += 1;
                                return (*outSerialPoint)[iv][1 + idim]; // pointVecData
                            },
                            0.0,
                            0);
                    }
                    else if (config.dataIOControl.outPltMode == 1)
                    {
                        reader->PrintSerialPartVTKDataArray(
                            fname,
                            std::max(NOUTS_C - cDim, 0), std::min(NOUTS_C, 1),
                            std::max(NOUTSPoint_C - cDim, 0), std::min(NOUTSPoint_C, 1), //! vectors number is not cDim but 1
                            [&](int idata)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return names[idata]; // cellNames
                            },
                            [&](int idata, index iv)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return (*outDist)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            {
                                return "Velo"; // cellVecNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                return idim < cDim ? (*outDist)[iv][1 + idim] : 0.0; // cellVecData
                            },
                            [&](int idata)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return names[idata]; // pointNames
                            },
                            [&](int idata, index iv)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return outDistPointPair[iv][idata]; // pointData
                            },
                            [&](int idata)
                            {
                                return "Velo"; // pointVecNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                return idim < cDim ? outDistPointPair[iv][1 + idim] : 0.0; // pointVecData
                            },
                            0.0,
                            1);
                    }
                }
            }

            if (config.dataIOControl.outBndData)
            {
                for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
                {
                    // TU recu =
                    //     vfv->GetIntPointDiffBaseValue(iCell, -1, -1, -1, std::array<int, 1>{0}, 1) *
                    //     uRec[iCell];
                    // recu += u[iCell];
                    // recu = EulerEvaluator::CompressRecPart(u[iCell], recu);
                    index iCell = mesh->bnd2cell[iBnd][0];
                    index iFace = mesh->bnd2face[iBnd];

                    TU recu = u[iCell];
                    TVec velo = (recu(Seq123).array() / recu(0)).matrix();
                    real vsqr = velo.squaredNorm();
                    real asqr, p, H;
                    Gas::IdealGasThermal(recu(I4), recu(0), vsqr, eval.settings.idealGasProperty.gamma, p, asqr, H);
                    // DNDS_assert(asqr > 0);
                    real M = std::sqrt(std::abs(vsqr / asqr));
                    real T = p / recu(0) / eval.settings.idealGasProperty.Rgas;

                    (*outDistBnd)[iBnd][0] = recu(0);
                    for (int i = 0; i < dim; i++)
                        (*outDistBnd)[iBnd][i + 1] = velo(i);
                    (*outDistBnd)[iBnd][I4 + 0] = p;
                    (*outDistBnd)[iBnd][I4 + 1] = T;
                    (*outDistBnd)[iBnd][I4 + 2] = M;
                    for (int i = I4 + 1; i < nVars; i++)
                    {
                        (*outDistBnd)[iBnd][2 + i] = recu(i) / recu(0); // 4 is additional amount offset, not Index of last flow variable (I4)
                    }

                    (*outDistBnd)[iBnd](Eigen::seq(nVars + 2, nOUTSBnd - 5)) = eval.fluxBnd.at(iBnd);
                    (*outDistBnd)[iBnd](nOUTSBnd - 4) = mesh->GetFaceZone(iFace);
                    (*outDistBnd)[iBnd](Eigen::seq(nOUTSBnd - 3, nOUTSBnd - 1)) = vfv->GetFaceNorm(iFace, 0) * vfv->GetFaceArea(iFace);

                    // (*outDist)[iCell][8] = (*vfv->SOR_iCell2iScan)[iCell];//!using SOR rb seq instead
                }

                int NOUTS_C{0}, NOUTSPoint_C{0};
                NOUTS_C = nOUTSBnd;

                if (config.dataIOControl.outPltMode == 0)
                {
                    outDist2SerialTransBnd.startPersistentPull();
                    outDist2SerialTransBnd.waitPersistentPull();
                }

                std::vector<std::string> names;
                if constexpr (dim == 2)
                    names = {
                        "R", "U", "V", "P", "T", "M"};
                else
                    names = {
                        "R", "U", "V", "W", "P", "T", "M"};
                for (int i = I4 + 1; i < nVars; i++)
                {
                    names.push_back("V" + std::to_string(i - I4));
                }
                for (int i = 0; i < nVars; i++)
                {
                    names.push_back("F" + std::to_string(i));
                }
                names.push_back("FaceZone");
                names.push_back("N0");
                names.push_back("N1");
                names.push_back("N2");

                if (config.dataIOControl.outPltTecplotFormat)
                {
                    if (config.dataIOControl.outPltMode == 0)
                    {
                        readerBnd->PrintSerialPartPltBinaryDataArray(
                            fname + "_bnd",
                            NOUTS_C, 0,
                            [&](int idata)
                            { return names[idata]; }, // cellNames
                            [&](int idata, index iv)
                            {
                                return (*outSerialBnd)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            { return "ERROR"; }, // pointNames
                            [&](int idata, index in)
                            { return std::nan("0"); }, // pointData
                            0.0,
                            0);
                    }
                    else if (config.dataIOControl.outPltMode == 1)
                    {

                        readerBnd->PrintSerialPartPltBinaryDataArray(
                            fname + "_bnd",
                            NOUTS_C, NOUTSPoint_C,
                            [&](int idata)
                            { return names[idata]; }, // cellNames
                            [&](int idata, index iv)
                            {
                                return (*outDistBnd)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            { return "ERROR"; }, // pointNames
                            [&](int idata, index in)
                            { return std::nan("0"); }, // pointData
                            0.0,
                            1);
                    }
                }

                const int cDim = dim;
                if (config.dataIOControl.outPltVTKFormat)
                {
                    if (config.dataIOControl.outPltMode == 0)
                    {
                        readerBnd->PrintSerialPartVTKDataArray(
                            fname + "_bnd",
                            NOUTS_C - cDim - 3, 2,
                            0, 0, //! vectors number is not cDim but 2
                            [&](int idata)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return names[idata]; // cellNames
                            },
                            [&](int idata, index iv)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return (*outSerialBnd)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            {
                                return idata == 0 ? "Velo" : "Norm"; // cellVecNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                if (idata == 0)
                                    return idim < cDim ? (*outSerialBnd)[iv][1 + idim] : 0; // cellVecData
                                else
                                    return (*outSerialBnd)[iv][nOUTSBnd - 3 + idim];
                            },
                            [&](int idata)
                            {
                                return "error"; // pointNames
                            },
                            [&](int idata, index iv)
                            {
                                return std::nan("0"); // pointData
                            },
                            [&](int idata)
                            {
                                return "error"; // pointNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                return std::nan("0"); // pointData
                            },
                            0.0,
                            0);
                    }
                    else if (config.dataIOControl.outPltMode == 1)
                    {
                        readerBnd->PrintSerialPartVTKDataArray(
                            fname + "_bnd",
                            NOUTS_C - cDim - 3, 2,
                            0, 0, //! vectors number is not cDim but 2
                            [&](int idata)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return names[idata]; // cellNames
                            },
                            [&](int idata, index iv)
                            {
                                idata = idata > 0 ? idata + cDim : 0;
                                return (*outDistBnd)[iv][idata]; // cellData
                            },
                            [&](int idata)
                            {
                                return idata == 0 ? "Velo" : "Norm"; // cellVecNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                if (idata == 0)
                                    return idim < cDim ? (*outDistBnd)[iv][1 + idim] : 0; // cellVecData
                                else
                                    return (*outDistBnd)[iv][nOUTSBnd - 3 + idim];
                            },
                            [&](int idata)
                            {
                                return "error"; // pointNames
                            },
                            [&](int idata, index iv)
                            {
                                return std::nan("0"); // pointData
                            },
                            [&](int idata)
                            {
                                return "error"; // pointNames
                            },
                            [&](int idata, index iv, int idim)
                            {
                                return std::nan("0"); // pointData
                            },
                            0.0,
                            1);
                    }
                }
            }
        }

        void WriteSerializer(SerializerBase *serializer, const std::string &name)
        {
            auto cwd = serializer->GetCurrentPath();
            serializer->CreatePath(name);
            serializer->GoToPath(name);

            u.WriteSerialize(serializer, "meanValue");

            serializer->GoToPath(cwd);

            nlohmann::ordered_json configJson;
            config.ReadWriteJson(configJson, nVars, false);
            serializer->WriteString("lastConfig", configJson.dump());

            if (config.dataIOControl.serializerSaveURec)
            {
                serializer->WriteInt("hasReconstructionValue", 1);
                uRec.WriteSerialize(serializer, "recValue");
            }
            else
                serializer->WriteInt("hasReconstructionValue", 0);
        }
    };
}