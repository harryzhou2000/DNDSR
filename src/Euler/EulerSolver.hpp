#pragma once
#include "Gas.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"

#include "Solver/Linear.hpp"
#include "EulerEvaluator.hpp"
#include "DNDS/JsonUtil.hpp"

#include <iomanip>
#include <functional>
#include <tuple>

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
        ssp<EulerEvaluator<model>> pEval;

        ArrayDOFV<nVars_Fixed> u, uInc, uIncRHS, uTemp, rhsTemp, wAveraged, uAveraged;
        ArrayRECV<nVars_Fixed> uRec, uRecNew, uRecNew1, uRecOld, uRec1, uRecInc, uRecInc1;
        ArrayDOFV<nVars_Fixed> JD, JD1, JSource, JSource1;
        ArrayDOFV<1> alphaPP, alphaPP1, betaPP, betaPP1, alphaPP_tmp;

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

        ssp<BoundaryHandler<model>> pBCHandler;

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
                bool useRHSfPP = false;
                real rhsfPPScale = 1.0;
                int odeCode = 0;
                real tEnd = veryLargeReal;
                real odeSetting1 = 0;
                real odeSetting2 = 0;
                real odeSetting3 = 0;
                bool partitonMeshOnly = false;
                real dtIncreaseLimit = 2;
                int dtIncreaseAfterCount = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    TimeMarchControl,
                    dtImplicit, nTimeStep,
                    steadyQuit, useRestart,
                    useImplicitPP, useRHSfPP, rhsfPPScale,
                    odeCode, tEnd, odeSetting1, odeSetting2, odeSetting3,
                    partitonMeshOnly,
                    dtIncreaseLimit, dtIncreaseAfterCount)
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
                int zeroRecForSteps = 0;
                int zeroRecForStepsInternal = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ImplicitReconstructionControl,
                    nInternalRecStep, zeroGrads,
                    recLinearScheme, nGmresSpace, nGmresIter,
                    recThreshold, nRecConsolCheck,
                    nRecMultiplyForZeroedGrad,
                    storeRecInc, dampRecIncDTau,
                    zeroRecForSteps,
                    zeroRecForStepsInternal)
            } implicitReconstructionControl;

            struct OutputControl
            {
                int nConsoleCheck = 1;
                int nConsoleCheckInternal = 1;
                int consoleOutputMode = 0; // 0 for basic, 1 for wall force out
                int consoleOutputEveryFix = 0;
                int nDataOut = 10000;
                int nDataOutC = 50;
                int nDataOutInternal = 10000;
                int nDataOutCInternal = 1;
                int nRestartOut = INT_MAX;
                int nRestartOutC = INT_MAX;
                int nRestartOutInternal = INT_MAX;
                int nRestartOutCInternal = INT_MAX;
                int nTimeAverageOut = INT_MAX;
                int nTimeAverageOutC = INT_MAX;
                real tDataOut = veryLargeReal;

                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    OutputControl,
                    nConsoleCheck,
                    nConsoleCheckInternal,
                    consoleOutputMode,
                    consoleOutputEveryFix,
                    nDataOut, nDataOutC,
                    nDataOutInternal, nDataOutCInternal,
                    nRestartOut, nRestartOutC,
                    nRestartOutInternal, nRestartOutCInternal,
                    nTimeAverageOut, nTimeAverageOutC,
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
                int nSmoothDTau = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ImplicitCFLControl,
                    CFL,
                    nForceLocalStartStep,
                    nCFLRampStart,
                    nCFLRampLength,
                    CFLRampEnd,
                    useLocalDt,
                    nSmoothDTau)
            } implicitCFLControl;

            struct ConvergenceControl
            {
                int nTimeStepInternal = 20;
                int nTimeStepInternalMin = 5;
                real rhsThresholdInternal = 1e-10;
                real res_base = 0;
                bool useVolWiseResidual = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    ConvergenceControl,
                    nTimeStepInternal, nTimeStepInternalMin,
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

                std::vector<std::string> outCellScalarNames{};

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
                    outCellScalarNames,
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
                bool preserveLimited = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    LimiterControl,
                    useLimiter, usePPRecLimiter,
                    smoothIndicatorProcedure, limiterProcedure,
                    nPartialLimiterStart, nPartialLimiterStartLocal,
                    preserveLimited)
            } limiterControl;

            struct LinearSolverControl
            {
                int jacobiCode = 1; // 0 for jacobi, 1 for gs
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
                    jacobiCode,
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

            struct TimeAverageControl
            {
                bool enabled = false;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    TimeAverageControl,
                    enabled)
            } timeAverageControl;

            struct Others
            {
                int nFreezePassiveInner = 0;
                DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                    Others,
                    nFreezePassiveInner)
            } others;

            nlohmann::ordered_json eulerSettings = nlohmann::ordered_json::object();
            nlohmann::ordered_json vfvSettings = nlohmann::ordered_json::object();
            nlohmann::ordered_json bcSettings = nlohmann::ordered_json::object();

            void
            ReadWriteJson(nlohmann::ordered_json &jsonObj, int nVars, bool read)
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
                __DNDS__json_to_config(timeAverageControl);
                __DNDS__json_to_config(others);
                __DNDS__json_to_config(restartState);

                __DNDS__json_to_config(eulerSettings);
                __DNDS__json_to_config(vfvSettings);
                __DNDS__json_to_config(bcSettings);
                if (read)
                {
                    DNDS_assert(eulerSettings.is_object());
                    DNDS_assert(vfvSettings.is_object());
                    DNDS_assert(bcSettings.is_array());
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
                bcSettings = BoundaryHandler<model>();
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
                PrintConfig(true);
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
        }

        void ReadMeshAndInitialize()
        {
            DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize 1 nvars " + std::to_string(nVars));
            output_stamp = getTimeStamp(mpi);
            if (!config.dataIOControl.uniqueStamps)
                output_stamp = "";
            if (mpi.rank == 0)
                log() << "=== Got Time Stamp: [" << output_stamp << "] ===" << std::endl;
            // Debug::MPIDebugHold(mpi);

            int gDimLocal = gDim; //! or else the linker breaks down here (with clang++ or g++, -g -O0,2; c++ non-optimizer bug?)

            DNDS_MAKE_SSP(pBCHandler);
            auto &BCHandler = *pBCHandler;
            // using tBC = typename BoundaryHandler<model>;
            BCHandler = config.bcSettings;

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
                reader->ReadFromCGNSSerial(config.dataIOControl.meshFile,
                                           [&](const std::string &name) -> Geom::t_index
                                           { return BCHandler.GetIDFromName(name); });
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

            if (config.timeMarchControl.partitonMeshOnly)
            {
                auto meshOutName = std::string(config.dataIOControl.meshFile) + "_part_" + std::to_string(mpi.size) + ".dir";
                std::filesystem::path meshOutDir{meshOutName};
                std::filesystem::create_directories(meshOutDir);
                std::string meshPartPath = DNDS::getStringForcePath(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

                DNDS::SerializerJSON serializerJSON;
                serializerJSON.SetUseCodecOnUint8(true);
                DNDS::SerializerBase *serializer = &serializerJSON;

                serializer->OpenFile(meshPartPath, false);
                mesh->WriteSerialize(serializer, "meshPart");
                serializer->CloseFile();
                return; //** mesh preprocess only (without transformation)
            }

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
                meshBnd->TransformCoords(
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
                meshBnd->TransformCoords(
                    [&](const Geom::tPoint &p)
                    { return p * scale; });

                for (auto &i : mesh->periodicInfo.translation)
                    i *= scale;
                for (auto &i : mesh->periodicInfo.rotationCenter)
                    i *= scale;
            }
            /// @todo //todo: upgrade to optional
            if (config.dataIOControl.outPltMode == 0)
                reader->coordSerialOutTrans.pullOnce(),
                    readerBnd->coordSerialOutTrans.pullOnce();
            vfv->ConstructMetrics();
            vfv->ConstructBaseAndWeight(
                [&](Geom::t_index id, int iOrder) -> real
                {
                    auto type = BCHandler.GetTypeFromID(id);
                    if (type == BCSpecial || type == BCOut)
                        return 0;
                    if (type == BCFar)
                    {
                        // use Dirichlet type
                        if (iOrder > 0)
                            return 0;
                        return 1;
                    }
                    if (type == BCWallInvis)
                    {
                        // // suppress higher order
                        // return 1;
                        // use Dirichlet type
                        if (iOrder > 0)
                            return 0;
                        return 1;
                    }
                    // others: use Dirichlet type
                    if (iOrder > 0)
                        return 0;
                    return 1;
                });
            vfv->ConstructRecCoeff();

            vfv->BuildUDof(u, nVars);
            vfv->BuildUDof(uInc, nVars);
            vfv->BuildUDof(uIncRHS, nVars);
            vfv->BuildUDof(uTemp, nVars);
            vfv->BuildUDof(rhsTemp, nVars);
            if (config.timeAverageControl.enabled)
            {
                vfv->BuildUDof(wAveraged, nVars);
                vfv->BuildUDof(uAveraged, nVars);
            }

            vfv->BuildURec(uRec, nVars);
            if (config.timeMarchControl.odeCode == 401)
                vfv->BuildURec(uRec1, nVars);
            vfv->BuildURec(uRecNew, nVars);
            vfv->BuildURec(uRecNew1, nVars);
            vfv->BuildURec(uRecOld, nVars);
            vfv->BuildScalar(ifUseLimiter);
            vfv->BuildUDof(betaPP, 1);
            vfv->BuildUDof(alphaPP, 1);
            vfv->BuildUDof(alphaPP_tmp, 1);
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

            DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize 2 nvars " + std::to_string(nVars));
            /*******************************/
            // initialize pEval
            DNDS_MAKE_SSP(pEval, mesh, vfv, pBCHandler);
            EulerEvaluator<model> &eval = *pEval;
            eval.settings.jsonSettings = config.eulerSettings;
            eval.settings.ReadWriteJSON(eval.settings.jsonSettings, nVars, true);
            /*******************************/
            // ** initialize output Array

            DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize 3 nvars " + std::to_string(nVars));

            // update output number
            DNDS_assert(config.dataIOControl.outCellScalarNames.size() < 128);
            nOUTS += config.dataIOControl.outCellScalarNames.size();

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
            DNDS_MPI_InsertCheck(mpi, "ReadMeshAndInitialize -1 nvars " + std::to_string(nVars));
        }

        void RunImplicitEuler();

        void PrintConfig(bool updateCommit = false)
        {
            /***********************************************************/
            // if these objects are existent, extract settings from them
            if (vfv)
                config.vfvSettings = vfv->settings;
            if (pEval)
                pEval->settings.ReadWriteJSON(config.eulerSettings, nVars, false);
            if (pBCHandler)
                config.bcSettings = *pBCHandler;
            /***********************************************************/
            config.ReadWriteJson(gSetting, nVars, false);
            if (mpi.rank == 0)
            {
                std::ofstream logConfig(config.dataIOControl.outLogName + "_" + output_stamp + ".config.json");
                gSetting["___Compile_Time_Defines"] = DNDS_Defines_state;
                gSetting["___Runtime_PartitionNumber"] = mpi.size;
                gSetting["___Commit_ID"] = DNDS_MACRO_TO_STRING(DNDS_CURRENT_COMMIT_HASH);
                // if (updateCommit)
                // {
                //     std::ifstream commitIDFile("commitID.txt");
                //     if (commitIDFile)
                //     {
                //         std::string commitHash;
                //         commitIDFile >> commitHash;
                //         gSetting["___Commit_ID"] = commitHash;
                //     }
                // }
                logConfig << std::setw(4) << gSetting;
                logConfig.close();
            }
        }
        void ReadRestart(std::string fname)
        {
            if (mpi.rank == 0)
                log() << fmt::format("=== Reading Restart From [{}]", fname) << std::endl;
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
            if (mpi.rank == 0)
                log() << fmt::format("=== Read Restart") << std::endl;
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
            config.restartState.lastRestartFile = getStringForcePath(outPath);
            PrintConfig();
        }

        using tAdditionalCellScalarList = tCellScalarList;

        enum PrintDataMode
        {
            PrintDataLatest = 0,
            PrintDataTimeAverage = 1,
        };

        void PrintData(const std::string &fname,
                       const tCellScalarFGet &odeResidualF,
                       tAdditionalCellScalarList &additionalCellScalars,
                       TEval &eval, PrintDataMode mode = PrintDataLatest);

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