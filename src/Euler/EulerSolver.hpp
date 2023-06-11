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
            int outPltMode = 0;   // 0 = serial, 1 = dist plt
            int readMeshMode = 0; // 0 = serial cgns, 1 = dist json
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
            int smoothIndicatorProcedure = 0;
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
            __gs_to_config(outPltMode);
            __gs_to_config(readMeshMode);
            __gs_to_config(uniqueStamps);
            __gs_to_config(err_dMax);
            __gs_to_config(res_base);
            __gs_to_config(useVolWiseResidual);
            __gs_to_config(useLocalDt);
            __gs_to_config(useLimiter);
            __gs_to_config(smoothIndicatorProcedure);
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

            // TODO: BC settings

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
            DNDS_assert(config.readMeshMode == 0 || config.readMeshMode == 1);
            DNDS_assert(config.outPltMode == 0 || config.outPltMode == 1);

            if (config.readMeshMode == 0)
            {
                reader->ReadFromCGNSSerial(config.meshFile); // TODO: add bnd mapping here
                reader->BuildCell2Cell();
                reader->MeshPartitionCell2Cell();
                reader->PartitionReorderToMeshCell2Cell();
                if (config.outPltMode == 0)
                {
                    reader->BuildSerialOut();
                    mesh->BuildGhostPrimary();
                    mesh->AdjGlobal2LocalPrimary();
                }
            }
            else
            {
                std::filesystem::path meshPath{config.meshFile};
                auto meshOutName = std::string(config.meshFile) + "_part_" + std::to_string(mpi.size) + ".dir";
                std::filesystem::path meshOutDir{meshOutName};
                // std::filesystem::create_directories(meshOutDir); // reading not writing
                std::string meshPartPath = std::string(meshOutDir / (std::string("part_") + std::to_string(mpi.rank) + ".json"));

                SerializerJSON serializerJSON;
                serializerJSON.SetUseCodecOnUint8(true);
                SerializerBase *serializer = &serializerJSON;
                serializer->OpenFile(meshPartPath, true);
                mesh->ReadSerialize(serializer, "meshPart");
                serializer->CloseFile();
                if (config.outPltMode == 0)
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

            vfv->BuildURec(uRec, nVars);
            vfv->BuildURec(uRecNew, nVars);
            vfv->BuildURec(uRecNew1, nVars);
            vfv->BuildURec(uOld, nVars);
            vfv->BuildScalar(ifUseLimiter);

            DNDS_MAKE_SSP(outDist, mpi);
            outDist->Resize(mesh->NumCell(), nOUTS);
            if (config.outPltMode == 0)
            {
                //! serial mesh specific output method
                DNDS_MAKE_SSP(outSerial, mpi);
                outDist2SerialTrans.setFatherSon(outDist, outSerial);
                DNDS_assert(reader->mode == Geom::MeshReaderMode::SerialOutput);
                outDist2SerialTrans.BorrowGGIndexing(reader->cell2nodeSerialOutTrans);
                outDist2SerialTrans.createMPITypes();
                outDist2SerialTrans.initPersistentPull();
            }
        }

        void RunImplicitEuler();

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

            if (config.outPltMode == 0)
            {
                outDist2SerialTrans.startPersistentPull();
                outDist2SerialTrans.waitPersistentPull();
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
            if (config.outPltMode == 0)
            {
                reader->PrintSerialPartPltBinaryDataArray(
                    fname, nOUTS, //! oprank = 0
                    [&](int idata)
                    { return names[idata]; },
                    [&](int idata, index iv)
                    {
                        return (*outSerial)[iv][idata];
                    },
                    0.0,
                    0);
            }
            else if (config.outPltMode == 1)
            {
                reader->PrintSerialPartPltBinaryDataArray(
                    fname, nOUTS, //! oprank = 0
                    [&](int idata)
                    { return names[idata]; },
                    [&](int idata, index iv)
                    {
                        return (*outDist)[iv][idata];
                    },
                    0.0,
                    1);
            }
            else
            {
                DNDS_assert(false);
            }
        }
    };

}