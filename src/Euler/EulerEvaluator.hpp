#pragma once
#include "Gas.hpp"
#include "Geom/Mesh.hpp"
#include "CFV/VariationalReconstruction.hpp"

#include <iomanip>
#include <functional>

#define JSON_ASSERT DNDS_assert
#include "json.hpp"
#include "DNDS/JsonUtil.hpp"

#include "Euler.hpp"
#include "DNDS/SerializerBase.hpp"

// #define DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
// // #define DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM
// // #define DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM

// // #ifdef DNDS_FV_EULEREVALUATOR_IGNORE_SOURCE_TERM // term dependency
// // // #define DNDS_FV_EULEREVALUATOR_USE_SCALAR_JACOBIAN
// // #endif

namespace DNDS::Euler
{

    template <EulerModel model>
    class EulerEvaluator
    {
    public:
        static const int nVars_Fixed = getNVars_Fixed(model);
        static const int dim = getDim_Fixed(model);
        static const int gDim = getGeomDim_Fixed(model);
        static const auto I4 = dim + 1;

#define DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS                            \
    static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>); \
    static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);     \
    static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>);

        typedef Eigen::Vector<real, dim> TVec;
        typedef Eigen::Matrix<real, dim, dim> TMat;
        typedef Eigen::Vector<real, nVars_Fixed> TU;
        typedef Eigen::Matrix<real, nVars_Fixed, nVars_Fixed> TJacobianU;
        typedef Eigen::Matrix<real, dim, nVars_Fixed> TDiffU;
        typedef Eigen::Matrix<real, nVars_Fixed, dim> TDIffUTransposed;

    public:
        // static const int gdim = 2; //* geometry dim

    private:
        int nVars = 5;

        bool passiveDiscardSource = false;

    public:
        void setPassiveDiscardSource(bool n) { passiveDiscardSource = n; }

    private:
    public:
        ssp<Geom::UnstructuredMesh> mesh;
        ssp<CFV::VariationalReconstruction<gDim>> vfv; //! gDim -> 3 for intellisense
        int kAv = 0;

        std::vector<real> lambdaCell;
        std::vector<real> lambdaFace;
        std::vector<real> lambdaFaceC;
        std::vector<real> lambdaFaceVis;
        std::vector<real> deltaLambdaFace;
        std::vector<Eigen::Matrix<real, 10, 5>> dFdUFace;

        // todo: improve to contiguous

        std::vector<Eigen::Vector<real, nVars_Fixed>> jacobianCellSourceDiag;
        std::vector<Eigen::Matrix<real, nvarsFixedMultiply<nVars_Fixed, 2>(), nVars_Fixed>> jacobianFace;
        std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCell;
        std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCellInv;
        std::vector<real> jacobianCell_Scalar;
        std::vector<real> jacobianCellInv_Scalar;

        // std::vector<Eigen::Vector<real, nVars_Fixed>> jacobianCellSourceDiag_Fixed;
        // std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianFace_Fixed;
        // std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCell_Fixed;
        // std::vector<Eigen::Matrix<real, nVars_Fixed, nVars_Fixed>> jacobianCellInv_Fixed;

        std::vector<std::vector<real>> dWall;

        // ArrayVDOF<25> dRdUrec;
        // ArrayVDOF<25> dRdb;

        Eigen::Vector<real, -1> fluxWallSum;
        index nFaceReducedOrder = 0;

        struct Setting
        {
            nlohmann::ordered_json jsonSettings;
            Gas::RiemannSolverType rsType = Gas::Roe;

            struct IdealGasProperty
            {
                real gamma = 1.4;
                real Rgas = 1;
                real muGas = 1;
                real prGas = 0.72;
                real CpGas = Rgas * gamma / (gamma - 1);
                real TRef = 273.15;
                real CSutherland = 110.4;

                void ReadWriteJSON(nlohmann::ordered_json &jsonObj, bool read)
                {
                    __DNDS__json_to_config(gamma);
                    __DNDS__json_to_config(Rgas);
                    __DNDS__json_to_config(muGas);
                    __DNDS__json_to_config(prGas);
                    __DNDS__json_to_config(TRef);
                    __DNDS__json_to_config(CSutherland);
                    CpGas = Rgas * gamma / (gamma - 1);
                }
            } idealGasProperty;

            Eigen::Vector<real, -1> farFieldStaticValue = Eigen::Vector<real, 5>{1, 0, 0, 0, 2.5};

            struct BoxInitializer
            {
                real x0, x1, y0, y1, z0, z1;
                Eigen::Vector<real, -1> v;
                void ReadWriteJSON(nlohmann::ordered_json &jsonObj, int nVars, bool read)
                {
                    __DNDS__json_to_config(x0);
                    __DNDS__json_to_config(x1);
                    __DNDS__json_to_config(y0);
                    __DNDS__json_to_config(y1);
                    __DNDS__json_to_config(z0);
                    __DNDS__json_to_config(z1);
                    __DNDS__json_to_config(v);
                    // std::cout << "here2" << std::endl;
                    if (read)
                        DNDS_assert(v.size() == nVars);
                    // if (read)
                    //     v = JsonGetEigenVector(jsonObj["v"]), DNDS_assert(v.size() == nVars);
                    // else
                    //     jsonObj["v"] = EigenVectorGetJson(v);
                }
            };
            std::vector<BoxInitializer> boxInitializers;

            struct PlaneInitializer
            {
                real a, b, c, h;
                Eigen::Vector<real, -1> v;
                void ReadWriteJSON(nlohmann::ordered_json &jsonObj, int nVars, bool read)
                {
                    __DNDS__json_to_config(a);
                    __DNDS__json_to_config(b);
                    __DNDS__json_to_config(c);
                    __DNDS__json_to_config(h);
                    __DNDS__json_to_config(v);
                    if (read)
                        DNDS_assert(v.size() == nVars);
                    // if (read)
                    //     v = JsonGetEigenVector(jsonObj["v"]), DNDS_assert(v.size() == nVars);
                    // else
                    //     jsonObj["v"] = EigenVectorGetJson(v);
                }
            };

            std::vector<PlaneInitializer> planeInitializers;

            int specialBuiltinInitializer = 0;

            Eigen::Vector<real, 3> constMassForce = Eigen::Vector<real, 3>{0, 0, 0};

            bool ignoreSourceTerm = false;
            bool useScalarJacobian = false;

            /***************************************************************************************************/
            /***************************************************************************************************/

            void ReadWriteSerializer(bool read, SerializerBase *serializer, const std::string &name)
            {
                auto cwd = serializer->GetCurrentPath();
                if (!read)
                    serializer->CreatePath(name);
                serializer->GoToPath(name);
                // TODO: find some convenient solution

                serializer->GoToPath(cwd);
            }

            void ReadWriteJSON(nlohmann::ordered_json &jsonObj, int nVars, bool read)
            {

                //********* root entries

                __DNDS__json_to_config(useScalarJacobian);
                __DNDS__json_to_config(ignoreSourceTerm);
                __DNDS__json_to_config(specialBuiltinInitializer);
                Gas::RiemannSolverType riemannSolverType = rsType;
                __DNDS__json_to_config(riemannSolverType);
                rsType = riemannSolverType;
                // std::cout << rsType << std::endl;
                __DNDS__json_to_config(constMassForce);
                if (read)
                    DNDS_assert(constMassForce.size() == 3);
                __DNDS__json_to_config(farFieldStaticValue);
                if (read)
                    DNDS_assert(farFieldStaticValue.size() == nVars);

                //********* box entries
                if (read)
                {
                    boxInitializers.clear();
                    DNDS_assert(jsonObj["boxInitializers"].is_array());
                    for (auto &boxJson : jsonObj["boxInitializers"])
                    {
                        DNDS_assert(boxJson.is_object());
                        BoxInitializer box;
                        box.ReadWriteJSON(boxJson, nVars, read);
                        boxInitializers.push_back(box);
                    }
                }
                else
                {
                    jsonObj["boxInitializers"] = nlohmann::ordered_json::array();
                    for (auto &b : boxInitializers)
                    {
                        nlohmann::ordered_json j;
                        b.ReadWriteJSON(j, nVars, read);
                        jsonObj["boxInitializers"].push_back(j);
                    }
                }

                //********* plane entries
                if (read)
                {
                    planeInitializers.clear();
                    DNDS_assert(jsonObj["planeInitializers"].is_array());
                    for (auto &planeJson : jsonObj["planeInitializers"])
                    {
                        DNDS_assert(planeJson.is_object());
                        PlaneInitializer p;
                        p.ReadWriteJSON(planeJson, nVars, read);
                        planeInitializers.push_back(p);
                    }
                }
                else
                {
                    jsonObj["planeInitializers"] = nlohmann::ordered_json::array();
                    for (auto &p : planeInitializers)
                    {
                        nlohmann::ordered_json j;
                        p.ReadWriteJSON(j, nVars, read);
                        jsonObj["planeInitializers"].push_back(j);
                    }
                }

                //********* idealGasProperty
                if (read)
                    DNDS_assert(jsonObj["idealGasProperty"].is_object());
                idealGasProperty.ReadWriteJSON(jsonObj["idealGasProperty"], read);
            }

        } settings;

        EulerEvaluator(const decltype(mesh) &Nmesh, const decltype(vfv) &Nvfv)
            : mesh(Nmesh), vfv(Nvfv), kAv(Nvfv->settings.maxOrder + 1)
        {
            nVars = getNVars(model);

            lambdaCell.resize(mesh->NumCellProc()); // but only dist part are used, ghost part to not judge for it in facial iter
            lambdaFace.resize(mesh->NumFaceProc());
            lambdaFaceC.resize(mesh->NumFaceProc());
            lambdaFaceVis.resize(lambdaFace.size());
            deltaLambdaFace.resize(lambdaFace.size());

            dFdUFace.resize(lambdaFace.size());

            jacobianFace.resize(lambdaFace.size(), typename decltype(jacobianFace)::value_type(nVars * 2, nVars));
            jacobianCell.resize(lambdaCell.size(), typename decltype(jacobianCell)::value_type(nVars, nVars));
            jacobianCellInv.resize(lambdaCell.size(), typename decltype(jacobianCellInv)::value_type(nVars, nVars));
            using jacobianCellSourceDiagElemType = typename decltype(jacobianCellSourceDiag)::value_type;
            jacobianCellSourceDiag.resize(lambdaCell.size(), jacobianCellSourceDiagElemType::Zero(nVars)); // zeroed
            jacobianCell_Scalar.resize(lambdaCell.size());
            jacobianCellInv_Scalar.resize(lambdaCell.size());

            // jacobianFace_Fixed.resize(lambdaFace.size());
            // jacobianCell_Fixed.resize(lambdaCell.size());
            // jacobianCellInv_Fixed.resize(lambdaCell.size());
            // jacobianCellSourceDiag_Fixed.resize(lambdaCell.size()); // zeroed
            // for (auto &i : jacobianCellSourceDiag_Fixed)
            //     i.setZero();

            // vfv->BuildRec(dRdUrec);
            // vfv->BuildRec(dRdb);

            //! wall dist code!!!
            ///@todo implement wall dist
            real maxD = 0.1;
            dWall.resize(mesh->NumCellProc(), std::vector<real>{UnInitReal});
        }

        /******************************************************/
        void EvaluateDt(
            std::vector<real> &dt,
            ArrayDOFV<nVars_Fixed> &u,
            real CFL, real &dtMinall, real MaxDt = 1,
            bool UseLocaldt = false);
        /**
         * @brief
         * \param rhs overwritten;
         *
         */
        void EvaluateRHS(
            ArrayDOFV<nVars_Fixed> &rhs,
            ArrayDOFV<nVars_Fixed> &u,
            ArrayRECV<nVars_Fixed> &uRec,
            real t);

        void LUSGSMatrixInit(
            std::vector<real> &dTau, real dt, real alphaDiag,
            ArrayDOFV<nVars_Fixed> &u,
            ArrayRECV<nVars_Fixed> &uRec,
            int jacobianCode,
            real t);

        void LUSGSMatrixVec(
            real alphaDiag,
            ArrayDOFV<nVars_Fixed> &u,
            ArrayDOFV<nVars_Fixed> &uInc,
            ArrayDOFV<nVars_Fixed> &AuInc);

        /**
         * @brief to use LUSGS, use LUSGSForward(..., uInc, uInc); uInc.pull; LUSGSBackward(..., uInc, uInc);
         * the underlying logic is that for index, ghost > dist, so the forward uses no ghost,
         * and ghost should be pulled before using backward;
         * to use Jacobian instead of LUSGS, use LUSGSForward(..., uInc, uIncNew); LUSGSBackward(..., uInc, uIncNew); uIncNew.pull; uInc = uIncNew;
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSForward(
            real alphaDiag,
            ArrayDOFV<nVars_Fixed> &rhs,
            ArrayDOFV<nVars_Fixed> &u,
            ArrayDOFV<nVars_Fixed> &uInc,
            ArrayDOFV<nVars_Fixed> &uIncNew);

        /**
         * @brief
         * \param uIncNew overwritten;
         *
         */
        void UpdateLUSGSBackward(
            real alphaDiag,
            ArrayDOFV<nVars_Fixed> &rhs,
            ArrayDOFV<nVars_Fixed> &u,
            ArrayDOFV<nVars_Fixed> &uInc,
            ArrayDOFV<nVars_Fixed> &uIncNew);

        void FixUMaxFilter(ArrayDOFV<nVars_Fixed> &u);

        void EvaluateResidual(
            Eigen::Vector<real, -1> &res,
            ArrayDOFV<nVars_Fixed> &rhs,
            index P = 1, bool volWise = false);

        /******************************************************/

        real muEff(const TU &U) // TODO: more than sutherland law
        {
            return std::nan("1");
        }

        void UFromCell2Face(TU &u, index iFace, index iCell, rowsize if2c)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (!mesh->isPeriodic)
                return;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return;
            if (if2c < 0)
                if2c = vfv->CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                u(Seq123) = mesh->periodicInfo.TransVectorBack(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                u(Seq123) = mesh->periodicInfo.TransVector(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
        }

        void UFromFace2Cell(TU &u, index iFace, index iCell, rowsize if2c)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (!mesh->isPeriodic)
                return;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return;
            if (if2c < 0)
                if2c = vfv->CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                u(Seq123) = mesh->periodicInfo.TransVector(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                u(Seq123) = mesh->periodicInfo.TransVectorBack(Eigen::Vector<real, dim>{u(Seq123)}, faceID);
        }

        void DiffUFromCell2Face(TDiffU &u, index iFace, index iCell, rowsize if2c)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            if (!mesh->isPeriodic)
                return;
            auto faceID = mesh->GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return;
            if (if2c < 0)
                if2c = vfv->CellIsFaceBack(iCell, iFace) ? 0 : 1;
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
            {
                u(Seq012, Eigen::all) = mesh->periodicInfo.TransVectorBack<dim, nVars_Fixed>(u(Seq012, Eigen::all), faceID);
                u(Eigen::all, Seq123) = mesh->periodicInfo.TransVectorBack<dim, dim>(u(Eigen::all, Seq123).transpose(), faceID).transpose();
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
            {
                u(Seq012, Eigen::all) = mesh->periodicInfo.TransVector<dim, nVars_Fixed>(u(Seq012, Eigen::all), faceID);
                u(Eigen::all, Seq123) = mesh->periodicInfo.TransVector<dim, dim>(u(Eigen::all, Seq123).transpose(), faceID).transpose();
            }
        }

        TU fluxFace(
            const TU &ULxy,
            const TU &URxy,
            const TU &ULMeanXy,
            const TU &URMeanXy,
            const TDiffU &DiffUxy,
            const TVec &unitNorm,
            const TMat &normBase,
            TU &FLfix,
            TU &FRfix,
            Geom::t_index btype,
            typename Gas::RiemannSolverType rsType,
            index iFace, int ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS

            TU UR = URxy;
            TU UL = ULxy;
            UR(Seq123) = normBase(Seq012, Seq012).transpose() * UR(Seq123);
            UL(Seq123) = normBase(Seq012, Seq012).transpose() * UL(Seq123);
            // if (btype == BoundaryType::Wall_NoSlip)
            //     UR(Seq123) = -UL(Seq123);
            // if (btype == BoundaryType::Wall_Euler)
            //     UR(1) = -UL(1);

            TU UMeanXy = 0.5 * (ULxy + URxy);

            real pMean, asqrMean, Hmean;
            real gamma = settings.idealGasProperty.gamma;
            Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                 gamma, pMean, asqrMean, Hmean);

            // ! refvalue:

            real muRef = settings.idealGasProperty.muGas;

            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
            real mufPhy, muf;
            mufPhy = muf = settings.idealGasProperty.muGas *
                           std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                           (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                           (T + settings.idealGasProperty.CSutherland);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
            real fnu1 = 0.;
            if constexpr (model == NS_SA)
            {
                real cnu1 = 7.1;
                real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Chi < 10) //*negative fix
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
#endif
                real Chi3 = std::pow(Chi, 3);
                fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
                muf *= std::max((1 + Chi * fnu1), 1.0);
            }

            real k = settings.idealGasProperty.CpGas * (muf - mufPhy) / 0.9 +
                     settings.idealGasProperty.CpGas * mufPhy / settings.idealGasProperty.prGas;
            TDiffU VisFlux;
            VisFlux.resizeLike(DiffUxy);
            VisFlux.setZero();
            Gas::ViscousFlux_IdealGas<dim>(
                UMeanXy, DiffUxy, unitNorm, btype == Geom::BC_ID_DEFAULT_WALL,
                settings.idealGasProperty.gamma,
                muf,
                k,
                settings.idealGasProperty.CpGas,
                VisFlux);

            // if (mesh->face2cellLocal[iFace][0] == 10756)
            // {
            //     std::cout << "Face " << iFace << " " << mesh->face2cellLocal[iFace][1] << std::endl;
            //     std::cout << DiffUxy << std::endl;
            //     std::cout << VisFlux << std::endl;
            //     std::cout << unitNorm << std::endl;
            //     std::cout << unitNorm.transpose() * VisFlux << std::endl;
            //     std::cout << muf << " " << k << std::endl;

            // }
            // if (iFace == 16404)
            // {
            //     std::cout << std::setprecision(10);
            //     std::cout << "Face " << iFace << " " << mesh->face2cellLocal[iFace][0] << " " << mesh->face2cellLocal[iFace][1] << std::endl;
            //     std::cout << DiffUxy << std::endl;
            //     std::cout << VisFlux << std::endl;
            //     std::cout << unitNorm << std::endl;
            //     std::cout << unitNorm.transpose() * VisFlux << std::endl;
            //     std::cout << muf << " " << k << std::endl;
            //     std::cout << lambdaFace[iFace] << std::endl;
            //     exit(-1);

            // }
            if constexpr (model == NS_SA)
            {
                real sigma = 2. / 3.;
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - UMeanXy(I4 + 1) * muRef / UMeanXy(0) * diffRho) / UMeanXy(0);

                real cn1 = 16;
                real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (UMeanXy(I4 + 1) < 0)
                {
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                }
#endif
                VisFlux(Seq012, {I4 + 1}) = diffNu * (mufPhy + UMeanXy(I4 + 1) * muRef * fn) / sigma / muRef;
            }
#endif

            TU finc;
            finc.resizeLike(ULxy);

            {
                TU wLMean, wRMean;
                TU ULMean = ULMeanXy;
                TU URMean = URMeanXy;
                ULMean(Seq123) = normBase(Seq012, Seq012).transpose() * ULMean(Seq123);
                URMean(Seq123) = normBase(Seq012, Seq012).transpose() * URMean(Seq123);
                Gas::IdealGasThermalConservative2Primitive<dim>(ULMean, wLMean, gamma);
                Gas::IdealGasThermalConservative2Primitive<dim>(URMean, wRMean, gamma);
                Gas::GasInviscidFlux<dim>(ULMean, wLMean(Seq123), wLMean(I4), FLfix);
                Gas::GasInviscidFlux<dim>(URMean, wRMean(Seq123), wRMean(I4), FRfix);
                FLfix(Seq123) = normBase * FLfix(Seq123);
                FRfix(Seq123) = normBase * FRfix(Seq123);
                if (model == NS_SA)
                {
                    FLfix(I4 + 1) = wLMean(1) * ULMean(I4 + 1);
                    FRfix(I4 + 1) = wRMean(1) * URMean(I4 + 1); // F_5 = rhoNut * un
                }
                // FLfix *= 0;
                // FRfix *= 0;
            }

            auto exitFun = [&]()
            {
                std::cout << "face at" << vfv->GetFaceQuadraturePPhys(iFace, -1) << '\n';
                std::cout << "UL" << UL.transpose() << '\n';
                std::cout << "UR" << UR.transpose() << std::endl;
            };

            real lam0{0}, lam123{0}, lam4{0};
            lam123 = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5;

            // std::cout << "HERE" << std::endl;
            if (rsType == Gas::RiemannSolverType::HLLEP)
                Gas::HLLEPFlux_IdealGas<dim>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun);
            else if (rsType == Gas::RiemannSolverType::HLLC)
                Gas::HLLCFlux_IdealGas_HartenYee<dim>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun);
            else if (rsType == Gas::RiemannSolverType::Roe)
                Gas::RoeFlux_IdealGas_HartenYee<dim>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M1)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 1>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M2)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 2>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M3)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 3>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M4)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 4>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun, lam0, lam123, lam4);
            else if (rsType == Gas::RiemannSolverType::Roe_M5)
                Gas::RoeFlux_IdealGas_HartenYee<dim, 5>(
                    UL, UR, settings.idealGasProperty.gamma, finc, deltaLambdaFace[iFace],
                    exitFun, lam0, lam123, lam4);
            else
                DNDS_assert(false);
                // std::cout << "HERE2" << std::endl;
                // if (btype == BoundaryType::Wall_NoSlip || btype == BoundaryType::Wall_Euler)
                //     finc(0) = 0; //! enforce mass leak = 0

#ifndef USE_ENTROPY_FIXED_LAMBDA_IN_SA
            lam123 = (std::abs(UL(1) / UL(0)) + std::abs(UR(1) / UR(0))) * 0.5; //! high fix
                                                                                // lam123 = std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5; //! low fix
#endif

            if constexpr (model == NS_SA)
            {
                // real lambdaFaceCC = sqrt(std::abs(asqrMean)) + std::abs(UL(1) / UL(0) + UR(1) / UR(0)) * 0.5;
                real lambdaFaceCC = lam123; //! using velo instead of velo + a
                finc(I4 + 1) = ((UL(1) / UL(0) * UL(I4 + 1) + UR(1) / UR(0) * UR(I4 + 1)) -
                                (UR(I4 + 1) - UL(I4 + 1)) * lambdaFaceCC) *
                               0.5;
            }

            finc(Seq123) = normBase * finc(Seq123);
#ifndef DNDS_FV_EULEREVALUATOR_IGNORE_VISCOUS_TERM
            finc -= VisFlux.transpose() * unitNorm * 1;
#endif

            if (finc.hasNaN() || (!finc.allFinite()))
            {
                std::cout << finc.transpose() << std::endl;
                std::cout << ULxy.transpose() << std::endl;
                std::cout << URxy.transpose() << std::endl;
                std::cout << DiffUxy << std::endl;
                std::cout << unitNorm << std::endl;
                std::cout << normBase << std::endl;
                std::cout << T << std::endl;
                std::cout << muf << std::endl;
                std::cout << pMean << std::endl;
                DNDS_assert(false);
            }

            return -finc;
        }

        TU source(
            const TU &UMeanXy,
            const TDiffU &DiffUxy,
            index iCell, index ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
            TU ret;
            ret.resizeLike(UMeanXy);
            ret.setZero();
            return ret;
#endif
            if constexpr (model == NS || model == NS_2D || model == NS_3D)
            {
                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();
                ret(Seq123) = settings.constMassForce(Seq012) * UMeanXy(0);
                return ret;
            }
            else if constexpr (model == NS_SA)
            {
                real d = std::min(dWall[iCell][ig], std::pow(veryLargeReal, 1. / 6.));
                real cb1 = 0.1355;
                real cb2 = 0.622;
                real sigma = 2. / 3.;
                real cnu1 = 7.1;
                real cnu2 = 0.7;
                real cnu3 = 0.9;
                real cw2 = 0.3;
                real cw3 = 2;
                real kappa = 0.41;
                real rlim = 10;
                real cw1 = cb1 / sqr(kappa) + (1 + cb2) / sigma;

                real ct3 = 1.2;
                real ct4 = 0.5;

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(I4 + 1) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = settings.idealGasProperty.muGas *
                               std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                               (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                               (T + settings.idealGasProperty.CSutherland);

                real Chi = (UMeanXy(I4 + 1) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
                Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, dim, dim> Omega = 0.5 * (diffU.transpose() - diffU);
                real S = Omega.norm() * std::sqrt(2);
                real Sbar = nuh / (sqr(kappa) * sqr(d)) * fnu2;

                real Sh;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Sbar < -cnu2 * S)
                    Sh = S + S * (sqr(cnu2) * S + cnu3 * Sbar) / ((cnu3 - 2 * cnu2) * S - Sbar);
                else //*negative fix
#endif
                    Sh = S + Sbar;

                real r = std::min(nuh / (Sh * sqr(kappa * d) + verySmallReal), rlim);
                real g = r + cw2 * (std::pow(r, 6) - r);
                real fw = g * std::pow((1 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1. / 6.);

                real ft2 = ct3 * std::exp(-ct4 * sqr(Chi));
#ifdef USE_NS_SA_NEGATIVE_MODEL
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d); //! modified >>
                real P = cb1 * (1 - ft2) * Sh * nuh;                         //! modified >>
#else
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d);
                real P = cb1 * (1 - ft2) * Sh * nuh;
#endif
                real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (UMeanXy(I4 + 1) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }
#endif

                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();
                ret(Seq123) = settings.constMassForce * UMeanXy(0);

                if (passiveDiscardSource)
                    P = D = 0;
                ret(I4 + 1) = UMeanXy(0) * (P - D + diffNu.squaredNorm() * cb2 / sigma) / muRef -
                              (UMeanXy(I4 + 1) * fn * muRef + mufPhy) / (UMeanXy(0) * sigma) * diffRho.dot(diffNu) / muRef;
                // std::cout << "P, D " << P / muRef << " " << D / muRef << " " << diffNu.squaredNorm() << std::endl;
                if (ret.hasNaN())
                {
                    std::cout << P << std::endl;
                    std::cout << D << std::endl;
                    std::cout << UMeanXy(0) << std::endl;
                    std::cout << Sh << std::endl;
                    std::cout << nuh << std::endl;
                    std::cout << g << std::endl;
                    std::cout << r << std::endl;
                    std::cout << S << std::endl;
                    std::cout << d << std::endl;
                    std::cout << fnu2 << std::endl;
                    std::cout << mufPhy << std::endl;
                    DNDS_assert(false);
                }
                // if (passiveDiscardSource)
                //     ret(Eigen::seq(5, Eigen::last)).setZero();
                return ret;
            }
            else
            {
                DNDS_assert(false);
            }
        }

        // zeroth means needs not derivative
        TU sourceJacobianDiag(
            const TU &UMeanXy,
            const TDiffU &DiffUxy,
            index iCell, index ig)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
#ifdef DNDS_FV_EULEREVALUATOR_SOURCE_TERM_ZERO
            TU ret;
            ret.resizeLike(UMeanXy);
            ret.setZero();
            return ret;
#endif
            if constexpr (model == NS || model == NS_2D || model == NS_3D)
            {
            }
            else if constexpr (model == NS_SA)
            {
                real d = std::min(dWall[iCell][ig], std::pow(veryLargeReal, 1. / 6.));
                real cb1 = 0.1355;
                real cb2 = 0.622;
                real sigma = 2. / 3.;
                real cnu1 = 7.1;
                real cnu2 = 0.7;
                real cnu3 = 0.9;
                real cw2 = 0.3;
                real cw3 = 2;
                real kappa = 0.41;
                real rlim = 10;
                real cw1 = cb1 / sqr(kappa) + (1 + cb2) / sigma;

                real ct3 = 1.2;
                real ct4 = 0.5;

                real pMean, asqrMean, Hmean;
                real gamma = settings.idealGasProperty.gamma;
                Gas::IdealGasThermal(UMeanXy(I4), UMeanXy(0), (UMeanXy(Seq123) / UMeanXy(0)).squaredNorm(),
                                     gamma, pMean, asqrMean, Hmean);
                // ! refvalue:
                real muRef = settings.idealGasProperty.muGas;

                real nuh = UMeanXy(I4 + 1) * muRef / UMeanXy(0);

                real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * UMeanXy(0));
                real mufPhy, muf;
                mufPhy = muf = settings.idealGasProperty.muGas *
                               std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                               (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                               (T + settings.idealGasProperty.CSutherland);

                real Chi = (UMeanXy(I4 + 1) * muRef / mufPhy);
                real fnu1 = std::pow(Chi, 3) / (std::pow(Chi, 3) + std::pow(cnu1, 3));
                real fnu2 = 1 - Chi / (1 + Chi * fnu1);

                Eigen::Matrix<real, dim, 1> velo = UMeanXy(Seq123) / UMeanXy(0);
                Eigen::Matrix<real, dim, 1> diffRhoNu = DiffUxy(Seq012, {I4 + 1}) * muRef;
                Eigen::Matrix<real, dim, 1> diffRho = DiffUxy(Seq012, {0});
                Eigen::Matrix<real, dim, 1> diffNu = (diffRhoNu - nuh * diffRho) / UMeanXy(0);
                Eigen::Matrix<real, dim, dim> diffRhoU = DiffUxy(Seq012, Seq123);
                Eigen::Matrix<real, dim, dim> diffU = (diffRhoU - diffRho * velo.transpose()) / UMeanXy(0);

                Eigen::Matrix<real, dim, dim> Omega = 0.5 * (diffU.transpose() - diffU);
                real S = Omega.norm() * std::sqrt(2);
                real Sbar = nuh / (sqr(kappa) * sqr(d)) * fnu2;

                real Sh;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Sbar < -cnu2 * S)
                    Sh = S + S * (sqr(cnu2) * S + cnu3 * Sbar) / ((cnu3 - 2 * cnu2) * S - Sbar);
                else
#endif
                    Sh = S + Sbar;

                real r = std::min(nuh / (Sh * sqr(kappa * d) + verySmallReal), rlim);
                real g = r + cw2 * (std::pow(r, 6) - r);
                real fw = g * std::pow((1 + std::pow(cw3, 6)) / (std::pow(g, 6) + std::pow(cw3, 6)), 1. / 6.);

                real ft2 = ct3 * std::exp(-ct4 * sqr(Chi));
                real D = (cw1 * fw - cb1 / sqr(kappa) * ft2) * sqr(nuh / d); //! modified >>
                real P = cb1 * (1 - ft2) * Sh * nuh;                         //! modified >>
                // real D = cw1 * fw * sqr(nuh / d);
                // real P = cb1 * Sh * nuh;
                real fn = 1;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (UMeanXy(I4 + 1) < 0)
                {
                    real cn1 = 16;
                    real Chi = UMeanXy(I4 + 1) * muRef / mufPhy;
                    fn = (cn1 + std::pow(Chi, 3)) / (cn1 - std::pow(Chi, 3));
                    P = cb1 * (1 - ct3) * S * nuh;
                    D = -cw1 * sqr(nuh / d);
                }
#endif

                TU ret;
                ret.resizeLike(UMeanXy);
                ret.setZero();

                if (passiveDiscardSource)
                    P = D = 0;
                ret(I4 + 1) = -std::min(UMeanXy(0) * (P * 0 - D * 2) / muRef / (UMeanXy(I4 + 1) + verySmallReal), -verySmallReal);
                // std::cout << ret(I4+1) << std::endl;

                if (ret.hasNaN())
                {
                    std::cout << P << std::endl;
                    std::cout << D << std::endl;
                    std::cout << UMeanXy(0) << std::endl;
                    std::cout << Sh << std::endl;
                    std::cout << nuh << std::endl;
                    std::cout << g << std::endl;
                    std::cout << r << std::endl;
                    std::cout << S << std::endl;
                    std::cout << d << std::endl;
                    std::cout << fnu2 << std::endl;
                    std::cout << mufPhy << std::endl;
                    std::cout << UMeanXy.transpose() << std::endl;
                    std::cout << pMean << std::endl;
                    std::cout << ret.transpose() << std::endl;

                    DNDS_assert(false);
                }
                // if (passiveDiscardSource)
                //     ret(Eigen::seq(5, Eigen::last)).setZero();
                return ret;
            }
            else
            {
                DNDS_assert(false);
            }

            TU Ret;
            Ret.resizeLike(UMeanXy);
            Ret.setZero();
            return Ret;
        }

        /**
         * @brief inviscid flux approx jacobian (flux term not reconstructed / no riemann)
         *
         */
        auto fluxJacobian0_Right(
            TU &UR,
            const TVec &uNorm,
            Geom::t_index btype)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            DNDS_assert(dim == 3); // only for 3D!!!!!!!!
            const TU &U = UR;
            const TVec &n = uNorm;

            real rhoun = n.dot(U({1, 2, 3}));
            real rhousqr = U({1, 2, 3}).squaredNorm();
            real gamma = settings.idealGasProperty.gamma;
            TJacobianU subFdU;
            subFdU.resize(nVars, nVars);

            subFdU.setZero();
            subFdU(0, 1) = n(1 - 1);
            subFdU(0, 2) = n(2 - 1);
            subFdU(0, 3) = n(3 - 1);
            subFdU(1, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(2 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(1 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 1) = (rhoun + U(2 - 1) * n(1 - 1) * 2.0 - U(2 - 1) * gamma * n(1 - 1)) / U(1 - 1);
            subFdU(1, 2) = (U(2 - 1) * n(2 - 1)) / U(1 - 1) - (U(3 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 3) = (U(2 - 1) * n(3 - 1)) / U(1 - 1) - (U(4 - 1) * n(1 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(1, 4) = n(1 - 1) * (gamma - 1.0);
            subFdU(2, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(3 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(2 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 1) = (U(3 - 1) * n(1 - 1)) / U(1 - 1) - (U(2 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 2) = (rhoun + U(3 - 1) * n(2 - 1) * 2.0 - U(3 - 1) * gamma * n(2 - 1)) / U(1 - 1);
            subFdU(2, 3) = (U(3 - 1) * n(3 - 1)) / U(1 - 1) - (U(4 - 1) * n(2 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(2, 4) = n(2 - 1) * (gamma - 1.0);
            subFdU(3, 0) = -1.0 / (U(1 - 1) * U(1 - 1)) * U(4 - 1) * rhoun + (1.0 / (U(1 - 1) * U(1 - 1)) * n(3 - 1) * (gamma - 1.0) * (rhousqr - U(1 - 1) * U(5 - 1) * 2.0)) / 2.0 + (U(5 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 1) = (U(4 - 1) * n(1 - 1)) / U(1 - 1) - (U(2 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 2) = (U(4 - 1) * n(2 - 1)) / U(1 - 1) - (U(3 - 1) * n(3 - 1) * (gamma - 1.0)) / U(1 - 1);
            subFdU(3, 3) = (rhoun + U(4 - 1) * n(3 - 1) * 2.0 - U(4 - 1) * gamma * n(3 - 1)) / U(1 - 1);
            subFdU(3, 4) = n(3 - 1) * (gamma - 1.0);
            subFdU(4, 0) = 1.0 / (U(1 - 1) * U(1 - 1) * U(1 - 1)) * rhoun * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma);
            subFdU(4, 1) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(1 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(2 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 2) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(2 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(3 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 3) = 1.0 / (U(1 - 1) * U(1 - 1)) * n(3 - 1) * (-rhousqr + (U(2 - 1) * U(2 - 1)) * gamma + (U(3 - 1) * U(3 - 1)) * gamma + (U(4 - 1) * U(4 - 1)) * gamma - U(1 - 1) * U(5 - 1) * gamma * 2.0) * (-1.0 / 2.0) - 1.0 / (U(1 - 1) * U(1 - 1)) * U(4 - 1) * rhoun * (gamma - 1.0);
            subFdU(4, 4) = (gamma * rhoun) / U(1 - 1);

            real un = rhoun / U(0);

            if constexpr (model == NS_SA)
            {
                subFdU(5, 5) = un;
                subFdU(5, 0) = -un * U(5) / U(0);
                subFdU(5, 1) = n(0) * U(5) / U(0);
                subFdU(5, 2) = n(1) * U(5) / U(0);
                subFdU(5, 3) = n(2) * U(5) / U(0);
            }
            return subFdU;
        }

        /**
         * @brief inviscid flux approx jacobian (flux term not reconstructed / no riemann)
         *
         */
        TU fluxJacobian0_Right_Times_du(
            const TU &U,
            const TVec &n,
            Geom::t_index btype,
            const TU &dU, real lambdaMain, real lambdaC)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            real gamma = settings.idealGasProperty.gamma;
            TVec velo = U(Seq123) / U(0);
            real p, H, asqr;
            Gas::IdealGasThermal(U(I4), U(0), velo.squaredNorm(), gamma, p, asqr, H);
            TVec dVelo;
            real dp;
            Gas::IdealGasUIncrement<dim>(U, dU, velo, gamma, dVelo, dp);
            TU dF(U.size());
            Gas::GasInviscidFluxFacialIncrement<dim>(
                U, dU,
                n,
                velo, dVelo,
                dp, p,
                dF);
            dF(Seq01234) -= lambdaMain * dU(Seq01234);
            if constexpr (model == NS_SA)
            {
                dF(I4 + 1) = dU(I4 + 1) * n.dot(velo) + U(I4 + 1) * n.dot(dVelo);
                dF(I4 + 1) -= dU(I4 + 1) * lambdaMain;
            }
            return dF;
        }

        TU generateBoundaryValue(
            TU &ULxy, //! warning, possible that UL is also modified
            const TVec &uNorm,
            const TMat &normBase,
            const TVec &pPhysics,
            real t,
            Geom::t_index btype,
            bool fixUL = false)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS

            TU URxy;

            if (btype == Geom::BC_ID_DEFAULT_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR ||
                btype == Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR)
            {
                DNDS_assert(ULxy(0) > 0);
                if (btype == Geom::BC_ID_DEFAULT_FAR)
                {
                    const TU &far = settings.farFieldStaticValue;

                    real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
                    real vsqr = (ULxy(Seq123) / ULxy(0)).squaredNorm();
                    real gamma = settings.idealGasProperty.gamma;
                    real asqr, H, p;
                    Gas::IdealGasThermal(ULxy(I4), ULxy(0), vsqr, gamma, p, asqr, H);

                    DNDS_assert(asqr >= 0);
                    real a = std::sqrt(asqr);

                    if (un - a > 0) // full outflow
                    {
                        URxy = ULxy;
                    }
                    else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                    }
                    else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                        farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                    }
                    else // full inflow
                    {
                        URxy = settings.farFieldStaticValue;
                    }
                    // URxy = settings.farFieldStaticValue; //!override
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_DMR_FAR)
                {
                    DNDS_assert(dim > 1);
                    URxy = settings.farFieldStaticValue;
                    real uShock = 10;
                    if constexpr (dim == 3) //* manual static dispatch
                    {
                        if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                             pPhysics(1) / std::tan(pi / 3)) > 0)
                            URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{1.4, 0, 0, 0, 2.5};
                        else
                            URxy({0, 1, 2, 3, 4}) = Eigen::Vector<real, 5>{8, 57.157676649772960, -33, 0, 5.635e2};
                    }
                    else
                    {
                        if (((pPhysics(0) - uShock / std::sin(pi / 3) * t - 1. / 6.) -
                             pPhysics(1) / std::tan(pi / 3)) > 0)
                            URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{1.4, 0, 0, 2.5};
                        else
                            URxy({0, 1, 2, 3}) = Eigen::Vector<real, 4>{8, 57.157676649772960, -33, 5.635e2};
                    }
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_RT_FAR)
                {
                    DNDS_assert(dim > 1);
                    Eigen::VectorXd far = settings.farFieldStaticValue;
                    real gamma = settings.idealGasProperty.gamma;
                    real un = ULxy(Seq123).dot(uNorm) / ULxy(0);
                    real vsqr = (ULxy(Seq123) / ULxy(0)).squaredNorm();
                    real asqr, H, p;
                    Gas::IdealGasThermal(ULxy(I4), ULxy(0), vsqr, gamma, p, asqr, H);

                    DNDS_assert(asqr >= 0);
                    real a = std::sqrt(asqr);
                    real v = -0.025 * a * cos(pPhysics(0) * 8 * pi);

                    if (pPhysics(1) < 0.5)
                    {

                        real rho = 2;
                        real p = 1;
                        far(0) = rho;
                        far(1) = 0;
                        far(2) = rho * v;
                        far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                    }
                    else
                    {
                        real rho = 1;
                        real p = 2.5;
                        far(0) = rho;
                        far(1) = 0;
                        far(2) = rho * v;
                        far(I4) = 0.5 * rho * sqr(v) + p / (gamma - 1);
                    }

                    if (un - a > 0) // full outflow
                    {
                        URxy = ULxy;
                    }
                    else if (un > 0) //  1 sonic outflow, 1 sonic inflow, other outflow (subsonic out)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        ULxyPrimitive(I4) = farPrimitive(I4); // using far pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(ULxyPrimitive, URxy, gamma);
                    }
                    else if (un + a > 0) //  1 sonic outflow, 1 sonic inflow, other inflow (subsonic in)
                    {
                        TU farPrimitive, ULxyPrimitive;
                        Gas::IdealGasThermalConservative2Primitive<dim>(far, farPrimitive, gamma);
                        Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                        // farPrimitive(0) = ULxyPrimitive(0); // using inner density
                        farPrimitive(I4) = ULxyPrimitive(I4); // using inner pressure
                        Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                    }
                    else // full inflow
                    {
                        URxy = far;
                    }
                    URxy = far; //! override
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_IV_FAR)
                {
                    real chi = 5;
                    real gamma = settings.idealGasProperty.gamma;
                    real xc = 5 + t;
                    real yc = 5 + t;
                    real r = std::sqrt(sqr(pPhysics(0) - xc) + sqr(pPhysics(1) - yc));
                    real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r));
                    real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - xc);
                    real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - yc);
                    real T = dT + 1;
                    real ux = dux + 1;
                    real uy = duy + 1;
                    real S = 1;
                    real rho = std::pow(T / S, 1 / (gamma - 1));
                    real p = T * rho;

                    real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                    // std::cout << T << " " << rho << std::endl;
                    URxy.setZero();
                    URxy(0) = rho;
                    URxy(1) = rho * ux;
                    URxy(2) = rho * uy;
                    URxy(dim + 1) = E;
                }
                else if (btype == Geom::BC_ID_DEFAULT_SPECIAL_2DRiemann_FAR)
                {
                    real gamma = settings.idealGasProperty.gamma;
                    real bdL = 0.0; // left
                    real bdR = 1.0; // right
                    real bdD = 0.0; // down
                    real bdU = 1.0; // up

                    real phi1 = -0.663324958071080;
                    real phi2 = -0.422115882408869;
                    real location = 0.8;
                    real p1 = location + phi1 * t;
                    real p2 = location + phi2 * t;
                    real rho, u, v, pre;
                    TU ULxyPrimitive;

                    Gas::IdealGasThermalConservative2Primitive<dim>(ULxy, ULxyPrimitive, gamma);
                    real rhoL = ULxyPrimitive(0);
                    real uL = ULxyPrimitive(1);
                    real vL = ULxyPrimitive(2);
                    real preL = ULxyPrimitive(I4);
                    TU farPrimitive = ULxyPrimitive;

                    static const real bTol = 1e-9;
                    if (std::abs(pPhysics(0) - bdL) < bTol)
                    { // left, phi2
                        if (pPhysics(1) <= p2)
                        { // region 3
                            rho = 0.137992831541219;
                            u = 1.206045378311055;
                            v = 1.206045378311055;
                            pre = 0.029032258064516;
                        }
                        else
                        { // region 2
                            rho = 0.532258064516129;
                            u = 1.206045378311055;
                            v = 0.0;
                            pre = 0.3;
                        }
                    }
                    else if (std::abs(pPhysics(0) - bdR) < bTol)
                    { // right, phi1
                        if (pPhysics(1) <= p1)
                        { // region 4
                            // rho = 0.532258064516129;
                            // u = 0.0;
                            // v = 1.206045378311055;
                            // pre = 0.3;
                            rho = rhoL;
                            u = -uL;
                            v = vL;
                            pre = preL;
                        }
                        else
                        { // region 1
                            // rho = 1.5;
                            // u = 0.0;
                            // v = 0.0;
                            // pre = 1.5;
                            rho = rhoL;
                            u = -uL;
                            v = vL;
                            pre = preL;
                        }
                    }
                    else if (std::abs(pPhysics(1) - bdU) < bTol)
                    { // up, phi1
                        if (pPhysics(0) <= p1)
                        { // region 2
                            // rho = 0.532258064516129;
                            // u = 1.206045378311055;
                            // v = 0.0;
                            // pre = 0.3;
                            rho = rhoL;
                            u = uL;
                            v = -vL;
                            pre = preL;
                        }
                        else
                        { // region 1
                            // rho = 1.5;
                            // u = 0.0;
                            // v = 0.0;
                            // pre = 1.5;
                            rho = rhoL;
                            u = uL;
                            v = -vL;
                            pre = preL;
                        }
                    }
                    else if (std::abs(pPhysics(1) - bdD) < bTol)
                    { // down, phi2
                        if (pPhysics(0) <= p2)
                        { // region 3
                            rho = 0.137992831541219;
                            u = 1.206045378311055;
                            v = 1.206045378311055;
                            pre = 0.029032258064516;
                        }
                        else
                        { // region 4
                            rho = 0.532258064516129;
                            u = 0.0;
                            v = 1.206045378311055;
                            pre = 0.3;
                        }
                    }
                    else
                    {
                        rho = u = v = pre = std::nan("1");
                        DNDS_assert(false); // not valid boundary pos
                    }
                    farPrimitive(0) = rho;
                    farPrimitive(1) = u, farPrimitive(2) = v;
                    farPrimitive(I4) = pre;
                    Gas::IdealGasThermalPrimitive2Conservative<dim>(farPrimitive, URxy, gamma);
                }
                else
                    DNDS_assert(false);
            }
            else if (btype == Geom::BC_ID_DEFAULT_WALL_INVIS)
            {
                URxy = ULxy;
                URxy(Seq123) -= URxy(Seq123).dot(uNorm) * uNorm;
            }
            else if (btype == Geom::BC_ID_DEFAULT_WALL)
            {
                URxy = ULxy;
                URxy(Seq123) *= -1;
                if (model == NS_SA)
                {
                    URxy(I4 + 1) *= -1;
#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
                    if (fixUL)
                        ULxy(I4 + 1) = URxy(I4 + 1) = 0; //! modifing UL
#endif
                }
            }
            else
            {
                DNDS_assert(false);
            }
            return URxy;
        }

        inline TU CompressRecPart(
            const TU &umean,
            const TU &uRecInc,
            bool &compressed)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            // if (umean(0) + uRecInc(0) < 0)
            // {
            //     std::cout << umean.transpose() << std::endl
            //               << uRecInc.transpose() << std::endl;
            //     DNDS_assert(false);
            // }
            // return umean + uRecInc; // ! no compress shortcut
            // return umean; // ! 0th order shortcut

            // // * Compress Method
            // real compressT = 0.00001;
            // real eFixRatio = 0.00001;
            // Eigen::Vector<real, 5> ret;

            // real compress = 1.0;
            // if ((umean(0) + uRecInc(0)) < umean(0) * compressT)
            //     compress *= umean(0) * (1 - compressT) / uRecInc(0);

            // ret = umean + uRecInc * compress;

            // real Ek = ret({1, 2, 3}).squaredNorm() * 0.5 / (verySmallReal + ret(0));
            // real eT = eFixRatio * Ek;
            // real e = ret(4) - Ek;
            // if (e < 0)
            //     e = eT * 0.5;
            // else if (e < eT)
            //     e = (e * e + eT * eT) / (2 * eT);
            // ret(4) = e + Ek;
            // // * Compress Method

            // TU ret = umean + uRecInc;
            // real eK = ret(Seq123).squaredNorm() * 0.5 / (verySmallReal + std::abs(ret(0)));
            // real e = ret(I4) - eK;
            // if (e <= 0 || ret(0) <= 0)
            //     ret = umean, compressed = true;
            // if constexpr (model == NS_SA)
            //     if (ret(I4 + 1) < 0)
            //         ret = umean, compressed = true;

            bool rhoFixed = false;
            bool eFixed = false;
            TU ret = umean + uRecInc;
            // do rho fix
            if (ret(0) < 0)
            {
                // rhoFixed = true;
                // TVec veloOld = umean(Seq123) / umean(0);
                // real eOld = umean(I4) - 0.5 * veloOld.squaredNorm() * umean(0);
                // ret(0) = umean(0) * std::exp(uRecInc(0) / (umean(0) + verySmallReal));
                // ret(Seq123) = veloOld * ret(0);
                // ret(I4) = eOld + 0.5 * veloOld.squaredNorm() * ret(0);

                ret = umean;
                compressed = true;
            }

            real eK = ret(Seq123).squaredNorm() * 0.5 / (verySmallReal + ret(0));
            real e = ret(I4) - eK;
            if (e < 0)
            {
                // eFixed = true;
                // real eOld = umean(I4) - eK;
                // real eNew = eOld * std::exp(eOld / (umean(I4) - eK));
                // ret(I4) = eNew + eK;

                ret = umean;

                compressed = true;
            }

#ifdef USE_NS_SA_NUT_REDUCED_ORDER
            if constexpr (model == NS_SA)
                if (ret(I4 + 1) < 0)
                    ret(I4 + 1) = umean(I4 + 1), compressed = true;
#endif

            return ret;
        }

        inline TU CompressInc(
            const TU &u,
            const TU &uInc,
            const TU &rhs)
        {
            DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
            TU ret = uInc;

            /** A intuitive fix **/ //! need positive perserving technique!
            // DNDS_assert(u(0) > 0);
            // if (u(0) + ret(0) <= 0)
            // {
            //     real declineV = ret(0) / (u(0) + verySmallReal);
            //     real newrho = u(0) * std::exp(declineV);
            //     ret(0) = newrho - u(0);
            // }
            // real rhoEinternal = u(I4) - 0.5 * u(Seq123).squaredNorm() / u(0);
            // real ek = 0.5 * (u(Seq123) + ret(Seq123)).squaredNorm() / (u(0) + ret(0));
            // real rhoEinternalNew = u(I4) + ret(I4) - ek;
            // if (rhoEinternalNew <= 0)
            // {
            //     real declineV = (rhoEinternalNew - rhoEinternal) / (rhoEinternal + verySmallReal);
            //     real newrhoEinteralNew = std::exp(declineV) * rhoEinternal;
            //     ret(I4) = newrhoEinteralNew + ek;
            // }
            /** A intuitive fix **/

            if constexpr (model == NS_SA)
            {
                if (u(I4 + 1) + ret(I4 + 1) < 0)
                {
                    // std::cout << "Fixing SA inc " << std::endl;

                    DNDS_assert(u(I4 + 1) >= 0); //! might be bad using gmeres, add this to gmres inc!
                    real declineV = ret(I4 + 1) / (u(I4 + 1) + 1e-6);
                    real newu5 = u(I4 + 1) * std::exp(declineV);
                    // ! refvalue:
                    real muRef = settings.idealGasProperty.muGas;
                    newu5 = std::max(1e-6, newu5);
                    ret(I4 + 1) = newu5 - u(I4 + 1);
                }
            }

            return ret;
        }

        void InitializeUDOF(ArrayDOFV<nVars_Fixed> &u)
        {
            Eigen::VectorXd initConstVal = this->settings.farFieldStaticValue;
            u.setConstant(initConstVal);
            if (model == EulerModel::NS_SA)
            {
                for (int iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    auto c2f = mesh->cell2face[iCell];
                    for (int ic2f = 0; ic2f < c2f.size(); ic2f++)
                    {
                        index iFace = c2f[ic2f];
                        if (mesh->GetFaceZone(iFace) == Geom::BC_ID_DEFAULT_WALL)
                            u[iCell](I4 + 1) *= 1.0; // ! not fixing first layer!
                    }
                }
            }

            switch (settings.specialBuiltinInitializer)
            {
            case 1: // for RT problem
                DNDS_assert(model == NS || model == NS_2D || model == NS_3D);
                if constexpr (model == NS || model == NS_2D)
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    {
                        Geom::tPoint pos = vfv->GetCellBary(iCell);
                        real gamma = settings.idealGasProperty.gamma;
                        real rho = 2;
                        real p = 1 + 2 * pos(1);
                        if (pos(1) >= 0.5)
                        {
                            rho = 1;
                            p = 1.5 + pos(1);
                        }
                        real v = -0.025 * sqrt(gamma * p / rho) * std::cos(8 * pi * pos(0));
                        if constexpr (dim == 3)
                            u[iCell] = Eigen::Vector<real, 5>{rho, 0, rho * v, 0, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                        else
                            u[iCell] = Eigen::Vector<real, 4>{rho, 0, rho * v, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                    }
                else if constexpr (model == NS_3D)
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    {
                        Geom::tPoint pos = vfv->GetCellBary(iCell);
                        real gamma = settings.idealGasProperty.gamma;
                        real rho = 2;
                        real p = 1 + 2 * pos(1);
                        if (pos(1) >= 0.5)
                        {
                            rho = 1;
                            p = 1.5 + pos(1);
                        }
                        real v = -0.025 * sqrt(gamma * p / rho) * std::cos(8 * pi * pos(0)) * std::cos(8 * pi * pos(2));
                        u[iCell] = Eigen::Vector<real, 5>{rho, 0, rho * v, 0, 0.5 * rho * sqr(v) + p / (gamma - 1)};
                    }
                break;
            case 2: // for IV10 problem
                DNDS_assert(model == NS || model == NS_2D);
                if constexpr (model == NS || model == NS_2D)
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    {
                        Geom::tPoint pos = vfv->GetCellBary(iCell);
                        real chi = 5;
                        real gamma = settings.idealGasProperty.gamma;
                        auto c2n = mesh->cell2node[iCell];
                        auto gCell = vfv->GetCellQuad(iCell);
                        TU um;
                        um.setZero();
                        // Eigen::MatrixXd coords;
                        // mesh->GetCoords(c2n, coords);
                        gCell.IntegrationSimple(
                            um,
                            [&](TU &inc, int ig)
                            {
                                // std::cout << coords<< std::endl << std::endl;
                                // std::cout << DiNj << std::endl;
                                Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);
                                real rm = 8;
                                real r = std::sqrt(sqr(pPhysics(0) - 5) + sqr(pPhysics(1) - 5));
                                real dT = -(gamma - 1) / (8 * gamma * sqr(pi)) * sqr(chi) * std::exp(1 - sqr(r)) * (1 - 1. / std::exp(std::max(sqr(rm) - sqr(r), 0.0)));
                                real dux = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * -(pPhysics(1) - 5) * (1 - 1. / std::exp(std::max(sqr(rm) - sqr(r), 0.0)));
                                real duy = chi / 2 / pi * std::exp((1 - sqr(r)) / 2) * +(pPhysics(0) - 5) * (1 - 1. / std::exp(std::max(sqr(rm) - sqr(r), 0.0)));
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
                        u[iCell] = um / vfv->GetCellVol(iCell); // mean value
                    }
            case 3: // for taylor-green vortex problem
                DNDS_assert(model == NS_3D);
                if constexpr (model == NS_3D)
                {
                    for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                    {
                        Geom::tPoint pos = vfv->GetCellBary(iCell);
                        real M0 = 0.1;
                        real gamma = settings.idealGasProperty.gamma;
                        auto c2n = mesh->cell2node[iCell];
                        auto gCell = vfv->GetCellQuad(iCell);
                        TU um;
                        um.setZero();
                        // Eigen::MatrixXd coords;
                        // mesh->GetCoords(c2n, coords);
                        gCell.IntegrationSimple(
                            um,
                            [&](TU &inc, int ig)
                            {
                                // std::cout << coords<< std::endl << std::endl;
                                // std::cout << DiNj << std::endl;
                                Geom::tPoint pPhysics = vfv->GetCellQuadraturePPhys(iCell, ig);
                                real x{pPhysics(0)}, y{pPhysics(1)}, z{pPhysics(2)};
                                real ux = std::sin(x) * std::cos(y) * std::cos(z);
                                real uy = -std::cos(x) * std::sin(y) * std::cos(z);
                                real p = 1. / (gamma * sqr(M0)) + 1. / 16 * ((std::cos(2 * x) + std::cos(2 * y)) * (2 + std::cos(2 * z)));
                                real rho = gamma * sqr(M0) * p;
                                real E = 0.5 * (sqr(ux) + sqr(uy)) * rho + p / (gamma - 1);

                                // std::cout << T << " " << rho << std::endl;
                                inc.setZero();
                                inc(0) = rho;
                                inc(1) = rho * ux;
                                inc(2) = rho * uy;
                                inc(dim + 1) = E;

                                inc *= vfv->GetCellJacobiDet(iCell, ig); // don't forget this
                            });
                        u[iCell] = um / vfv->GetCellVol(iCell); // mean value
                    }
                }
                break;
            case 0:
                break;
            default:
                log() << "Wrong specialBuiltinInitializer" << std::endl;
                DNDS_assert(false);
                break;
            }

            // Box
            for (auto &i : settings.boxInitializers)
            {
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    if (pos(0) > i.x0 && pos(0) < i.x1 &&
                        pos(1) > i.y0 && pos(1) < i.y1 &&
                        pos(2) > i.z0 && pos(2) < i.z1)
                    {
                        u[iCell] = i.v;
                    }
                }
            }

            // Plane
            for (auto &i : settings.planeInitializers)
            {
                for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
                {
                    Geom::tPoint pos = vfv->GetCellBary(iCell);
                    if (pos(0) * i.a + pos(1) * i.b + pos(2) * i.c + i.h > 0)
                    {
                        // std::cout << pos << std::endl << i.a << i.b << std::endl << i.h <<std::endl;
                        // DNDS_assert(false);
                        u[iCell] = i.v;
                    }
                }
            }
        }
    };
}