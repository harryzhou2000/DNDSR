#pragma once
#include <json.hpp>

#include "Euler.hpp"
#include "Gas.hpp"
#include "DNDS/JsonUtil.hpp"

namespace DNDS::Euler
{
    template <EulerModel model>
    struct EulerEvaluatorSettings
    {
        static const int nVarsFixed = getnVarsFixed(model);
        static const int dim = getDim_Fixed(model);
        static const int gDim = getGeomDim_Fixed(model);
        static const auto I4 = dim + 1;
        
        nlohmann::ordered_json jsonSettings;

        Gas::RiemannSolverType rsType = Gas::Roe;
        Gas::RiemannSolverType rsTypeAux = Gas::UnknownRS;
        int rsMeanValueEig = 0;
        int nCentralSmoothStep = 0;
        int rsRotateScheme = 0;

        struct IdealGasProperty
        {
            real gamma = 1.4;
            real Rgas = 1;
            real muGas = 1;
            real prGas = 0.72;
            real CpGas = Rgas * gamma / (gamma - 1);
            real TRef = 273.15;
            real CSutherland = 110.4;
            int muModel = 1; // 0=constant, 1=sutherland, 2=constant_nu

            void ReadWriteJSON(nlohmann::ordered_json &jsonObj, bool read)
            {
                __DNDS__json_to_config(gamma);
                __DNDS__json_to_config(Rgas);
                __DNDS__json_to_config(muGas);
                __DNDS__json_to_config(prGas);
                __DNDS__json_to_config(TRef);
                __DNDS__json_to_config(CSutherland);
                __DNDS__json_to_config(muModel);
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
                    DNDS_assert_info(v.size() == nVars, "initial value dimension incorrect");
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
                    DNDS_assert_info(v.size() == nVars, "initial value dimension incorrect");
                // if (read)
                //     v = JsonGetEigenVector(jsonObj["v"]), DNDS_assert(v.size() == nVars);
                // else
                //     jsonObj["v"] = EigenVectorGetJson(v);
            }
        };

        std::vector<PlaneInitializer> planeInitializers;

        int specialBuiltinInitializer = 0;

        real uRecBetaCompressPower = 11;
        real uRecAlphaCompressPower = 2;

        Eigen::Vector<real, 3> constMassForce = Eigen::Vector<real, 3>{0, 0, 0};

        bool ignoreSourceTerm = false;
        bool useScalarJacobian = false;

        Eigen::Vector<real, -1> refU;
        Eigen::Vector<real, -1> refUPrim;

        /***************************************************************************************************/
        /***************************************************************************************************/

        void ReadWriteJSON(nlohmann::ordered_json &jsonObj, int nVars, bool read)
        {

            //********* root entries

            __DNDS__json_to_config(useScalarJacobian);
            __DNDS__json_to_config(ignoreSourceTerm);
            __DNDS__json_to_config(specialBuiltinInitializer);
            __DNDS__json_to_config(uRecAlphaCompressPower);
            __DNDS__json_to_config(uRecBetaCompressPower);
            Gas::RiemannSolverType riemannSolverType = rsType;
            __DNDS__json_to_config(riemannSolverType);
            rsType = riemannSolverType;
            Gas::RiemannSolverType riemannSolverTypeAux = rsTypeAux;
            __DNDS__json_to_config(riemannSolverTypeAux);
            rsTypeAux = riemannSolverTypeAux;
            // std::cout << rsType << std::endl;
            __DNDS__json_to_config(rsMeanValueEig);
            __DNDS__json_to_config(rsRotateScheme);
            __DNDS__json_to_config(nCentralSmoothStep);
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

            if (read)
            {
                DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
                refU = farFieldStaticValue;
                refUPrim = refU;
                Gas::IdealGasThermalConservative2Primitive<dim>(refU, refUPrim, idealGasProperty.gamma);
                refU(Seq123).setConstant(refU(Seq123).norm());
                refUPrim(Seq123).setConstant(refUPrim(Seq123).norm());
            }
        }
    };
}