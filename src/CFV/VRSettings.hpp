#pragma once

#include "DNDS/Defines.hpp"
#include "DNDS/MPI.hpp"
#include "Geom/Quadrature.hpp"
#include "Geom/Mesh.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatirx.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"

#include "BaseFunction.hpp"
#include "Limiters.hpp"

#define JSON_ASSERT DNDS_assert
#include "json.hpp"

#include "Eigen/Dense"

#include "DNDS/JsonUtil.hpp"

namespace DNDS::CFV
{
    /**
     * @brief
     * A means to translate nlohmann json into c++ primitive data types and back;
     * and stores then during computation.
     *
     */
    struct VRSettings
    {
        using json = nlohmann::ordered_json;

        int maxOrder{3};            /// @brief polynomial degree of reconstruction
        int intOrder{5};            /// @brief integration order globally set @note this is actually reduced somewhat
        bool cacheDiffBase = false; /// @brief if cache the base function values on each of the quadrature points
        uint8_t cacheDiffBaseSize = UINT8_MAX;

        real jacobiRelax = 1.0; /// @brief VR SOR/Jacobi iteration relaxation factor
        bool SORInstead = true; /// @brief use SOR instead of relaxed Jacobi iteration

        real smoothThreshold = 0.01; /// @brief limiter's smooth indicator threshold
        real WBAP_nStd = 10;         /// @brief n used in WBAP limiters
        bool normWBAP = false;       /// @brief if switch to normWBAP
        int limiterBiwayAlter = 0;   /// @brief 0=wbap-L2-biway, 1=minmod-biway
        int subs2ndOrder = 0;        /// @brief 0: vfv; 1: gauss rule; 2: least square

        struct BaseSettings
        {
            bool localOrientation = false;
            bool anisotropicLengths = false;
            DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                BaseSettings,
                localOrientation,
                anisotropicLengths)
        } baseSettings;

        struct FunctionalSettings
        {
            enum ScaleType
            {
                UnknownScale = -1,
                MeanAACBB = 0,
                BaryDiff = 1,
            } scaleType = BaryDiff;

            real scaleMultiplier = 1.0;

            enum DirWeightScheme
            {
                UnknownDirWeight = -1,
                Factorial = 0,
                HQM_OPT = 1,
                ManualDirWeight = 999,
            } dirWeightScheme = Factorial;

            int dirWeightCombPowV = 1;

            Eigen::VectorXd manualDirWeights;

            enum GeomWeightScheme
            {
                UnknownGeomWeight = -1,
                GWNone = 0,
                HQM_SD = 1,
                SD_Power = 2,
            } geomWeightScheme = GWNone;

            real geomWeightBias = 0;
            real geomWeightPower = 0.5;
            real geomWeightPower1 = 0;
            real geomWeightPower2 = 0;

            bool useAnisotropicFunctional = false;

            enum AnisotropicType
            {
                UnknownAnisotropic = -1,
                InertiaCoord = 0,
                InertiaCoordBB = 1,
            } anisotropicType = InertiaCoord;

            real inertiaWeightPower = 1.0;

            real greenGauss1Weight = 0.0;
            real greenGauss1Bias = 0.5;
            real greenGauss1Penalty = 0.0;
            int greenGaussSpacial = 0; // 1 for uniform weight

            DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
                FunctionalSettings,
                scaleType,
                scaleMultiplier,
                dirWeightScheme, dirWeightCombPowV, manualDirWeights,
                geomWeightScheme,
                geomWeightPower, geomWeightPower1, geomWeightPower2,
                useAnisotropicFunctional,
                anisotropicType,
                inertiaWeightPower,
                geomWeightBias,
                greenGauss1Weight,
                greenGauss1Bias,
                greenGauss1Penalty,
                greenGaussSpacial)
            FunctionalSettings()
            {
                manualDirWeights.resize(5);
                manualDirWeights
                    << 1,
                    1, 0.5, 1. / 6, 1. / 24;
            }
        } functionalSettings;

        VRSettings()
        {
        }

        VRSettings(int dim)
        {
            cacheDiffBaseSize = dim + 1;
        }

        /**
         * @brief write any data into jsonSetting member
         *
         */
        void WriteIntoJson(json &jsonSetting) const
        {
            jsonSetting["maxOrder"] = maxOrder;
            jsonSetting["intOrder"] = intOrder;

            jsonSetting["cacheDiffBase"] = cacheDiffBase;
            jsonSetting["cacheDiffBaseSize"] = cacheDiffBaseSize;
            jsonSetting["jacobiRelax"] = jacobiRelax;
            jsonSetting["SORInstead"] = SORInstead;

            jsonSetting["smoothThreshold"] = smoothThreshold;
            jsonSetting["WBAP_nStd"] = WBAP_nStd;
            jsonSetting["normWBAP"] = normWBAP;
            jsonSetting["limiterBiwayAlter"] = limiterBiwayAlter;
            jsonSetting["subs2ndOrder"] = subs2ndOrder;
            jsonSetting["baseSettings"] = baseSettings;
            jsonSetting["functionalSettings"] = functionalSettings;
        }

        /**
         * @brief read any data from jsonSetting member
         *
         */
        void ParseFromJson(const json &jsonSetting)
        {
            maxOrder = jsonSetting["maxOrder"]; ///@todo //TODO: update to better
            intOrder = jsonSetting["intOrder"];
            cacheDiffBase = jsonSetting["cacheDiffBase"];
            cacheDiffBaseSize = jsonSetting["cacheDiffBaseSize"];
            jacobiRelax = jsonSetting["jacobiRelax"];
            SORInstead = jsonSetting["SORInstead"];

            smoothThreshold = jsonSetting["smoothThreshold"];
            WBAP_nStd = jsonSetting["WBAP_nStd"];
            normWBAP = jsonSetting["normWBAP"];
            limiterBiwayAlter = jsonSetting["limiterBiwayAlter"];
            subs2ndOrder = jsonSetting["subs2ndOrder"];
            baseSettings = jsonSetting["baseSettings"];
            functionalSettings = jsonSetting["functionalSettings"];
        }
        friend void from_json(const json &j, VRSettings &s)
        {
            s.ParseFromJson(j);
        }

        friend void to_json(json &j, const VRSettings &s)
        {
            s.WriteIntoJson(j);
        }
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        VRSettings::FunctionalSettings::ScaleType,
        {{VRSettings::FunctionalSettings::UnknownScale, nullptr},
         {VRSettings::FunctionalSettings::MeanAACBB, "MeanAACBB"},
         {VRSettings::FunctionalSettings::BaryDiff, "BaryDiff"}})

    NLOHMANN_JSON_SERIALIZE_ENUM(
        VRSettings::FunctionalSettings::DirWeightScheme,
        {{VRSettings::FunctionalSettings::UnknownDirWeight, nullptr},
         {VRSettings::FunctionalSettings::Factorial, "Factorial"},
         {VRSettings::FunctionalSettings::HQM_OPT, "HQM_OPT"},
         {VRSettings::FunctionalSettings::ManualDirWeight, "ManualDirWeight"}})

    NLOHMANN_JSON_SERIALIZE_ENUM(
        VRSettings::FunctionalSettings::GeomWeightScheme,
        {{VRSettings::FunctionalSettings::UnknownGeomWeight, nullptr},
         {VRSettings::FunctionalSettings::GWNone, "GWNone"},
         {VRSettings::FunctionalSettings::HQM_SD, "HQM_SD"},
         {VRSettings::FunctionalSettings::SD_Power, "SD_Power"}})

    NLOHMANN_JSON_SERIALIZE_ENUM(
        VRSettings::FunctionalSettings::AnisotropicType,
        {{VRSettings::FunctionalSettings::UnknownAnisotropic, nullptr},
         {VRSettings::FunctionalSettings::InertiaCoord, "InertiaCoord"},
         {VRSettings::FunctionalSettings::InertiaCoordBB, "InertiaCoordBB"}})
}