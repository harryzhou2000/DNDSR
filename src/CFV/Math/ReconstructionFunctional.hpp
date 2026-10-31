#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Config/ConfigEnum.hpp"
#include "Geom/Geometric.hpp"

namespace DNDS::CFV
{
    /// Isotropic functional configuration, independent of the polynomial basis
    /// and of mesh/field ownership. Non-unit dirWeightCombPowV is a legacy
    /// compatibility metric and does not promise rotational invariance.
    struct ReconstructionFunctionalSettings
    {
        enum class DirectionalWeight
        {
            Invalid,
            Factorial,
            HQM_OPT,
            TEST_OPT,
            ManualDirWeight
        };
        enum class GeometricWeight
        {
            Invalid,
            GWNone,
            HQM_SD,
            SD_Power
        };
        enum class LengthScale
        {
            Invalid,
            BaryDiff,
            MeanAACBB,
            CellMax
        };
        DirectionalWeight dirWeightScheme = DirectionalWeight::Factorial;
        GeometricWeight geomWeightScheme = GeometricWeight::GWNone;
        LengthScale scaleType = LengthScale::BaryDiff;
        std::array<real, 4> manualDirWeights{1, 1, 0.5, 1. / 6};
        real scaleMultiplier = 1, geomWeightBias = 0, geomWeightPower = 0.5, geomWeightPower1 = 0, geomWeightPower2 = 0;
        int dirWeightCombPowV = 1;
        DNDS_DECLARE_CONFIG(ReconstructionFunctionalSettings)
        {
            DNDS_FIELD(dirWeightScheme, "Factorial/HQM_OPT/TEST_OPT/manual derivative weights");
            DNDS_FIELD(geomWeightScheme, "GWNone/HQM_SD/SD_Power geometric weights");
            DNDS_FIELD(scaleType, "Isotropic functional length selection");
            DNDS_FIELD(manualDirWeights, "Unsquared derivative weights by order");
            DNDS_FIELD(scaleMultiplier, "Multiplier on the functional derivative length");
            DNDS_FIELD(geomWeightBias, "Geometric weight bias");
            DNDS_FIELD(geomWeightPower, "HQM_SD geometric power");
            DNDS_FIELD(geomWeightPower1, "SD_Power face measure exponent");
            DNDS_FIELD(geomWeightPower2, "SD_Power barycenter distance exponent");
            DNDS_FIELD(dirWeightCombPowV, "Legacy tensor multiplicity exponent; one is Frobenius invariant");
        }
        void Validate() const
        {
            DNDS_check_throw_info(int(dirWeightScheme) >= int(DirectionalWeight::Factorial) && int(dirWeightScheme) <= int(DirectionalWeight::ManualDirWeight) &&
                                      int(geomWeightScheme) >= int(GeometricWeight::GWNone) && int(geomWeightScheme) <= int(GeometricWeight::SD_Power) &&
                                      int(scaleType) >= int(LengthScale::BaryDiff) && int(scaleType) <= int(LengthScale::CellMax) &&
                                      std::isfinite(scaleMultiplier) && scaleMultiplier > 0 &&
                                      std::isfinite(geomWeightBias) && std::isfinite(geomWeightPower) &&
                                      std::isfinite(geomWeightPower1) && std::isfinite(geomWeightPower2),
                                  "invalid isotropic functional configuration");
            for (auto weight : manualDirWeights)
                DNDS_check_throw_info(std::isfinite(weight) && weight >= 0, "invalid manual derivative weight");
        }
    };
    NLOHMANN_JSON_SERIALIZE_ENUM(ReconstructionFunctionalSettings::DirectionalWeight,
                                 {{ReconstructionFunctionalSettings::DirectionalWeight::Invalid, nullptr},
                                  {ReconstructionFunctionalSettings::DirectionalWeight::Factorial, "Factorial"},
                                  {ReconstructionFunctionalSettings::DirectionalWeight::HQM_OPT, "HQM_OPT"},
                                  {ReconstructionFunctionalSettings::DirectionalWeight::TEST_OPT, "TEST_OPT"},
                                  {ReconstructionFunctionalSettings::DirectionalWeight::ManualDirWeight, "ManualDirWeight"}})
    NLOHMANN_JSON_SERIALIZE_ENUM(ReconstructionFunctionalSettings::GeometricWeight,
                                 {{ReconstructionFunctionalSettings::GeometricWeight::Invalid, nullptr},
                                  {ReconstructionFunctionalSettings::GeometricWeight::GWNone, "GWNone"},
                                  {ReconstructionFunctionalSettings::GeometricWeight::HQM_SD, "HQM_SD"},
                                  {ReconstructionFunctionalSettings::GeometricWeight::SD_Power, "SD_Power"}})
    NLOHMANN_JSON_SERIALIZE_ENUM(ReconstructionFunctionalSettings::LengthScale,
                                 {{ReconstructionFunctionalSettings::LengthScale::Invalid, nullptr},
                                  {ReconstructionFunctionalSettings::LengthScale::BaryDiff, "BaryDiff"},
                                  {ReconstructionFunctionalSettings::LengthScale::MeanAACBB, "MeanAACBB"},
                                  {ReconstructionFunctionalSettings::LengthScale::CellMax, "CellMax"}})

    namespace Math
    {
        /// Unsquared per-order weights, matching the legacy isotropic schemes.
        /// TEST_OPT's span inputs are face-major lengths, not cell axes.
        inline std::array<real, 4> ReconstructionOrderWeights(const ReconstructionFunctionalSettings &settings,
                                                              int dim, int degree, real area, real distance, const Geom::tPoint &faceSpan)
        {
            using S = ReconstructionFunctionalSettings;
            settings.Validate();
            DNDS_check_throw_info((dim == 2 || dim == 3) && degree >= 1 && degree <= 3 &&
                                      area > 0 && distance > 0 && std::isfinite(area) && std::isfinite(distance),
                                  "invalid functional face geometry");
            std::array<real, 4> weights{1, 1, 0.5, 1. / 6};
            real geometric = 1;
            if (settings.geomWeightScheme == S::GeometricWeight::HQM_SD)
                geometric = settings.geomWeightBias + std::pow(std::pow(area, 1. / (dim - 1)) / distance, 0.5 * settings.geomWeightPower);
            else if (settings.geomWeightScheme == S::GeometricWeight::SD_Power)
                geometric = settings.geomWeightBias + std::pow(area, 0.5 * settings.geomWeightPower1) * std::pow(distance, 0.5 * settings.geomWeightPower2);
            if (settings.dirWeightScheme == S::DirectionalWeight::HQM_OPT)
            {
                if (degree == 2)
                    weights[1] = 0.4643, weights[2] = 0.1559;
                if (degree == 3)
                    weights[1] = 0.5295, weights[2] = weights[3] = 0.2117;
            }
            else if (settings.dirWeightScheme == S::DirectionalWeight::ManualDirWeight ||
                     (settings.dirWeightScheme == S::DirectionalWeight::TEST_OPT && degree != 3))
                weights = settings.manualDirWeights;
            else if (settings.dirWeightScheme == S::DirectionalWeight::TEST_OPT)
            {
                geometric = 1; // Legacy TEST_OPT includes its own geometric fit.
                if (dim == 2)
                {
                    const real ratio = area / distance, logarithm = std::log10(ratio);
                    weights[0] = std::sqrt(std::pow(1.00745 + ratio, 1.004));
                    weights[1] = weights[0] * std::sqrt(0.359 - 0.0674 * std::exp(-sqr(logarithm - 5.18e-11) / (2 * sqr(0.124))));
                    weights[2] = weights[0] * std::sqrt(0.0676 - 0.0507 * std::exp(-sqr(logarithm - 3.15e-10) / (2 * sqr(0.137))));
                    weights[3] = weights[0] * std::sqrt(0.0529 - 0.0487 * std::exp(-sqr(logarithm - 1.33e-10) / (2 * sqr(0.167))));
                }
                else
                {
                    const real a = faceSpan[0] / distance, b = faceSpan[1] / distance, logarithm = std::log10(std::hypot(a, b));
                    DNDS_check_throw_info(a > 0 && b > 0, "TEST_OPT requires two nonzero face spans");
                    weights[0] = std::sqrt(std::pow(1.0031 + 0.86 * std::pow(a * b, 0.5327), 0.98132));
                    weights[1] = weights[0] * std::sqrt(0.3604 - 0.0774 * std::exp(-0.5 * sqr((logarithm - 1.8e-10) / 0.1325)));
                    weights[2] = weights[0] * std::sqrt(0.0676 - 0.0547 * std::exp(-0.5 * sqr((logarithm + 2.74e-10) / 0.1376)));
                    weights[3] = weights[0] * std::sqrt(5.299e-2 - 4.63e-2 * std::exp(-0.5 * sqr((logarithm - 9.028e-12) / 0.175)));
                }
            }
            for (auto &weight : weights)
            {
                weight *= geometric;
                DNDS_check_throw_info(std::isfinite(weight) && weight >= 0, "functional weight is not finite and nonnegative");
            }
            return weights;
        }
    }
}
