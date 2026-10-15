#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "Euler/LimitedVR.hpp"
#include "doctest.h"

using namespace DNDS::Euler;

TEST_CASE("Weak LVR branch preserves uniform-strip quadratics and rejects contacts and expansion")
{
    LimitedVRSettings s;
    s.weakAlphaMax = 1;
    // q=1+x^2, centroids -h/2,+h/2 and a shared face at zero.
    CHECK(EvaluateLimitedVRWeakAlpha(1.0025, 1.0025, -0.005, -0.005,
                                     0.1, 0, 1, s) == 0);
    // Shifted quadratic: the two primitive-gradient increments integrate the mean difference exactly.
    CHECK(EvaluateLimitedVRWeakAlpha(1.0225, 1.0025, -0.015, 0.005,
                                     0.1, 0, 1, s) == 0);
    // For p=x^3 on a unit uniform strip, cell means adjacent to x=0 are
    // -1/4 and +1/4 and the two centered-GG face increments are +1 and -1.
    // Adding a positive constant leaves r=|1/2-1-1|/(1/2+1+1)=0.6.
    CHECK(EvaluateLimitedVRWeakAlpha(1.75, 2.25, 1, -1,
                                     1, 0, 1, s) == doctest::Approx(1));
    CHECK(EvaluateLimitedVRWeakAlpha(1, 1, 0, 0, 1, 1, 10, s) == 0);
    CHECK(EvaluateLimitedVRWeakAlpha(1.01, 1, -0.0025, 0.0025,
                                     0, 0.01, 1, s) == 0);
}

TEST_CASE("Weak LVR branch detects rescaled shocks above its explicit noise floor")
{
    LimitedVRSettings s;
    s.weakAlphaMax = 1;
    for (DNDS::real jump : {1e-2, 1e-6})
        CHECK(EvaluateLimitedVRWeakAlpha(1 + jump, 1, -.25 * jump, .25 * jump,
                                         jump, 0, 1, s) == doctest::Approx(1));
    CHECK(EvaluateLimitedVRWeakAlpha(1 + 1e-12, 1, 0, 0, 1e-12, 0, 1, s) == 0);
    s.weakAlphaMax = 0;
    CHECK(EvaluateLimitedVRWeakAlpha(1.01, 1, 0, 0, .01, 0, 1, s) == 0);
}

TEST_CASE("Limited variational reconstruction cubic ramp has exact endpoints and bounded transition")
{
    CHECK(LimitedVRCubicRamp(0.0, 0.1, 0.3) == 0.0);
    CHECK(LimitedVRCubicRamp(0.1, 0.1, 0.3) == 0.0);
    CHECK(LimitedVRCubicRamp(0.2, 0.1, 0.3) == doctest::Approx(0.5));
    CHECK(LimitedVRCubicRamp(0.3, 0.1, 0.3) == 1.0);
    CHECK(LimitedVRCubicRamp(1.0, 0.1, 0.3) == 1.0);
    CHECK_THROWS(LimitedVRCubicRamp(0.2, 0.3, 0.3));
}

TEST_CASE("Weak LVR alpha relaxation preserves its fixed point and damps updates")
{
    CHECK(IsLimitedVRStageStartIteration(1));
    CHECK(IsLimitedVRStageStartIteration(0));
    CHECK(IsLimitedVRStageStartIteration(std::numeric_limits<int>::max()));
    CHECK_FALSE(IsLimitedVRStageStartIteration(2));
    CHECK(RelaxLimitedVRAlpha(0.2, 0.8, 1.0) == doctest::Approx(0.8));
    CHECK(RelaxLimitedVRAlpha(0.2, 0.8, 0.5) == doctest::Approx(0.5));
    CHECK(RelaxLimitedVRAlpha(0.2, 0.8, 0.25) == doctest::Approx(0.35));
    CHECK(RelaxLimitedVRAlpha(0.6, 0.6, 0.25) == doctest::Approx(0.6));
    CHECK_THROWS(RelaxLimitedVRAlpha(0.2, 0.8, 0.0));
    CHECK_THROWS(RelaxLimitedVRAlpha(0.2, 0.8, 1.1));
    CHECK(UpdateLimitedVRWeakAlpha(0.2, 0.8, false, false, 1.0) == doctest::Approx(0.2));
    CHECK(UpdateLimitedVRWeakAlpha(0.2, 0.8, true, false, 1.0) == doctest::Approx(0.8));
    CHECK(UpdateLimitedVRWeakAlpha(0.2, 0.8, true, false, 0.5) == doctest::Approx(0.5));
    CHECK(UpdateLimitedVRWeakAlpha(0.2, 0.8, true, false, 1.0, 0.9) == doctest::Approx(0.8));
    CHECK(UpdateLimitedVRWeakAlpha(0.8, 0.2, true, false, 1.0, 0.9) == doctest::Approx(0.74));
    CHECK(UpdateLimitedVRWeakAlpha(0.8, 0.2, false, false, 1.0, 0.9) == doctest::Approx(0.8));
    CHECK(UpdateLimitedVRWeakAlpha(0.8, 0.2, true, true, 1.0, 0.9) == doctest::Approx(0.2));
    CHECK(UpdateLimitedVRWeakAlpha(0.8, 0.2, true, true, 0.5, 0.0) == doctest::Approx(0.2));
    CHECK_THROWS(UpdateLimitedVRWeakAlpha(0.8, 0.2, true, false, 1.0, 1.0));
}

TEST_CASE("Limited variational reconstruction sensor rejects contacts shear and expansions")
{
    LimitedVRSettings settings;
    CHECK(EvaluateLimitedVRFaceSensor(1.01, 1.0, 0.01, 0.0, 1.0, 1.0, settings).alpha == 0.0);
    CHECK(EvaluateLimitedVRFaceSensor(1.0, 1.0, 1.0, 0.0, 1.0, 1.0, settings).alpha == 0.0);
    CHECK(EvaluateLimitedVRFaceSensor(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, settings).alpha == 0.0);
    CHECK(EvaluateLimitedVRFaceSensor(1.0, 2.0, 0.0, 1.0, 1.0, 1.0, settings).alpha == 0.0);
}

TEST_CASE("Limited variational reconstruction sensor has a bounded intermediate response")
{
    LimitedVRSettings settings;
    auto sensor = EvaluateLimitedVRFaceSensor(1.15, 1.0, 0.12, 0.0, 1.0, 1.0, settings);
    CHECK(sensor.pressureJump > settings.pressureJumpStart);
    CHECK(sensor.compression > settings.compressionStart);
    CHECK(sensor.alpha > 0.0);
    CHECK(sensor.alpha < settings.alphaMax);
}

TEST_CASE("Limited variational reconstruction alternative gates cover pressure-only shock tails")
{
    LimitedVRSettings settings;
    settings.pressureJumpStart = 0.005;
    settings.pressureJumpFull = 0.12;
    settings.compressionStart = 0.005;
    settings.compressionFull = 0.09;

    settings.gateMode = LimitedVRSettings::GateProductCubic;
    CHECK(EvaluateLimitedVRFaceSensor(1.04, 1.0, 0.0, 0.0, 1.0, 1.0, settings).alpha == 0.0);

    settings.gateMode = LimitedVRSettings::GateUnionCubic;
    auto unionSensor = EvaluateLimitedVRFaceSensor(1.04, 1.0, 0.0, 0.0, 1.0, 1.0, settings);
    CHECK(unionSensor.alpha > 0.0);
    CHECK(unionSensor.alpha < settings.alphaMax);

    settings.gateMode = LimitedVRSettings::GateCombinedRational;
    auto rationalSensor = EvaluateLimitedVRFaceSensor(1.04, 1.0, 0.0, 0.0, 1.0, 1.0, settings);
    CHECK(rationalSensor.alpha > 0.0);
    CHECK(rationalSensor.alpha < settings.alphaMax);

    CHECK(EvaluateLimitedVRFaceSensor(1.001, 1.0, 0.0, 0.0, 1.0, 1.0, settings).alpha == 0.0);
}

TEST_CASE("Limited variational reconstruction coupled rational gate requires both indicators")
{
    LimitedVRSettings settings;
    settings.pressureJumpStart = 0.035;
    settings.pressureJumpFull = 0.16;
    settings.compressionStart = 0.025;
    settings.compressionFull = 0.13;
    settings.alphaMax = 0.4;
    settings.gateMode = LimitedVRSettings::GateCoupledRational;

    auto pressureOnly = EvaluateLimitedVRFaceSensor(1.04, 1.0, 0.0, 0.0, 1.0, 1.0, settings);
    CHECK(pressureOnly.pressureJump > settings.pressureJumpStart);
    CHECK(pressureOnly.compression == 0.0);
    CHECK(pressureOnly.alpha == 0.0);

    auto coupledTail = EvaluateLimitedVRFaceSensor(1.04, 1.0, 0.04, 0.0, 1.0, 1.0, settings);
    CHECK(coupledTail.pressureJump > settings.pressureJumpStart);
    CHECK(coupledTail.compression > settings.compressionStart);
    CHECK(coupledTail.alpha > 0.0);
    CHECK(coupledTail.alpha < settings.alphaMax);
}

TEST_CASE("Limited variational reconstruction saturated rational gate has a flat shock core")
{
    LimitedVRSettings settings;
    settings.pressureJumpStart = 0.025;
    settings.pressureJumpFull = 0.12;
    settings.compressionStart = 0.01;
    settings.compressionFull = 0.09;
    settings.alphaMax = 0.4;
    settings.gateMode = LimitedVRSettings::GateCoupledRationalSaturated;

    auto tail = EvaluateLimitedVRFaceSensor(1.04, 1.0, 0.04, 0.0, 1.0, 1.0, settings);
    CHECK(tail.alpha > 0.0);
    CHECK(tail.alpha < settings.alphaMax);

    auto core = EvaluateLimitedVRFaceSensor(1.3, 1.0, 0.3, 0.0, 1.0, 1.0, settings);
    CHECK(core.alpha == doctest::Approx(settings.alphaMax));
}

TEST_CASE("Limited variational reconstruction compressive shock saturates and is orientation invariant")
{
    LimitedVRSettings settings;
    auto forward = EvaluateLimitedVRFaceSensor(10.0, 1.0, 2.0, 0.0, 1.0, 1.0, settings);
    auto reversed = EvaluateLimitedVRFaceSensor(1.0, 10.0, 0.0, -2.0, 1.0, 1.0, settings);
    CHECK(forward.alpha == doctest::Approx(settings.alphaMax));
    CHECK(reversed.alpha == doctest::Approx(forward.alpha));
    CHECK(reversed.pressureJump == doctest::Approx(forward.pressureJump));
    CHECK(reversed.compression == doctest::Approx(forward.compression));
}

TEST_CASE("Limited variational reconstruction density-only contact gate is separate and orientation invariant")
{
    LimitedVRSettings settings;
    settings.contactDensityJumpStart = 0.2;
    settings.contactDensityJumpFull = 1.0;
    settings.contactAlphaMax = 0.1;

    auto below = EvaluateLimitedVRContactFaceSensor(1.0, 1.1, settings);
    CHECK(below.densityJump < settings.contactDensityJumpStart);
    CHECK(below.alpha == 0.0);

    auto forward = EvaluateLimitedVRContactFaceSensor(1.0, 2.0, settings);
    auto reversed = EvaluateLimitedVRContactFaceSensor(2.0, 1.0, settings);
    CHECK(forward.densityJump == doctest::Approx(2.0 / 3.0));
    CHECK(forward.alpha > 0.0);
    CHECK(forward.alpha < settings.contactAlphaMax);
    CHECK(reversed.densityJump == doctest::Approx(forward.densityJump));
    CHECK(reversed.alpha == doctest::Approx(forward.alpha));

    auto saturated = EvaluateLimitedVRContactFaceSensor(1.0, 10.0, settings);
    CHECK(saturated.alpha == doctest::Approx(settings.contactAlphaMax));

    auto shockSensor = EvaluateLimitedVRFaceSensor(1.0, 1.0, 0.5, 0.5, 1.0, 1.0, settings);
    CHECK(shockSensor.alpha == 0.0);
    CHECK(forward.alpha > shockSensor.alpha);
}

TEST_CASE("Limited variational reconstruction contact gate can be disabled exactly")
{
    LimitedVRSettings settings;
    settings.contactAlphaMax = 0.0;
    auto sensor = EvaluateLimitedVRContactFaceSensor(1.0, 100.0, settings);
    CHECK(sensor.densityJump > settings.contactDensityJumpFull);
    CHECK(sensor.alpha == 0.0);
}

TEST_CASE("Limited variational reconstruction combines shock and contact gates by maximum")
{
    CHECK(CombineLimitedVRAlpha(0.4, 0.1) == doctest::Approx(0.4));
    CHECK(CombineLimitedVRAlpha(0.1, 0.4) == doctest::Approx(0.4));
    CHECK(CombineLimitedVRAlpha(0.0, 0.0) == 0.0);
}

TEST_CASE("Limited variational reconstruction settings parse and validate")
{
    nlohmann::ordered_json config = LimitedVRSettings{};
    CHECK(config["o2ReferenceLimiter"] == LimitedVRSettings::O2ReferenceLimiterBarth);
    CHECK(config["gateMode"] == LimitedVRSettings::GateProductCubic);
    CHECK(config["gateHaloLayers"] == 0);
    CHECK(config["gateHaloSaturatedOnly"] == false);
    CHECK(config["contactDensityJumpStart"] == doctest::Approx(0.2));
    CHECK(config["contactDensityJumpFull"] == doctest::Approx(0.8));
    CHECK(config["contactAlphaMax"] == 0.0);
    CHECK(config["contactUseCellMeanJump"] == false);
    CHECK(config["contactUseExtremumCorrection"] == false);
    CHECK(config["contactExtremumUseCellMeanSupport"] == false);
    CHECK(config["contactExtremumStart"] == doctest::Approx(0.001));
    CHECK(config["contactExtremumFull"] == doctest::Approx(0.008));
    CHECK(config["contactSupportStart"] == doctest::Approx(0.02));
    CHECK(config["contactSupportFull"] == doctest::Approx(0.15));
    CHECK(config["contactExtremumRetention"] == doctest::Approx(0.95));
    CHECK(config["contactExtremumHaloLayers"] == 1);
    CHECK(config["alphaUpdateRelaxation"] == 1.0);
    CHECK(config["weakAlphaUpdateRelaxation"] == 1.0);
    CHECK(config["weakAlphaDecayRetention"] == 0.0);
    CHECK(config["weakFreezeWithinImplicitStage"] == false);
    CHECK(config["oscillationExtremumStart"] == doctest::Approx(0.001));
    CHECK(config["oscillationExtremumFull"] == doctest::Approx(0.008));
    CHECK(config["oscillationSupportStart"] == doctest::Approx(0.02));
    CHECK(config["oscillationSupportFull"] == doctest::Approx(0.15));
    CHECK(config["oscillationRetention"] == doctest::Approx(0.95));
    CHECK(config["oscillationTransverseTolerance"] == doctest::Approx(1e-8));
    CHECK(config["oscillationMinTransverseNeighbours"] == 2);
    CHECK(config["o2ReferenceUsePP"] == true);
    config["alphaMax"] = 1.0;
    config["pressureJumpStart"] = 0.05;
    config["pressureJumpFull"] = 0.50;
    auto settings = config.get<LimitedVRSettings>();
    CHECK(settings.alphaMax == 1.0);
    CHECK(settings.o2ReferenceLimiter == LimitedVRSettings::O2ReferenceLimiterBarth);
    CHECK(settings.o2ReferenceUsePP);
    CHECK(settings.validate().empty());

    config["o2ReferenceLimiter"] = LimitedVRSettings::O2ReferenceLimiterWBAP;
    config["o2ReferenceUsePP"] = false;
    config["contactUseCellMeanJump"] = true;
    config["contactUseExtremumCorrection"] = true;
    config["contactExtremumUseCellMeanSupport"] = true;
    config["contactExtremumHaloLayers"] = 2;
    settings = config.get<LimitedVRSettings>();
    CHECK(settings.o2ReferenceLimiter == LimitedVRSettings::O2ReferenceLimiterWBAP);
    CHECK_FALSE(settings.o2ReferenceUsePP);
    CHECK(settings.contactUseCellMeanJump);
    CHECK(settings.contactUseExtremumCorrection);
    CHECK(settings.contactExtremumUseCellMeanSupport);
    CHECK(settings.contactExtremumHaloLayers == 2);
    CHECK(settings.validate().empty());

    config["o2ReferenceLimiter"] = 2;
    CHECK_THROWS(config.get<LimitedVRSettings>());

    config["o2ReferenceLimiter"] = LimitedVRSettings::O2ReferenceLimiterBarth;
    config["gateMode"] = 9;
    CHECK_THROWS(config.get<LimitedVRSettings>());

    config["gateMode"] = LimitedVRSettings::GateProductCubic;
    config["gateHaloLayers"] = 5;
    CHECK_THROWS(config.get<LimitedVRSettings>());

    config["gateHaloLayers"] = 0;
    config["alphaUpdateRelaxation"] = 0.0;
    settings = config.get<LimitedVRSettings>();
    CHECK_FALSE(settings.validate().empty());

    config["alphaUpdateRelaxation"] = 1.0;
    config["weakAlphaUpdateRelaxation"] = 0.0;
    settings = config.get<LimitedVRSettings>();
    CHECK_FALSE(settings.validate().empty());

    config["weakAlphaUpdateRelaxation"] = 1.0;
    config["pressureJumpFull"] = 0.05;
    settings = config.get<LimitedVRSettings>();
    CHECK_FALSE(settings.validate().empty());

    config["pressureJumpFull"] = 0.50;
    config["contactDensityJumpStart"] = 0.5;
    config["contactDensityJumpFull"] = 0.5;
    settings = config.get<LimitedVRSettings>();
    CHECK_FALSE(settings.validate().empty());

    config["contactDensityJumpStart"] = 0.2;
    config["contactDensityJumpFull"] = 0.8;
    config["contactExtremumHaloLayers"] = 5;
    CHECK_THROWS(config.get<LimitedVRSettings>());

    config["contactExtremumHaloLayers"] = 1;
    config["contactSupportFull"] = config["contactSupportStart"];
    settings = config.get<LimitedVRSettings>();
    CHECK_FALSE(settings.validate().empty());
}

TEST_CASE("LVR input presets expand before explicit overrides and preserve legacy input")
{
    nlohmann::ordered_json layer = {{"limiterControl", {{"limitedVR", {{"preset", "v3"}, {"alphaMax", 1.0}, {"contactAlphaMax", 1.0}}}}}};
    ExpandLimitedVRPreset(layer);
    const auto &section = layer["limiterControl"]["limitedVR"];
    CHECK_FALSE(section.contains("preset"));
    auto settings = section.get<LimitedVRSettings>();
    CHECK(settings.alphaMax == 1);
    CHECK(settings.contactAlphaMax == 1);
    CHECK(settings.gateMode == LimitedVRSettings::GateProductCubicWithExtremumCorrection);
    CHECK(settings.o2ReferenceLimiter == LimitedVRSettings::O2ReferenceLimiterWBAP);
    CHECK(settings.contactExtremumHaloLayers == 2);
    CHECK(settings.weakAlphaMax == 0);
    CHECK(settings.validate().empty());
    const auto resolved = layer;
    ExpandLimitedVRPreset(layer);
    CHECK(layer == resolved);
    layer["limiterControl"]["limitedVR"]["preset"] = "unknown";
    CHECK_THROWS(ExpandLimitedVRPreset(layer));
}
