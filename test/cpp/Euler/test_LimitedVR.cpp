#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "Euler/LimitedVR.hpp"
#include "doctest.h"

using namespace DNDS::Euler;

TEST_CASE("LVR selectors write names, accept retained integer aliases and reject removed choices")
{
    using json = nlohmann::ordered_json;
    CHECK(json(ReconstructionLimiter::WBAP3) == "3WBAP");
    CHECK(json(ReconstructionLimiter::CWBAP) == "CWBAP");
    CHECK(json(ReconstructionLimiter::LVR) == "LVR");
    for (const auto choice : {ReconstructionLimiter::WBAP3, ReconstructionLimiter::CWBAP, ReconstructionLimiter::LVR})
    {
        CHECK(json(choice).get<ReconstructionLimiter>() == choice);
        CHECK(json(static_cast<int>(choice)).get<ReconstructionLimiter>() == choice);
    }
    for (const auto choice : {O2ReferenceLimiter::Barth, O2ReferenceLimiter::WBAP})
    {
        CHECK(json(choice).get<O2ReferenceLimiter>() == choice);
        CHECK(json(static_cast<int>(choice)).get<O2ReferenceLimiter>() == choice);
    }
    for (const auto choice : {LimitedVRGateMode::ProductCubic, LimitedVRGateMode::PressureExtremum})
    {
        CHECK(json(choice).get<LimitedVRGateMode>() == choice);
        CHECK(json(static_cast<int>(choice)).get<LimitedVRGateMode>() == choice);
    }
    for (int mode = 1; mode <= 7; ++mode)
        CHECK_THROWS_WITH(json(mode).get<LimitedVRGateMode>(), doctest::Contains("experimental modes 1..7 were removed"));
    for (const json &invalid : {json(nullptr), json(false), json(0.0), json(-1), json(999), json("typo")})
    {
        CHECK_THROWS(invalid.get<ReconstructionLimiter>());
        CHECK_THROWS(invalid.get<O2ReferenceLimiter>());
        CHECK_THROWS(invalid.get<LimitedVRGateMode>());
    }
    CHECK_THROWS(json("GateUnionCubic").get<LimitedVRGateMode>());
    CHECK_THROWS(json(2).get<O2ReferenceLimiter>());
    CHECK_THROWS(json(3).get<ReconstructionLimiter>());

    // Test the registered metadata without regenerating checked-in schema files.
    const auto fields = LimitedVRSettings::schema().at("properties");
    CHECK(fields.at("gateMode").at("type") == "string");
    CHECK(fields.at("gateMode").at("enum") == json::array({"ProductCubic", "PressureExtremum"}));
    CHECK(fields.at("o2ReferenceLimiter").at("enum") == json::array({"Barth", "WBAP"}));
    CHECK(fields.at("penaltyTarget").at("enum") == json::array({"O2", "3WBAP", "CWBAP"}));
    for (const auto choice : {LimitedVRTarget::O2, LimitedVRTarget::WBAP3, LimitedVRTarget::CWBAP})
        CHECK(json(choice).get<LimitedVRTarget>() == choice);
    CHECK_THROWS(json("unknown").get<LimitedVRTarget>());
    json legacy = LimitedVRSettings{};
    legacy["gateMode"] = 8;
    legacy["o2ReferenceLimiter"] = 1;
    json saved = legacy.get<LimitedVRSettings>();
    CHECK(saved.at("gateMode") == "PressureExtremum");
    CHECK(saved.at("o2ReferenceLimiter") == "WBAP");
}

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
    CHECK(config["o2ReferenceLimiter"] == "WBAP");
    CHECK(config["gateMode"] == "PressureExtremum");
    CHECK(config["gateHaloLayers"] == 0);
    CHECK(config["gateHaloSaturatedOnly"] == false);
    CHECK(config["contactDensityJumpStart"] == doctest::Approx(0.5));
    CHECK(config["contactDensityJumpFull"] == doctest::Approx(0.8));
    CHECK(config["contactAlphaMax"] == 1.0);
    CHECK(config["contactUseCellMeanJump"] == false);
    CHECK(config["contactUseExtremumCorrection"] == true);
    CHECK(config["contactExtremumUseCellMeanSupport"] == true);
    CHECK(config["contactExtremumStart"] == doctest::Approx(0.001));
    CHECK(config["contactExtremumFull"] == doctest::Approx(0.008));
    CHECK(config["contactSupportStart"] == doctest::Approx(0.005));
    CHECK(config["contactSupportFull"] == doctest::Approx(0.06));
    CHECK(config["contactExtremumRetention"] == doctest::Approx(0.95));
    CHECK(config["contactExtremumHaloLayers"] == 2);
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
    CHECK(config["penaltyTarget"] == "O2");
    CHECK(config["highOrderTargetAlphaThreshold"] == doctest::Approx(0.01));
    config["alphaMax"] = 1.0;
    config["pressureJumpStart"] = 0.05;
    config["pressureJumpFull"] = 0.50;
    auto settings = config.get<LimitedVRSettings>();
    CHECK(settings.alphaMax == 1.0);
    CHECK(settings.o2ReferenceLimiter == O2ReferenceLimiter::WBAP);
    CHECK(settings.o2ReferenceUsePP);
    CHECK(settings.validate().empty());

    config["o2ReferenceLimiter"] = O2ReferenceLimiter::WBAP;
    config["o2ReferenceUsePP"] = false;
    config["contactUseCellMeanJump"] = true;
    config["contactUseExtremumCorrection"] = true;
    config["contactExtremumUseCellMeanSupport"] = true;
    config["contactExtremumHaloLayers"] = 2;
    settings = config.get<LimitedVRSettings>();
    CHECK(settings.o2ReferenceLimiter == O2ReferenceLimiter::WBAP);
    CHECK_FALSE(settings.o2ReferenceUsePP);
    CHECK(settings.contactUseCellMeanJump);
    CHECK(settings.contactUseExtremumCorrection);
    CHECK(settings.contactExtremumUseCellMeanSupport);
    CHECK(settings.contactExtremumHaloLayers == 2);
    CHECK(settings.validate().empty());

    config["o2ReferenceLimiter"] = 2;
    CHECK_THROWS(config.get<LimitedVRSettings>());

    config["o2ReferenceLimiter"] = O2ReferenceLimiter::Barth;
    config["gateMode"] = 9;
    CHECK_THROWS(config.get<LimitedVRSettings>());

    config["gateMode"] = LimitedVRGateMode::ProductCubic;
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
    for (const std::string name : {"v1", "v2", "v3"})
    {
        nlohmann::ordered_json preset = {{"limiterControl", {{"limitedVR", {{"preset", name}}}}}};
        ExpandLimitedVRPreset(preset);
        const auto selected = preset["limiterControl"]["limitedVR"].get<LimitedVRSettings>();
        CHECK(selected.alphaMax == (name == "v3" ? 1.0 : 0.5));
        CHECK(selected.contactAlphaMax == (name == "v3" ? 1.0 : 0.0));
        CHECK(selected.contactUseExtremumCorrection == (name == "v3"));
        CHECK(selected.o2ReferenceLimiter == (name == "v1" ? O2ReferenceLimiter::Barth : O2ReferenceLimiter::WBAP));
        CHECK(selected.gateMode == (name == "v1" ? LimitedVRGateMode::ProductCubic : LimitedVRGateMode::PressureExtremum));
        CHECK(selected.validate().empty());
    }
    nlohmann::ordered_json layer = {{"limiterControl", {{"limitedVR", {{"preset", "v3"}, {"alphaMax", 1.0}, {"contactAlphaMax", 1.0}}}}}};
    ExpandLimitedVRPreset(layer);
    const auto &section = layer["limiterControl"]["limitedVR"];
    CHECK_FALSE(section.contains("preset"));
    auto settings = section.get<LimitedVRSettings>();
    CHECK(settings.alphaMax == 1);
    CHECK(settings.contactAlphaMax == 1);
    CHECK(settings.gateMode == LimitedVRGateMode::PressureExtremum);
    CHECK(settings.o2ReferenceLimiter == O2ReferenceLimiter::WBAP);
    CHECK(settings.contactExtremumHaloLayers == 2);
    CHECK(settings.weakAlphaMax == 0);
    CHECK(settings.validate().empty());
    const auto resolved = layer;
    ExpandLimitedVRPreset(layer);
    CHECK(layer == resolved);
    layer["limiterControl"]["limitedVR"]["preset"] = "unknown";
    CHECK_THROWS(ExpandLimitedVRPreset(layer));
}
