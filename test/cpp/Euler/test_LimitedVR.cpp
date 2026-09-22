#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "Euler/LimitedVR.hpp"
#include "doctest.h"

using namespace DNDS::Euler;

TEST_CASE("Limited variational reconstruction cubic ramp has exact endpoints and bounded transition")
{
    CHECK(LimitedVRCubicRamp(0.0, 0.1, 0.3) == 0.0);
    CHECK(LimitedVRCubicRamp(0.1, 0.1, 0.3) == 0.0);
    CHECK(LimitedVRCubicRamp(0.2, 0.1, 0.3) == doctest::Approx(0.5));
    CHECK(LimitedVRCubicRamp(0.3, 0.1, 0.3) == 1.0);
    CHECK(LimitedVRCubicRamp(1.0, 0.1, 0.3) == 1.0);
    CHECK_THROWS(LimitedVRCubicRamp(0.2, 0.3, 0.3));
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

TEST_CASE("Limited variational reconstruction settings parse and validate")
{
    nlohmann::ordered_json config = LimitedVRSettings{};
    CHECK(config["o2ReferenceLimiter"] == LimitedVRSettings::O2ReferenceLimiterBarth);
    CHECK(config["gateMode"] == LimitedVRSettings::GateProductCubic);
    CHECK(config["gateHaloLayers"] == 0);
    CHECK(config["gateHaloSaturatedOnly"] == false);
    CHECK(config["alphaUpdateRelaxation"] == 1.0);
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
    settings = config.get<LimitedVRSettings>();
    CHECK(settings.o2ReferenceLimiter == LimitedVRSettings::O2ReferenceLimiterWBAP);
    CHECK_FALSE(settings.o2ReferenceUsePP);
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
    config["pressureJumpFull"] = 0.05;
    settings = config.get<LimitedVRSettings>();
    CHECK_FALSE(settings.validate().empty());
}
