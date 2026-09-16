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
    config["pressureJumpFull"] = 0.05;
    settings = config.get<LimitedVRSettings>();
    CHECK_FALSE(settings.validate().empty());
}
