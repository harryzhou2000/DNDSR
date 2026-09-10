#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "Euler/ReactiveSplitIndicator.hpp"
#include "doctest.h"

using namespace DNDS::Euler;

TEST_CASE("Reactive split indicator endpoints and bounds")
{
    ReactiveSplitIndicatorSettings settings;
    settings.chiOverride = 0.0;
    CHECK(ReactiveSplitChi(1.0, settings) == doctest::Approx(0.0));
    settings.chiOverride = 1.0;
    CHECK(ReactiveSplitChi(1.0, settings) == doctest::Approx(1.0));
    settings.chiOverride = -1.0;
    settings.switchShape = 0;
    CHECK(ReactiveSplitChi(0.0, settings) == 1.0);
    CHECK(ReactiveSplitChi(0.01, settings) == 0.0);

    settings.strangSnapTolerance = 0.0;
    settings.coupledSnapTolerance = 0.0;
    CHECK(ReactiveSplitChi(0.0, settings) == doctest::Approx(0.9933071490757153));
    CHECK(ReactiveSplitChi(0.01, settings) == doctest::Approx(0.0066928509242847));

    settings.strangSnapTolerance = 0.01;
    settings.coupledSnapTolerance = 0.01;
    settings.chiOverride = 0.005;
    CHECK(ReactiveSplitChi(1.0, settings) == doctest::Approx(0.005));
}

TEST_CASE("Reactive split coupled score is shock suppressed")
{
    ReactiveSplitIndicatorSettings settings;
    DNDS::real smooth = ReactiveSplitCoupledScore(0.2, 0.4, 0.0, settings);
    DNDS::real shocked = ReactiveSplitCoupledScore(0.2, 0.4, 0.8, settings);
    CHECK(smooth > 0.0);
    CHECK(smooth < 1.0);
    CHECK(shocked < smooth * 1.0e-3);
}

TEST_CASE("Reactive split activity saturation exposes a0 b0 and p")
{
    ReactiveSplitIndicatorSettings settings;
    CHECK(ReactiveSplitSaturate(2.0) == doctest::Approx(2.0 / 3.0));
    CHECK(ReactiveSplitSaturate(2.0, 2.0, 0.5) == doctest::Approx(0.5));
    CHECK(ReactiveSplitSaturate(0.0, 2.0, 0.5) == 0.0);

    settings.chemicalActivityThreshold = 2.0;
    settings.diffusiveActivityThreshold = 4.0;
    settings.activitySaturationExponent = 0.5;
    CHECK(ReactiveSplitCoupledScore(2.0, 4.0, 0.0, settings) == doctest::Approx(0.25));
    settings.chemicalActivityThreshold = 1.0;
    settings.diffusiveActivityThreshold = 1.0;
    CHECK(ReactiveSplitCoupledScore(0.01, 0.01, 0.0, settings) >
          ReactiveSplitCoupledScore(0.01, 0.01, 0.0, ReactiveSplitIndicatorSettings{}));
    CHECK_THROWS(ReactiveSplitSaturate(1.0, 0.0, 1.0));
    CHECK_THROWS(ReactiveSplitSaturate(1.0, 1.0, 0.0));
}

TEST_CASE("Reactive split activity tuning is parsed and validated")
{
    nlohmann::ordered_json config = ReactiveSplitIndicatorSettings{};
    config["chemicalActivityThreshold"] = 5.6234;
    config["diffusiveActivityThreshold"] = 1.7783;
    config["activitySaturationExponent"] = 0.5;
    config["strangSnapTolerance"] = 0.02;
    config["coupledSnapTolerance"] = 0.03;
    auto settings = config.get<ReactiveSplitIndicatorSettings>();
    CHECK(settings.chemicalActivityThreshold == doctest::Approx(5.6234));
    CHECK(settings.diffusiveActivityThreshold == doctest::Approx(1.7783));
    CHECK(settings.activitySaturationExponent == doctest::Approx(0.5));
    CHECK(settings.strangSnapTolerance == doctest::Approx(0.02));
    CHECK(settings.coupledSnapTolerance == doctest::Approx(0.03));

    config["chemicalActivityThreshold"] = 0.0;
    settings = config.get<ReactiveSplitIndicatorSettings>();
    CHECK_FALSE(settings.validate().empty());

    config["chemicalActivityThreshold"] = 1.0;
    config["strangSnapTolerance"] = 0.6;
    config["coupledSnapTolerance"] = 0.4;
    settings = config.get<ReactiveSplitIndicatorSettings>();
    CHECK_FALSE(settings.validate().empty());
}

TEST_CASE("Hill switch is the default with an exact zero tail and configurable midpoint")
{
    ReactiveSplitIndicatorSettings settings;
    CHECK(ReactiveSplitChi(0.0, settings) == 1.0);
    CHECK(settings.switchShape == 1);
    settings.hillExponent = 3.0;
    CHECK(ReactiveSplitChi(0.0, settings) == 1.0);
    CHECK(ReactiveSplitCoupledFraction(0.0, settings) == 0.0);
    CHECK(ReactiveSplitCoupledFraction(settings.coupledThreshold, settings) == doctest::Approx(0.5));
    CHECK(ReactiveSplitCoupledFraction(0.02, settings) > 0.98);
    CHECK(ReactiveSplitCoupledFraction(0.001, settings) < 0.01);
    settings.strangBias = 2.0;
    CHECK(ReactiveSplitCoupledFraction(settings.coupledThreshold, settings) < 0.12);
    settings.chiOverride = 1.0;
    CHECK(ReactiveSplitChi(settings.coupledThreshold, settings) == 1.0);

    settings.chiOverride = -1.0;
    settings.switchShape = 0;
    settings.strangBias = 1.0;
    settings.strangSnapTolerance = 0.0;
    CHECK(ReactiveSplitChi(0.0, settings) == doctest::Approx(0.9933071490757153));
}

TEST_CASE("Neighbor expansion is bounded, decaying, and shock gated")
{
    ReactiveSplitIndicatorSettings settings;
    settings.spatialDecay = 0.65;
    DNDS::real expanded = ReactiveSplitExpandedCoupledFraction(0.1, 0.8, 0.0, settings);
    CHECK(expanded == doctest::Approx(0.52));
    CHECK(ReactiveSplitExpandedCoupledFraction(0.7, 0.8, 0.0, settings) == doctest::Approx(0.7));
    CHECK(ReactiveSplitExpandedCoupledFraction(0.1, 0.8, 0.8, settings) == doctest::Approx(0.1));
}
