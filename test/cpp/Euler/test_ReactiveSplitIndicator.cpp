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
    CHECK(ReactiveSplitChi(0.0, settings) > 0.99);
    CHECK(ReactiveSplitChi(0.1, settings) < 1.0e-10);
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

TEST_CASE("Hill switch has exact zero tail and a configurable midpoint")
{
    ReactiveSplitIndicatorSettings settings;
    settings.switchShape = 1;
    settings.hillExponent = 3.0;
    CHECK(ReactiveSplitCoupledFraction(0.0, settings) == 0.0);
    CHECK(ReactiveSplitCoupledFraction(settings.coupledThreshold, settings) == doctest::Approx(0.5));
    CHECK(ReactiveSplitCoupledFraction(0.02, settings) > 0.98);
    CHECK(ReactiveSplitCoupledFraction(0.001, settings) < 0.01);
    settings.strangBias = 2.0;
    CHECK(ReactiveSplitCoupledFraction(settings.coupledThreshold, settings) < 0.12);
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
