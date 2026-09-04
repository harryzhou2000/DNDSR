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
