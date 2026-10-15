#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "Euler/ReactiveSplitIndicator.hpp"
#include "doctest.h"

using DNDS::Euler::ReactiveSplitIndicatorMode;

TEST_CASE("Reactive split strategy names round trip and reject invalid configuration")
{
    using DNDS::Euler::ReactiveSplitIndicatorSettings;
    const auto names = DNDS_ENUM_ALLOWED_VALUES(ReactiveSplitIndicatorMode);
    REQUIRE(names.size() == 6);
    const auto schema = ReactiveSplitIndicatorSettings::schema();
    const auto &modeSchema = schema["properties"]["indicatorMode"];
    CHECK(modeSchema["enum"] == nlohmann::ordered_json(names));
    const auto description = modeSchema["description"].get<std::string>();
    for (const auto &name : names)
        CHECK(description.find(name) != std::string::npos);
    for (const auto *recipe : {"broadA", "BA-CE6", "Balance/B-E2", "localRXX", "Kinetic/H01P2"})
        CHECK(description.find(recipe) != std::string::npos);
    nlohmann::ordered_json config = ReactiveSplitIndicatorSettings{};
    CHECK(config["indicatorMode"] == "Activity");
    for (const auto &name : names)
    {
        config["indicatorMode"] = name;
        const auto settings = config.get<ReactiveSplitIndicatorSettings>();
        CHECK(settings.validate().empty());
        nlohmann::ordered_json restored = settings;
        CHECK(restored["indicatorMode"] == name);
    }
    for (const auto &invalid : nlohmann::ordered_json::array({"unknown", 0, 5, nullptr}))
    {
        config["indicatorMode"] = invalid;
        CHECK_FALSE(config.get<ReactiveSplitIndicatorSettings>().validate().empty());
    }
}

TEST_CASE("Chemical response ratio is step independent and has no auxiliary gates")
{
    DNDS::Euler::ReactiveSplitIndicatorSettings settings;
    settings.indicatorMode = ReactiveSplitIndicatorMode::ChemicalResponse;
    settings.ratioThreshold = .1;
    settings.ratioExponent = 2;
    CHECK(settings.validate().empty());
    using DNDS::Euler::ReactiveSplitChemicalResponseFraction;
    CHECK(ReactiveSplitChemicalResponseFraction(0, 0, settings) == 0);
    CHECK(ReactiveSplitChemicalResponseFraction(1, 0, settings) == 0);
    CHECK(ReactiveSplitChemicalResponseFraction(0, 1, settings) == 1);
    CHECK(ReactiveSplitChemicalResponseFraction(1, .1, settings) == doctest::Approx(.5));
    CHECK(ReactiveSplitChemicalResponseFraction(1, 1, settings) == doctest::Approx(1. / 1.01));
    CHECK(ReactiveSplitChemicalResponseFraction(1e-200, 1e-201, settings) == doctest::Approx(.5));
    CHECK(ReactiveSplitChemicalResponseFraction(1e200, 1e199, settings) == doctest::Approx(.5));
    settings.chemicalActivityThreshold = 1e10;
    settings.shockScale = 1e-10;
    CHECK(ReactiveSplitChemicalResponseFraction(1, .1, settings) == doctest::Approx(.5));
    CHECK_THROWS(ReactiveSplitChemicalResponseFraction(-1, 1, settings));
    nlohmann::ordered_json config = settings;
    CHECK(config.get<DNDS::Euler::ReactiveSplitIndicatorSettings>().indicatorMode == ReactiveSplitIndicatorMode::ChemicalResponse);
    settings.spatialPasses = 1;
    CHECK_FALSE(settings.validate().empty());
    settings.spatialPasses = 0;
    settings.balanceSmoothingPasses = 1;
    CHECK_FALSE(settings.validate().empty());
    settings.balanceSmoothingPasses = 0;
    settings.ratioThreshold = 0;
    CHECK_FALSE(settings.validate().empty());
}

TEST_CASE("Local matched-rate ratio guards activity and forbids spatial expansion")
{
    DNDS::Euler::ReactiveSplitIndicatorSettings settings;
    settings.indicatorMode = ReactiveSplitIndicatorMode::MatchedRateRatio;
    settings.chemicalActivityThreshold = .001;
    settings.ratioThreshold = .1;
    settings.ratioExponent = 4;
    CHECK(settings.validate().empty());
    using DNDS::Euler::ReactiveSplitMatchedRatioFraction;
    CHECK(ReactiveSplitMatchedRatioFraction(0, 1, 0, settings) == 0);
    CHECK(ReactiveSplitMatchedRatioFraction(1, 0, 0, settings) == 0);
    CHECK(ReactiveSplitMatchedRatioFraction(.001, .0001, 0, settings) == doctest::Approx(.25));
    CHECK(ReactiveSplitMatchedRatioFraction(.001, .0001, .08, settings) == doctest::Approx(.125));
    CHECK(ReactiveSplitMatchedRatioFraction(1e300, 1e-300, 0, settings) == 0);
    CHECK(ReactiveSplitMatchedRatioFraction(1e300, 1e300, 0, settings) == doctest::Approx(1.0 / 1.0001));
    CHECK(ReactiveSplitMatchedRatioFraction(.01, .001, 0, settings) >
          ReactiveSplitMatchedRatioFraction(.001, .0001, 0, settings));
    settings.spatialPasses = 1;
    CHECK_FALSE(settings.validate().empty());
    settings.spatialPasses = 0;
    settings.balanceExpansionPasses = 1;
    CHECK_FALSE(settings.validate().empty());
    settings.balanceExpansionPasses = 0;
    settings.ratioThreshold = 0;
    CHECK_FALSE(settings.validate().empty());
}
using namespace DNDS::Euler;

TEST_CASE("Balance chain filters preserve constants and defer endpoint snapping")
{
    ReactiveSplitIndicatorSettings settings;
    CHECK(settings.balanceExpansionPasses == 0);
    CHECK(settings.balanceSmoothingPasses == 0);
    settings.indicatorMode = ReactiveSplitIndicatorMode::TransportBalance;
    settings.balanceExpansionPasses = 2;
    settings.balanceSmoothingPasses = 1;
    settings.strangSnapTolerance = 0.05;
    CHECK(settings.validate().empty());
    nlohmann::ordered_json config = settings;
    auto restored = config.get<ReactiveSplitIndicatorSettings>();
    CHECK(restored.balanceExpansionPasses == 2);
    CHECK(restored.balanceSmoothingPasses == 1);
    for (int count = 0; count <= 2; ++count)
    {
        CHECK(ReactiveSplitBalanceSmooth(0.3, count * 0.3, count) == doctest::Approx(0.3));
        CHECK(ReactiveSplitBalanceSmooth(0, 0, count) == 0);
        CHECK(ReactiveSplitBalanceSmooth(1, count, count) == 1);
    }
    CHECK(ReactiveSplitBalanceSmooth(0.8, 0.4, 2) == doctest::Approx(0.5));
    CHECK(ReactiveSplitBalanceSmooth(0.8, 0.4, 1) == doctest::Approx(0.7));
    CHECK_THROWS(ReactiveSplitBalanceSmooth(0.3, 0.9, 3));
    DNDS::real filtered = ReactiveSplitBalanceSmooth(0.04, 0.12, 2);
    CHECK(filtered == doctest::Approx(0.05));
    CHECK(ReactiveSplitSnapChi(1.0 - ReactiveSplitBalanceSmooth(0.04, 0.08, 2), settings) == 1);
    CHECK(ReactiveSplitSnapChi(1.0 - ReactiveSplitBalanceSmooth(0.04, 0.16, 2), settings) < 1);
    settings.indicatorMode = ReactiveSplitIndicatorMode::Activity;
    CHECK_FALSE(settings.validate().empty());
}

TEST_CASE("Transport balance has matched-rate limits and independent configuration")
{
    ReactiveSplitIndicatorSettings settings;
    settings.indicatorMode = ReactiveSplitIndicatorMode::TransportBalance;
    settings.coupledThreshold = 0.001;
    settings.hillExponent = 2;
    CHECK(settings.validate().empty());
    CHECK(ReactiveSplitBalanceScore(0, 1, 0, settings) == 0);
    CHECK(ReactiveSplitBalanceScore(1, 0, 0, settings) == 0);
    CHECK(ReactiveSplitBalanceScore(1, 2, 0, settings) == doctest::Approx(0.5));
    CHECK(ReactiveSplitBalanceScore(1e-8, 2e-8, 0, settings) == doctest::Approx(2e-16).epsilon(1e-6).scale(1e-16));
    CHECK(ReactiveSplitBalanceScore(1e12, 2e12, 0, settings) == doctest::Approx(2));
    CHECK(ReactiveSplitBalanceScore(1e300, 2e300, 0, settings) == doctest::Approx(2));
    CHECK(ReactiveSplitBalanceScore(1, 2, 0.8, settings) < 1e-4);
    CHECK(ReactiveSplitChi(ReactiveSplitBalanceScore(0, 2, 0, settings), settings) == 1);
    CHECK(ReactiveSplitChi(ReactiveSplitBalanceScore(1e12, 2e12, 0, settings), settings) == 0);
    CHECK_THROWS(ReactiveSplitBalanceScore(-1, 1, 0, settings));
    settings.spatialPasses = 1;
    CHECK_FALSE(settings.validate().empty());
    settings.spatialPasses = 0;
    settings.switchShape = 0;
    CHECK_FALSE(settings.validate().empty());
    settings.switchShape = 1;
    settings.coupledThreshold = 0;
    CHECK_FALSE(settings.validate().empty());
}

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

TEST_CASE("Reactive split v2 rate ratio is scale invariant and configurable")
{
    ReactiveSplitIndicatorSettings settings;
    settings.indicatorMode = ReactiveSplitIndicatorMode::StiffnessRatio;
    settings.ratioThreshold = 0.1;
    settings.ratioExponent = 2.0;
    settings.strangSnapTolerance = 0.0;
    settings.coupledSnapTolerance = 0.0;

    CHECK(ReactiveSplitChiStiffnessRatio(0.1, 1.0, settings) == doctest::Approx(0.5));
    CHECK(ReactiveSplitChiStiffnessRatio(1.0, 10.0, settings) == doctest::Approx(0.5));
    CHECK(ReactiveSplitChiStiffnessRatio(0.01, 1.0, settings) == doctest::Approx(1.0 / 101.0));

    settings.strangSnapTolerance = 0.01;
    settings.coupledSnapTolerance = 0.01;
    CHECK(ReactiveSplitChiStiffnessRatio(0.01, 1.0, settings) == 0.0);
    CHECK(ReactiveSplitChiStiffnessRatio(1.0, 1.0, settings) == 1.0);
    settings.chiOverride = 0.25;
    CHECK(ReactiveSplitChiStiffnessRatio(0.0, 0.0, settings) == doctest::Approx(0.25));

    nlohmann::ordered_json config = ReactiveSplitIndicatorSettings{};
    config["indicatorMode"] = "StiffnessRatio";
    config["ratioThreshold"] = 0.2;
    config["ratioExponent"] = 3.0;
    settings = config.get<ReactiveSplitIndicatorSettings>();
    CHECK(settings.indicatorMode == ReactiveSplitIndicatorMode::StiffnessRatio);
    CHECK(settings.ratioThreshold == doctest::Approx(0.2));
    CHECK(settings.ratioExponent == doctest::Approx(3.0));
    CHECK(settings.validate().empty());

    config["ratioThreshold"] = 0.0;
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

TEST_CASE("Chemistry escape preserves baseline and selects stiff low diffusion chemistry")
{
    ReactiveSplitIndicatorSettings settings;
    settings.indicatorMode = ReactiveSplitIndicatorMode::ChemistryEscape;
    settings.chemicalActivityThreshold = 10;
    settings.diffusiveActivityThreshold = 2;
    settings.activitySaturationExponent = 0.5;
    settings.strangSnapTolerance = 0.05;
    CHECK(settings.validate().empty());
    CHECK(ReactiveSplitChiChemistryEscape(0.1, 1, 0, settings) == ReactiveSplitChi(0.1, settings));
    CHECK(ReactiveSplitChiChemistryEscape(0.1, 0, 1e6, settings) == 1);
    CHECK(ReactiveSplitChiChemistryEscape(0.1, 1e4, 1e6, settings) == 0);
    CHECK(ReactiveSplitChiChemistryEscape(0, 0, 0, settings) == 1);
    settings.strangSnapTolerance = 0;
    settings.coupledSnapTolerance = 0;
    CHECK(ReactiveSplitChiChemistryEscape(0.005, 1e-4, 100, settings) == doctest::Approx(0.625));
    settings.chiOverride = 0.3;
    CHECK(ReactiveSplitChiChemistryEscape(1, 0, 1e6, settings) == doctest::Approx(0.3));
    settings.spatialPasses = 1;
    CHECK_FALSE(settings.validate().empty());
    settings.spatialPasses = 0;
    settings.escapeRatioThreshold = 0;
    CHECK_FALSE(settings.validate().empty());
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
