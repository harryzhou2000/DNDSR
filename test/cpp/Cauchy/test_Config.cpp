#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Config/SimulationConfig.hpp"

namespace C = DNDS::Cauchy;
TEST_CASE("Cauchy root resolves profiles overrides CLI and condition defaults with provenance")
{
    const auto input = C::Json::parse(R"json({
        "profiles": {"gas": {"equation": {"gamma": 1.6}, "execution": {"layout": "soa"}}},
        "defaults": {"equation": {"gamma": 1.5}},
        "domains": [
            {"name": "left", "ranks": [2, 0], "profiles": ["gas"],
             "overrides": {"equation": {"gamma": 1.45}, "boundaries": {"7": {"kind": "farfield"}},
                 "numerics": {"cfv": {"boundaryZoneWeights": [[7, [1, 0.5, 0, 0]]]}}}},
            {"name": "right", "ranks": [1, 3]}
        ]
    })json");
    const auto resolved = C::SimulationConfig::Resolve(input, 4, {{"/domains/0/settings/equation/gamma", 1.4}});
    REQUIRE(resolved.Domains().size() == 2);
    CHECK(resolved.Domains()[0].settings.equation.gamma == 1.4);
    CHECK(resolved.Domains()[1].settings.equation.gamma == 1.5);
    CHECK(resolved.Domains()[0].settings.execution.layout == "soa");
    CHECK(resolved.Domains()[1].settings.execution.layout == "aos");
    CHECK(resolved.Domains()[0].settings.boundaries.at("7").primitive[4] == 1);
    CHECK(resolved.Domains()[0].settings.numerics.cfv.boundaryZoneWeights.at(7)[1] == 0.5);
    CHECK(resolved.RankGroups()[0].ranks == std::vector<int>{2, 0});
    CHECK(resolved.Provenance().at("/domains/0/settings/equation/gamma") == "CLI /domains/0/settings/equation/gamma");
    CHECK(resolved.Provenance().at("/domains/0/settings/execution/layout") == "profile gas");
    CHECK(resolved.Provenance().at("/domains/1/settings/equation/gamma") == "solver defaults");
    CHECK(resolved.Provenance().at("/domains/0/settings/boundaries/7/primitive/4") == "schema default");
    auto changed = input;
    changed["defaults"]["equation"]["gamma"] = 1.8;
    CHECK(resolved.Domains()[1].settings.equation.gamma == 1.5);
}
TEST_CASE("Cauchy rejects unknown nested keys unused contradictory settings and unsupported capabilities")
{
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"equation", {{"gama", 1.4}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"cfv", {{"functional", {{"badWeight", 1}}}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"boundaries", {{"7", {{"badBC", 1}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"boundaries", {{"07", C::Json::object()}}}}}}, 2));
    CHECK_NOTHROW(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"riemann", {{"fixScale", 0}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"flux", "Roe_M3"}, {"riemann", {{"fixScale", 0}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"equation", {{"viscosity", 0.1}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"dt", -1}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"steps", 2.5}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"domains", C::Json::array({{{"name", "bad"}, {"ranks", {0, 0}}}})}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve(C::Json::object(), 2, {{"/domains/0/settings/numerics/mispelled", 1}}));
    const C::Json unsupported{{"defaults", {{"execution", {{"device", "cuda"}}}, {"equation", {{"eos", "realGas"}}}}}};
    try
    {
        C::SimulationConfig::Resolve(unsupported, 2);
        FAIL("missing capabilities were accepted");
    }
    catch (const std::exception &error)
    {
        CHECK(std::string(error.what()).find("backend cuda") != std::string::npos);
        CHECK(std::string(error.what()).find("EOS realGas") != std::string::npos);
    }
}
