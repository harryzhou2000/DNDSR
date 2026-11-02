#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Config/SimulationConfig.hpp"

namespace C = DNDS::Cauchy;
TEST_CASE("Strict implicit VR acceptance needs an explicit positive reconstruction tolerance")
{
    C::Json input{{"defaults", {{"implicit", {{"requireConvergedReconstruction", true}}}, {"numerics", {{"reconstruction", "variational"}}}}}};
    CHECK_THROWS(C::SimulationConfig::Resolve(input, 1));
    input["defaults"]["numerics"]["reconstructionTolerance"] = 1e-10;
    CHECK(C::SimulationConfig::Resolve(input, 1).Domains()[0].settings.implicit.requireConvergedReconstruction);
}
TEST_CASE("Steady config resolves Euler's BDF1 infinite timestep override with provenance")
{
    const auto config = C::SimulationConfig::Resolve({{"time", {{"steady", true}, {"method", "esdirk4"}, {"steps", 123}, {"finalTime", 10}, {"cfl", 1}, {"maxRetries", 3}}}}, 2);
    CHECK(config.Time().steady);
    CHECK(config.Time().method == "bdf1");
    CHECK(config.Time().steps == 1);
    CHECK(config.Time().finalTime == 0);
    CHECK(config.Time().cfl == 0);
    CHECK(config.Time().maxRetries == 0);
    CHECK(config.Root()["time"]["method"] == "bdf1");
    CHECK(config.Provenance().at("/time/method") == "BDF1 steady limit");
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"consoleInterval", -1}}}}}}, 2));
}
TEST_CASE("Reconstruction PP controls are explicit and reject unsupported compatibility combinations")
{
    C::NumericsSettings settings;
    CHECK(settings.positivityReconstruction == "Euler");
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivity = true;
    settings.positivityCompressToMean = false;
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivityReconstruction = "Segment";
    CHECK_THROWS(settings.ValidatePositivity());
    settings.positivityCompressToMean = true;
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivityBetaPower = -1;
    CHECK_THROWS(settings.ValidatePositivity());
    settings.positivityBetaPower = 2;
    CHECK_THROWS(settings.ValidatePositivity());
    settings.positivityReconstruction = "Euler";
    CHECK_NOTHROW(settings.ValidatePositivity());
    settings.positivityReconstruction = "unknown";
    CHECK_THROWS(settings.ValidatePositivity());
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"positivityReconstruction", "unknown"}}}}}}, 2));
}
TEST_CASE("Cauchy implicit controls resolve through the root and reject contradictory paths")
{
    for (const std::string method : {"ssprk3", "bdf1", "bdf2", "trapz", "esdirk2", "esdirk3", "esdirk4", "ditr"})
    {
        const auto config = C::SimulationConfig::Resolve({{"time", {{"method", method}}}, {"defaults", {{"implicit", {{"linear", {{"preconditioner", "ilu"}}}}}}}}, 2);
        CHECK(config.Time().method == method);
        CHECK(config.Domains()[0].settings.implicit.linear.preconditioner == "ilu");
        CHECK(config.Domains()[0].settings.implicit.sgs.exchangeBetweenSweeps == false);
    }
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "unavailable"}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "ditr"}, {"ditr", {{"fraction", .9}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "ditr"}, {"ditr", {{"iterations", 0}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"time", {{"method", "ditr"}, {"ditr", {{"absoluteTolerance", -1}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"roeDissipation", true}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"iterations", 0}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"linear", {{"preconditioner", "ilu"}}}, {"sgs", {{"exchangeBetweenSweeps", true}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"positivity", {{"rhsRelax", 0}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"positivity", {{"rhsReduction", "unknown"}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"multigrid", {{"levels", 3}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"implicit", {{"multigrid", {{"levels", 1}}}}}}}}, 2));
    CHECK_NOTHROW(C::SimulationConfig::Resolve({{"time", {{"method", "bdf1"}}}, {"defaults", {{"numerics", {{"reconstruction", "variational"}, {"cfv", {{"maxOrder", 3}}}}}, {"implicit", {{"multigrid", {{"levels", 2}}}}}}}}, 2));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"numerics", {{"reconstruction", "variational"}, {"cfv", {{"maxOrder", 3}}}}}, {"implicit", {{"multigrid", {{"levels", 2}}}}}}}}, 2));
    const auto sor = C::SimulationConfig::Resolve({{"defaults", {{"execution", {{"overlap", false}}}, {"numerics", {{"cfv", {{"SORInstead", true}, {"coreFirst", false}}}}}}}}, 2);
    CHECK(sor.Domains()[0].settings.numerics.cfv.coreFirst);
    CHECK(sor.Root()["domains"][0]["settings"]["numerics"]["cfv"]["coreFirst"].get<bool>());
    CHECK(sor.Provenance().at("/domains/0/settings/numerics/cfv/coreFirst") == "Cauchy SOR logical ordering");
}
TEST_CASE("Cauchy partial analytic boundary profile receives nested schema defaults")
{
    const auto config = C::SimulationConfig::Resolve({{"defaults", {{"boundaries", {{"7", {{"kind", "analytic"}, {"profile", {{"kind", "doubleMachReflection"}}}}}}}}}}, 1);
    const auto &profile = config.Domains()[0].settings.boundaries.at("7").profile;
    CHECK(profile.kind == "doubleMachReflection");
    CHECK(profile.strength == 5);
    CHECK(profile.period == 10);
    CHECK_THROWS(C::SimulationConfig::Resolve({{"defaults", {{"mesh", {{"boundaryIds", {{"FAR", 1}}}}}}}}, 1));
    CHECK(C::SimulationConfig::Resolve(C::Json::object(), 1).Output().hdfDeflateLevel == 0);
    CHECK_THROWS(C::SimulationConfig::Resolve({{"output", {{"hdfDeflateLevel", 10}}}}, 1));
    CHECK_THROWS(C::SimulationConfig::Resolve({{"output", {{"hdfDeflateLevel", -1}}}}, 1));
}
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
