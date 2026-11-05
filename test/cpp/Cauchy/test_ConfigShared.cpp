#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Config/SimulationConfig.hpp"

TEST_CASE("Hidden consumers serialize complete configuration registered in the shared library")
{
    namespace C = DNDS::Cauchy;
    const C::Json input{{"defaults", {{"initialPrimitive", {.8, .1, -.02, .03, 1.}}, {"numerics", {{"reconstruction", "variational"}, {"cfv", {{"maxOrder", 3}, {"intOrder", 6}}}}}}}};
    // Resolve registers this type in cauchy_shared. Serialization below is
    // compiled in a hidden-visibility consumer, as are Python extensions.
    const auto resolved = C::SimulationConfig::Resolve(input, 1);
    REQUIRE(resolved.Domains().front().settings.numerics.cfv.maxOrder == 3);
    const C::Json serialized = resolved.Domains().front().settings;
    REQUIRE(serialized.is_object());
    REQUIRE(serialized.at("numerics").is_object());
    CHECK(serialized.at("numerics").at("reconstruction") == "variational");
    CHECK(serialized.at("numerics").at("cfv").at("maxOrder") == 3);
    const auto roundtrip = C::SimulationConfig::Resolve({{"defaults", serialized}}, 1);
    CHECK(roundtrip.Domains().front().settings.initialPrimitive[0] == .8);
    CHECK(roundtrip.Domains().front().settings.numerics.cfv.intOrder == 6);
    const auto ditr = C::SimulationConfig::Resolve({{"time", {{"method", "ditr"}, {"ditr", {{"relativeTolerance", 0}}}}}}, 1);
    const C::Json serializedTime = ditr.Time();
    CHECK(serializedTime["ditr"]["iterations"].is_null());
    CHECK(serializedTime["ditr"]["relativeTolerance"] == 0);
    const auto inherited = ditr.Time().ditr.ResolveControls(ditr.Domains().front().settings.implicit);
    CHECK(inherited.iterations == ditr.Domains().front().settings.implicit.iterations);
    CHECK(inherited.relativeTolerance == 0);
}
