#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Euler/Gas.hpp"
#include <limits>

namespace C = DNDS::Cauchy;
static C::Riemann::AcousticState Prepare(const C::PerfectGas &gas, const C::FlowState &state)
{
    const auto primitive = gas.Primitive(state);
    return {{state, primitive.pressure}, primitive.soundSpeed};
}
TEST_CASE("Flow5 retains transverse momentum on planar geometry and exact physical flux")
{
    const C::PerfectGas gas{1.4, 2., 0., 0.72};
    const C::FlowState primitive{2, 3, -4, 5, 7};
    const auto state = gas.Conservative(primitive);
    REQUIRE(gas.Admissible(state));
    CHECK(state[4] == doctest::Approx(67.5)); // 7/.4 + .5*2*(9+16+25)
    const auto back = gas.Primitive(state);
    CHECK(back.density == 2);
    CHECK(back.pressure == doctest::Approx(7));
    CHECK(back.temperature == doctest::Approx(1.75));
    CHECK(back.velocity == C::Vector3{3, -4, 5});
    const auto flux = C::Riemann::PhysicalFlux(Prepare(gas, state).flow, {1, 0, 0});
    const C::FlowState exact{6, 25, -24, 30, 223.5};
    for (int j = 0; j < 5; ++j)
        CHECK(flux[j] == doctest::Approx(exact[j]));
    CHECK_FALSE(gas.Admissible({0, 0, 0, 0, 1}));
    CHECK_FALSE(gas.Admissible({1, 3, 0, 0, 4}));
    CHECK_FALSE(gas.Admissible({1, 0, 0, 0, std::numeric_limits<double>::infinity()}));
}
TEST_CASE("Inviscid kernels agree with legacy flux and conserve under oriented face reversal")
{
    const C::PerfectGas gas;
    const C::Vector3 normal{1. / 3, 2. / 3, 2. / 3}, reverse{-1. / 3, -2. / 3, -2. / 3};
    for (int sample = 0; sample < 40; ++sample)
    {
        const auto left = gas.Conservative({0.5 + .1 * sample, .3 * sample - 2, .7, -1.1, 2. + .2 * sample});
        const auto right = gas.Conservative({1.4, -0.2, .3 - .1 * sample, 1.3, 1.1});
        Eigen::Matrix<double, 5, 1> state, legacy;
        Eigen::Vector3d velocity, n;
        for (int j = 0; j < 5; ++j)
            state[j] = left[j];
        for (int d = 0; d < 3; ++d)
        {
            velocity[d] = left[1 + d] / left[0];
            n[d] = normal[d];
        }
        DNDS::Euler::Gas::GasInviscidFlux_XY<3>(state, velocity, Eigen::Vector3d::Zero(), n, gas.Pressure(left), legacy);
        const auto leftInput = Prepare(gas, left), rightInput = Prepare(gas, right);
        const auto physical = C::Riemann::PhysicalFlux(leftInput.flow, normal);
        const auto same = C::Riemann::Rusanov(leftInput, leftInput, normal);
        const auto numerical = C::Riemann::Rusanov(leftInput, rightInput, normal), flipped = C::Riemann::Rusanov(rightInput, leftInput, reverse);
        for (int j = 0; j < 5; ++j)
        {
            CHECK(physical[j] == doctest::Approx(legacy[j]).epsilon(2e-14));
            CHECK(same[j] == doctest::Approx(physical[j]).epsilon(2e-14));
            CHECK(numerical[j] == doctest::Approx(-flipped[j]).epsilon(2e-14));
        }
    }
}
TEST_CASE("Newtonian Fourier kernels reproduce analytic shear heat flux and legacy three-dimensional stress")
{
    const C::PerfectGas gas{1.4, 2, .03, .72};
    const auto state = gas.Conservative({1.2, 0, 0, 0, 2.4});
    C::FlowGradient gradient{};
    gradient[1][1] = 4;
    auto flux = gas.ViscousFlux(gas.Primitive(state), gradient, {0, 1, 0});
    CHECK(flux[1] == doctest::Approx(.12));
    CHECK(flux[2] == 0);
    CHECK(flux[4] == 0);
    gradient = {};
    gradient[4][0] = 3;
    flux = gas.ViscousFlux(gas.Primitive(state), gradient, {1, 0, 0});
    const double cp = gas.gamma * gas.gasConstant / (gas.gamma - 1), k = gas.viscosity * cp / gas.prandtl;
    CHECK(flux[4] == doctest::Approx(k * 3));
    gradient = {};
    gradient[1][1] = 2;
    gradient[2][0] = -2; // Rigid rotation: strain/stress identically zero.
    for (auto value : gas.ViscousFlux(gas.Primitive(state), gradient, {1, 0, 0}))
        CHECK(value == 0);

    const auto moving = gas.Conservative({1.2, .4, -.3, .2, 2.4});
    const auto primitive = gas.Primitive(moving);
    Eigen::Matrix<double, 5, 1> legacyState, legacyFlux;
    Eigen::Matrix<double, 3, 5> legacyGradient;
    Eigen::Vector3d normal(1. / 3, 2. / 3, -2. / 3);
    for (int j = 0; j < 5; ++j)
    {
        legacyState[j] = moving[j];
        for (int d = 0; d < 3; ++d)
            gradient[j][d] = .13 * (j + 1) - .2 * (d + 1);
    }
    for (int d = 0; d < 3; ++d)
    {
        for (int j = 0; j < 4; ++j)
            legacyGradient(d, j) = gradient[j][d];
        legacyGradient(d, 4) = gas.gasConstant * (moving[0] * gradient[4][d] + primitive.temperature * gradient[0][d]);
    }
    DNDS::Euler::Gas::ViscousFlux_IdealGas<3>(legacyState, legacyGradient, normal, false, gas.gamma, gas.gamma, gas.viscosity, 0, false, k, cp, legacyFlux);
    const auto result = gas.ViscousFlux(primitive, gradient, {normal[0], normal[1], normal[2]});
    for (int j = 0; j < 5; ++j)
        CHECK(result[j] == doctest::Approx(legacyFlux[j]).epsilon(3e-14));
}
TEST_CASE("Boundary values and exterior states implement slip no-slip and thermal wall contracts")
{
    const C::PerfectGas gas{1.4, 2, .03, .72};
    const auto state = gas.Conservative({2, 3, -4, 5, 7});
    const C::Vector3 normal{0, 1, 0};
    C::BoundaryCondition wall;
    wall.kind = C::BoundaryKind::SlipWall;
    const auto slipFace = gas.Primitive(wall.FaceValue(gas, state, normal));
    const auto slipOutside = gas.Primitive(wall.Exterior(gas, state, normal));
    CHECK(slipFace.velocity == C::Vector3{3, 0, 5});
    CHECK(slipOutside.velocity == C::Vector3{3, 4, 5});
    CHECK(slipFace.pressure == doctest::Approx(7));
    wall.kind = C::BoundaryKind::NoSlipWall;
    wall.velocity = {.5, 0, -.2};
    wall.isothermal = true;
    wall.temperature = .8;
    const auto face = gas.Primitive(wall.FaceValue(gas, state, normal));
    CHECK(face.velocity == wall.velocity);
    CHECK(face.temperature == doctest::Approx(.8));
    CHECK(face.pressure == doctest::Approx(7));
    CHECK(face.density == doctest::Approx(7 / (2 * .8)));
    const auto outside = gas.Primitive(wall.Exterior(gas, state, normal));
    CHECK(outside.velocity == C::Vector3{-2, 4, -5.4});
    CHECK(outside.density == 2);
    C::BoundarySettings config;
    C::BoundarySystem system({{"7", config}}, gas);
    CHECK_NOTHROW(system.ValidateUsed({7}));
    CHECK_THROWS(system.ValidateUsed({}));
    CHECK_THROWS(system.At(8));
}
