#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include "Cauchy/Physics/Positivity.hpp"
#include "Cauchy/Physics/ImplicitPositivity.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Euler/Gas.hpp"
#include <limits>

namespace C = DNDS::Cauchy;
TEST_CASE("Implicit energy compression preserves Euler scheme-one mathematics with explicit energy references")
{
    const C::PerfectGas gas{1.4, 1, 0, .72};
    const auto state = gas.Conservative({1.2, 4, -.7, .3, .8});
    const std::array<C::FlowState, 4> increments{{{-.3, 1, -2, .4, -4}, {.1, 8, 2, -.2, -2}, {0, 0, 0, 0, -.2}, {-.9, -3, 0, 0, -10}}};
    for (const auto &increment : increments)
        for (double baseOld : {0., -.5, .2})
        {
            const double baseNew = baseOld + .01;
            Eigen::Matrix<double, 5, 1> u, delta;
            for (int j = 0; j < 5; ++j)
            {
                u[j] = state[j];
                delta[j] = increment[j];
            }
            const double expected = DNDS::Euler::Gas::IdealGasGetCompressionRatioPressure<3, 1, 5>(u, delta, 1e-12, baseOld, baseNew);
            CHECK(C::InternalEnergyCompression(state, increment, 1e-12, baseOld, baseNew) == doctest::Approx(expected).epsilon(2e-14));
        }
}
TEST_CASE("Implicit flow compression enforces density and both temperature and energy constraints")
{
    const C::ThermalBounds bounds{.1, 2., .4};
    const C::FlowState state{1, 2, -.3, .1, 6};
    for (const C::FlowState increment : {C::FlowState{-2, 3, 0, 0, -4}, C::FlowState{1, 8, 1, -.4, 0}, C::FlowState{-.7, -1, 0, 0, -5}, C::FlowState{0, 0, 0, 0, -10}})
        for (double relax : {1., .5, .1})
        {
            const double alpha = C::FlowIncrementFraction(state, increment, bounds, relax);
            CHECK(alpha >= 0);
            CHECK(alpha < 1);
            auto updated = state;
            for (int j = 0; j < 5; ++j)
                updated[j] += alpha * increment[j];
            CHECK(C::ThermallyAdmissible(updated, bounds));
            const auto fixed = C::FixFlowIncrement(state, increment, bounds);
            updated = state;
            for (int j = 0; j < 5; ++j)
                updated[j] += fixed[j];
            CHECK(C::ThermallyAdmissible(updated, bounds));
        }
    const C::FlowState safe{.01, .01, 0, 0, .1};
    CHECK(C::FlowIncrementFraction(state, safe, bounds) == 1);
    CHECK(C::FixFlowIncrement(state, safe, bounds) == safe);
    // A shifted reference energy remains valid; temperature PP is never a
    // hardcoded positivity check on the stored total/internal energy itself.
    const C::ThermalBounds shifted{.1, -10, -std::numeric_limits<double>::infinity()};
    const C::FlowState shiftedState{1, 3, 0, 0, -2}, shiftedIncrement{0, 6, 0, 0, 0};
    const auto fixed = C::FixFlowIncrement(shiftedState, shiftedIncrement, shifted);
    CHECK(C::ThermallyAdmissible(C::PositivityDetail::Add(shiftedState, fixed), shifted));
}
TEST_CASE("Equation-owned physical flux Jacobian differentiates all three momentum components")
{
    for (double gamma : {1.4, 1.67})
    {
        C::EquationSettings settings;
        settings.gamma = gamma;
        const auto equation = C::EquationSystem::Resolve(settings);
        for (const C::Vector3 normal : {C::Vector3{1, 0, 0}, C::Vector3{.6, 0, -.8}, C::Vector3{0, -1, 0}})
        {
            const auto state = equation.Gas().Conservative({1.3, .7, -.2, .4, 1.1});
            const auto matrix = equation.PrepareFluxJacobian(state, normal);
            for (int column = 0; column < 5; ++column)
            {
                auto plus = state, minus = state;
                const double step = 1e-6 * std::max(1., std::abs(state[column]));
                plus[column] += step;
                minus[column] -= step;
                const auto left = C::Riemann::PhysicalFlux(equation.PrepareFluxState(plus).flow, normal);
                const auto right = C::Riemann::PhysicalFlux(equation.PrepareFluxState(minus).flow, normal);
                for (int row = 0; row < 5; ++row)
                    CHECK(matrix[column * 5 + row] == doctest::Approx((left[row] - right[row]) / (2 * step)).epsilon(2e-8).scale(1));
            }
        }
    }
}

TEST_CASE("Physical boundary Jacobians match conservative exterior-state derivatives")
{
    for (double gamma : {1.4, 1.67})
        for (const std::string kind : {"extrapolate", "farfield", "slipWall", "noSlipWall", "analytic", "doubleMachBottom"})
        {
            const C::PerfectGas gas{gamma, 1, .02, .72};
            C::BoundarySettings settings;
            settings.kind = kind;
            settings.primitive = {2, -.3, .2, .1, 1.4};
            settings.velocity = {.04, -.02, .03};
            C::BoundarySystem boundaries({{"7", settings}}, gas);
            const auto &boundary = boundaries.At(7);
            for (const C::Vector3 normal : {C::Vector3{1, 0, 0}, C::Vector3{.6, .8, 0}})
                for (double speed : {-4., -.2, .2, 4.})
                    for (double x : {.1, .5})
                    {
                        const auto inside = gas.Conservative({1.3, speed * normal[0] - .1 * normal[1], speed * normal[1] + .1 * normal[0], .15, 1.1});
                        const C::Vector3 point{x, .2, .3};
                        const auto matrix = boundary.ExteriorJacobian(gas, inside, normal, point, .2);
                        for (int column = 0; column < 5; ++column)
                        {
                            auto plus = inside, minus = inside;
                            const double h = 1e-6 * std::max(1., std::abs(inside[column]));
                            plus[column] += h;
                            minus[column] -= h;
                            const auto a = boundary.Exterior(gas, plus, normal, point, .2);
                            const auto b = boundary.Exterior(gas, minus, normal, point, .2);
                            for (int row = 0; row < 5; ++row)
                                CHECK(matrix[column * 5 + row] == doctest::Approx((a[row] - b[row]) / (2 * h)).epsilon(3e-8).scale(1));
                        }
                    }
        }
}
TEST_CASE("Thermal PP uses an internal-energy temperature bound including shifted energy references")
{
    const C::ThermalBounds bounds{.1, -10, -std::numeric_limits<double>::infinity()};
    const C::FlowState base{1, 3, 0, 0, -2}; // internal=-6.5, above shifted Tmin energy=-10
    CHECK(C::ThermallyAdmissible(base, bounds));
    CHECK(C::AdmissibleFraction(C::FlowState{1, 0, 0, 0, 2}, C::FlowState{-1, 0, 0, 0, 2}, bounds) == doctest::Approx(.45).epsilon(2e-9));
    const C::FlowState target{1, 7, 0, 0, -2};
    CHECK_FALSE(C::ThermallyAdmissible(target, bounds));
    const double exact = (4 - 3.) / (7 - 3.); // kinetic <=8, so |momentum|<=4
    const auto fraction = C::AdmissibleFraction(base, target, bounds);
    CHECK(fraction == doctest::Approx(exact).epsilon(2e-9));
    C::FlowState state;
    for (int j = 0; j < 5; ++j)
        state[j] = (1 - fraction) * base[j] + fraction * target[j];
    CHECK(C::ThermallyAdmissible(state, bounds));
    struct NonlinearBound
    {
        double densityFloor = .1;
        double MinimumInternalEnergyDensity(const C::FlowState &u) const { return u[0] * u[0]; }
    } nonlinear;
    const C::FlowState a{1, 0, 0, 0, 2}, b{3, 0, 0, 0, 2};
    CHECK(C::AdmissibleFraction(a, b, nonlinear) == doctest::Approx((std::sqrt(2.) - 1) / 2).epsilon(2e-9));
    const C::PerfectGas gas{1.4, 2, 0, .72};
    const C::ThermalBounds temperature{.1, 2 * .5 / .4, 0};
    CHECK_FALSE(C::ThermallyAdmissible(gas.Conservative({1, 100, 0, 0, .5}), temperature)); // T=.25 <.5, yet p>0
    CHECK(C::ThermallyAdmissible(gas.Conservative({1, 100, 0, 0, 2}), temperature));
    for (int sample = 1; sample < 40; ++sample)
    {
        const double momentum = 4 + sample * .2;
        const C::FlowState point{1, momentum, 0, 0, -2};
        const double expected = 1 / (momentum - 3);
        CHECK(C::AdmissibleFraction(base, point, bounds) == doctest::Approx(expected).epsilon(2e-9));
    }
    struct MixtureBound
    {
        double densityFloor = .1;
        double MinimumInternalEnergyDensity(const std::array<double, 6> &u) const
        {
            return -10 * u[0] + 2 * u[5];
        } // shifted species energy, affine at Tmin
    } mixture;
    CHECK(C::ThermallyAdmissible(std::array<double, 6>{1, 3, 0, 0, -2, .25}, mixture));
    CHECK_FALSE(C::ThermallyAdmissible(std::array<double, 6>{1, 3, 0, 0, -2, 2}, mixture));
    const C::ThermalBounds tinyTemperature{1e-12, 2.5e-10, 0};
    C::FlowState nearlyCold{1, 1, 0, 0, .5 + 2.5e-10 + 8 * std::numeric_limits<double>::epsilon()};
    CHECK(C::ThermallyAdmissible(nearlyCold, tinyTemperature));
    CHECK_FALSE(C::ThermallyInterior(nearlyCold, tinyTemperature));
    auto tooFast = nearlyCold;
    tooFast[1] += .1;
    CHECK(C::AdmissibleFraction(nearlyCold, tooFast, tinyTemperature) == 0);
}
TEST_CASE("Cold mean limiter characteristics retain a reliable inverse without altering physical sound speed")
{
    const auto equation = C::EquationSystem::Resolve(C::EquationSettings{});
    const auto state = equation.Gas().Conservative({1, 1, .2, .1, 1e-10});
    const auto physical = equation.PrepareFluxState(state);
    const auto basis = equation.PrepareCharacteristics(state, state, {1, 0, 0});
    const Eigen::Map<const Eigen::Matrix<double, 5, 5>> l(basis.forward.data()), r(basis.inverse.data());
    CHECK((l * r).isApprox(Eigen::Matrix<double, 5, 5>::Identity(), 1e-7));
    CHECK(basis.componentwiseFallback);
    CHECK(physical.soundSpeed * physical.soundSpeed == doctest::Approx(1.4 * equation.Gas().Pressure(state)).epsilon(1e-12));
}
TEST_CASE("Cauchy prescribed DMR shock and boundary stage times match canonical conservative data")
{
    C::FlowProfileSettings profile;
    profile.kind = "doubleMachReflection";
    const C::PerfectGas gas;
    const auto ahead = C::ProfileValue(profile, {1, 0, 0, 0, 1}, gas, {1, .5, 0}, 0);
    const auto behind = C::ProfileValue(profile, {1, 0, 0, 0, 1}, gas, {1, .5, 0}, .1);
    const C::FlowState unshocked{1.4, 0, 0, 0, 2.5};
    for (int j = 0; j < 5; ++j)
        CHECK(ahead[j] == doctest::Approx(unshocked[j]).epsilon(2e-14));
    const C::FlowState shocked{8, 57.157676649772960, -33, 0, 563.5};
    for (int j = 0; j < 5; ++j)
        CHECK(behind[j] == doctest::Approx(shocked[j]).epsilon(2e-14));
    C::BoundarySettings settings;
    settings.kind = "doubleMachBottom";
    settings.profile = profile;
    C::BoundarySystem boundaries({{"1", settings}}, gas);
    const auto &condition = boundaries.At(1);
    const auto prescribed = condition.Exterior(gas, ahead, {0, -1, 0}, {.1, 0, 0}, .05);
    for (int j = 0; j < 5; ++j)
        CHECK(prescribed[j] == doctest::Approx(shocked[j]).epsilon(2e-14));
    const auto wall = condition.Exterior(gas, shocked, {0, -1, 0}, {.5, 0, 0}, .05);
    CHECK(wall[2] == -shocked[2]);
    CHECK(wall[4] == doctest::Approx(shocked[4]));
}
TEST_CASE("Cauchy beta-five vortex oracle advects periodically and preserves isentropy")
{
    C::FlowProfileSettings profile;
    profile.kind = "isentropicVortex";
    const C::PerfectGas gas;
    const auto a = C::ProfileValue(profile, {1, 1, 1, 0, 1}, gas, {5, 5, 0}, 0);
    const auto b = C::ProfileValue(profile, {1, 1, 1, 0, 1}, gas, {5, 5, 0}, 10);
    CHECK(a[0] == doctest::Approx(.49380732389534654));
    CHECK(gas.Pressure(a) == doctest::Approx(std::pow(a[0], gas.gamma)));
    CHECK(a == b);
}
TEST_CASE("Cauchy farfield respects supersonic directions and subsonic incoming component count")
{
    C::PerfectGas gas;
    C::BoundarySettings settings;
    settings.kind = "farfield";
    settings.primitive = {2, -3, .1, .2, 4};
    C::BoundarySystem boundaries({{"1", settings}}, gas);
    const auto &bc = boundaries.At(1);
    const C::Vector3 normal{1, 0, 0};
    for (double speed : {-4., -.2, .2, 4.})
    {
        const auto inside = gas.Conservative({1, speed, .5, -.1, 1});
        const auto value = gas.Primitive(bc.Exterior(gas, inside, normal));
        if (speed > 2)
            CHECK(bc.Exterior(gas, inside, normal) == inside);
        if (speed < -2)
            CHECK(value.density == 2);
        if (speed == .2)
        {
            CHECK(value.pressure == doctest::Approx(4));
            CHECK(value.density == 1);
        }
        if (speed == -.2)
        {
            CHECK(value.pressure == doctest::Approx(1));
            CHECK(value.density == 2);
            CHECK(value.velocity[0] == -3);
        }
    }
}
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
