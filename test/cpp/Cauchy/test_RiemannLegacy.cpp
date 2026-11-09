#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Physics/EquationSystem.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Euler/Gas.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"

namespace C = DNDS::Cauchy;
namespace F = C::Riemann;
namespace L = DNDS::Euler::Gas;
TEST_CASE("Scalar convection uses the caller's fixed dissipation independently of flow flux")
{
    // Two scalars share a frozen face speed. Acoustic changes/flow selection
    // cannot change this result; reversing the face reverses its flux.
    for (const auto &pair : {std::pair<double, double>{2, 5}, {0, 3}})
    {
        const double a = pair.first, b = pair.second, uL = .3, uR = -.2, lambda = .4;
        const double expected = .5 * (uL * a + uR * b - lambda * (b - a));
        CHECK(C::AdvectedScalarFlux(a, b, uL, uR, lambda) == doctest::Approx(expected));
        CHECK(C::AdvectedScalarFlux(b, a, -uR, -uL, lambda) == doctest::Approx(-expected));
        CHECK(C::AdvectedScalarFlux(a, b, .7, .7, .7) == doctest::Approx(.7 * a));
        CHECK(C::AdvectedScalarFlux(a, b, -.7, -.7, .7) == doctest::Approx(-.7 * b));
    }
}
TEST_CASE("Every Cauchy Riemann selection matches legacy perfect gas facial and mean state behavior")
{
    C::EquationSettings settings;
    const auto equation = C::EquationSystem::Resolve(settings);
    const auto &gas = equation.Gas();
    const std::pair<const char *, L::RiemannSolverType> solvers[]{
        {"Roe", L::Roe}, {"Roe_M1", L::Roe_M1}, {"Roe_M2", L::Roe_M2}, {"Roe_M3", L::Roe_M3}, {"Roe_M4", L::Roe_M4}, {"Roe_M5", L::Roe_M5}, {"Roe_M6", L::Roe_M6}, {"Roe_M7", L::Roe_M7}, {"Roe_M8", L::Roe_M8}, {"Roe_M9", L::Roe_M9}, {"HLLC", L::HLLC}, {"HLLEP", L::HLLEP}, {"HLLEP_V1", L::HLLEP_V1}, {"AUSMPWPlus", L::AUSMPWPlus}, {"AUSMPlusUp", L::AUSMPlusUp}, {"AUSMPlusUp2", L::AUSMPlusUp2}};
    for (const auto &[name, legacyKind] : solvers)
        for (int sample = 0; sample < 24; ++sample)
        {
            INFO("flux=" << name);
            CAPTURE(sample);
            const bool equal = sample % 6 == 0;
            const double speed = sample < 12 ? .03 * sample : sample % 2 ? 10
                                                                         : -10;
            const auto left = gas.Conservative({1 + .03 * sample, speed, .3, -.2, 1 + .07 * sample});
            const auto right = equal ? left : gas.Conservative({.4 + .02 * sample, -.3 * speed, .1, .4, .3 + .03 * sample});
            const auto meanL = sample % 3 == 0 ? left : gas.Conservative({1.4, .7 * speed, .1, -.1, 1.3});
            const auto meanR = equal ? meanL : sample % 3 == 0 ? right
                                                               : gas.Conservative({.8, -.2 * speed, .2, .3, .7});
            const C::Vector3 normal = sample % 2 ? C::Vector3{1. / 3, 2. / 3, -2. / 3} : C::Vector3{1, 0, 0};
            const C::Vector3 grid = sample % 4 == 0 ? C::Vector3{.1, -.2, .05} : C::Vector3{};
            const auto flux = F::RiemannFlux::Resolve(name);
            const auto faceL = equation.PrepareFluxState(left), faceR = equation.PrepareFluxState(right);
            const auto prepared = equation.PrepareRiemann(flux, faceL, faceR, meanL, meanR, normal, grid);
            const auto input = prepared.Input(faceL, faceR, normal, grid);
            F::FluxSettings controls{.35, 1.2, .85, .12, sample % 2 ? .15 : 1e300};
            flux.Validate(input, controls);
            const auto result = flux.Evaluate(input, controls);
            Eigen::Matrix<double, 5, 1> oldL, oldR, oldML, oldMR, reference;
            Eigen::Vector3d n, vg;
            for (int j = 0; j < 5; ++j)
            {
                oldL[j] = left[j];
                oldR[j] = right[j];
                oldML[j] = meanL[j];
                oldMR[j] = meanR[j];
            }
            for (int d = 0; d < 3; ++d)
            {
                n[d] = normal[d];
                vg[d] = grid[d];
            }
            double minus = 0, contact = 0, plus = 0;
            L::InviscidFlux_IdealGas_Dispatcher<3>(legacyKind, oldL, oldR, oldML, oldMR, vg, n, gas.gamma, gas.gamma, reference, controls.dLambda, controls.fixScale, controls.incrementScale, [] {}, minus, contact, plus, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, controls.referenceMach, controls.pressureStencilMin);
            for (int j = 0; j < 5; ++j)
                CHECK(result.flux[j] == doctest::Approx(reference[j]).epsilon(5e-12).scale(1));
            CHECK(result.acousticMinus == doctest::Approx(minus).epsilon(1e-13));
            CHECK(result.contact == doctest::Approx(contact).epsilon(1e-13));
            CHECK(result.acousticPlus == doctest::Approx(plus).epsilon(1e-13));
        }
}
TEST_CASE("Every selected flux exposes explicit common-input requirements and missing data errors")
{
    const F::AcousticState state{{{1, 0, 0, 0, 3}, 1}, 2};
    for (const auto *name : {"Roe", "Roe_M1", "Roe_M2", "Roe_M3", "Roe_M4", "Roe_M5", "Roe_M6", "Roe_M7", "Roe_M8", "Roe_M9",
                             "HLLC", "HLLEP", "HLLEP_V1", "AUSMPWPlus", "AUSMPlusUp", "AUSMPlusUp2"})
    {
        const auto flux = F::RiemannFlux::Resolve(name);
        const F::FluxInput input{state, state, {1, 0, 0}};
        CHECK_THROWS(flux.Validate(input));
    }
    CHECK_NOTHROW(F::RiemannFlux::Resolve("rusanov").Validate({state, state, {1, 0, 0}}));
    CHECK_FALSE(F::RiemannFlux::Available("unknown"));
}
TEST_CASE("All Euler fluxes satisfy stationary face consistency and reversal independently of legacy")
{
    C::EquationSettings settings;
    const auto equation = C::EquationSystem::Resolve(settings);
    const auto &gas = equation.Gas();
    const auto left = equation.PrepareFluxState(gas.Conservative({1.2, .4, -.2, .3, 1.1}));
    const auto right = equation.PrepareFluxState(gas.Conservative({.8, -.3, .1, .5, .7}));
    const C::Vector3 normal{1. / 3, 2. / 3, -2. / 3}, reverse{-1. / 3, -2. / 3, 2. / 3};
    for (const auto *name : {"rusanov", "Roe", "Roe_M1", "Roe_M2", "Roe_M3", "Roe_M4", "Roe_M5", "Roe_M6", "Roe_M7", "Roe_M8", "Roe_M9",
                             "HLLC", "HLLEP", "HLLEP_V1", "AUSMPWPlus", "AUSMPlusUp", "AUSMPlusUp2"})
    {
        CAPTURE(name);
        const auto solver = F::RiemannFlux::Resolve(name);
        const auto forwardData = equation.PrepareRiemann(solver, left, right, left.flow.conservative, right.flow.conservative, normal);
        const auto reverseData = equation.PrepareRiemann(solver, right, left, right.flow.conservative, left.flow.conservative, reverse);
        const auto equalData = equation.PrepareRiemann(solver, left, left, left.flow.conservative, left.flow.conservative, normal);
        const auto forward = solver.Evaluate(forwardData.Input(left, right, normal)).flux;
        const auto backward = solver.Evaluate(reverseData.Input(right, left, reverse)).flux;
        const auto equal = solver.Evaluate(equalData.Input(left, left, normal)).flux;
        const auto physical = F::PhysicalFlux(left.flow, normal);
        for (int j = 0; j < 5; ++j)
        {
            CHECK(forward[j] == doctest::Approx(-backward[j]).epsilon(2e-12).scale(1));
            CHECK(equal[j] == doctest::Approx(physical[j]).epsilon(2e-12).scale(1));
        }
    }
}
