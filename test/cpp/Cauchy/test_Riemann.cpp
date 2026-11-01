#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
// Deliberately no PerfectGas/EOS/energy-model header in this independent test.
#include "Cauchy/Riemann/RiemannFlux.hpp"

namespace C = DNDS::Cauchy;
namespace F = DNDS::Cauchy::Riemann;
TEST_CASE("Riemann module uses supplied pressure and acoustic data independently of energy")
{
    const F::AcousticState left{{{2, 6, -8, 10, 100}, 7}, 11};
    const F::AcousticState right{{{1, -2, 1, 3, -50}, 19}, 5};
    // Negative conservative energy is intentional: a shifted energy origin
    // need not resemble a perfect-gas thermal energy. Positivity is EOS-owned.
    const C::FlowState leftFlux{6, 25, -24, 30, 321};
    const C::FlowState rightFlux{-2, 23, -2, -6, 62};
    const auto solver = F::RiemannFlux::Resolve("rusanov");
    const auto requirements = solver.RequiredInputs();
    CHECK(requirements.pressure);
    CHECK(requirements.soundSpeed);
    CHECK_FALSE(requirements.enthalpy);
    const auto flux = solver.Evaluate({left, right, {1, 0, 0}}).flux;
    // Independently supplied acoustic speeds give alpha=max(3+11,2+5)=14.
    for (int j = 0; j < 5; ++j)
        CHECK(flux[j] == doctest::Approx(.5 * (leftFlux[j] + rightFlux[j] - 14 * (right.flow.conservative[j] - left.flow.conservative[j]))));
    auto changed = left;
    changed.flow.pressure += 4;
    const auto pressureFlux = solver.Evaluate({changed, right, {1, 0, 0}}).flux;
    CHECK(pressureFlux[0] == flux[0]);
    CHECK(pressureFlux[1] - flux[1] == doctest::Approx(2));
    CHECK(pressureFlux[4] - flux[4] == doctest::Approx(6));
    changed = left;
    changed.soundSpeed += 2;
    const auto acousticFlux = solver.Evaluate({changed, right, {1, 0, 0}}).flux;
    for (int j = 0; j < 5; ++j)
        CHECK(acousticFlux[j] - flux[j] == doctest::Approx(left.flow.conservative[j] - right.flow.conservative[j]));
    CHECK_THROWS(F::RiemannFlux::Resolve("unknown"));
}
TEST_CASE("Energy-origin shifts and oriented reversal preserve the supplied Euler flux convention")
{
    F::AcousticState left{{{2, 0, 4, -2, 8}, 3}, 2};
    F::AcousticState right{{{1, 0, -3, 2, 4}, 5}, 7};
    const auto original = F::Rusanov(left, right, {1, 0, 0});
    const auto reversed = F::Rusanov(right, left, {-1, 0, 0});
    for (int j = 0; j < 5; ++j)
        CHECK(reversed[j] == -original[j]);
    const double offset = -1234;
    left.flow.conservative[4] += offset * left.flow.conservative[0];
    right.flow.conservative[4] += offset * right.flow.conservative[0];
    const auto shifted = F::Rusanov(left, right, {1, 0, 0});
    for (int j = 0; j < 4; ++j)
        CHECK(shifted[j] == original[j]);
    CHECK(shifted[4] == doctest::Approx(original[4] + offset * original[0]));
    const auto equal = F::Rusanov(left, left, {1, 0, 0});
    CHECK(equal == C::FlowState{0, 3, 0, 0, 0});
}
TEST_CASE("Every flux accepts external thermodynamics and transforms covariantly under energy-origin changes")
{
    const F::AcousticState left{{{2, 6, -8, 10, 100}, 7}, 11};
    const F::AcousticState right{{{1, -2, 1, 3, -50}, 19}, 5};
    // Deliberately independent data: no ideal-gas interpretation of these
    // energies, pressures or acoustic scales is assumed by the flux module.
    const F::RoeData roe{{.5, -.7, .1}, 9, 12, .375, {1, .2, -.3, .1, -.07}};
    const F::WaveBounds bounds{-12, 13};
    const F::AUSMData ausm{8};
    constexpr double offset = -1234;
    for (const auto *name : {"rusanov", "Roe", "Roe_M1", "Roe_M2", "Roe_M3", "Roe_M4", "Roe_M5", "Roe_M6", "Roe_M7", "Roe_M8", "Roe_M9",
                             "HLLC", "HLLEP", "HLLEP_V1", "AUSMPWPlus", "AUSMPlusUp", "AUSMPlusUp2"})
    {
        CAPTURE(name);
        const auto solver = F::RiemannFlux::Resolve(name);
        F::FluxInput input{left, right, {1, 0, 0}, {}, &left, &right, &roe, &bounds, &ausm};
        solver.Validate(input);
        const auto original = solver.Evaluate(input).flux;
        auto shiftedLeft = left, shiftedRight = right;
        shiftedLeft.flow.conservative[4] += offset * left.flow.conservative[0];
        shiftedRight.flow.conservative[4] += offset * right.flow.conservative[0];
        auto shiftedRoe = roe;
        shiftedRoe.enthalpy += offset;
        shiftedRoe.contactEnergy += offset;
        // Inverse-basis covector follows the same conservative variable change.
        shiftedRoe.contactProjection[0] -= offset * roe.contactProjection[4];
        input.left = shiftedLeft;
        input.right = shiftedRight;
        input.leftMean = &shiftedLeft;
        input.rightMean = &shiftedRight;
        input.roe = &shiftedRoe;
        solver.Validate(input);
        const auto shifted = solver.Evaluate(input).flux;
        for (int j = 0; j < 4; ++j)
            CHECK(shifted[j] == doctest::Approx(original[j]).epsilon(2e-12).scale(1));
        CHECK(shifted[4] == doctest::Approx(original[4] + offset * original[0]).epsilon(2e-12).scale(1));
    }
}
