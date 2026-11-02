#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Solver/FieldTimeStepper.hpp"
#include "Solver/FieldDITR.hpp"
#include <Eigen/LU>

namespace A = DNDS::Algebra;
namespace O = DNDS::ODE;
namespace R = DNDS::Runtime;
using Field = A::DofField<DNDS::real>;
namespace
{
    double Value(const Field &field)
    {
        const auto read = field.Owned().Read();
        return read.View<R::UniformTableRows>()(0, 0);
    }
    auto Evaluate()
    {
        return [](Field &state, Field &rate, double time, int)
        {
            const double value = Value(state);
            auto write = rate.Owned().Write();
            write.View()(0, 0) = -2 * value + 2 * std::cos(time) - std::sin(time);
        };
    }
    auto Solve()
    {
        return [](Field &state, const Field &source, double mass, double scale, double time, int)
        {
            const double value = (Value(source) + scale * (2 * std::cos(time) - std::sin(time))) / (mass + 2 * scale);
            auto write = state.Owned().Write();
            write.View()(0, 0) = value;
            return true;
        };
    }
    double Error(const std::string &method, int steps)
    {
        const A::DofSpace space(1);
        Field state(space, 1), trial(space, 1);
        A::Fill(state, 1.);
        O::FieldTimeStepper stepper(state, method);
        const double dt = 1. / steps;
        for (int i = 0; i < steps; ++i)
        {
            const double oldValue = Value(state);
            REQUIRE(stepper.Prepare(state, trial, i * dt, dt, Evaluate(), Solve()));
            CHECK(Value(state) == oldValue);
            A::Copy(state, trial);
            stepper.Commit();
        }
        return std::abs(Value(state) - std::cos(1.));
    }
}
TEST_CASE("Native implicit time methods converge with independent nonautonomous exact stages")
{
    const std::vector<std::pair<std::string, double>> methods{{"bdf1", .8}, {"bdf2", 1.8}, {"trapz", 1.9}, {"esdirk2", 1.8}, {"esdirk3", 2.8}, {"esdirk4", 3.5}};
    for (const auto &[method, minimumOrder] : methods)
    {
        const double coarse = Error(method, 8), fine = Error(method, 16);
        CAPTURE(method);
        CAPTURE(coarse);
        CAPTURE(fine);
        CHECK(coarse > 0);
        CHECK(fine > 0);
        CHECK(std::log2(coarse / fine) >= minimumOrder);
    }
}
TEST_CASE("Native BDF1 steady limit has zero physical mass and history source at a frozen time")
{
    const A::DofSpace space(2);
    Field initial(space, 2), trial(space, 2);
    A::Fill(initial, 3.);
    O::FieldTimeStepper stepper(initial, "bdf1");
    stepper.RestoreHistory(initial, .125);
    Eigen::Matrix2d matrix;
    matrix << -2, 1, .5, -3;
    const Eigen::Vector2d forcing(1, 2), exact = matrix.fullPivLu().solve(-forcing);
    const auto version = initial.Owned().Version();
    int calls = 0;
    const auto solve = [&](Field &state, const Field &source, double mass, double scale, double time, int stage)
    {
        ++calls;
        CHECK(mass == 0);
        CHECK(scale == 1);
        CHECK(time == 2.75);
        CHECK(stage == 0);
        const auto read = source.Owned().Read();
        auto write = state.Owned().Write();
        for (DNDS::index iRow = 0; iRow < 2; ++iRow)
            for (int j = 0; j < 2; ++j)
            {
                CHECK(read.View()(iRow, j) == 0);
                write.View()(iRow, j) = exact[j];
            }
        return calls != 1; // Rejected preparation must preserve accepted history.
    };
    CHECK_FALSE(stepper.PrepareSteady(initial, trial, 2.75, solve));
    CHECK(stepper.HasHistory());
    CHECK(stepper.PreviousDt() == .125);
    CHECK(initial.Owned().Version() == version);
    CHECK_THROWS(stepper.Commit());
    stepper.Discard();
    REQUIRE(stepper.PrepareSteady(initial, trial, 2.75, solve));
    CHECK(initial.Owned().Version() == version);
    stepper.Commit();
    CHECK_FALSE(stepper.HasHistory());
    CHECK(stepper.PreviousDt() == 0);
    const auto read = trial.Owned().Read();
    for (DNDS::index iRow = 0; iRow < 2; ++iRow)
        for (int j = 0; j < 2; ++j)
            CHECK(read.View()(iRow, j) == doctest::Approx(exact[j]));
    O::FieldTimeStepper wrong(initial, "esdirk4");
    CHECK_THROWS(wrong.PrepareSteady(initial, trial, 2.75, solve));
    CHECK_THROWS(stepper.PrepareSteady(initial, initial, 2.75, solve));
}
TEST_CASE("Native time target comparison excludes only relative roundoff remainders")
{
    CHECK(O::FieldTimeStepper::ReachedTimeTarget(300 - 2e-13, 300));
    CHECK(O::FieldTimeStepper::ReachedTimeTarget(300, 300));
    CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(299.9, 300));
    CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(0, 1e-20));
    CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(1, 1.00001));
    for (const auto [steps, dt] : {std::pair<int, double>{800, .00125}, {2500, .1}})
    {
        double time = 0;
        for (int iStep = 0; iStep < steps - 1; ++iStep)
            time += dt;
        CHECK_FALSE(O::FieldTimeStepper::ReachedTimeTarget(time, steps * dt));
        time += dt;
        CHECK(O::FieldTimeStepper::ReachedTimeTarget(time, steps * dt));
    }
}
TEST_CASE("Native BDF history is committed explicitly and retries use the last accepted step ratio")
{
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    O::FieldTimeStepper stepper(state, "bdf2");
    CHECK_FALSE(stepper.HasHistory());
    CHECK_THROWS(stepper.PreviousState());
    CHECK_THROWS(stepper.Commit());
    CHECK(stepper.Prepare(state, trial, 0, .1, Evaluate(), Solve()));
    const double first = Value(trial);
    stepper.Discard();
    CHECK_FALSE(stepper.HasHistory());
    CHECK_THROWS(stepper.Commit());
    CHECK(stepper.Prepare(state, trial, 0, .1, Evaluate(), Solve()));
    CHECK(Value(trial) == first);
    A::Copy(state, trial);
    stepper.Commit();
    CHECK(stepper.HasHistory());
    CHECK(stepper.PreviousDt() == .1);
    CHECK(Value(stepper.PreviousState()) == 1);
    // Rejected .2 trial must not replace accepted .1 history. The next .05
    // trial has r=.5, independent of the failed attempt's r=2.
    CHECK_FALSE(stepper.Prepare(state, trial, .1, .2, Evaluate(), [](auto &, const auto &, auto, auto, auto, auto)
                                { return false; }));
    CHECK(stepper.PreviousDt() == .1);
    CHECK(Value(stepper.PreviousState()) == 1);
    double observedMass = 0, observedScale = 0, observedSource = 0;
    CHECK(stepper.Prepare(state, trial, .1, .05, Evaluate(), [&](Field &x, const Field &source, double mass, double scale, double time, int stage)
                          {
        observedMass = mass; observedScale = scale; observedSource = Value(source);
        return Solve()(x, source, mass, scale, time, stage); }));
    CHECK(observedMass == 20);
    CHECK(observedScale == .75);
    CHECK(observedSource == doctest::Approx((1.125 * first - .125) * 20));
    O::FieldTimeStepper restored(state, "bdf2");
    restored.RestoreHistory(stepper.PreviousState(), stepper.PreviousDt());
    Field other(space, 1);
    CHECK(restored.Prepare(state, other, .1, .05, Evaluate(), Solve()));
    CHECK(Value(other) == Value(trial));
    stepper.ResetHistory();
    CHECK_FALSE(stepper.HasHistory());
    CHECK_THROWS(stepper.Prepare(state, state, .1, .05, Evaluate(), Solve()));
    CHECK_THROWS(O::FieldTimeStepper(state, "unknown"));
}
TEST_CASE("Native DITR solves both original coupled equations with an independent dense oracle")
{
    const A::DofSpace space(1);
    Field state(space, R::RowLayout::TiledColMajor(0, 1, 8)), trial(space, 1);
    A::Fill(state, .7);
    int factories = 0, resets = 0;
    const auto factory = [&](Field &mid)
    {
        ++factories;
        CHECK(mid.Owned().SharedStorage() != trial.Owned().SharedStorage());
        return O::FieldDITR::StageOperator{Evaluate(), Solve(), [&]
                                           { ++resets; }};
    };
    O::FieldDITR ditr(state, {}, factory);
    const double t = .3, h = .2, a = .55;
    // Fixed numeric Hermite/integration coefficients, independent of the
    // production coefficient builder and transformed mid-equation code.
    const double c0 = .42525, c1 = .57475, c2 = .111375, c3 = -.136125;
    const double w0 = 13. / 66, w1 = 200. / 297, w2 = 7. / 54;
    const auto g = [](double x)
    { return 2 * std::cos(x) - std::sin(x); };
    Eigen::Matrix2d matrix;
    matrix << 1 / h, -c1 / h + 2 * c3, 2 * w1, 1 / h + 2 * w2;
    Eigen::Vector2d forcing;
    forcing << c0 * .7 / h + c2 * (-2 * .7 + g(t)) + c3 * g(t + h),
        .7 / h + w0 * (-2 * .7 + g(t)) + w1 * g(t + a * h) + w2 * g(t + h);
    const Eigen::Vector2d exact = matrix.fullPivLu().solve(forcing);
    double lastH = 0, lastI = 0;
    const auto check = [&](const Field &hermite, const Field &integral, DNDS::index)
    {
        lastH = Value(hermite);
        lastI = Value(integral);
        return std::max(std::abs(lastH), std::abs(lastI)) < 2e-13;
    };
    REQUIRE(ditr.Prepare(state, trial, t, h, Evaluate(), Solve(), check));
    CHECK(Value(trial) == doctest::Approx(exact[1]).epsilon(2e-13));
    CHECK(Value(state) == .7);
    CHECK(ditr.LastReport().converged);
    CHECK(ditr.LastReport().iterations > 0);
    CHECK(std::abs(lastH) < 2e-13);
    CHECK(std::abs(lastI) < 2e-13);
    const double first = Value(trial);
    REQUIRE(ditr.Prepare(state, trial, t, h, Evaluate(), Solve(), check));
    CHECK(Value(trial) == first);
    CHECK(factories == 1);
    CHECK(resets == 2);
}
TEST_CASE("Native DITR nonautonomous convergence and capped retry distinguish physical acceptance")
{
    const auto error = [](int steps, double fraction)
    {
        const A::DofSpace space(1);
        Field state(space, 1), trial(space, 1);
        A::Fill(state, 1.);
        O::DITRSettings settings;
        settings.fraction = fraction;
        O::FieldDITR ditr(state, settings, [](Field &)
                          { return O::FieldDITR::StageOperator{Evaluate(), Solve(), {}}; });
        const auto check = [](const Field &h, const Field &i, DNDS::index)
        { return std::max(std::abs(Value(h)), std::abs(Value(i))) < 2e-13; };
        for (int step = 0; step < steps; ++step)
        {
            REQUIRE(ditr.Prepare(state, trial, double(step) / steps, 1. / steps, Evaluate(), Solve(), check));
            A::Copy(state, trial);
        }
        return std::abs(Value(state) - std::cos(1.));
    };
    for (double fraction : {.55, .5})
    {
        // At alpha=.55 the independent exact 2x2 method has orders2.736,
        // 2.883,2.945 on8/16/32/64: the third-order leading coefficient is
        // small near alpha=.5. Use the asymptotic pair and a stricter bound.
        const double coarse = error(32, fraction), fine = error(64, fraction);
        CAPTURE(fraction);
        CAPTURE(coarse);
        CAPTURE(fine);
        CHECK(coarse > 0);
        CHECK(fine > 0);
        CHECK(std::log2(coarse / fine) > (fraction == .5 ? 3.7 : 2.9));
    }
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    O::DITRSettings settings;
    settings.iterations = 5;
    O::FieldDITR capped(state, settings, [](Field &)
                        { return O::FieldDITR::StageOperator{Evaluate(), Solve(), {}}; });
    const auto check = [](const Field &h, const Field &i, DNDS::index)
    { return std::max(std::abs(Value(h)), std::abs(Value(i))) < 2e-10; };
    CHECK_FALSE(capped.Prepare(state, trial, 0, .2, Evaluate(), Solve(), check));
    CHECK_FALSE(capped.LastReport().converged);
    CHECK(capped.LastReport().iterations == 5);
    CHECK(Value(state) == 1);
    CHECK(capped.Prepare(state, trial, 0, .001, Evaluate(), Solve(), check));
    CHECK(Value(trial) == doctest::Approx(std::cos(.001)).epsilon(2e-10));
    CHECK_THROWS(capped.Prepare(state, state, 0, .01, Evaluate(), Solve(), check));
    CHECK_THROWS(O::FieldDITR(state, {}, {}));
}
TEST_CASE("DITR rejects intermediate rebinding and stateful rate callbacks before accepted-state mutation")
{
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    const auto check = [](const Field &h, const Field &i, DNDS::index)
    { return std::max(std::abs(Value(h)), std::abs(Value(i))) < 1e-12; };
    O::FieldDITR rebound(state, {}, [&](Field &mid)
                         { mid = state; return O::FieldDITR::StageOperator{Evaluate(), Solve(), {}}; });
    const auto generation = state.Owned().Version();
    CHECK_THROWS(rebound.Prepare(state, trial, 0, .1, Evaluate(), Solve(), check));
    CHECK(state.Owned().Version() == generation);
    CHECK_THROWS(rebound.Prepare(state, trial, 0, .1, Evaluate(), Solve(), check));
    O::FieldDITR mutation(state, {}, [](Field &)
                          { return O::FieldDITR::StageOperator{Evaluate(), Solve(), {}}; });
    const auto bad = [](Field &input, Field &output, double, int)
    { A::Scale(input, 2.); A::Copy(output, input); };
    CHECK_THROWS(mutation.Prepare(state, trial, 0, .1, bad, Solve(), check));
    CHECK(state.Owned().Version() == generation);
    CHECK(Value(state) == 1);
    CHECK_THROWS(O::DITRCoefficients(0));
    CHECK_THROWS(O::DITRCoefficients(1));
    O::DITRSettings settings;
    settings.iterations = 0;
    CHECK_THROWS(O::FieldDITR(state, settings, [](Field &)
                              { return O::FieldDITR::StageOperator{Evaluate(), Solve(), {}}; }));
}
TEST_CASE("Native time workspace integrates DITR rejection history and fresh intermediate bindings")
{
    const A::DofSpace space(1);
    Field state(space, 1), trial(space, 1);
    A::Fill(state, 1.);
    std::vector<R::ArrayStorage::Shared> bindings;
    const auto factory = [&](Field &mid)
    {
        bindings.push_back(mid.Owned().SharedStorage());
        return O::FieldDITR::StageOperator{Evaluate(), Solve(), {}};
    };
    O::FieldTimeStepper stepper(state, "ditr", {}, {}, factory);
    const auto check = [](const Field &h, const Field &i, DNDS::index)
    { return std::max(std::abs(Value(h)), std::abs(Value(i))) < 2e-13; };
    CHECK_THROWS(stepper.Prepare(state, trial, 0, .1, Evaluate(), Solve())); // explicit coupled checker required
    REQUIRE(stepper.Prepare(state, trial, 0, .1, Evaluate(), Solve(), check));
    CHECK(stepper.HasPendingTrial());
    CHECK_FALSE(stepper.HasHistory());
    REQUIRE(stepper.LastDITRReport());
    CHECK(stepper.LastDITRReport()->converged);
    const double expected = Value(trial);
    stepper.Discard();
    CHECK_FALSE(stepper.HasPendingTrial());
    CHECK_THROWS(stepper.Commit());
    REQUIRE(stepper.Prepare(state, trial, 0, .1, Evaluate(), Solve(), check));
    CHECK(Value(trial) == expected);
    CHECK(bindings.size() == 1);
    auto fresh = stepper.Fresh();
    REQUIRE(fresh->Prepare(state, trial, 0, .1, Evaluate(), Solve(), check));
    REQUIRE(bindings.size() == 2);
    CHECK(bindings[0] != bindings[1]);
    CHECK(Value(trial) == expected);
    A::Copy(state, trial);
    fresh->Commit();
    CHECK(fresh->HasHistory());
    CHECK(fresh->PreviousDt() == .1);
    CHECK(Value(fresh->PreviousState()) == 1);
    const auto bad = [](Field &input, Field &output, double, int)
    { A::Scale(input, 2.); A::Copy(output, input); };
    CHECK_THROWS(fresh->Prepare(state, trial, .1, .1, bad, Solve(), check));
    CHECK_FALSE(fresh->HasPendingTrial());
    CHECK(fresh->PreviousDt() == .1);
    REQUIRE(fresh->Prepare(state, trial, .1, .1, Evaluate(), Solve(), check));
    CHECK(bindings.size() == 3);
    CHECK(Value(state) == expected);
}
