#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Solver/FieldTimeStepper.hpp"

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
