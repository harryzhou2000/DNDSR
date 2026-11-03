#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/State/DerivedFields.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/Physics/EquationSystem.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using Q = C::DerivedQuantity;

TEST_CASE("Derived point refresh fills selected properties with lazy provenance and stale temperature guesses")
{
    C::DomainSettings settings;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .003;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    for (DNDS::index rows : {0, 35})
        for (int layout = 0; layout < 3; ++layout)
        {
            INFO("rows=" << rows << ", layout=" << layout);
            const auto prototype = layout == 0   ? R::RowLayout::Dense(0, 5, 7)
                                   : layout == 1 ? R::RowLayout::ColMajor(0, 5, {16, 1})
                                                 : R::RowLayout::TiledColMajor(0, 5, 32);
            A::DofSpace space(size_t(rows), 0, "derived fixture");
            auto state = C::EquationDofs::Coupled(equation.Schema(), A::DofField<DNDS::real>(space, prototype));
            {
                auto write = state.WriteOwned();
                const auto u = write.Block("flow");
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                {
                    const double rho = 1 + .01 * iRow, velocity = .3 + .001 * iRow, pressure = 1 + .02 * iRow;
                    u(iRow, 0) = rho;
                    u(iRow, 1) = rho * velocity;
                    u(iRow, 2) = -.1 * rho;
                    u(iRow, 3) = .02 * rho;
                    u(iRow, 4) = pressure / .4 + .5 * rho * (velocity * velocity + .01 + .0004);
                }
            }
            C::DerivedFields derived(space, {Q::Temperature, Q::Pressure, Q::MolecularViscosity});
            const auto sourceVersion = state.CoupledField().Owned().Version();
            CHECK_FALSE(derived.Current(state, equation));
            CHECK(C::RefreshDerivedFields(state, equation, derived));
            CHECK(state.ComponentNumber() == 5);
            CHECK(state.CoupledField().Owned().Version() == sourceVersion);
            CHECK(derived.Values().Layout().Width() == 3);
            {
                auto read = derived.ReadCurrent(state, equation);
                const auto temperature = read.Field(Q::Temperature), pressure = read.Field(Q::Pressure), viscosity = read.Field(Q::MolecularViscosity);
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                {
                    const double rho = 1 + .01 * iRow, p = 1 + .02 * iRow;
                    CHECK(temperature(iRow, 0) == doctest::Approx(p / (rho * equation.Gas().gasConstant)));
                    CHECK(pressure(iRow, 0) == doctest::Approx(p));
                    CHECK(viscosity(iRow, 0) == .003);
                }
            }
            const auto version = derived.Values().Version();
            CHECK_FALSE(C::RefreshDerivedFields(state, equation, derived, true));
            CHECK(derived.Values().Version() == version);
            {
                auto write = state.WriteOwned();
                const auto u = write.Block("flow");
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                    u(iRow, 4) += .25;
            }
            CHECK_FALSE(derived.Current(state, equation));
            CHECK_THROWS(derived.ReadCurrent(state, equation));
            {
                auto guess = derived.ReadTemperatureGuess(state, equation);
                const auto t = guess.Field(Q::Temperature);
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                    CHECK(t(iRow, 0) == doctest::Approx((1 + .02 * iRow) / ((1 + .01 * iRow) * equation.Gas().gasConstant)));
            }
            auto alias = derived;
            CHECK(C::RefreshDerivedFields(state, equation, derived, true));
            CHECK(alias.Current(state, equation));
            {
                auto read = alias.ReadCurrent(state, equation);
                const auto p = read.Field(Q::Pressure);
                for (DNDS::index iRow = 0; iRow < rows; ++iRow)
                    CHECK(p(iRow, 0) == doctest::Approx(1 + .02 * iRow + .1));
            }
            C::DerivedFields disabled(space, {});
            CHECK_FALSE(disabled.Enabled());
            CHECK_FALSE(C::RefreshDerivedFields(state, equation, disabled));
            CHECK(disabled.Values().RowCount() == 0);
        }
}
