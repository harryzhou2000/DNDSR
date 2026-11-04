#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Solver/FAS.hpp"

namespace M = DNDS::Multigrid;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
using Field = M::Field;
using I = DNDS::index;
namespace
{
    void Set(Field &field, const std::vector<double> &values)
    {
        REQUIRE(I(values.size()) == I(field.Space().OwnedRows()));
        auto write = field.Owned().Write();
        for (I iRow = 0; iRow < I(values.size()); ++iRow)
            write.View<R::UniformTableRows>()(iRow, 0) = values[iRow];
    }
    std::vector<double> Get(const Field &field)
    {
        const auto read = field.Owned().Read();
        std::vector<double> values(field.Space().OwnedRows());
        for (I iRow = 0; iRow < I(values.size()); ++iRow)
            values[iRow] = read.View<R::UniformTableRows>()(iRow, 0);
        return values;
    }
    M::FASTransfer PairTransfer()
    {
        const auto restrict = [](double scale)
        {
            return [scale](const Field &source, Field &destination)
            {
                const auto values = Get(source);
                std::vector<double> result(destination.Space().OwnedRows());
                REQUIRE(values.size() == 2 * result.size());
                for (I iRow = 0; iRow < I(result.size()); ++iRow)
                    result[iRow] = scale * (values[2 * iRow] + values[2 * iRow + 1]);
                Set(destination, result);
            };
        };
        return {restrict(.5), restrict(1), [](const Field &source, Field &destination)
                {
                    const auto values = Get(source);
                    std::vector<double> result(destination.Space().OwnedRows());
                    REQUIRE(result.size() == 2 * values.size());
                    for (I iRow = 0; iRow < I(result.size()); ++iRow)
                        result[iRow] = values[iRow / 2];
                    Set(destination, result);
                }};
    }
    auto Multiply(double coefficient)
    {
        return [coefficient](Field &source, Field &output)
        { A::Copy(output, source); A::Scale(output, coefficient); };
    }
    auto ExactDiagonal(double coefficient, std::vector<double> &observed)
    {
        return [coefficient, &observed](Field &state, const Field &source, I)
        {
            observed = Get(source);
            A::Copy(state, source);
            A::Scale(state, 1 / coefficient);
            return I(1);
        };
    }
    auto NoSmooth()
    {
        return [](Field &, const Field &, I)
        { return I(0); };
    }
}
TEST_CASE("FAS distinguishes state and residual restriction and preserves incoming references through three levels")
{
    const A::DofSpace fineSpace(4), coarseSpace(2), bottomSpace(1);
    Field fine(fineSpace, R::RowLayout::Dense(0, 1, 3)), rhs(fineSpace, 1);
    Field coarse(coarseSpace, R::RowLayout::ColMajor(0, 1, {8, 1}));
    Field bottom(bottomSpace, R::RowLayout::TiledColMajor(0, 1, 8));
    Set(fine, {1, 3, 5, 7});
    Set(rhs, {4, 8, 12, 16});
    const auto version = rhs.Owned().Version();
    std::vector<double> observed;
    M::FASCycle two({{fine, Multiply(2), NoSmooth(), {}}, {coarse, Multiply(3), ExactDiagonal(3, observed), {}}}, {PairTransfer()}, {0, 0, 1});
    const auto first = two.Apply(rhs);
    REQUIRE(observed.size() == 2);
    CHECK(observed[0] == 10);
    CHECK(observed[1] == 22);
    const auto actual = Get(fine);
    const std::vector<double> expected{7. / 3, 13. / 3, 19. / 3, 25. / 3};
    for (I iRow = 0; iRow < 4; ++iRow)
        CHECK(actual[iRow] == doctest::Approx(expected[iRow]).epsilon(1e-14));
    CHECK(first.cycles == 1);
    CHECK(first.operatorEvaluations == 2);
    CHECK(first.corrections == std::vector<I>{0, 1});
    CHECK(rhs.Owned().Version() == version);
    Set(fine, {1, 3, 5, 7});
    const auto middleSmooth = [](Field &state, const Field &source, I count)
    {
        const auto b = Get(source);
        auto u = Get(state);
        for (I sweep = 0; sweep < count; ++sweep)
            for (I iRow = 0; iRow < I(u.size()); ++iRow)
                u[iRow] += (3. / 16) * (b[iRow] / 3 - u[iRow]);
        Set(state, u);
        return count;
    };
    M::FASCycle three({{fine, Multiply(2), NoSmooth(), {}}, {coarse, Multiply(3), middleSmooth, {}}, {bottom, Multiply(5), ExactDiagonal(5, observed), {}}},
                      {PairTransfer(), PairTransfer()}, {1, 0, 1});
    const auto report = three.Apply(rhs);
    REQUIRE(observed.size() == 1);
    CHECK(observed[0] == doctest::Approx(27.75));
    // The middle pre-sweep changes the restricted state by .25. The incoming
    // reference must remain [2,6], giving total coarse correction1.55, rather
    // than the child-only correction1.30 from an overwritten reference.
    const auto result = Get(fine);
    for (I iRow = 0; iRow < 4; ++iRow)
        CHECK(result[iRow] == doctest::Approx(1 + 2 * iRow + 1.55).epsilon(1e-14));
    CHECK(report.corrections == std::vector<I>{0, 1, 1});
    CHECK(report.operatorEvaluations == 4);
}
TEST_CASE("Identity FAS cancels a common large linear term before forming tau")
{
    const A::DofSpace space(1);
    Field fine(space, 1), coarse(space, 1), source(space, 1);
    Set(fine, {4});
    Set(source, {2});
    std::vector<double> observed;
    M::FASLevel upper{fine, Multiply(1e16 - 2), NoSmooth(), {}, {}, Multiply(-2)};
    const auto inspectSource = [&](Field &, const Field &rhs, I)
    {
        observed = Get(rhs);
        return I(0); // Inspect assembly only; no mock convergence claim.
    };
    M::FASLevel lower{coarse, Multiply(1e16 - 3), inspectSource, {}, {}, Multiply(-3)};
    const auto copy = [](const Field &a, Field &b)
    { A::Copy(b, a); };
    M::FASCycle cycle({upper, lower}, {{copy, copy, copy, true}}, {0, 0, 1});
    cycle.Apply(source);
    // b_c=b_f+(-3*4)-(-2*4)=-2. Forming complete operators first
    // loses this difference against their common 4e16 time contribution.
    CHECK(observed == std::vector<double>{-2});
    CHECK(Get(fine) == std::vector<double>{4});
    Set(source, {1e16});
    upper.applyForIdentityTau = Multiply(-.25);
    lower.applyForIdentityTau = Multiply(.25);
    M::FASCycle history({upper, lower}, {{copy, copy, copy, true}}, {0, 0, 1});
    history.Apply(source);
    // (+1)-(-1)=2 survives one addition to resOther. Adding the two
    // individual unit terms to 1e16 loses both of them to rounding.
    CHECK(observed == std::vector<double>{1e16 + 2});
}
TEST_CASE("Nonlinear FAS coarse forcing preserves a known fine solution")
{
    const A::DofSpace fineSpace(4), coarseSpace(2);
    Field fine(fineSpace, 1), rhs(fineSpace, 1), coarse(coarseSpace, 1);
    Set(fine, {1, 2, 3, 4});
    Set(rhs, {2, 5, 10, 17});
    const auto nonlinear = [](double offset)
    {
        return [offset](Field &state, Field &output)
        {
            auto values = Get(state);
            for (auto &value : values)
                value = value * value + offset;
            Set(output, values);
        };
    };
    std::vector<double> forcing;
    const auto solve = [&](Field &state, const Field &source, I)
    {
        forcing = Get(source);
        auto values = forcing;
        for (auto &value : values)
            value = std::sqrt(value - 2);
        Set(state, values);
        return I(1);
    };
    M::FASCycle fas({{fine, nonlinear(1), NoSmooth(), {}}, {coarse, nonlinear(2), solve, {}}}, {PairTransfer()}, {0, 0, 1});
    fas.Apply(rhs);
    CHECK(forcing == std::vector<double>{4.25, 14.25});
    CHECK(Get(fine) == std::vector<double>{1, 2, 3, 4});
}
TEST_CASE("FAS rejects ambiguous ownership controls aliases and stateful operator callbacks")
{
    const A::DofSpace space(2), coarseSpace(1);
    Field state(space, 1), rhs(space, 1), coarse(coarseSpace, 1);
    Set(state, {1, 2});
    Set(rhs, {3, 4});
    const auto level = M::FASLevel{state, Multiply(2), NoSmooth(), {}};
    CHECK_THROWS(M::FASCycle({level}, {}, {0, 0, 0}));
    CHECK_THROWS(M::FASCycle({level, level}, {PairTransfer()}));
    M::FASCycle single({level}, {}, {0, 0, 1});
    CHECK_THROWS(single.Apply(state));
    CHECK_THROWS(single.Apply(rhs, 0));
    const auto bad = [](Field &input, Field &output)
    { A::Scale(input, 2.); A::Copy(output, input); };
    M::FASCycle unsafe({{state, bad, NoSmooth(), {}}, {coarse, Multiply(2), NoSmooth(), {}}}, {PairTransfer()}, {0, 0, 1});
    CHECK_THROWS(unsafe.Apply(rhs));
    CHECK_THROWS(unsafe.Apply(rhs));
    auto badTransfer = PairTransfer();
    badTransfer.restrictState = [](const Field &input, Field &output)
    { output = input; };
    M::FASCycle rebound({level, {coarse, Multiply(2), NoSmooth(), {}}}, {badTransfer}, {0, 0, 1});
    CHECK_THROWS(rebound.Apply(rhs));
    const auto layout = R::RowLayout::Dense(2, 1);
    const auto backing = R::Buffer::Allocate(4 * sizeof(double));
    Field viewA(space, R::Array<double>::FromStorage(layout, backing.Subspan(0, 2 * sizeof(double))), R::Array<double>(R::RowLayout::Dense(0, 1)));
    Field viewB(space, R::Array<double>::FromStorage(layout, backing.Subspan(2 * sizeof(double), 2 * sizeof(double))), viewA.Ghost());
    Set(viewA, {1, 2});
    Set(viewB, {3, 4});
    REQUIRE(viewA.Owned().Identity() == viewB.Owned().Identity());
    REQUIRE(viewA.Owned().SharedStorage() != viewB.Owned().SharedStorage());
    M::FASCycle rangeRebound({{viewA, Multiply(2), [&](Field &input, const Field &, I)
                               { input = viewB; return I(0); },
                               {}}},
                             {}, {0, 0, 1});
    CHECK_THROWS(rangeRebound.Apply(rhs));
}
TEST_CASE("Selected FAS solves principal equations and freezes other components across strided levels")
{
    for (I rows : {I(0), I(3)})
    {
        const A::DofSpace fineSpace(rows), coarseSpace(rows);
        Field fine(fineSpace, R::RowLayout::ColMajor(0, 3, {8, 1})), rhs(fineSpace, 3);
        Field coarse(coarseSpace, R::RowLayout::TiledColMajor(0, 3, 8));
        {
            auto u = fine.Owned().Write(), b = rhs.Owned().Write();
            for (I iRow = 0; iRow < rows; ++iRow)
                for (I j = 0; j < 3; ++j)
                {
                    u.View<R::UniformTableRows>()(iRow, j) = j == 1 ? -0. : .1 * (iRow + j);
                    // The deliberately incompatible frozen equation must not
                    // enter the principal solve through a full block inverse.
                    b.View<R::UniformTableRows>()(iRow, j) = j == 1 ? 1000 : 1 + iRow + j;
                }
        }
        const auto apply = [](double shift)
        {
            return [shift](Field &input, Field &output)
            {
                const auto read = input.Owned().Read();
                auto write = output.Owned().Write();
                const auto u = read.View<R::UniformTableRows>();
                const auto v = write.View<R::UniformTableRows>();
                for (I iRow = 0; iRow < u.RowCount(); ++iRow)
                {
                    v(iRow, 0) = (4 + shift) * u(iRow, 0) + 2 * u(iRow, 1) + u(iRow, 2);
                    v(iRow, 1) = 2 * u(iRow, 0) + 3 * u(iRow, 1) + 2 * u(iRow, 2);
                    v(iRow, 2) = u(iRow, 0) + 2 * u(iRow, 1) + (5 + shift) * u(iRow, 2);
                }
            };
        };
        const auto full = [](Field &, const Field &, I) -> I
        { throw std::runtime_error("full smoother called for subset"); };
        I selectedCalls = 0;
        const auto selected = [&](Field &input, const Field &source, I count, const std::vector<I> &components)
        {
            CHECK(components == std::vector<I>{0, 2});
            ++selectedCalls;
            const auto read = source.Owned().Read();
            auto write = input.Owned().Write();
            const auto b = read.View<R::UniformTableRows>();
            const auto u = write.View<R::UniformTableRows>();
            for (I iRow = 0; iRow < u.RowCount(); ++iRow)
            {
                const double b0 = b(iRow, 0) - 2 * u(iRow, 1), b2 = b(iRow, 2) - 2 * u(iRow, 1);
                // Independent inverse of [[5,1],[1,6]], not full 3x3.
                u(iRow, 0) = (6 * b0 - b2) / 29;
                u(iRow, 2) = (5 * b2 - b0) / 29;
            }
            return count;
        };
        const auto identity = [](const Field &a, Field &b)
        {
            const auto read = a.Owned().Read();
            auto write = b.Owned().Write();
            for (I iRow = 0; iRow < I(a.Space().OwnedRows()); ++iRow)
                for (I j = 0; j < 3; ++j)
                    write.View<R::UniformTableRows>()(iRow, j) = read.View<R::UniformTableRows>()(iRow, j);
        };
        const auto prolong = [identity](const Field &a, Field &b)
        {
            identity(a, b);
            auto write = b.Owned().Write();
            for (I iRow = 0; iRow < I(b.Space().OwnedRows()); ++iRow)
                write.View<R::UniformTableRows>()(iRow, 1) = 1234;
        };
        M::FASCycle cycle({{fine, apply(0), full, {}, selected}, {coarse, apply(1), full, {}, selected}},
                          {{identity, identity, prolong}}, {0, 0, 1, {2, 0}});
        const auto work = cycle.Apply(rhs);
        CHECK(work.cycles == 1);
        CHECK(selectedCalls == 1);
        const auto read = fine.Owned().Read();
        const auto u = read.View<R::UniformTableRows>();
        for (I iRow = 0; iRow < rows; ++iRow)
        {
            const double old0 = .1 * iRow, old2 = .1 * (iRow + 2);
            const double r0 = 1 + iRow - 4 * old0 - old2, r2 = 3 + iRow - old0 - 5 * old2;
            CHECK(u(iRow, 0) == doctest::Approx(old0 + (6 * r0 - r2) / 29).epsilon(1e-14));
            CHECK(u(iRow, 2) == doctest::Approx(old2 + (5 * r2 - r0) / 29).epsilon(1e-14));
            CHECK(std::signbit(u(iRow, 1)));
        }
    }
}
TEST_CASE("Selected FAS rejects invalid selections and callbacks that write frozen means")
{
    const A::DofSpace space(2), coarseSpace(1);
    Field state(space, 2), rhs(space, 2), coarse(coarseSpace, 2);
    A::Fill(state, 1.);
    A::Fill(rhs, 2.);
    auto level = M::FASLevel{state, Multiply(2), NoSmooth(), {}};
    CHECK_THROWS(M::FASCycle({level}, {}, {0, 0, 1, {0}}));
    level.smoothSelected = [](Field &u, const Field &, I, const auto &)
    { A::Scale(u, 2.); return I(1); };
    CHECK_THROWS(M::FASCycle({level}, {}, {0, 0, 1, {0, 0}}));
    CHECK_THROWS(M::FASCycle({level}, {}, {0, 0, 1, {2}}));
    M::FASCycle badSmoother({level}, {}, {0, 0, 1, {0}});
    CHECK_THROWS(badSmoother.Apply(rhs));
    CHECK_THROWS(badSmoother.Apply(rhs));
    level.smoothSelected = [](Field &, const Field &, I, const auto &)
    { return I(0); };
    level.update = [](Field &u, const Field &)
    { A::Scale(u, 2.); };
    auto other = level;
    other.state = coarse;
    const auto transfer = [](const Field &, Field &output)
    { A::Fill(output, 0.); };
    M::FASCycle badUpdate({level, other}, {{transfer, transfer, transfer}}, {0, 0, 1, {0}});
    CHECK_THROWS(badUpdate.Apply(rhs));
    // Explicit full selection takes the original full-state callback path.
    level.update = {};
    level.smooth = NoSmooth();
    M::FASCycle full({level}, {}, {0, 0, 1, {1, 0}});
    CHECK_NOTHROW(full.Apply(rhs));
}
