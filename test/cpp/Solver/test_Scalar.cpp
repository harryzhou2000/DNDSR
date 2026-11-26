/**
 * @file test_Scalar.cpp
 * @brief Unit tests for BisectSolveLower in Solver/Scalar.hpp.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Solver/Scalar.hpp"

#include <cmath>

using namespace DNDS;

TEST_CASE("BisectSolveLower: x^2 = 4 => x = 2")
{
    auto F = [](double x)
    { return x * x; };
    double v = Scalar::BisectSolveLower(F, 0.0, 3.0, 4.0, 100);
    CHECK(v == doctest::Approx(2.0).epsilon(1e-10));
}

TEST_CASE("BisectSolveLower: few iterations gives lower accuracy")
{
    auto F = [](double x)
    { return x * x; };
    double v = Scalar::BisectSolveLower(F, 0.0, 3.0, 4.0, 5);
    // 5 iterations: interval halves 5 times: 3/2^5 = 0.09375
    CHECK(std::abs(v - 2.0) < 0.1);
    CHECK(std::abs(v - 2.0) > 1e-6); // should not be exact with only 5 iters
}

TEST_CASE("BisectSolveLowerDetailed exits early when exact midpoint hits target")
{
    auto F = [](double x)
    { return x; };
    const auto res = Math::BisectSolveLowerDetailed(F, 0.0, 4.0, 2.0, 100);
    CHECK(res.converged);
    CHECK(res.value == doctest::Approx(2.0));
    CHECK(res.iterations == 1);
}

TEST_CASE("BisectMonotone reports convergence and failure cleanly")
{
    auto F = [](double x)
    { return x * x; };
    const auto ok = Math::BisectMonotone(
        F, 0.0, 4.0, 9.0, 60,
        [](double, double f, double, double)
        { return std::abs(f - 9.0) < 1e-10; });
    CHECK(ok.converged);
    CHECK(ok.value == doctest::Approx(3.0).epsilon(1e-10));
    CHECK(ok.iterations < 60);

    const auto fail = Math::BisectMonotone(
        F, 0.0, 4.0, 10.0, 2,
        [](double, double f, double, double)
        { return std::abs(f - 10.0) < 1e-12; });
    CHECK_FALSE(fail.converged);
    CHECK(fail.iterations == 2);
}
