/**
 * @file test_ResidualCFLDriver.cpp
 * @brief Focused tests for the residual-based implicit CFL law.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Euler/ResidualCFLDriver.hpp"

#include <cmath>
#include <limits>

using namespace DNDS::Euler;

namespace
{
    Eigen::Vector<DNDS::real, Eigen::Dynamic> Vector(std::initializer_list<DNDS::real> values)
    {
        Eigen::Vector<DNDS::real, Eigen::Dynamic> result(values.size());
        Eigen::Index i = 0;
        for (DNDS::real value : values)
            result(i++) = value;
        return result;
    }

    ResidualCFLControl BaseControl()
    {
        ResidualCFLControl control;
        control.CFLMin = 10;
        control.CFLMax = 1000;
        control.CFLOrd = 0.2;
        control.alpha = 0.5;
        control.pMGSafetyFactor = 1;
        return control;
    }
}

TEST_CASE("residual CFL initializes from the first component-wise norms")
{
    ResidualCFLDriver driver;
    const auto update = driver.Update(
        Vector({2, 4}), Vector({3, 5}), BaseControl(), 3);

    CHECK(driver.HasReference());
    CHECK(update.referenceInitializedThisUpdate);
    CHECK(update.CFL == doctest::Approx(10));
    CHECK(update.xi == doctest::Approx(1));
    CHECK(update.xi2 == doctest::Approx(1));
    CHECK(update.xiInf == doctest::Approx(1));
    CHECK_FALSE(update.usedLInfBranch);
}

TEST_CASE("residual CFL grows from the worst normalized L2 residual")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    static_cast<void>(driver.Update(
        Vector({2, 4}), Vector({3, 5}), control, 3));

    const auto update = driver.Update(
        Vector({1, 1}), Vector({1.5, 2.5}), control, 3);

    CHECK(update.xi2 == doctest::Approx(0.5));
    CHECK(update.xiInf == doctest::Approx(0.5));
    CHECK(update.xi == doctest::Approx(0.5));
    CHECK_FALSE(update.usedLInfBranch);
    CHECK(update.CFL == doctest::Approx(10 / std::sqrt(0.5)).epsilon(1e-13));
}

TEST_CASE("residual CFL clips xi2 at one while Linf does not diverge")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    static_cast<void>(driver.Update(
        Vector({2, 4}), Vector({3, 5}), control, 3));

    const auto update = driver.Update(
        Vector({4, 2}), Vector({1.5, 2.5}), control, 3);

    CHECK(update.xi2 == doctest::Approx(2));
    CHECK(update.xiInf == doctest::Approx(0.5));
    CHECK(update.xi == doctest::Approx(1));
    CHECK(update.CFL == doctest::Approx(control.CFLMin));
}

TEST_CASE("residual CFL uses Linf as a divergence tripwire")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    static_cast<void>(driver.Update(
        Vector({2, 4}), Vector({3, 5}), control, 3));

    const auto update = driver.Update(
        Vector({0.2, 0.4}), Vector({6, 2.5}), control, 3);
    const DNDS::real expectedPhi = std::exp(
        control.alpha * (1 - 2) * control.CFLMin /
        (control.CFLMin - control.CFLOrd));
    const DNDS::real expected = control.CFLOrd +
                                expectedPhi * (control.CFLMin - control.CFLOrd);

    CHECK(update.xi2 == doctest::Approx(0.1));
    CHECK(update.xiInf == doctest::Approx(2));
    CHECK(update.xi == doctest::Approx(2));
    CHECK(update.usedLInfBranch);
    CHECK(update.CFL == doctest::Approx(expected).epsilon(1e-13));
}

TEST_CASE("residual CFL applies the published pMG level factor")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    control.pMGSafetyFactor = 2;
    static_cast<void>(driver.Update(
        Vector({2}), Vector({3}), control, 3, 2, 0));

    const auto update = driver.Update(
        Vector({1}), Vector({6}), control, 3, 2, 0);
    const DNDS::real levelFactor = 6;
    const DNDS::real expectedPhi = std::exp(
        control.alpha * (1 - 2) * levelFactor * control.CFLMin /
        (control.CFLMin - control.CFLOrd));
    const DNDS::real expected = control.CFLOrd +
                                expectedPhi * (control.CFLMin - control.CFLOrd);

    CHECK(update.levelFactor == doctest::Approx(levelFactor));
    CHECK(update.CFL == doctest::Approx(expected).epsilon(1e-13));
}

TEST_CASE("residual CFL applies a sub-unity safety factor on a single grid")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    control.pMGSafetyFactor = 0.5;
    static_cast<void>(driver.Update(
        Vector({2}), Vector({3}), control, 3));

    const auto update = driver.Update(
        Vector({1}), Vector({6}), control, 3);
    const DNDS::real expectedPhi = std::exp(
        control.alpha * (1 - 2) * control.pMGSafetyFactor * control.CFLMin /
        (control.CFLMin - control.CFLOrd));
    const DNDS::real expected = control.CFLOrd +
                                expectedPhi * (control.CFLMin - control.CFLOrd);

    CHECK(update.levelFactor == doctest::Approx(0.5));
    CHECK(update.CFL == doctest::Approx(expected).epsilon(1e-13));
}

TEST_CASE("residual CFL caps the zero-residual limit")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    static_cast<void>(driver.Update(
        Vector({2, 4}), Vector({3, 5}), control, 3));

    const auto update = driver.Update(
        Vector({0, 0}), Vector({0, 0}), control, 3);

    CHECK(update.xi == doctest::Approx(0));
    CHECK(update.CFL == doctest::Approx(control.CFLMax));
    CHECK(update.limitedByCFLMax);
}

TEST_CASE("residual CFL handles a zero reference component deterministically")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    static_cast<void>(driver.Update(
        Vector({0, 2}), Vector({0, 3}), control, 3));

    const auto unchangedZero = driver.Update(
        Vector({0, 1}), Vector({0, 1.5}), control, 3);
    CHECK(unchangedZero.xi == doctest::Approx(0.5));

    const auto activatedZero = driver.Update(
        Vector({1, 1}), Vector({1, 1.5}), control, 3);
    CHECK(std::isinf(activatedZero.xi));
    CHECK(activatedZero.usedLInfBranch);
    CHECK(activatedZero.CFL == doctest::Approx(control.CFLOrd));
}

TEST_CASE("residual CFL reset captures a new first residual")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    static_cast<void>(driver.Update(
        Vector({2}), Vector({3}), control, 3));
    static_cast<void>(driver.Update(
        Vector({1}), Vector({1.5}), control, 3));

    driver.Reset();
    CHECK_FALSE(driver.HasReference());
    const auto update = driver.Update(
        Vector({100}), Vector({200}), control, 3);
    CHECK(update.referenceInitializedThisUpdate);
    CHECK(update.CFL == doctest::Approx(control.CFLMin));
}

TEST_CASE("residual CFL derives CFLOrd from reconstruction degree")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    control.CFLOrd = 0;

    const auto update = driver.Update(
        Vector({2}), Vector({3}), control, 3);
    CHECK(update.CFLOrd == doctest::Approx(1.0 / 7.0));
}

TEST_CASE("residual CFL rejects invalid inputs")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();

    CHECK_THROWS(static_cast<void>(driver.Update(
        Vector({1}), Vector({1, 2}), control, 3)));
    CHECK_THROWS(static_cast<void>(driver.Update(
        Vector({std::numeric_limits<DNDS::real>::quiet_NaN()}),
        Vector({1}), control, 3)));

    control.CFLMax = control.CFLMin / 2;
    CHECK_THROWS(static_cast<void>(driver.Update(
        Vector({1}), Vector({1}), control, 3)));
}
