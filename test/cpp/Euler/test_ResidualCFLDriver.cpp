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

TEST_CASE("residual CFL can select equations for ratio maxima")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    control.equationIndices = {1};
    static_cast<void>(driver.Update(
        Vector({2, 4, 8}), Vector({3, 5, 10}), control, 3));

    const auto update = driver.Update(
        Vector({20, 1, 80}), Vector({30, 2.5, 100}), control, 3);

    CHECK(update.xi2 == doctest::Approx(0.25));
    CHECK(update.xiInf == doctest::Approx(0.5));
    CHECK(update.xi == doctest::Approx(0.25));
    CHECK_FALSE(update.usedLInfBranch);
    CHECK(update.CFL == doctest::Approx(20));
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

TEST_CASE("external CFL factor scales both the residual safety term and CFL cap")
{
    ResidualCFLDriver driver;
    auto control = BaseControl();
    control.pMGSafetyFactor = 2;
    static_cast<void>(driver.Update(
        Vector({2}), Vector({3}), control, 1, 3, 1, 4));

    const auto divergence = driver.Update(
        Vector({1}), Vector({6}), control, 1, 3, 1, 4);
    CHECK(divergence.levelFactor == doctest::Approx(8));
    CHECK(divergence.CFLMaxEffective == doctest::Approx(4000));

    const auto zeroResidual = driver.Update(
        Vector({0}), Vector({0}), control, 1, 3, 1, 4);
    CHECK(zeroResidual.CFL == doctest::Approx(4000));
    CHECK(zeroResidual.limitedByCFLMax);
}

TEST_CASE("residual CFL reuses one fine sample with independent pMG level laws")
{
    ResidualCFLDriver driver;
    auto fineControl = BaseControl();
    static_cast<void>(driver.Update(
        Vector({2, 4, 8}), Vector({3, 5, 10}), fineControl, 3, 3, 3));
    static_cast<void>(driver.Update(
        Vector({1, 8, 4}), Vector({1.5, 15, 5}), fineControl, 3, 3, 3));

    auto level1Control = BaseControl();
    level1Control.CFLMin = 20;
    level1Control.alpha = 1;
    level1Control.pMGSafetyFactor = 2;
    level1Control.equationIndices = {0, 2};
    const auto level1 = driver.EvaluateCurrent(level1Control, 1, 3, 1);

    CHECK(level1.xi2 == doctest::Approx(0.5));
    CHECK(level1.xiInf == doctest::Approx(0.5));
    CHECK(level1.CFL == doctest::Approx(40));
    CHECK(level1.levelFactor == doctest::Approx(6));

    auto level2Control = BaseControl();
    level2Control.CFLMin = 30;
    level2Control.CFLOrd = 0.5;
    level2Control.alpha = 0.25;
    level2Control.pMGSafetyFactor = 0.5;
    level2Control.equationIndices = {1};
    const auto level2 = driver.EvaluateCurrent(level2Control, 0, 3, 0);
    const DNDS::real expectedPhi = std::exp(
        level2Control.alpha * (1 - 3) * 2 * level2Control.CFLMin /
        (level2Control.CFLMin - level2Control.CFLOrd));
    const DNDS::real expected = level2Control.CFLOrd +
                                expectedPhi * (level2Control.CFLMin - level2Control.CFLOrd);

    CHECK(level2.xi2 == doctest::Approx(2));
    CHECK(level2.xiInf == doctest::Approx(3));
    CHECK(level2.CFL == doctest::Approx(expected).epsilon(1e-13));
    CHECK(level2.levelFactor == doctest::Approx(2));
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

    control = BaseControl();
    control.equationIndices = {-1};
    CHECK_THROWS(static_cast<void>(driver.Update(
        Vector({1}), Vector({1}), control, 3)));

    control.equationIndices = {1};
    CHECK_THROWS(static_cast<void>(driver.Update(
        Vector({1}), Vector({1}), control, 3)));
}
