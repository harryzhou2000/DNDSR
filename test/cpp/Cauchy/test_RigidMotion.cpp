#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/Motion/RigidMotion.hpp"
#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Riemann/RiemannFlux.hpp"
#include "Cauchy/Boundary/BoundarySystem.hpp"

namespace C = DNDS::Cauchy;

TEST_CASE("Rigid frames preserve absolute energy and provide consistent mesh velocity and rotation derivatives")
{
    C::RigidMotionSettings settings;
    settings.angularVelocity = {0, 0, .17};
    settings.translationVelocity = {.1, -.2, 0};
    settings.center = {2, -1, 0};
    const C::RigidMotion motion(settings, 2);
    constexpr double time = .7, h = 1e-6;
    const auto frame = motion.At(time), plus = motion.At(time + h), minus = motion.At(time - h);
    const C::Vector3 point{3, 2, 0};
    const auto position = frame.WorldPoint(point), velocity = frame.WorldVector(frame.GridVelocity(point));
    const auto xPlus = plus.WorldPoint(point), xMinus = minus.WorldPoint(point);
    const double a = .17 * time, c = std::cos(a), s = std::sin(a);
    CHECK(position[0] == doctest::Approx(2 + .1 * time + c - 3 * s));
    CHECK(position[1] == doctest::Approx(-1 - .2 * time + s + 3 * c));
    for (int d = 0; d < 3; ++d)
        CHECK(velocity[d] == doctest::Approx((xPlus[d] - xMinus[d]) / (2 * h)).epsilon(1e-8));
    const C::PerfectGas gas;
    const auto world = gas.Conservative({1.2, .4, -.2, .3, 1.1});
    const auto local = frame.LocalState(world), recovered = frame.WorldState(local);
    for (int j = 0; j < 5; ++j)
        CHECK(recovered[j] == doctest::Approx(world[j]));
    CHECK(local[4] == world[4]);
    const auto localPlus = plus.LocalState(world), localMinus = minus.LocalState(world);
    const auto source = frame.RotationSource(local);
    for (int j = 0; j < 5; ++j)
        CHECK(source[j] == doctest::Approx((localPlus[j] - localMinus[j]) / (2 * h)).epsilon(1e-8).scale(1));

    // Independent 3D tilted-axis source and negative-residual Jacobian oracle.
    C::RigidMotionSettings tilted;
    tilted.angularVelocity = {.2, -.3, .4};
    const auto rotating = C::RigidMotion(tilted, 3).At(.2);
    const auto f = rotating.RotationSource(world);
    const auto matrix = rotating.NegativeSourceJacobian();
    // The frozen kernel record contains only values, without a Geom pose ABI.
    // Compare tilted rotation as well as translation against the host owner.
    for (const auto &owner : {frame, rotating, C::RigidFrame{}})
    {
        const auto parameters = owner.KernelParameters();
        for (int d = 0; d < 3; ++d)
        {
            CHECK(parameters.GridVelocity(point)[d] == doctest::Approx(owner.GridVelocity(point)[d]).epsilon(1e-14).scale(1));
            CHECK(parameters.WorldPoint(point)[d] == doctest::Approx(owner.WorldPoint(point)[d]).epsilon(1e-14).scale(1));
            CHECK(parameters.LocalVector(point)[d] == doctest::Approx(owner.LocalVector(point)[d]).epsilon(1e-14).scale(1));
        }
        for (int j = 0; j < 5; ++j)
        {
            CHECK(parameters.WorldState(world)[j] == doctest::Approx(owner.WorldState(world)[j]).epsilon(1e-14).scale(1));
            CHECK(parameters.LocalState(world)[j] == doctest::Approx(owner.LocalState(world)[j]).epsilon(1e-14).scale(1));
            CHECK(parameters.RotationSource(world)[j] == doctest::Approx(owner.RotationSource(world)[j]).epsilon(1e-14).scale(1));
        }
        CHECK(parameters.NegativeSourceJacobian() == owner.NegativeSourceJacobian());
    }
    CHECK(f[1] == doctest::Approx(.3 * world[3] + .4 * world[2]));
    CHECK(f[2] == doctest::Approx(-.4 * world[1] + .2 * world[3]));
    CHECK(f[3] == doctest::Approx(-.2 * world[2] - .3 * world[1]));
    CHECK(f[0] == 0);
    CHECK(f[4] == 0);
    for (int column = 0; column < 5; ++column)
    {
        auto uPlus = world, uMinus = world;
        uPlus[column] += h;
        uMinus[column] -= h;
        const auto fPlus = rotating.RotationSource(uPlus), fMinus = rotating.RotationSource(uMinus);
        for (int row = 0; row < 5; ++row)
            CHECK(matrix[row + 5 * column] == doctest::Approx(-(fPlus[row] - fMinus[row]) / (2 * h)).epsilon(1e-9).scale(1));
    }
    const C::Vector3 normal{.6, .8, 0}, grid = frame.GridVelocity(point);
    const auto fixed = C::Riemann::PhysicalFlux({local, gas.Pressure(local)}, normal);
    const auto moving = C::Riemann::PhysicalFlux({local, gas.Pressure(local)}, normal, grid);
    const double wn = .6 * grid[0] + .8 * grid[1];
    for (int j = 0; j < 5; ++j)
        CHECK(moving[j] == doctest::Approx(fixed[j] - wn * local[j]));
    for (auto kind : {C::BoundaryKind::SlipWall, C::BoundaryKind::NoSlipWall, C::BoundaryKind::Farfield})
    {
        C::BoundaryCondition boundary;
        boundary.kind = kind;
        boundary.farfield = world;
        const auto jacobian = boundary.ExteriorJacobian(
            gas, local, normal, point, time,
            DNDS::OptionalRef<const C::RigidFrame>(frame));
        for (int column = 0; column < 5; ++column)
        {
            auto uPlus = local, uMinus = local;
            uPlus[column] += h;
            uMinus[column] -= h;
            const auto bPlus = boundary.Exterior(
                gas, uPlus, normal, point, time,
                DNDS::OptionalRef<const C::RigidFrame>(frame));
            const auto bMinus = boundary.Exterior(
                gas, uMinus, normal, point, time,
                DNDS::OptionalRef<const C::RigidFrame>(frame));
            for (int row = 0; row < 5; ++row)
                CHECK(jacobian[row + 5 * column] == doctest::Approx((bPlus[row] - bMinus[row]) / (2 * h)).epsilon(1e-8).scale(1));
        }
    }
    CHECK_THROWS(C::RigidMotion(tilted, 2));
}
