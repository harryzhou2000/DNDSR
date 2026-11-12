#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Cauchy/State/RANSState.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"

TEST_CASE("Reduced-order SA reconstruction compresses its scalar about the mean")
{
    struct Constant
    {
        double value;
        double operator()(DNDS::index, DNDS::index) const { return value; }
    };
    const DNDS::Cauchy::RANSReconstructionView policy{DNDS::Physics::RANS::ModelKind::SA, 1};
    CHECK(policy.Compute(Constant{.2}, Constant{-.2}, 0, 0, 1) == doctest::Approx(.45));
    CHECK(policy.Compute(Constant{0}, Constant{-.2}, 0, 0, 1) == 0);
    CHECK(policy.Compute(Constant{1e-30}, Constant{1e-30}, 0, 0, 1) == 1);
}
