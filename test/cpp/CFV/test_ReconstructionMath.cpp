#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "CFV/Math/TaylorBasis.hpp"
#include "CFV/Math/ReconstructionFunctional.hpp"

namespace M = DNDS::CFV::Math;
using S = DNDS::CFV::ReconstructionFunctionalSettings;
using Point = DNDS::Geom::tPoint;

TEST_CASE("Taylor derivative pullback preserves Frobenius tensor norms through degree three")
{
    for (int dim : {2, 3})
    {
        M::TaylorBasis basis(dim, 3);
        M::SmallMatrix values;
        const auto rotation = Eigen::AngleAxisd(0.53, dim == 2 ? Point::UnitZ().eval() : Point(1, 2, 3).normalized()).toRotationMatrix();
        basis.Derivatives(values, {0.2, -0.3, 0.4}, {0, 0, 0}, 0.7);
        auto rotated = values;
        basis.Pullback(rotated, rotation);
        int squareMode = -1;
        for (int mode = 0; mode < basis.Size(); ++mode)
            if (basis.MultiIndex(mode) == std::array<int, 3>{2, 0, 0})
                squareMode = mode;
        REQUIRE(squareMode >= 0);
        for (int direction = 0; direction < dim; ++direction)
            CHECK(rotated(1 + direction, squareMode) == doctest::Approx(values(1, squareMode) * rotation(0, direction)).epsilon(1e-12));
        const double orderWeights[]{1, 1, 1, 1};
        double weights[20];
        M::FunctionalWeights(weights, dim, 3, 1, orderWeights);
        for (int degree = 0; degree <= 3; ++degree)
            for (int mode = 0; mode < basis.Size(); ++mode)
            {
                double before = 0, after = 0;
                for (int derivative = 0; derivative < basis.Size(); ++derivative)
                    if (basis.Degree(derivative) == degree)
                    {
                        before += weights[derivative] * DNDS::sqr(values(derivative, mode));
                        after += weights[derivative] * DNDS::sqr(rotated(derivative, mode));
                    }
                CHECK(after == doctest::Approx(before).epsilon(1e-12));
            }
        basis.Pullback(rotated, rotation.transpose());
        CHECK(rotated.isApprox(values, 1e-12));
    }
}

TEST_CASE("Central isotropic weight schemes preserve configured legacy formulas")
{
    S settings;
    auto values = M::ReconstructionOrderWeights(settings, 2, 3, 2, 4, {2, 1, 0});
    CHECK((values == std::array<double, 4>{1, 1, 0.5, 1. / 6}));
    settings.dirWeightScheme = S::DirectionalWeight::HQM_OPT;
    settings.geomWeightScheme = S::GeometricWeight::HQM_SD;
    settings.geomWeightBias = 1;
    values = M::ReconstructionOrderWeights(settings, 2, 2, 2, 4, {2, 1, 0});
    CHECK(values[0] == doctest::Approx(1 + std::pow(0.5, 0.25)));
    CHECK(values[1] / values[0] == doctest::Approx(0.4643));
    settings.dirWeightScheme = S::DirectionalWeight::ManualDirWeight;
    settings.manualDirWeights = {2, 3, 4, 5};
    settings.geomWeightScheme = S::GeometricWeight::SD_Power;
    settings.geomWeightBias = 0.1;
    settings.geomWeightPower1 = 2;
    settings.geomWeightPower2 = -2;
    values = M::ReconstructionOrderWeights(settings, 3, 3, 2, 4, {2, 1, 0});
    CHECK(values[2] == doctest::Approx(4 * (0.1 + 2. / 4)));
    settings.dirWeightScheme = S::DirectionalWeight::TEST_OPT;
    auto first = M::ReconstructionOrderWeights(settings, 3, 3, 2, 4, {2, 1, 0});
    settings.geomWeightBias = 100;
    CHECK(first == M::ReconstructionOrderWeights(settings, 3, 3, 2, 4, {2, 1, 0}));
    settings.dirWeightScheme = static_cast<S::DirectionalWeight>(999);
    CHECK_THROWS(settings.Validate());
}
