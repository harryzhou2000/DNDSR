#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/GradientPlan.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "../Geom/InterfaceFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Standalone FV gradients recover all three spatial directions on a volume cell")
{
    auto mesh = InterfaceFixture::SingleCell(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
    {
        auto plan = C::GradientPlan::Prepare(*mesh, metrics, method);
        auto mean = metrics.BuildField(1), gradient = metrics.BuildField(3);
        auto boundary = plan.BuildBoundaryValues(1);
        const G::tPoint exact{2, -0.7, 0.3};
        {
            auto geometry = metrics.AcquireRead();
            auto u = mean.Owned().Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                u.View()(iCell, 0) = 1 + exact.dot(geometry.HostView().GetCellBary(iCell));
            auto b = boundary.Write();
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
                b.View()(iPoint, 0) = 1 + exact.dot(plan.GetBoundaryQuadrature(iPoint).point);
        }
        metrics.BindHalo(mean).Begin().Wait();
        plan.Evaluate(mean, boundary, gradient);
        auto read = gradient.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I d = 0; d < 3; ++d)
                CHECK(read.View()(iCell, d) == doctest::Approx(exact[d]).epsilon(1e-12));
    }
}

TEST_CASE("Standalone gradients recover linear fields with physical boundary samples")
{
    for (bool skew : {false, true})
    {
        auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
        auto mesh = fixture.mesh;
        if (skew)
            mesh->TransformCoords([](const G::tPoint &p) -> G::tPoint
                                  { return {p[0] + 0.23 * p[1], 0.7 * p[1], p[2]}; });
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
        {
            if (skew && method == C::GradientMethod::GreenGauss)
                continue; // uncorrected GG is not linearly exact here
            auto plan = C::GradientPlan::Prepare(*mesh, metrics, method);
            auto mean = metrics.BuildField(2), gradient = metrics.BuildField(4);
            auto boundary = plan.BuildBoundaryValues(2);
            const auto function = [](const G::tPoint &p, I component)
            { return component == 0 ? 1 + 2 * p[0] - 0.7 * p[1] : -3 + 0.2 * p[0] + 4 * p[1]; };
            {
                auto geometry = metrics.AcquireRead();
                auto write = mean.Owned().Write();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 2; ++component)
                        write.View()(iCell, component) = function(geometry.HostView().GetCellBary(iCell), component);
                auto values = boundary.Write();
                for (I point = 0; point < plan.NumBoundaryQuadrature(); ++point)
                    for (I component = 0; component < 2; ++component)
                        values.View()(point, component) = function(plan.GetBoundaryQuadrature(point).point, component);
            }
            metrics.BindHalo(mean).Begin().Wait();
            plan.Evaluate(mean, boundary, gradient);
            const double expected[]{2, -0.7, 0.2, 4};
            {
                auto read = gradient.Owned().Read();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 4; ++component)
                        CHECK(read.View()(iCell, component) == doctest::Approx(expected[component]).epsilon(1e-11));
            }
            const auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
            auto coefficients = metrics.BuildField(2 * reconstruction.NRecDOF());
            plan.ConvertToCoefficients(gradient, reconstruction, coefficients);
            {
                auto geometry = metrics.AcquireRead();
                auto read = coefficients.Owned().Read();
                const I n = reconstruction.NRecDOF();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 2; ++component)
                        for (I mode = 0; mode < n; ++mode)
                            CHECK(read.View()(iCell, component * n + mode) == doctest::Approx(
                                                                                  mode < 2 ? expected[component * 2 + mode] * geometry.HostView().cells(iCell, 7) : 0)
                                                                                  .epsilon(1e-11));
            }
            R::Array<double> wrong(R::RowLayout::Dense(plan.NumBoundaryQuadrature(), 3));
            CHECK_THROWS(plan.Evaluate(mean, wrong, gradient));
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
