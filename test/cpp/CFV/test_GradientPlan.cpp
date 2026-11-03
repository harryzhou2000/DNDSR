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

TEST_CASE("Gradient core phases exclude active MPI ghost writes across table layouts")
{
    // At np2 both ranks own cells; np4/8 additionally exercise an empty owner.
    // Three or more complete grid rows per active rank ensure a real core and
    // dependent region, instead of a vacuous empty-core split comparison.
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, mpi.size > 2, 24);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    const auto partition = metrics.CellWorkPartition(), shared = metrics.CellWorkPartition();
    CHECK(&partition.Core() == &shared.Core());
    CHECK(&partition.HaloDependent() == &shared.HaloDependent());
    if (metrics.NumCell())
    {
        REQUIRE_FALSE(partition.Core().empty());
        if (mpi.size > 1)
            REQUIRE_FALSE(partition.HaloDependent().empty());
    }
    const double expected[]{2, -.7, -.2, .3, .6, -.1, -.7, 1.2};
    const auto value = [&](const G::tPoint &point, I component)
    { return 1 + component + expected[2 * component] * point[0] + expected[2 * component + 1] * point[1]; };
    for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
    {
        auto plan = C::GradientPlan::Prepare(*fixture.mesh, metrics, method);
        auto boundary = plan.BuildBoundaryValues(4);
        {
            auto write = boundary.Write();
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
                for (I component = 0; component < 4; ++component)
                    write.View()(iPoint, component) = value(plan.GetBoundaryQuadrature(iPoint).point, component);
        }
        for (int layout = 0; layout < 3; ++layout)
        {
            const auto shape = [&](I width)
            {
                return layout == 0 ? R::RowLayout::Dense(0, width, width + 2) : layout == 1 ? R::RowLayout::ColMajor(0, width, {16, 1})
                                                                                            : R::RowLayout::TiledColMajor(0, width, 16);
            };
            auto mean = metrics.BuildField(shape(4)), full = metrics.BuildField(shape(8)), split = metrics.BuildField(shape(8));
            {
                auto geometry = metrics.AcquireRead();
                auto write = mean.Owned().Write();
                const auto u = write.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < 4; ++component)
                        u(iCell, component) = value(geometry.HostView().GetCellBary(iCell), component);
                auto output = split.Owned().Write();
                const auto out = output.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I j = 0; j < 8; ++j)
                        out(iCell, j) = 123456;
            }
            const auto version = mean.Owned().Version();
            auto halo = metrics.BindHalo(mean);
            halo.Begin().Wait();
            plan.Evaluate(mean, boundary, full, {1});
            auto pending = halo.Begin();
            CHECK_NOTHROW(plan.Evaluate(mean, boundary, split, {1}, C::ReconstructionRegion::Core));
            {
                auto read = split.Owned().Read();
                const auto out = read.View<R::UniformTableRows>();
                for (auto iCell : partition.Core())
                    for (I j = 0; j < 8; ++j)
                        CHECK(out(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-11));
                for (auto iCell : partition.HaloDependent())
                    for (I j = 0; j < 8; ++j)
                        CHECK(out(iCell, j) == 123456);
            }
            if (metrics.Cells().NumGhost())
            {
                const auto before = split.Owned().Version();
                CHECK_THROWS(plan.Evaluate(mean, boundary, split, {1}, C::ReconstructionRegion::HaloDependent));
                CHECK_THROWS(plan.Evaluate(mean, boundary, split, {1}));
                CHECK(split.Owned().Version() == before);
            }
            pending.Wait();
            plan.Evaluate(mean, boundary, split, {1}, C::ReconstructionRegion::HaloDependent);
            {
                auto a = full.Owned().Read(), b = split.Owned().Read();
                const auto reference = a.View<R::UniformTableRows>(), actual = b.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I j = 0; j < 8; ++j)
                    {
                        CHECK(actual(iCell, j) == reference(iCell, j));
                        CHECK(actual(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-11));
                    }
            }
            CHECK(mean.Owned().Version() == version);
            CHECK_THROWS(plan.Evaluate(mean, boundary, split, {1}, static_cast<C::ReconstructionRegion>(100)));
            if (metrics.NumCell() && metrics.Cells().NumGhost())
            {
                // Core deliberately never acquires the input ghost read, so
                // explicit output/input alias validation must cover it too.
                const auto inputBytes = size_t(mean.Ghost().RowCount() * 4) * sizeof(double);
                const auto outputBytes = size_t(split.Owned().RowCount() * 8) * sizeof(double);
                auto buffer = R::Buffer::Allocate(std::max(inputBytes, outputBytes));
                auto aliasedInput = mean, aliasedOutput = split;
                aliasedInput.Ghost() = R::Array<double>::FromStorage(R::RowLayout::Dense(mean.Ghost().RowCount(), 4), buffer.Subspan(0, inputBytes));
                aliasedOutput.Owned() = R::Array<double>::FromStorage(R::RowLayout::Dense(split.Owned().RowCount(), 8), buffer.Subspan(0, outputBytes));
                CHECK_THROWS_WITH_AS(plan.Evaluate(aliasedInput, boundary, aliasedOutput, {1}, C::ReconstructionRegion::Core),
                                     doctest::Contains("gradient output overlaps"), std::exception);
            }
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
