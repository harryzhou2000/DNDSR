#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Distributed polynomial sampling evaluates on cell owners and returns query order")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto mesh = fixture.mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 1;
    auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto mean = metrics.BuildField(2);
    auto coefficients = metrics.BuildField(2 * reconstruction.NRecDOF());
    {
        auto geometry = metrics.AcquireRead();
        const auto view = geometry.HostView();
        auto u = mean.Owned().Write();
        auto r = coefficients.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto center = view.GetCellCent(iCell);
            u.View()(iCell, 0) = 1 + 2 * center[0] - 0.5 * center[1] + 10 * mpi.rank;
            u.View()(iCell, 1) = -3 + 0.7 * center[0] + 4 * center[1] + 3 * mpi.rank;
            const double scale = view.cells(iCell, 7);
            r.View()(iCell, 0) = 2 * scale;
            r.View()(iCell, 1) = -0.5 * scale;
            r.View()(iCell, 2) = 0.7 * scale;
            r.View()(iCell, 3) = 4 * scale;
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(coefficients).Begin().Wait();
    std::vector<I> selected(size_t(mesh->Entities(G::EntityKind::Bnd).NumOwned()));
    std::iota(selected.begin(), selected.end(), I(0));
    auto reference = G::BoundaryPatch::Create(mesh, selected);
    // The empty geometry owner also submits queries to remote cell owners.
    R::Array<double> points(R::RowLayout::Dense(3, 3));
    R::Array<I> ids(R::RowLayout::Dense(3, 1));
    for (int step = 0; step < 2; ++step)
    {
        G::RigidPose pose;
        pose.rotation = Eigen::AngleAxisd(0.4 * step, G::tPoint::UnitZ()).toRotationMatrix();
        pose.translation = {0.2 * step, 0.1 * step, 0};
        auto patch = reference.WithPose(pose);
        {
            auto p = points.Write();
            auto key = ids.Write();
            for (I iQuery = 0; iQuery < 3; ++iQuery)
            {
                const auto query = pose.Apply({0, (double((mpi.rank + 3 * iQuery + step) % 10) + 0.31) * 0.1, 0});
                for (int d = 0; d < 3; ++d)
                    p.View()(iQuery, d) = query[d];
                key.View()(iQuery, 0) = 40 - 11 * iQuery;
            }
        }
        for (auto mode : {G::InterfaceRoutingMode::CoarseBoxes, G::InterfaceRoutingMode::AllGatherBoundary})
        {
            const auto map = G::DistributedInterfacePointMap::Match(patch, points, ids, mode);
            C::DistributedInterfaceSamplingPlan sampler(reconstruction, map);
            auto work = sampler.BuildWorkspace(2);
            R::Array<double> values(R::RowLayout::Dense(3, 2));
            sampler.Evaluate(mean, coefficients, work, values);
            {
                auto read = values.Read();
                for (I iQuery = 0; iQuery < 3; ++iQuery)
                {
                    const auto match = map.Get(iQuery);
                    const auto p = match.geometry.donorPoint;
                    CHECK(read.View()(iQuery, 0) == doctest::Approx(1 + 2 * p[0] - 0.5 * p[1] + 10 * match.donorRank).epsilon(1e-12));
                    CHECK(read.View()(iQuery, 1) == doctest::Approx(-3 + 0.7 * p[0] + 4 * p[1] + 3 * match.donorRank).epsilon(1e-12));
                }
            }
            // Reuse a persistent result binding; only donor evaluation changes
            // the scratch output between communication operations.
            auto binding = sampler.BindResult(work, values);
            sampler.EvaluateDonors(mean, coefficients, work);
            {
                auto write = work.donorValues.Write();
                for (I iRow = 0; iRow < work.donorValues.RowCount(); ++iRow)
                    for (I component = 0; component < 2; ++component)
                        write.View()(iRow, component) += 0.25;
            }
            binding.Begin().Wait();
            auto read = values.Read();
            for (I iQuery = 0; iQuery < 3; ++iQuery)
            {
                const auto match = map.Get(iQuery);
                const auto p = match.geometry.donorPoint;
                CHECK(read.View()(iQuery, 0) == doctest::Approx(1.25 + 2 * p[0] - 0.5 * p[1] + 10 * match.donorRank).epsilon(1e-12));
                CHECK(read.View()(iQuery, 1) == doctest::Approx(-2.75 + 0.7 * p[0] + 4 * p[1] + 3 * match.donorRank).epsilon(1e-12));
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
