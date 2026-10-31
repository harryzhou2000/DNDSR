#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "NativeMeshOverlapFixture.hpp"
#include "NativeMeshFacadeFixture.hpp"

using namespace DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Distributed sliding matches exact Cartesian donors across geometry and cell owners")
{
    for (auto backend : {BoundingBoxBackend::Native, BoundingBoxBackend::CGAL})
        for (bool emptyPeer : {false, true})
        {
            auto fixture = OverlapTest::BuildNativeFixture(mpi, emptyPeer);
            auto mesh = fixture.mesh;
            const int active = emptyPeer && mpi.size > 1 ? mpi.size - 1 : mpi.size;
            std::vector<I> selected(size_t(mesh->Entities(EntityKind::Bnd).NumOwned()));
            std::iota(selected.begin(), selected.end(), I(0));
            const auto reference = BoundaryPatch::Create(mesh, selected, {}, backend);
            const auto unchanged = mesh->versions.Stamp(R::Change::Geometry);
            const I count = !emptyPeer && mpi.size > 1 && mpi.rank == mpi.size - 1 ? 0 : 3;
            R::Array<double> points(R::RowLayout::Dense(count, 3));
            R::Array<I> ids(R::RowLayout::Dense(count, 1));
            for (int step = 0; step < 3; ++step)
            {
                RigidPose pose;
                pose.rotation = Eigen::AngleAxisd(0.3 * step, tPoint::UnitZ()).toRotationMatrix();
                pose.translation = {0.2 * step, -0.1 * step, 0.03 * step};
                auto patch = reference.WithPose(pose);
                {
                    auto p = points.Write();
                    auto key = ids.Write();
                    for (I iQuery = 0; iQuery < count; ++iQuery)
                    {
                        const I row = iQuery == 2 ? 5 : (3 * mpi.rank + iQuery + 3 * step) % 10;
                        const auto world = pose.Apply({0, (iQuery == 2 ? 0.5 : (double(row) + 0.37) * 0.1), 0});
                        for (int d = 0; d < 3; ++d)
                            p.View()(iQuery, d) = world[d];
                        key.View()(iQuery, 0) = 100 - 7 * iQuery;
                    }
                }
                auto routed = DistributedInterfacePointMap::Match(patch, points, ids);
                auto baseline = DistributedInterfacePointMap::Match(patch, points, ids, InterfaceRoutingMode::AllGatherBoundary);
                CHECK(routed.Size() == count);
                for (I iQuery = 0; iQuery < count; ++iQuery)
                {
                    const I row = iQuery == 2 ? 5 : (3 * mpi.rank + iQuery + 3 * step) % 10;
                    const auto match = routed.Get(iQuery), oracle = baseline.Get(iQuery);
                    CHECK(match.geometry.queryId == 100 - 7 * iQuery);
                    CHECK(match.geometry.donorOriginalId == 39 - row); // left boundary runs top to bottom
                    CHECK(match.geometry.donorOriginalId == oracle.geometry.donorOriginalId);
                    CHECK(match.donorRank == oracle.donorRank);
                    CHECK((match.geometry.donorPoint - tPoint{0, (iQuery == 2 ? 0.5 : (double(row) + 0.37) * 0.1), 0}).norm() < 1e-12);
                    for (int peer = 0; peer < active; ++peer)
                    {
                        const auto cells = DNDS::EvenSplitRange(peer, active, I(10));
                        if (row >= cells.first && row < cells.second)
                        {
                            CHECK(match.donorRank == peer);
                            CHECK(match.donorOwnedCell == 10 * (row - cells.first));
                        }
                        const auto boundaries = DNDS::EvenSplitRange(peer, active, I(40));
                        if (39 - row >= boundaries.first && 39 - row < boundaries.second)
                            CHECK(match.geometryRank == peer);
                    }
                }
                CHECK_NOTHROW(routed.ValidateFor(patch));
                CHECK_THROWS(routed.ValidateFor(reference));
                CHECK(unchanged.Valid()); // rigid placement did not alter internal geometry
            }
            {
                auto p = points.Write();
                for (I iQuery = 0; iQuery < count; ++iQuery)
                    p.View()(iQuery, 0) = 100;
            }
            CHECK_THROWS(DistributedInterfacePointMap::Match(reference, points, ids));
        }
}

TEST_CASE("Coarse boxes retain reference-tolerance projections outside physical endpoints")
{
    auto mesh = OverlapTest::BuildNativeFixture(mpi, true).mesh;
    std::vector<I> selected;
    {
        auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
        const auto view = read.HostView();
        for (I iBnd = 0; iBnd < mesh->NumBnd(); ++iBnd)
        {
            bool left = true;
            for (auto node : view.BoundaryNodes(iBnd))
                left = left && std::abs(view.Coord(node)[0]) < 1e-12;
            if (left)
                selected.push_back(iBnd);
        }
    }
    const auto patch = BoundaryPatch::Create(mesh, selected);
    R::Array<double> points(R::RowLayout::Dense(1, 3));
    R::Array<I> ids(R::RowLayout::Dense(1, 1));
    {
        auto p = points.Write();
        auto key = ids.Write();
        p.View()(0, 0) = 0;
        p.View()(0, 1) = -.0001;
        p.View()(0, 2) = 0;
        key.View()(0, 0) = 42;
    }
    InterfaceSearchSettings settings;
    settings.referenceTolerance = .01;
    settings.maxGap = 1e-12;
    for (int count : {1, 3, 10})
    {
        settings.coarseBoxesPerRank = count;
        const auto match = DistributedInterfacePointMap::Match(patch, points, ids, InterfaceRoutingMode::CoarseBoxes, settings).Get(0);
        CHECK(match.geometry.donorOriginalId == 39);
        CHECK((match.geometry.donorPoint - tPoint{0, -.0001, 0}).norm() < 1e-12);
    }
    const auto baseline = DistributedInterfacePointMap::Match(patch, points, ids, InterfaceRoutingMode::AllGatherBoundary, settings).Get(0);
    CHECK(baseline.geometry.donorOriginalId == 39);
    if (mpi.size == 1)
        CHECK(InterfacePointMap::Match(patch, points, ids, settings).Get(0).donorOriginalId == 39);
}

TEST_CASE("Distributed matching rejects overlapping interiors and inconsistent settings collectively")
{
    auto mesh = NativeFacadeFixture::Make(mpi).Publish(mpi);
    auto patch = BoundaryPatch::Create(mesh, {0});
    R::Array<double> points(R::RowLayout::Dense(1, 3));
    R::Array<I> ids(R::RowLayout::Dense(1, 1));
    {
        auto p = points.Write();
        p.View()(0, 0) = 0.5;
        p.View()(0, 1) = 0;
        p.View()(0, 2) = 0;
        auto key = ids.Write();
        key.View()(0, 0) = 7;
    }
    if (mpi.size > 1)
    {
        CHECK_THROWS(DistributedInterfacePointMap::Match(patch, points, ids));
        InterfaceSearchSettings inconsistent;
        if (mpi.rank == 0)
            inconsistent.maxGap = 0.1;
        CHECK_THROWS(DistributedInterfacePointMap::Match(patch, points, ids, InterfaceRoutingMode::CoarseBoxes, inconsistent));
    }
    else
        CHECK_NOTHROW(DistributedInterfacePointMap::Match(patch, points, ids));
    {
        // A rank-local input lease conflict must agree before candidate
        // communication, rather than leaving the other ranks blocked.
        std::optional<R::Array<double>::WriteLease> writer;
        if (mpi.rank == 0)
            writer.emplace(points.Write());
        CHECK_THROWS(DistributedInterfacePointMap::Match(patch, points, ids));
    }
    {
        auto p = points.Write();
        if (mpi.rank == 0)
            p.View()(0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    CHECK_THROWS(DistributedInterfacePointMap::Match(patch, points, ids));
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
