#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "RotatingVortexFixture.hpp"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"

namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("P1 Delaunay disk and periodic box-hole fixtures have complementary area and moving boundary matches")
{
    constexpr int segments = 48;
    const auto disk = CauchyTest::VortexPatch(mpi, true, 16, segments);
    const auto outer = CauchyTest::VortexPatch(mpi, false, 16, segments);
    std::optional<G::BoundaryPatch> patches[2];
    double volumes[2]{};
    int side = 0;
    for (const auto &mesh : {disk, outer})
    {
        const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            volumes[side] += metrics.GetCellVol(iCell);
        auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto view = read.HostView();
        std::vector<I> selected;
        for (I iBnd = 0; iBnd < view.NumBnd(); ++iBnd)
            if (view.GetBndZone(iBnd) == 101)
                selected.push_back(iBnd);
        patches[side].emplace(G::BoundaryPatch::Create(mesh, selected));
        I total = I(selected.size());
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        CHECK(total == segments);
        ++side;
    }
    MPI_Allreduce(MPI_IN_PLACE, volumes, 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
    const double area = .5 * segments * 1.5 * 1.5 * std::sin(2 * std::acos(-1.) / segments);
    CHECK(volumes[0] == doctest::Approx(area).epsilon(1e-12));
    CHECK(volumes[1] == doctest::Approx(100 - area).epsilon(1e-12));
    CHECK(outer->isPeriodic);
    G::RigidPose pose;
    pose.rotation = Eigen::AngleAxisd(.137, G::tPoint::UnitZ()).toRotationMatrix();
    pose.translation = G::tPoint{5, 5, 0} - pose.rotation * G::tPoint{5, 5, 0};
    const auto receiver = patches[0]->WithPose(pose).SurfaceGeometry();
    R::Array<double> points(R::RowLayout::Dense(receiver.Size(), 3));
    R::Array<I> ids(R::RowLayout::Dense(receiver.Size(), 1));
    {
        auto p = points.Write();
        auto key = ids.Write();
        for (I iPoint = 0; iPoint < receiver.Size(); ++iPoint)
        {
            const auto point = pose.Apply(receiver.Geometry(iPoint).Point(G::tPoint::Zero()));
            for (int d = 0; d < 3; ++d)
                p.View()(iPoint, d) = point[d];
            key.View()(iPoint, 0) = receiver.Element(iPoint).id;
        }
    }
    G::InterfaceSearchSettings search;
    search.coarseBoxesPerRank = 3;
    search.maxGap = .01;
    const auto sparse = G::DistributedInterfacePointMap::Match(*patches[1], points, ids, G::InterfaceRoutingMode::CoarseBoxes, search);
    const auto dense = G::DistributedInterfacePointMap::Match(*patches[1], points, ids, G::InterfaceRoutingMode::AllGatherBoundary, search);
    for (I iPoint = 0; iPoint < receiver.Size(); ++iPoint)
    {
        const auto a = sparse.Get(iPoint), b = dense.Get(iPoint);
        CHECK(a.geometry.donorOriginalId == b.geometry.donorOriginalId);
        CHECK(a.donorRank == b.donorRank);
        CHECK(a.geometry.distance < .01);
    }
}
int main(int argc, char **argv)
{
    if (DNDS::MPI::Init_thread(&argc, &argv) != MPI_SUCCESS)
        return 1;
    mpi.setWorld();
    int status;
    {
        doctest::Context context(argc, argv);
        status = context.run();
    }
    DNDS::MPI::Finalize();
    return status;
}
