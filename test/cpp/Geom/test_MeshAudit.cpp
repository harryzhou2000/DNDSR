#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Geom/Mesh/Mesh.hpp"
#include "Geom/Mesh/NativeMeshSmoothing.hpp"
#include "Geom/Mesh/NativeMeshCGNS.hpp"
#include "Geom/Mesh/NativeMeshTopology.hpp"
#include <filesystem>

using namespace DNDS;
using namespace DNDS::Geom;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context(argc, argv);
    int result = context.run();
    MPI_Finalize();
    return result;
}

static ssp<UnstructuredMesh> ReadSquare(MPIInfo mpi)
{
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
    auto state = ReadNativeCGNS(mpi, (root / "data/mesh/UniformSquare_10.cgns").string(), 2,
                                [](const std::string &) -> t_index
                                { return 21; }, 0, {1, true});
    return UnstructuredMesh::FromNative(mpi, PrepareNativeMeshTopology(*state, mpi));
}

TEST_CASE("Audit batch 2: wall quadrature preserves square distances")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    auto mesh = ReadSquare(mpi);
    double lower[2] = {1e30, 1e30}, upper[2] = {-1e30, -1e30};
    for (DNDS::index i = 0; i < mesh->NumNode(); ++i)
        for (int d = 0; d < 2; ++d)
        {
            lower[d] = std::min(lower[d], mesh->Coord(i)(d));
            upper[d] = std::max(upper[d], mesh->Coord(i)(d));
        }
    MPI_Allreduce(MPI_IN_PLACE, lower, 2, MPI_DOUBLE, MPI_MIN, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, upper, 2, MPI_DOUBLE, MPI_MAX, mpi.comm);
    UnstructuredMesh::WallDistOptions options;
    options.method = 1;
    options.minWallDist = 0;
    options.verbose = 0;
    for (int method : {0, 1, 20})
    {
        options.method = method;
        mesh->BuildNodeWallDist([](auto)
                                { return true; }, options);
        for (DNDS::index i = 0; i < mesh->NumNodeProc(); ++i)
        {
            const auto p = mesh->Coord(i);
            const double expected = std::min({p(0) - lower[0], upper[0] - p(0), p(1) - lower[1], upper[1] - p(1)});
            CHECK(mesh->NodeWallDist(i).allFinite());
            CHECK(mesh->NodeWallDist(i).norm() == doctest::Approx(expected).epsilon(1e-11));
        }
        mesh->BuildNodeWallDist([](auto)
                                { return false; }, options);
        for (DNDS::index i = 0; i < mesh->NumNodeProc(); ++i)
            CHECK(mesh->NodeWallDist(i).norm() == std::pow(DNDS::veryLargeReal, 0.25));
    }
}

TEST_CASE("Audit batch 2: native smoothers preserve zero displacement and coordinates")
{
    MPIInfo mpi(MPI_COMM_WORLD);
    auto linear = ReadSquare(mpi);
    auto mesh = std::make_shared<UnstructuredMesh>(mpi, 2);
    mesh->BuildO2FromO1Elevation(*linear);
    mesh->PrepareNativeTopology();
    mesh->ElevatedNodesGetBoundarySmooth([](auto)
                                         { return false; });
    // Exercise both linked implementations' common no-displacement contract.
    // A finite-iteration smoothing algorithm comparison is outside this test.
    auto workspace = mesh->BoundarySmoothingWorkspace();
    CHECK(workspace->TotalMoved() == 0);
    auto displacement = workspace->Displacement().owned.Read();
    DNDS::real normSquared = 0;
    for (size_t row = 0; row < displacement.View().RowCount(); ++row)
        for (size_t component = 0; component < 3; ++component)
            normSquared += std::pow(displacement.View()(row, component), 2);
    MPI_Allreduce(MPI_IN_PLACE, &normSquared, 1, DNDS_MPI_REAL, MPI_SUM, mpi.comm);
    CHECK(normSquared == 0);
    std::vector<tPoint> before;
    for (DNDS::index row = 0; row < mesh->NumNodeProc(); ++row)
        before.push_back(mesh->Coord(row));
    CHECK_NOTHROW(mesh->ElevatedNodesSolveInternalSmooth());
    mesh->ElevatedNodesGetBoundarySmooth([](auto)
                                         { return false; });
    CHECK_NOTHROW(mesh->ElevatedNodesSolveInternalSmoothV2());
    for (DNDS::index row = 0; row < mesh->NumNodeProc(); ++row)
        CHECK(mesh->Coord(row) == before[size_t(row)]);
    // Escaped value leases retain the old diagnostic allocation even after the
    // workspace's source association has been superseded by publication.
    for (size_t row = 0; row < displacement.View().RowCount(); ++row)
        for (size_t component = 0; component < 3; ++component)
            CHECK(displacement.View()(row, component) == 0);
}
