#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/WallDistanceSamples.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include "../CFV/ReconstructionFixture.hpp"

namespace C = DNDS::Cauchy;
namespace G = DNDS::Geom;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Nodal wall sampling retains P1 and P2 shapes on distributed cells and faces")
{
    for (bool quadratic : {false, true})
    {
        auto mesh = CFVTest::CartesianHexes(mpi, quadratic ? 1 : 2);
        if (quadratic)
        {
            auto elevated = DNDS::make_ssp<G::UnstructuredMesh>(mpi, 3);
            elevated->BuildO2FromO1Elevation(*mesh);
            elevated->PrepareNativeTopology();
            mesh = std::move(elevated);
        }
        G::UnstructuredMesh::WallDistOptions options;
        options.minWallDist = 0;
        mesh->BuildNodeWallDist([](auto zone)
                                { return zone == 7; }, options);
        const int order = 4;
        const auto samples = C::WallDistanceSamples::Construct(mesh, order);
        CHECK(samples.IntegrationOrder() == order);
        // Both nodal sets have only one nonzero distance, at the box center.
        // Tensor-product interpolation gives an independent closed-form oracle;
        // this deliberately differs from the exact nearest-wall distance inside.
        const auto expected = [&](const G::tPoint &point)
        {
            double value = .5;
            for (int d = 0; d < 3; ++d)
                value *= quadratic ? 4 * point[d] * (1 - point[d]) : 1 - std::abs(2 * point[d] - 1);
            return std::max(0., value);
        };
        {
            auto access = samples.AcquireRead();
            const auto values = access.HostView();
            auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            for (I iCell = 0; iCell < view.NumCellProc(); ++iCell)
            {
                const auto nodes = view.CellNodes(iCell);
                G::tPoint low = view.Coord(nodes[0]), high = low;
                for (I iNode = 1; iNode < I(nodes.size()); ++iNode)
                {
                    low = low.cwiseMin(view.Coord(nodes[iNode]));
                    high = high.cwiseMax(view.Coord(nodes[iNode]));
                }
                G::Elem::Quadrature quadrature(view.GetCellElement(iCell), order);
                REQUIRE(values.cells[iCell].size() == size_t(quadrature.GetNumPoints()));
                double sum = 0;
                for (int iPoint = 0; iPoint < quadrature.GetNumPoints(); ++iPoint)
                {
                    const auto param = std::get<0>(quadrature.GetQuadraturePointInfo(iPoint));
                    const G::tPoint point = (high + low) * .5 + (high - low).cwiseProduct(param) * .5;
                    const double value = expected(point);
                    CHECK(values.cells(iCell, iPoint) == doctest::Approx(value).epsilon(1e-12));
                    sum += value;
                }
                CHECK(values.cellMean(iCell, 0) == doctest::Approx(sum / quadrature.GetNumPoints()).epsilon(1e-12));
            }
            for (I iFace = 0; iFace < view.NumFaceProc(); ++iFace)
            {
                const auto nodes = view.FaceNodes(iFace);
                const G::tPoint origin = view.Coord(nodes[0]);
                const G::tPoint dx = view.Coord(nodes[1]) - origin, dy = view.Coord(nodes[3]) - origin;
                G::Elem::Quadrature quadrature(view.GetFaceElement(iFace), order);
                REQUIRE(values.faces[iFace].size() == size_t(quadrature.GetNumPoints()));
                for (int iPoint = 0; iPoint < quadrature.GetNumPoints(); ++iPoint)
                {
                    const auto param = std::get<0>(quadrature.GetQuadraturePointInfo(iPoint));
                    const G::tPoint point = origin + .5 * (1 + param[0]) * dx + .5 * (1 + param[1]) * dy;
                    CHECK(values.faces(iFace, iPoint) == doctest::Approx(expected(point)).epsilon(1e-12));
                }
            }
            CHECK_THROWS(mesh->versions.BeginMutation(DNDS::Runtime::Change::Geometry));
        }
        // Cache identity includes the wall publication, not just coordinates.
        mesh->BuildNodeWallDist([](auto)
                                { return false; }, options);
        CHECK_THROWS(samples.AcquireRead());
        const auto rebuilt = C::WallDistanceSamples::Construct(mesh, order);
        const auto access = rebuilt.AcquireRead();
        for (I iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
            CHECK(access.HostView().cellMean(iCell, 0) == doctest::Approx(std::pow(DNDS::veryLargeReal, .25)));
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
