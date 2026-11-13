#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/DistributedInterfaceSamplingPlan.hpp"
#include "DNDS/Runtime/Execution/CUDAExecution.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("Queued owner-side interface sampling agrees on inline worker and CUDA")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 1;
    settings.SORInstead = false;
    auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    constexpr I Width = 7;
    auto means = metrics.BuildField(Width);
    auto gradients = metrics.BuildField(Width * metrics.Dimension());
    auto coefficients = metrics.BuildField(Width * reconstruction.NRecDOF());
    {
        const auto geometry = metrics.AcquireRead();
        const auto cells = geometry.HostView();
        auto u = means.Owned().Write(), g = gradients.Owned().Write(), r = coefficients.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I j = 0; j < Width; ++j)
            {
                const auto center = cells.GetCellCent(iCell);
                u.View()(iCell, j) = 3 * j + 10 * mpi.rank + (j + 1) * center[0] - .5 * center[1];
                g.View()(iCell, 2 * j) = j + 1;
                g.View()(iCell, 2 * j + 1) = -.5;
                r.View()(iCell, 2 * j) = (j + 1) * cells.cells(iCell, 7);
                r.View()(iCell, 2 * j + 1) = -.5 * cells.cells(iCell, 7);
            }
    }
    std::vector<I> boundaries(size_t(fixture.mesh->Entities(G::EntityKind::Bnd).NumOwned()));
    std::iota(boundaries.begin(), boundaries.end(), I(0));
    auto patch = G::BoundaryPatch::Create(fixture.mesh, boundaries);
    R::Array<double> points(R::RowLayout::Dense(3, 3));
    R::Array<I> ids(R::RowLayout::Dense(3, 1));
    {
        auto p = points.Write();
        auto keys = ids.Write();
        for (I iQuery = 0; iQuery < 3; ++iQuery)
        {
            p.View()(iQuery, 0) = 0;
            p.View()(iQuery, 1) = (double((mpi.rank + 3 * iQuery) % 10) + .31) * .1;
            p.View()(iQuery, 2) = 0;
            keys.View()(iQuery, 0) = 100 - iQuery;
        }
    }
    auto map = G::DistributedInterfacePointMap::Match(patch, points, ids, G::InterfaceRoutingMode::CoarseBoxes);
    C::DistributedInterfaceSamplingPlan hostPlan(reconstruction, map);
    std::array<std::optional<R::Array<double>>, 3> expected;
    for (int backend = 0; backend < 3; ++backend)
    {
        const auto device = backend == 2 ? R::Device::CUDA(0) : R::Device::Host();
        const R::ExecutionContext placement{device, nullptr};
        const auto placedMetrics = metrics.Placed(device, placement);
        // The layout overload is also used by pMG coarse state allocation:
        // it must inherit placement just like BuildField(width).
        auto allocated = placedMetrics.BuildField(means.Owned().Layout());
        CHECK(allocated.Device() == device);
        CHECK_NOTHROW(placedMetrics.ValidateField(allocated, Width));
        auto plan = hostPlan.Placed(device, placement);
        if (backend == 2)
        {
            C::ReconstructionResidencyOptions options;
            options.retainHostPayloads = false;
            options.arbitraryPointSampling = true;
            // Moving interfaces rebuild sample points after the solver has
            // discarded the host VR matrix owner. Retained moments suffice.
            auto resident = reconstruction.Place(device, placement, options);
            plan = C::DistributedInterfaceSamplingPlan(metrics.Placed(device, placement), resident, map, placement);
        }
        const C::FVField u(means.Space(), means.Owned().To(device, placement), means.Ghost().To(device, placement));
        const C::FVField g(gradients.Space(), gradients.Owned().To(device, placement), gradients.Ghost().To(device, placement));
        const C::FVField r(coefficients.Space(), coefficients.Owned().To(device, placement), coefficients.Ghost().To(device, placement));
        R::EagerEngine engine(backend == 1 ? 1 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = backend == 2 ? engine.CreateStream(std::make_shared<R::CUDALauncher>(0)) : engine.CreateCPUStream();
        auto workspace = plan.BuildWorkspace(Width);
        R::Array<double> values(R::RowLayout::Dense(map.Size(), Width), device);
        auto binding = plan.BindResult(workspace, values);
        for (int mode = 0; mode < 3; ++mode)
        {
            const C::DonorInputs input = mode == 0 ? C::DonorInputs(u) : mode == 1 ? C::DonorInputs::Gradient(u, g)
                                                                                   : C::DonorInputs(u, r);
            plan.EvaluateDonors(engine, stream, input, workspace);
            // Current CUDA-aware MPI is host-controlled: explicitly observe the
            // producer before direct transfer; no hidden sampler staging.
            engine.Synchronize();
            binding.Begin(placement).Wait();
            auto host = values.To(R::Device::Host(), placement);
            if (backend == 0)
                expected[mode] = host;
            const auto actual = host.Read(), reference = expected[mode]->Read();
            for (I iQuery = 0; iQuery < map.Size(); ++iQuery)
                for (I j = 0; j < Width; ++j)
                {
                    CHECK(actual.View()(iQuery, j) == doctest::Approx(reference.View()(iQuery, j)).epsilon(1e-12).scale(1));
                    if (mode)
                    {
                        const auto match = map.Get(iQuery);
                        const auto point = match.geometry.donorPoint;
                        CHECK(actual.View()(iQuery, j) == doctest::Approx(3 * j + 10 * match.donorRank + (j + 1) * point[0] - .5 * point[1]).epsilon(1e-12).scale(1));
                    }
                }
        }
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    DNDS::MPI::Finalize();
    return result;
}
