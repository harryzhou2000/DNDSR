#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "ReconstructionFixture.hpp"
#include "RotatedSectorFixture.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

namespace
{
    void Fill(C::FVField &field, const G::UnstructuredMesh &mesh)
    {
        auto write = field.Owned().Write();
        const auto values = write.View();
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (I component = 0; component < field.Width(); ++component)
                values(iCell, component) = std::sin(.31 * mesh.OriginalEntityId(G::EntityKind::Cell, iCell) + .17 * component);
    }

    void Equal(const C::FVField &a, const C::FVField &b)
    {
        auto left = a.Owned().Read(), right = b.Owned().Read();
        for (I iCell = 0; iCell < a.Owned().RowCount(); ++iCell)
            for (I component = 0; component < a.Width(); ++component)
                CHECK(left.View()(iCell, component) == right.View()(iCell, component));
    }

    C::ReconstructionBoundaryValues Boundary(const C::VariationalReconstructionPlan &plan, I components)
    {
        auto boundary = plan.BuildBoundaryValues(components);
        auto write = boundary.values.Write();
        for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto point = plan.GetBoundaryQuadrature(iPoint).point;
            for (I component = 0; component < components; ++component)
                write.View()(iPoint, component) = std::cos(point[0] + .3 * point[1] - .2 * point[2] + component);
        }
        return boundary;
    }

    bool Allocated(const C::ReconstructionStorageReport &report, const std::string &name)
    {
        for (const auto &table : report.tables)
            if (table.name == name)
                return table.allocated;
        throw std::runtime_error("unreported reconstruction table: " + name);
    }
}

TEST_CASE("Sweep capability profiles preserve full Jacobi SOR and WBAP arithmetic in 2D and 3D")
{
    for (int dim : {2, 3})
    {
        auto mesh = dim == 2 ? G::OverlapTest::BuildNativeFixture(mpi, true, 4).mesh : CFVTest::CartesianHexes(mpi, 2);
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
        for (int degree : {1, 2, 3})
        {
            INFO("dimension=" << dim << " degree=" << degree);
            C::ReconstructionSettings settings;
            settings.maxOrder = degree;
            settings.SORInstead = degree == 2;
            settings.coreFirst = settings.SORInstead;
            settings.cacheDiffBase = degree != 1;
            auto full = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
            auto mean = metrics.BuildField(2);
            Fill(mean, *mesh);
            metrics.BindHalo(mean).Begin().Wait();
            auto reference = full.BuildWorkspace(2);
            auto referenceCoefficients = full.BuildCoefficients(2);
            Fill(referenceCoefficients, *mesh);
            metrics.BindHalo(referenceCoefficients).Begin().Wait();
            auto boundary = Boundary(full, 2);
            full.DoReconstructionIterTo(mean, referenceCoefficients, reference, boundary);

            for (bool wbap : {false, true})
            {
                auto lean = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, C::ReconstructionCapabilities::ForSweeps(wbap));
                auto work = lean.BuildWorkspace(2);
                auto coefficients = lean.BuildCoefficients(2);
                Fill(coefficients, *mesh);
                auto bc = Boundary(lean, 2);
                auto pending = metrics.BindHalo(coefficients).Begin();
                lean.DoReconstructionIterTo(mean, coefficients, work, bc, C::ReconstructionRegion::Core);
                pending.Wait();
                lean.DoReconstructionIterTo(mean, coefficients, work, bc, C::ReconstructionRegion::HaloDependent);
                Equal(reference.temporary, work.temporary);

                const auto report = lean.StorageReport(), baseline = full.StorageReport();
                for (const char *name : {"localMatrix", "localInverse", "rawMatrixB", "rawVectorB", "rawBoundaryInjection"})
                    CHECK_FALSE(Allocated(report, name));
                CHECK(Allocated(report, "secondary") == wbap);
                CHECK(report.retainedAllocationBytes <= baseline.retainedAllocationBytes);
                if (metrics.NumCell())
                    CHECK(report.retainedAllocationBytes < baseline.retainedAllocationBytes);
                CHECK(report.preparationRuntimeScratchBytes <= baseline.preparationRuntimeScratchBytes);
                CHECK(lean.GetInverseDiagnostics().rankCounts == full.GetInverseDiagnostics().rankCounts);
                // An unavailable operation fails before output acquisition or mutation.
                const auto version = work.temporary.Owned().Version();
                CHECK_THROWS_WITH(lean.ApplyLinearOperator(coefficients, work.temporary), doctest::Contains("VR operation was not prepared: original linear system"));
                CHECK(work.temporary.Owned().Version() == version);
                if (!wbap)
                    CHECK_THROWS(lean.DoLimiterWBAP_3OrderTo(coefficients, work, degree));
                else
                {
                    // Both limit the same nontrivial old coefficients, not the
                    // freshly computed sweep. Descending passes remain explicit.
                    auto limiterReference = full.BuildWorkspace(2);
                    auto limiterReferenceCoefficients = full.BuildCoefficients(2);
                    Fill(limiterReferenceCoefficients, *mesh);
                    metrics.BindHalo(limiterReferenceCoefficients).Begin().Wait();
                    full.DoLimiterWBAP_3OrderTo(limiterReferenceCoefficients, limiterReference, degree);
                    lean.DoLimiterWBAP_3OrderTo(coefficients, work, degree);
                    Equal(limiterReference.temporary, work.temporary);
                }
            }
        }
    }
}

TEST_CASE("Original operator and preconditioner profiles remain independent under complete SVD truncation")
{
    // Rotated periodic closure also exercises a different source/target frame,
    // and leaves ranks beyond the two owners empty in np4/np8 runs.
    auto mesh = CFVTest::Sector(mpi);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 3;
    settings.svdTolerance = 1;
    auto full = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    C::ReconstructionCapabilities rawOnly{false, false, true, false, false};
    auto raw = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, rawOnly);
    C::ReconstructionCapabilities inverseOnly{false, false, false, true, false};
    auto inverse = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, inverseOnly);
    auto x = metrics.BuildField(full.NRecDOF()), expected = metrics.BuildField(full.NRecDOF()), actual = metrics.BuildField(full.NRecDOF());
    Fill(x, *mesh);
    metrics.BindHalo(x).Begin().Wait();
    full.ApplyLinearOperator(x, expected);
    raw.ApplyLinearOperator(x, actual);
    Equal(expected, actual);
    CHECK_FALSE(Allocated(raw.StorageReport(), "localInverse"));
    CHECK_FALSE(Allocated(raw.StorageReport(), "inverseConditions"));
    CHECK_FALSE(Allocated(inverse.StorageReport(), "rawMatrixB"));
    C::ReconstructionCapabilities wbapOnly{false, true, false, false, false};
    auto limiter = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, wbapOnly);
    auto reference = full.BuildWorkspace(1), limited = limiter.BuildWorkspace(1);
    auto referenceCoefficients = full.BuildCoefficients(1), limitedCoefficients = limiter.BuildCoefficients(1);
    Fill(referenceCoefficients, *mesh);
    Fill(limitedCoefficients, *mesh);
    metrics.BindHalo(referenceCoefficients).Begin().Wait();
    metrics.BindHalo(limitedCoefficients).Begin().Wait();
    full.DoLimiterWBAP_3OrderTo(referenceCoefficients, reference, settings.maxOrder);
    limiter.DoLimiterWBAP_3OrderTo(limitedCoefficients, limited, settings.maxOrder);
    Equal(reference.temporary, limited.temporary);
    inverse.ApplyLocalPreconditioner(x, actual);
    auto read = actual.Owned().Read();
    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (I mode = 0; mode < full.NRecDOF(); ++mode)
            CHECK(read.View()(iCell, mode) == 0);
    CHECK_THROWS(inverse.GetInverseDiagnostics());
}

TEST_CASE("Inverse-only preparation includes self-periodic diagonal corrections")
{
    auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(NativePeriodicFixtures::QuadraticTorus(mpi)));
    mesh->PrepareNativeTopology();
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 3;
    auto full = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto inverse = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, {false, false, false, true, false});
    auto x = metrics.BuildField(full.NRecDOF()), expected = metrics.BuildField(full.NRecDOF()), actual = metrics.BuildField(full.NRecDOF());
    Fill(x, *mesh);
    full.ApplyLocalPreconditioner(x, expected);
    inverse.ApplyLocalPreconditioner(x, actual);
    Equal(expected, actual);
}

TEST_CASE("Sampling-only preparation is independent of matrix capabilities and ranks agree before setup")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true, 2).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionCapabilities sampling{false, false, false, false, false};
    if (mpi.size > 1)
    {
        auto disagree = sampling;
        disagree.sweeps = mpi.rank == 0;
        CHECK_THROWS(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, {}, disagree));
    }
    C::ReconstructionSettings settings;
    settings.cacheDiffBase = false;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, sampling);
    CHECK(plan.StorageReport().preparationRuntimeScratchBytes == 0);
    auto mean = metrics.BuildField(1), coefficient = metrics.BuildField(plan.NRecDOF());
    {
        auto u = mean.Owned().Write(), r = coefficient.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            u.View()(iCell, 0) = 3.7;
            std::fill(r.View()[iCell].begin(), r.View()[iCell].end(), 0.);
        }
    }
    R::Array<I> cells(R::RowLayout::Dense(metrics.NumCell(), 1));
    R::Array<double> points(R::RowLayout::Dense(metrics.NumCell(), 3)), values(R::RowLayout::Dense(metrics.NumCell(), 1));
    {
        auto c = cells.Write();
        auto p = points.Write();
        auto geometry = metrics.AcquireRead();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            c.View()(iCell, 0) = iCell;
            for (int d = 0; d < 3; ++d)
                p.View()(iCell, d) = geometry.HostView().GetCellCent(iCell)[d];
        }
    }
    plan.Evaluate(mean, coefficient, cells, points, values);
    auto read = values.Read();
    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        CHECK(read.View()(iCell, 0) == 3.7);
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Geometry);
        mutation.Commit();
    }
    CHECK_THROWS(plan.StorageReport()); // Reduced profiles retain the same geometry-generation guard.
}

int main(int argc, char **argv)
{
    const int status = DNDS::MPI::Init_thread(&argc, &argv);
    if (status != MPI_SUCCESS)
        return status;
    mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    DNDS::MPI::Finalize();
    return result;
}
