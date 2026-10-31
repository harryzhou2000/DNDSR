#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Runtime/EagerEngine.hpp"
#include "DNDS/Runtime/MPIProgress.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

static void Fill(C::FVField &field, double offset)
{
    auto write = field.Owned().Write();
    for (I iRow = 0; iRow < field.Owned().RowCount(); ++iRow)
        for (I component = 0; component < field.Width(); ++component)
            write.View()(iRow, component) = offset + 0.01 * (iRow + 5 * mpi.rank + component);
}
static void Equal(const C::FVField &left, const C::FVField &right)
{
    auto a = left.Owned().Read(), b = right.Owned().Read();
    for (I iRow = 0; iRow < left.Owned().RowCount(); ++iRow)
        for (I component = 0; component < left.Width(); ++component)
            CHECK(a.View()(iRow, component) == doctest::Approx(b.View()(iRow, component)).epsilon(1e-13));
}

TEST_CASE("Jacobi and WBAP core work overlaps real direct MPI and preserves unsplit results")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    auto mean = metrics.BuildField(2);
    auto reference = plan.BuildWorkspace(2), split = plan.BuildWorkspace(2);
    Fill(mean, 1);
    Fill(reference.coefficients, 0.2);
    Fill(split.coefficients, 0.2);
    auto boundary = plan.BuildBoundaryValues(2);
    {
        auto write = boundary.values.Write();
        for (I row = 0; row < boundary.values.RowCount(); ++row)
            for (I component = 0; component < 2; ++component)
                write.View()(row, component) = 1 + component;
    }
    auto meanHalo = metrics.BindHalo(mean);
    auto referenceHalo = metrics.BindHalo(reference.coefficients), splitHalo = metrics.BindHalo(split.coefficients);
    meanHalo.Begin().Wait();
    referenceHalo.Begin().Wait();
    plan.DoReconstructionIter(mean, reference, boundary);
    {
        auto pendingMean = meanHalo.Begin();
        auto pendingRec = splitHalo.Begin();
        // Even if MPI finishes quickly, active operation leases remain until
        // completion is observed. Core therefore proves no ghost reads/writes.
        plan.DoReconstructionIterTo(mean, split.coefficients, split, boundary, C::ReconstructionRegion::Core);
        if (mpi.size > 1 && mean.Owned().RowCount() > 0)
            CHECK_THROWS(plan.CommitReconstruction(split)); // owned send input still pinned
        pendingMean.Wait();
        pendingRec.Wait();
    }
    plan.DoReconstructionIterTo(mean, split.coefficients, split, boundary, C::ReconstructionRegion::HaloDependent);
    plan.CommitReconstruction(split);
    Equal(reference.coefficients, split.coefficients);
    for (int order = plan.Settings().maxOrder; order >= 1; --order)
    {
        referenceHalo.Begin().Wait();
        plan.DoLimiterWBAP_3Order(reference, order);
        auto pending = splitHalo.Begin();
        plan.DoLimiterWBAP_3OrderTo(split, order, nullptr, C::ReconstructionRegion::Core);
        pending.Wait();
        plan.DoLimiterWBAP_3OrderTo(split, order, nullptr, C::ReconstructionRegion::HaloDependent);
        plan.CommitReconstruction(split);
        Equal(reference.coefficients, split.coefficients);
    }
    R::Array<double> indicator(R::RowLayout::Dense(metrics.NumCell(), 1));
    {
        auto write = indicator.Write();
        for (I row = 0; row < metrics.NumCell(); ++row)
            write.View()(row, 0) = -1;
    }
    plan.DoLimiterWBAP_3OrderTo(split, 1, nullptr, C::ReconstructionRegion::All, &indicator, 0);
    plan.CommitReconstruction(split);
    Equal(reference.coefficients, split.coefficients); // inactive is exact passthrough
    referenceHalo.Begin().Wait();
    splitHalo.Begin().Wait();
    plan.DoLimiterWBAP_3Order(reference, 1);
    {
        auto write = indicator.Write();
        for (I row = 0; row < metrics.NumCell(); ++row)
            write.View()(row, 0) = 0;
    }
    plan.DoLimiterWBAP_3OrderTo(split, 1, nullptr, C::ReconstructionRegion::All, &indicator, 0);
    plan.CommitReconstruction(split);
    Equal(reference.coefficients, split.coefficients); // equality activates
    CHECK_THROWS(plan.DoLimiterWBAP_3OrderTo(split, 1, nullptr, C::ReconstructionRegion::All, nullptr,
                                             std::numeric_limits<double>::quiet_NaN()));
    if (metrics.NumCell())
    {
        {
            auto write = indicator.Write();
            write.View()(0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        CHECK_THROWS(plan.DoLimiterWBAP_3OrderTo(split, 1, nullptr, C::ReconstructionRegion::All, &indicator, 0));
    }
}

TEST_CASE("Core-first SOR uses a separate frozen send snapshot during split execution")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    C::ReconstructionSettings settings;
    settings.SORInstead = true;
    settings.coreFirst = true;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto mean = metrics.BuildField(1), snapshot = metrics.BuildField(plan.NRecDOF());
    auto reference = plan.BuildWorkspace(1), split = plan.BuildWorkspace(1);
    Fill(mean, 1);
    Fill(snapshot, 0.2);
    Fill(reference.coefficients, 0.2);
    Fill(split.coefficients, 0.2);
    auto boundary = plan.BuildBoundaryValues(1);
    {
        auto write = boundary.values.Write();
        for (I row = 0; row < boundary.values.RowCount(); ++row)
            write.View()(row, 0) = 1;
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(reference.coefficients).Begin().Wait();
    plan.DoReconstructionIter(mean, reference, boundary);
    auto snapshotHalo = metrics.BindHalo(snapshot);
    auto pending = snapshotHalo.Begin();
    plan.DoReconstructionIterTo(mean, snapshot, split, boundary, C::ReconstructionRegion::Core);
    pending.Wait();
    plan.DoReconstructionIterTo(mean, snapshot, split, boundary, C::ReconstructionRegion::HaloDependent);
    Equal(reference.coefficients, split.coefficients);
}

TEST_CASE("CFV split operators share eager ordering across inline and worker CPU launchers")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    auto mean = metrics.BuildField(1);
    auto reference = plan.BuildWorkspace(1);
    Fill(mean, 1);
    Fill(reference.coefficients, 0.3);
    auto boundary = plan.BuildBoundaryValues(1);
    {
        auto write = boundary.values.Write();
        for (I row = 0; row < boundary.values.RowCount(); ++row)
            write.View()(row, 0) = 1;
    }
    auto meanHalo = metrics.BindHalo(mean);
    meanHalo.Begin().Wait();
    metrics.BindHalo(reference.coefficients).Begin().Wait();
    plan.DoReconstructionIter(mean, reference, boundary);
    for (auto workers : {size_t(0), size_t(2)})
    {
        auto split = plan.BuildWorkspace(1);
        Fill(split.coefficients, 0.3);
        auto halo = metrics.BindHalo(split.coefficients);
        R::EagerEngine engine(workers);
        auto core = engine.CreateCPUStream(), ring = engine.CreateCPUStream();
        R::MPIProgress progress;
        const auto communication = progress.Import(halo.Begin());
        engine.Submit(core, [&](R::ExecutionContext)
                      {
            // Repeating a pure out-of-place core pass increases its work while
            // preserving the exact same numerical reference result.
            for (int repeat = 0; repeat < 8; ++repeat)
                plan.DoReconstructionIterTo(mean, split.coefficients, split, boundary, C::ReconstructionRegion::Core); });
        ring.WaitFor(core.Record());
        ring.WaitFor(communication);
        engine.Submit(ring, [&](R::ExecutionContext)
                      {
            plan.DoReconstructionIterTo(mean, split.coefficients, split, boundary, C::ReconstructionRegion::HaloDependent);
            plan.CommitReconstruction(split); });
        progress.Drain();
        engine.Synchronize();
        Equal(reference.coefficients, split.coefficients);
    }
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
