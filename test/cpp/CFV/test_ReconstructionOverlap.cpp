#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/Runtime/Communication/MPIProgress.hpp"
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
    auto referenceCoefficients = plan.BuildCoefficients(2), splitCoefficients = plan.BuildCoefficients(2);
    Fill(mean, 1);
    Fill(referenceCoefficients, 0.2);
    Fill(splitCoefficients, 0.2);
    auto boundary = plan.BuildBoundaryValues(2);
    {
        auto write = boundary.values.Write();
        for (I row = 0; row < boundary.values.RowCount(); ++row)
            for (I component = 0; component < 2; ++component)
                write.View()(row, component) = 1 + component;
    }
    auto meanHalo = metrics.BindHalo(mean);
    auto referenceHalo = metrics.BindHalo(referenceCoefficients), splitHalo = metrics.BindHalo(splitCoefficients);
    meanHalo.Begin().Wait();
    referenceHalo.Begin().Wait();
    plan.DoReconstructionIter(mean, referenceCoefficients, reference, boundary);
    {
        auto pendingMean = meanHalo.Begin();
        auto pendingRec = splitHalo.Begin();
        // Even if MPI finishes quickly, active operation leases remain until
        // completion is observed. Core therefore proves no ghost reads/writes.
        plan.DoReconstructionIterTo(mean, splitCoefficients, split, boundary, C::ReconstructionRegion::Core);
        if (mpi.size > 1 && mean.Owned().RowCount() > 0)
            CHECK_THROWS(plan.CommitReconstruction(splitCoefficients, split)); // owned send input still pinned
        pendingMean.Wait();
        pendingRec.Wait();
    }
    plan.DoReconstructionIterTo(mean, splitCoefficients, split, boundary, C::ReconstructionRegion::HaloDependent);
    plan.CommitReconstruction(splitCoefficients, split);
    Equal(referenceCoefficients, splitCoefficients);
    for (int order = plan.Settings().maxOrder; order >= 1; --order)
    {
        referenceHalo.Begin().Wait();
        plan.DoLimiterWBAP_3Order(referenceCoefficients, reference, order);
        auto pending = splitHalo.Begin();
        plan.DoLimiterWBAP_3OrderTo(splitCoefficients, split, order, std::nullopt, C::ReconstructionRegion::Core);
        pending.Wait();
        plan.DoLimiterWBAP_3OrderTo(splitCoefficients, split, order, std::nullopt, C::ReconstructionRegion::HaloDependent);
        plan.CommitReconstruction(splitCoefficients, split);
        Equal(referenceCoefficients, splitCoefficients);
    }
    R::Array<double> indicator(R::RowLayout::Dense(metrics.NumCell(), 1));
    {
        auto write = indicator.Write();
        for (I row = 0; row < metrics.NumCell(); ++row)
            write.View()(row, 0) = -1;
    }
    plan.DoLimiterWBAP_3OrderTo(splitCoefficients, split, 1, std::nullopt, C::ReconstructionRegion::All, std::cref(indicator), 0);
    plan.CommitReconstruction(splitCoefficients, split);
    Equal(referenceCoefficients, splitCoefficients); // inactive is exact passthrough
    referenceHalo.Begin().Wait();
    splitHalo.Begin().Wait();
    plan.DoLimiterWBAP_3Order(referenceCoefficients, reference, 1);
    {
        auto write = indicator.Write();
        for (I row = 0; row < metrics.NumCell(); ++row)
            write.View()(row, 0) = 0;
    }
    plan.DoLimiterWBAP_3OrderTo(splitCoefficients, split, 1, std::nullopt, C::ReconstructionRegion::All, std::cref(indicator), 0);
    plan.CommitReconstruction(splitCoefficients, split);
    Equal(referenceCoefficients, splitCoefficients); // equality activates
    CHECK_THROWS(plan.DoLimiterWBAP_3OrderTo(splitCoefficients, split, 1, std::nullopt, C::ReconstructionRegion::All, std::nullopt,
                                             std::numeric_limits<double>::quiet_NaN()));
    if (metrics.NumCell())
    {
        {
            auto write = indicator.Write();
            write.View()(0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        CHECK_THROWS(plan.DoLimiterWBAP_3OrderTo(splitCoefficients, split, 1, std::nullopt, C::ReconstructionRegion::All, std::cref(indicator), 0));
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
    auto referenceCoefficients = plan.BuildCoefficients(1), splitCoefficients = plan.BuildCoefficients(1);
    Fill(mean, 1);
    Fill(snapshot, 0.2);
    Fill(referenceCoefficients, 0.2);
    Fill(splitCoefficients, 0.2);
    auto boundary = plan.BuildBoundaryValues(1);
    {
        auto write = boundary.values.Write();
        for (I row = 0; row < boundary.values.RowCount(); ++row)
            write.View()(row, 0) = 1;
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(referenceCoefficients).Begin().Wait();
    plan.DoReconstructionIter(mean, referenceCoefficients, reference, boundary);
    auto snapshotHalo = metrics.BindHalo(snapshot);
    auto pending = snapshotHalo.Begin();
    plan.DoReconstructionIterTo(mean, snapshot, split, boundary, C::ReconstructionRegion::Core);
    pending.Wait();
    plan.DoReconstructionIterTo(mean, snapshot, split, boundary, C::ReconstructionRegion::HaloDependent);
    // The To operator fills update scratch. Publish only after both regions,
    // exactly as the production split path does, before observing coefficients.
    plan.CommitReconstruction(splitCoefficients, split);
    Equal(referenceCoefficients, splitCoefficients);
}

TEST_CASE("One-cell OpenMP SOR blocks reduce to Jacobi with frozen cross-block reads" *
          doctest::skip(DNDS::OMP::DistributedThreads() == 1))
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true, 6);
    const auto state = fixture.mesh->CurrentNativeState();
    const I owned = state->NumOwned(G::EntityKind::Cell);
    std::vector<I> starts;
    for (I iCell = 0; iCell <= owned; ++iCell)
        starts.push_back(iCell);
    if (!owned)
        starts.push_back(0);
    auto mesh = G::UnstructuredMesh::FromNative(mpi, state->WithLocalPartitions(std::move(starts)));
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.coreFirst = true;
    const auto jacobi = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    settings.SORInstead = true;
    const auto sor = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto means = metrics.BuildField(5);
    auto input = jacobi.BuildCoefficients(5);
    auto a = jacobi.BuildCoefficients(5), b = sor.BuildCoefficients(5);
    auto workA = jacobi.BuildWorkspace(5), workB = sor.BuildWorkspace(5);
    auto boundaryA = jacobi.BuildBoundaryValues(5), boundaryB = sor.BuildBoundaryValues(5);
    Fill(means, 1);
    Fill(input, .03);
    metrics.BindHalo(means).Begin().Wait();
    metrics.BindHalo(input).Begin().Wait();
    jacobi.DoReconstructionIterTo(means, input, workA, boundaryA);
    sor.DoReconstructionIterTo(means, input, workB, boundaryB, C::ReconstructionRegion::Core);
    sor.DoReconstructionIterTo(means, input, workB, boundaryB, C::ReconstructionRegion::HaloDependent);
    jacobi.CommitReconstruction(a, workA);
    sor.CommitReconstruction(b, workB);
    Equal(a, b);
}

TEST_CASE("CFV split operators share eager ordering across inline and worker CPU launchers")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    auto mean = metrics.BuildField(1);
    auto reference = plan.BuildWorkspace(1);
    auto referenceCoefficients = plan.BuildCoefficients(1);
    Fill(mean, 1);
    Fill(referenceCoefficients, 0.3);
    auto boundary = plan.BuildBoundaryValues(1);
    {
        auto write = boundary.values.Write();
        for (I row = 0; row < boundary.values.RowCount(); ++row)
            write.View()(row, 0) = 1;
    }
    auto meanHalo = metrics.BindHalo(mean);
    meanHalo.Begin().Wait();
    metrics.BindHalo(referenceCoefficients).Begin().Wait();
    plan.DoReconstructionIter(mean, referenceCoefficients, reference, boundary);
    for (auto workers : {size_t(0), size_t(2)})
    {
        auto split = plan.BuildWorkspace(1);
        auto splitCoefficients = plan.BuildCoefficients(1);
        Fill(splitCoefficients, 0.3);
        auto halo = metrics.BindHalo(splitCoefficients);
        R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
        auto core = engine.CreateCPUStream(), ring = engine.CreateCPUStream();
        R::MPIProgress progress;
        const auto communication = progress.Import(halo.Begin());
        // Repeating a pure out-of-place core pass increases its work while
        // preserving the exact same numerical reference result. Each record
        // retains its operands and declares reads/writes at submission.
        for (int repeat = 0; repeat < 8; ++repeat)
            plan.SubmitReconstructionIterTo(engine, core, mean, splitCoefficients, split, boundary,
                                            C::ReconstructionRegion::Core);
        ring.WaitFor(core.Record());
        ring.WaitFor(communication);
        plan.SubmitReconstructionIterTo(engine, ring, mean, splitCoefficients, split, boundary,
                                        C::ReconstructionRegion::HaloDependent);
        plan.SubmitCommitReconstruction(engine, ring, splitCoefficients, split);
        progress.Drain();
        engine.Synchronize();
        Equal(referenceCoefficients, splitCoefficients);
        auto limitedReference = plan.BuildWorkspace(1);
        auto limitedReferenceCoefficients = plan.BuildCoefficients(1);
        DNDS::Algebra::Copy(limitedReferenceCoefficients, referenceCoefficients);
        for (int order = plan.Settings().maxOrder; order >= 1; --order)
        {
            metrics.BindHalo(limitedReferenceCoefficients).Begin().Wait();
            halo.Begin().Wait();
            plan.DoLimiterWBAP_3Order(limitedReferenceCoefficients, limitedReference, order);
            plan.SubmitLimiterWBAP_3OrderTo(engine, ring, splitCoefficients, split, order, std::nullopt,
                                            C::ReconstructionRegion::All);
            plan.SubmitCommitReconstruction(engine, ring, splitCoefficients, split);
            engine.Synchronize();
            Equal(limitedReferenceCoefficients, splitCoefficients);
        }
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

TEST_CASE("CPU FV reconstruction limiter and linear actions agree across mixed DOF physical layouts")
{
    auto fixture = G::OverlapTest::BuildNativeFixture(mpi, false);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(fixture.mesh);
    for (bool sor : {false, true})
    {
        C::ReconstructionSettings settings;
        settings.maxOrder = 3;
        settings.SORInstead = sor;
        settings.coreFirst = true;
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        const I components = 5, width = plan.NRecDOF() * components;
        auto referenceMean = metrics.BuildField(components);
        auto reference = plan.BuildWorkspace(components);
        auto referenceCoefficients = plan.BuildCoefficients(components);
        auto boundary = plan.BuildBoundaryValues(components);
        Fill(referenceMean, 1.0);
        Fill(referenceCoefficients, .02);
        {
            auto write = boundary.values.Write();
            for (I iRow = 0; iRow < boundary.values.RowCount(); ++iRow)
                for (I iComp = 0; iComp < components; ++iComp)
                    write.View()(iRow, iComp) = 1 + .1 * iComp;
        }
        for (int format = 0; format < 2; ++format)
        {
            CAPTURE(sor);
            CAPTURE(format);
            Fill(referenceCoefficients, .02);
            const auto recShape = format == 0 ? R::RowLayout::ColMajor(0, width, {32, 1}) : R::RowLayout::TiledColMajor(0, width, 8);
            auto mean = metrics.BuildField(format == 0 ? R::RowLayout::TiledColMajor(0, components, 8) : R::RowLayout::ColMajor(0, components, {32, 1}));
            auto work = plan.BuildWorkspace(components, recShape);
            auto coefficients = plan.BuildCoefficients(components, recShape);
            Fill(mean, 1.0);
            Fill(coefficients, .02);
            auto refMeanHalo = metrics.BindHalo(referenceMean), meanHalo = metrics.BindHalo(mean);
            auto refHalo = metrics.BindHalo(referenceCoefficients), halo = metrics.BindHalo(coefficients);
            refMeanHalo.Begin().Wait();
            meanHalo.Begin().Wait();
            for (int iteration = 0; iteration < 4; ++iteration)
            {
                refHalo.Begin().Wait();
                halo.Begin().Wait();
                plan.DoReconstructionIter(referenceMean, referenceCoefficients, reference, boundary);
                plan.DoReconstructionIter(mean, coefficients, work, boundary);
                Equal(referenceCoefficients, coefficients);
            }
            auto transform = plan.BuildFaceTransform(0, components);
            for (int order = settings.maxOrder; order >= 1; --order)
            {
                refHalo.Begin().Wait();
                halo.Begin().Wait();
                plan.DoLimiterWBAP_3Order(referenceCoefficients, reference, order, std::cref(transform));
                plan.DoLimiterWBAP_3Order(coefficients, work, order, std::cref(transform));
                Equal(referenceCoefficients, coefficients);
            }
            refHalo.Begin().Wait();
            halo.Begin().Wait();
            auto refResult = metrics.BuildField(width), result = metrics.BuildField(recShape);
            plan.ApplyLinearOperator(referenceCoefficients, refResult);
            plan.ApplyLinearOperator(coefficients, result);
            Equal(refResult, result);
            plan.BuildRightHandSide(referenceMean, boundary, refResult);
            plan.BuildRightHandSide(mean, boundary, result);
            Equal(refResult, result);
            plan.ApplyLocalPreconditioner(referenceCoefficients, refResult);
            plan.ApplyLocalPreconditioner(coefficients, result);
            Equal(refResult, result);
        }
    }
}
