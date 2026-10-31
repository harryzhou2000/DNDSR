// Matched CPU preparation/sweep benchmark. Both methods consume the same mesh,
// cell means, boundary values and communication schedule in one MPI executable.
// Run on one allocated node with OMP_NUM_THREADS=1; timings are max-rank seconds.
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "CFV/VariationalReconstruction_LimiterProcedure.hxx"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include <iostream>

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;

template <int dim>
void Run(DNDS::MPIInfo mpi, const char *file, int degree, int sweeps, int trials, double period)
{
    G::PartitionOptions partition;
    partition.metisSeed = 42;
    G::AutoAppendName2ID names;
    G::NativePeriodicImportOptions periodic;
    periodic.frames.translation[1].map() = G::tPoint{period, 0, 0};
    periodic.frames.translation[2].map() = G::tPoint{0, period, 0};
    periodic.frames.translation[3].map() = G::tPoint{0, 0, period};
    auto mesh = G::ReadNativeMeshFromCGNS(mpi, dim, file, partition, [&](const std::string &name)
                                          { return names(name); }, period > 0 ? std::optional<G::NativePeriodicImportOptions>(periodic) : std::nullopt);
    mesh->PrepareNativeTopology();
    auto measure = [&](auto &&action)
    {
        MPI_Barrier(mpi.comm);
        const double start = MPI_Wtime();
        action();
        const double elapsed = MPI_Wtime() - start;
        double maximum = 0;
        MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return maximum;
    };
    I total = mesh->NumCell(), global = 0;
    MPI_Allreduce(&total, &global, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    for (int trial = 0; trial < trials; ++trial)
    {
        C::ReconstructionSettings settings;
        settings.maxOrder = degree, settings.intOrder = degree + 2;
        settings.boundaryDerivativeWeights = {1, 0, 0, 0}; // Matched physical-value BC with the legacy callback below.
        std::optional<C::FiniteVolumePlan> metrics;
        std::optional<C::VariationalReconstructionPlan> plan;
        C::VariationalReconstruction<dim> legacy(mpi, mesh);
        C::VRSettings old(dim);
        nlohmann::ordered_json json;
        old.WriteIntoJson(json);
        json["maxOrder"] = degree, json["intOrder"] = settings.intOrder;
        json["SORInstead"] = false, json["cacheDiffBase"] = true;
        legacy.parseSettings(json);
        double nativeMetric, nativePrepare, legacyMetric, legacyPrepare;
        const auto prepareNative = [&]
        {
            nativeMetric = measure([&]
                                   { metrics.emplace(C::FiniteVolumePlan::ConstructMetrics(mesh, settings.intOrder)); });
            nativePrepare = measure([&]
                                    { plan.emplace(C::VariationalReconstructionPlan::ConstructRecCoeff(*metrics, settings)); });
        };
        const auto prepareLegacy = [&]
        {
            legacyMetric = measure([&]
                                   { legacy.ConstructMetrics(); });
            legacyPrepare = measure([&]
                                    {
                legacy.ConstructBaseAndWeight([](G::t_index, int order) { return order == 0 ? 1. : 0.; });
                legacy.ConstructRecCoeff();
                if (mesh->IsPeriodic()) legacy.SetPeriodicTransformations(); });
        };
        if (trial % 2)
            prepareLegacy(), prepareNative();
        else
            prepareNative(), prepareLegacy();
        auto mean = metrics->BuildField(5);
        auto work = plan->BuildWorkspace(5);
        auto boundary = plan->BuildBoundaryValues(5);
        C::tUDof<5> oldMean;
        C::tURec<5> oldRec, oldNext, limited, scratch;
        C::tScalarPair indicator;
        legacy.BuildUDof(oldMean, 5), legacy.BuildURec(oldRec, 5), legacy.BuildURec(oldNext, 5);
        legacy.BuildURec(limited, 5), legacy.BuildURec(scratch, 5), legacy.BuildScalar(indicator);
        oldRec.setConstant(0), oldNext.setConstant(0);
        {
            auto write = mean.Owned().Write();
            auto values = write.template View<R::FixedRows<5>>();
            for (I iCell = 0; iCell < metrics->NumCell(); ++iCell)
                for (int v = 0; v < 5; ++v)
                    values(iCell, v) = oldMean[iCell](v) = std::sin(.03 * mesh->OriginalEntityId(G::EntityKind::Cell, iCell) + .2 * v);
            auto bc = boundary.values.Write();
            for (I i = 0; i < plan->NumBoundaryQuadrature(); ++i)
                for (int v = 0; v < 5; ++v)
                    bc.View()(i, v) = 0;
        }
        for (I iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
            indicator(iCell, 0) = 1;
        auto meanHalo = metrics->BindHalo(mean), halo = metrics->BindHalo(work.coefficients);
        meanHalo.Begin().Wait();
        oldMean.trans.startPersistentPull(), oldMean.trans.waitPersistentPull();
        typename C::VariationalReconstruction<dim>::template TFBoundary<5> bc =
            [](const auto &, const auto &, I, I, int, const G::tPoint &, const G::tPoint &, G::t_index)
        { return Eigen::Matrix<double, 5, 1>::Zero().eval(); };
        auto nativeSweep = [&]
        { halo.Begin().Wait(); plan->DoReconstructionIter(mean, work, boundary); };
        auto legacySweep = [&]
        {
            oldRec.trans.startPersistentPull(), oldRec.trans.waitPersistentPull();
            legacy.template DoReconstructionIter<5>(oldRec, oldNext, oldMean, bc, true);
            std::swap(oldRec, oldNext);
        };
        for (int i = 0; i < 5; ++i)
            nativeSweep(), legacySweep();
        double nativeIteration = 0, legacyIteration = 0;
        const auto iterateNative = [&]
        { nativeIteration = measure([&]
                                    { for (int i = 0; i < sweeps; ++i) nativeSweep(); }) /
                            sweeps; };
        const auto iterateLegacy = [&]
        { legacyIteration = measure([&]
                                    { for (int i = 0; i < sweeps; ++i) legacySweep(); }) /
                            sweeps; };
        if (trial % 2)
            iterateLegacy(), iterateNative();
        else
            iterateNative(), iterateLegacy();
        double difference = 0, maximum = 0;
        {
            auto read = work.coefficients.Owned().Read();
            auto values = read.template View<R::TableRows>();
            for (I iCell = 0; iCell < metrics->NumCell(); ++iCell)
                for (int v = 0; v < 5; ++v)
                    for (int mode = 0; mode < plan->NRecDOF(); ++mode)
                    {
                        const double delta = values(iCell, v * plan->NRecDOF() + mode) - oldRec[iCell](mode, v);
                        if (!std::isfinite(delta))
                            difference = std::numeric_limits<double>::infinity();
                        else
                            difference = std::max(difference, std::abs(delta));
                    }
        }
        MPI_Allreduce(&difference, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(std::isfinite(maximum) && maximum < 1e-8, "benchmark methods disagree");
        using Batch = typename C::VariationalReconstruction<dim>::template tLimitBatch<5>;
        typename C::VariationalReconstruction<dim>::template tFMEig<5> identity =
            [](const auto &, const auto &, const G::tPoint &, const Eigen::Ref<Batch> &a) -> Batch
        { return a; };
        halo.Begin().Wait();
        oldRec.trans.startPersistentPull(), oldRec.trans.waitPersistentPull();
        // One limiter application per preparation trial, from the matched
        // reconstruction above, including identical order-wise halo traffic.
        const double nativeLimiter = measure([&]
                                             {
            for (int order = degree; order >= 1; --order)
            { plan->DoLimiterWBAP_3Order(work, order); halo.Begin().Wait(); } });
        const double legacyLimiter = measure([&]
                                             { legacy.template DoLimiterWBAP_3<5>(oldMean, oldRec, limited, scratch, indicator, true, identity, identity, true); });
        if (mpi.rank == 0)
        {
            nlohmann::ordered_json report{{"cells", global}, {"np", mpi.size}, {"degree", degree}, {"trial", trial}, {"sweeps", sweeps}, {"native_metrics", nativeMetric}, {"legacy_metrics", legacyMetric}, {"native_prepare", nativePrepare}, {"legacy_prepare", legacyPrepare}, {"native_iteration", nativeIteration}, {"legacy_iteration", legacyIteration}, {"native_wbap", nativeLimiter}, {"legacy_wbap", legacyLimiter}, {"max_coefficient_error", maximum}};
            std::cout << "FV_BENCH " << report.dump() << std::endl;
        }
    }
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    try
    {
        DNDS_check_throw_info(argc == 7, "arguments: mesh dim degree sweeps trials periodic-length-or-zero");
        const int dim = std::stoi(argv[2]), degree = std::stoi(argv[3]), sweeps = std::stoi(argv[4]), trials = std::stoi(argv[5]);
        DNDS_check_throw_info(sweeps > 0 && trials > 0 && (dim == 2 || dim == 3), "invalid benchmark arguments");
        if (dim == 2)
            Run<2>(mpi, argv[1], degree, sweeps, trials, std::stod(argv[6]));
        else
            Run<3>(mpi, argv[1], degree, sweeps, trials, std::stod(argv[6]));
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
