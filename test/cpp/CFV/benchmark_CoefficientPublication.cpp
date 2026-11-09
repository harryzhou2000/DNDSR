// E3: production publication versus the same row copy without mesh acquisition.
// Preparation, reset and payload validation stay outside timed batches. This is
// a CFV field probe, not a reacting RHS or a buffer-swap implementation.
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "../Cauchy/CartesianFixture.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <functional>

namespace F = DNDS::CFV;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;

namespace
{
    R::RowLayout Layout(const std::string &name, I width)
    {
        return name == "aos" ? R::RowLayout::Dense(0, width) : R::RowLayout::ColMajor(0, width);
    }
    struct Field
    {
        F::FVField mean;
        F::ReconstructionWorkspace work;
        F::ReconstructionBoundaryValues boundary;
        R::TransferBinding meanHalo, coefficientHalo;
        uint64_t identity;
        Field(const F::VariationalReconstructionPlan &plan, I components, const std::string &layout)
            : mean(plan.Metrics().BuildField(Layout(layout, components))),
              work(plan.BuildWorkspace(components, Layout(layout, plan.NRecDOF() * components))),
              boundary(plan.BuildBoundaryValues(components)),
              meanHalo(plan.Metrics().BindHalo(mean)), coefficientHalo(plan.Metrics().BindHalo(work.coefficients)),
              identity(work.coefficients.Owned().Identity()) {}
    };
    double Value(int field, I global, I component)
    {
        return .03 * std::sin(.017 * global + .11 * component + field);
    }
    void Fill(F::FVField &field, const DNDS::Geom::EntitySpace &cells, int kind)
    {
        auto write = field.Owned().Write();
        const auto view = write.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < view.RowCount(); ++iCell)
            for (I component = 0; component < field.Width(); ++component)
                view(iCell, component) = Value(kind, cells.LocalToGlobal(iCell), component);
    }
    void Copy(Field &field)
    {
        auto source = field.work.temporary.Owned().Read();
        auto destination = field.work.coefficients.Owned().Write();
        const auto src = source.View<R::UniformTableRows>();
        const auto dst = destination.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < src.RowCount(); ++iCell)
            std::copy(src[iCell].begin(), src[iCell].end(), dst[iCell].begin());
    }
    void NativeCopy(Field &field)
    {
        // Bounded candidate, no swap/rebinding: these matching unpadded layouts
        // make the whole-buffer Runtime copy exactly the same logical payload.
        // Production commit remains unchanged until the measured decision.
        field.work.coefficients.Owned().CopyFrom(field.work.temporary.Owned()).Wait();
    }
    void CheckCopy(Field &field, const DNDS::Geom::EntitySpace &cells)
    {
        // A pre-existing binding must still address the published allocation.
        // The expected remote values use global IDs, independent of MPI routes.
        field.coefficientHalo.Begin().Wait();
        auto owned = field.work.coefficients.Owned().Read();
        auto ghost = field.work.coefficients.Ghost().Read();
        const auto local = owned.View<R::UniformTableRows>();
        const auto remote = ghost.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < cells.NumLocal(); ++iCell)
            for (I component = 0; component < field.work.coefficients.Width(); ++component)
                DNDS_check_throw_info((iCell < cells.NumOwned() ? local(iCell, component) : remote(iCell - cells.NumOwned(), component)) ==
                                          Value(2, cells.LocalToGlobal(iCell), component),
                                      "coefficient publication or retained binding changed logical values");
    }
    template <class Action>
    double Measure(const DNDS::MPIInfo &mpi, I repetitions, Action &&action)
    {
        for (int warm = 0; warm < 4; ++warm)
            action();
        MPI_Barrier(mpi.comm);
        const auto start = MPI_Wtime();
        for (I iteration = 0; iteration < repetitions; ++iteration)
            action();
        const double local = (MPI_Wtime() - start) / repetitions;
        double maximum = 0;
        MPI_Allreduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return maximum;
    }
    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 7, "arguments: grid-width degree flow5|split5+12 aos|soa copy-repetitions trials");
        const I width = std::stoll(argv[1]), repetitions = std::stoll(argv[5]);
        const int degree = std::stoi(argv[2]), trials = std::stoi(argv[6]);
        const std::string grouping = argv[3], layout = argv[4];
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        int provided = MPI_THREAD_SINGLE;
        MPI_Query_thread(&provided);
        DNDS_check_throw_info(width >= 4 && width <= 512 && width * width >= mpi.size && (degree == 2 || degree == 3) &&
                                  (grouping == "flow5" || grouping == "split5+12") && (layout == "aos" || layout == "soa") &&
                                  repetitions >= 8 && repetitions <= 100000 && trials > 0 && trials <= 8,
                              "coefficient probe arguments outside bounds");
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        F::FiniteVolumeMetricOptions metricOptions;
        metricOptions.integrationOrder = degree + 2;
        metricOptions.collectivePreflight = false;
        auto metrics = F::FiniteVolumePlan::ConstructMetrics(mesh, metricOptions);
        F::ReconstructionSettings settings;
        settings.maxOrder = degree;
        settings.intOrder = degree + 2;
        auto plan = F::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, F::ReconstructionCapabilities::ForSweeps(true));
        std::vector<std::unique_ptr<Field>> fields;
        fields.push_back(std::make_unique<Field>(plan, 5, layout));
        if (grouping == "split5+12")
            fields.push_back(std::make_unique<Field>(plan, 12, layout));
        struct Phase
        {
            std::string name;
            I repetitions;
            bool copy;
            std::function<void(Field &)> action;
        };
        const I kernels = std::max<I>(4, std::min<I>(32, repetitions / 100));
        std::vector<Phase> phases{
            {"production_commit", repetitions, true, [&](Field &field)
             { plan.CommitReconstruction(field.work); }},
            {"bare_row_copy", repetitions, true, Copy},
            {"native_array_copy", repetitions, true, NativeCopy},
            {"sweep_without_commit", kernels, false, [&](Field &field)
             { plan.DoReconstructionIterTo(field.mean, field.work.coefficients, field.work, field.boundary); }},
            {"wbap_without_commit", kernels, false, [&](Field &field)
             { plan.DoLimiterWBAP_3OrderTo(field.work, degree); }}};
        for (int trial = 0; trial < trials; ++trial)
        {
            J measurements;
            for (I ordinal = 0; ordinal < I(phases.size()); ++ordinal)
            {
                const auto &phase = phases[trial % 2 ? phases.size() - 1 - ordinal : ordinal];
                std::vector<uint64_t> meanVersions;
                std::vector<uint64_t> coefficientVersions;
                for (auto &field : fields)
                {
                    Fill(field->mean, metrics.Cells(), 0);
                    Fill(field->work.coefficients, metrics.Cells(), 1);
                    Fill(field->work.temporary, metrics.Cells(), 2);
                    field->meanHalo.Begin().Wait();
                    field->coefficientHalo.Begin().Wait();
                    meanVersions.push_back(field->mean.Owned().Version());
                    coefficientVersions.push_back(field->work.coefficients.Owned().Version());
                }
                const auto seconds = Measure(mpi, phase.repetitions, [&]
                                             { for (auto &field : fields) phase.action(*field); });
                for (I iField = 0; iField < I(fields.size()); ++iField)
                {
                    auto &field = *fields[iField];
                    DNDS_check_throw_info(field.mean.Owned().Version() == meanVersions[iField] &&
                                              field.work.coefficients.Owned().Identity() == field.identity,
                                          "probe mutated means or replaced the bound coefficient allocation");
                    DNDS_check_throw_info(phase.copy || field.work.coefficients.Owned().Version() == coefficientVersions[iField],
                                          "compute-only probe changed its frozen coefficient input");
                    if (phase.copy)
                        CheckCopy(field, metrics.Cells());
                    auto values = field.work.temporary.Owned().Read();
                    const auto view = values.View<R::UniformTableRows>();
                    for (I iCell = 0; iCell < view.RowCount(); ++iCell)
                        for (I component = 0; component < field.work.temporary.Width(); ++component)
                            DNDS_check_throw_info(std::isfinite(view(iCell, component)), "nonfinite reconstruction/limiter output");
                }
                measurements[phase.name] = {{"seconds", seconds}, {"repetitions", phase.repetitions}, {"calls_per_batch", fields.size()}};
            }
            I local[]{metrics.NumCell(), 16 * metrics.NumCell() * plan.NRecDOF() * (grouping == "flow5" ? 5 : 17)}, low[2], high[2];
            MPI_Allreduce(local, low, 2, MPI_INT64_T, MPI_MIN, mpi.comm);
            MPI_Allreduce(local, high, 2, MPI_INT64_T, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
                std::cout << "PLAN53_COEFFICIENT_COPY " << J{{"trial", trial}, {"np", mpi.size}, {"width", width}, {"degree", degree}, {"modes", plan.NRecDOF()}, {"grouping", grouping}, {"layout", layout}, {"owned_min", low[0]}, {"owned_max", high[0]}, {"logical_read_write_bytes_min", low[1]}, {"logical_read_write_bytes_max", high[1]}, {"preflight", false}, {"mpi_thread_provided", provided}, {"mean_generations_unchanged", true}, {"coefficient_allocation_retained", true}, {"retained_binding_payload_exact", true}, {"phases", measurements}}.dump() << std::endl;
        }
    }
}
int main(int argc, char **argv)
{
    const int initialized = DNDS::MPI::Init_thread(&argc, &argv);
    if (initialized != MPI_SUCCESS)
        return initialized;
    try
    {
        Run(argc, argv);
    }
    catch (const std::exception &error)
    {
        std::cerr << "coefficient probe: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
