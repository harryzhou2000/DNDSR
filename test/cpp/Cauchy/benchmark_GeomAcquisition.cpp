// E7: separate operator-entry ownership cost from numerical work. Opt-in only.
// Allocation counts cover C++ new on the calling thread, not Eigen malloc,
// allocator internals, MPI background threads or the number of Array leases.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "CartesianFixture.hpp"
#include <cstdlib>
#include <new>
#include <iostream>

namespace AllocationProbe
{
    thread_local bool active = false;
    thread_local uint64_t calls = 0, bytes = 0;
    void Count(size_t size)
    {
        if (active)
        {
            ++calls;
            bytes += size;
        }
    }
}
void *operator new(size_t size)
{
    if (auto *value = std::malloc(size ? size : 1))
    {
        AllocationProbe::Count(size);
        return value;
    }
    throw std::bad_alloc();
}
void *operator new[](size_t size) { return ::operator new(size); }
void operator delete(void *value) noexcept { std::free(value); }
void operator delete[](void *value) noexcept { std::free(value); }
void operator delete(void *value, size_t) noexcept { std::free(value); }
void operator delete[](void *value, size_t) noexcept { std::free(value); }
void *operator new(size_t size, std::align_val_t alignment)
{
    void *value = nullptr;
    if (posix_memalign(&value, size_t(alignment), size ? size : 1) != 0)
        throw std::bad_alloc();
    AllocationProbe::Count(size);
    return value;
}
void *operator new[](size_t size, std::align_val_t alignment) { return ::operator new(size, alignment); }
void operator delete(void *value, std::align_val_t) noexcept { std::free(value); }
void operator delete[](void *value, std::align_val_t) noexcept { std::free(value); }
void operator delete(void *value, size_t, std::align_val_t) noexcept { std::free(value); }
void operator delete[](void *value, size_t, std::align_val_t) noexcept { std::free(value); }

namespace C = DNDS::Cauchy;
namespace F = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;

namespace
{
    template <class Action>
    J Probe(const DNDS::MPIInfo &mpi, I repetitions, Action &&action)
    {
        for (int warm = 0; warm < 4; ++warm)
            action();
        // Count separately so counter increments do not distort the timing.
        AllocationProbe::calls = AllocationProbe::bytes = 0;
        AllocationProbe::active = true;
        for (int sample = 0; sample < 8; ++sample)
            action();
        AllocationProbe::active = false;
        uint64_t counts[]{AllocationProbe::calls, AllocationProbe::bytes};
        uint64_t minimum[2], maximum[2];
        MPI_Allreduce(counts, minimum, 2, MPI_UINT64_T, MPI_MIN, mpi.comm);
        MPI_Allreduce(counts, maximum, 2, MPI_UINT64_T, MPI_MAX, mpi.comm);
        MPI_Barrier(mpi.comm);
        const auto start = MPI_Wtime();
        for (I iteration = 0; iteration < repetitions; ++iteration)
            action();
        const double local = (MPI_Wtime() - start) / repetitions;
        double elapsed = 0;
        MPI_Allreduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return {{"seconds", elapsed}, {"repetitions", repetitions}, {"new_calls_min", minimum[0] / 8.}, {"new_calls_max", maximum[0] / 8.}, {"new_bytes_min", minimum[1] / 8.}, {"new_bytes_max", maximum[1] / 8.}};
    }

    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 4, "arguments: grid-width acquisition-repetitions trials");
        const I width = std::stoll(argv[1]), repetitions = std::stoll(argv[2]);
        const int trials = std::stoi(argv[3]);
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        DNDS_check_throw_info(width >= 4 && width <= 512 && width * width >= mpi.size && repetitions >= 8 && repetitions <= 100000 && trials > 0 && trials <= 8,
                              "bounded acquisition probe arguments invalid");
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .001;
        settings.implicit.collectiveChecks = false;
        settings.numerics.reconstruction = "variational";
        settings.numerics.limiter = "none";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 1;
        settings.numerics.positivity = false;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto &metrics = fv.Metrics();
        const auto &reconstruction = fv.Reconstruction();
        auto mean = metrics.BuildField(5);
        auto work = reconstruction.BuildWorkspace(5);
        auto boundary = reconstruction.BuildBoundaryValues(5);
        {
            auto values = mean.Owned().Write();
            auto coefficients = work.coefficients.Owned().Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto point = metrics.GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .01 * std::sin(6.28 * point[0]), 1, .1, 0, 1});
                for (I component = 0; component < 5; ++component)
                    values.View()(iCell, component) = value[component];
                for (I component = 0; component < work.coefficients.Width(); ++component)
                    coefficients.View()(iCell, component) = .001 * std::sin(.13 * component + .01 * iCell);
            }
        }
        metrics.BindHalo(mean).Begin().Wait();
        metrics.BindHalo(work.coefficients).Begin().Wait();
        const auto meanVersion = mean.Owned().Version(), coefficientVersion = work.coefficients.Owned().Version();
        C::AuxiliaryRequest request;
        request.meanThermodynamics = request.faceSpectralRadius = request.cellSpectralSum = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(mean, auxiliary);
        reconstruction.DoReconstructionIterTo(mean, work.coefficients, work, boundary);
        std::vector<double> reference;
        {
            auto read = work.temporary.Owned().Read();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (I component = 0; component < work.coefficients.Width(); ++component)
                    reference.push_back(read.View()(iCell, component));
        }
        volatile I observedRows = 0;
        struct Phase
        {
            std::string name;
            I repetitions;
            std::function<void()> action;
        };
        const I kernels = std::max<I>(4, std::min<I>(32, repetitions / 100));
        std::vector<Phase> phases{
            {"mesh_primary", repetitions, [&]
             { auto read = mesh->AcquireRead(G::MeshReadRequirements::Primary()); observedRows = read.HostView().NumCell(); }},
            {"mesh_cell_flux", repetitions, [&]
             { auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux()); observedRows = read.HostView().NumCell(); }},
            {"metric_read", repetitions, [&]
             { auto read = metrics.AcquireRead(); }},
            {"basis_read", repetitions, [&]
             { auto read = reconstruction.AcquireBasisRead(); }},
            {"rhs_acquire", repetitions, [&]
             { auto operation = fv.AcquireRHSOperator(); }},
            {"auxiliary_begin_abandon", repetitions, [&]
             { auto preparation = fv.BeginAuxValues(mean, auxiliary); }},
            {"auxiliary_complete", kernels, [&]
             { fv.PrepareAuxValues(mean, auxiliary); }},
            {"reconstruction_all", kernels, [&]
             { reconstruction.DoReconstructionIterTo(mean, work.coefficients, work, boundary); }},
            {"reconstruction_core", kernels, [&]
             { reconstruction.DoReconstructionIterTo(mean, work.coefficients, work, boundary, F::ReconstructionRegion::Core); }},
            {"reconstruction_dependent", kernels, [&]
             { reconstruction.DoReconstructionIterTo(mean, work.coefficients, work, boundary, F::ReconstructionRegion::HaloDependent); }}};
        for (int trial = 0; trial < trials; ++trial)
        {
            J measurements;
            for (I ordinal = 0; ordinal < I(phases.size()); ++ordinal)
            {
                const auto &phase = phases[trial % 2 ? phases.size() - 1 - ordinal : ordinal];
                measurements[phase.name] = Probe(mpi, phase.repetitions, phase.action);
            }
            DNDS_check_throw_info(observedRows == metrics.NumCell(), "unexpected mesh read row count");
            // Abandoned auxiliary begins publish no valid provenance. Refresh
            // before validating; the benchmark never consumes those outputs.
            fv.PrepareAuxValues(mean, auxiliary);
            fv.ValidateAuxValues(mean, auxiliary);
            reconstruction.DoReconstructionIterTo(mean, work.coefficients, work, boundary, F::ReconstructionRegion::Core);
            reconstruction.DoReconstructionIterTo(mean, work.coefficients, work, boundary, F::ReconstructionRegion::HaloDependent);
            {
                auto read = work.temporary.Owned().Read();
                for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                    for (I component = 0; component < work.coefficients.Width(); ++component)
                        DNDS_check_throw_info(std::isfinite(read.View()(iCell, component)) && read.View()(iCell, component) == reference[size_t(iCell * work.coefficients.Width() + component)],
                                              "split reconstruction differs from unsplit frozen output");
            }
            DNDS_check_throw_info(mean.Owned().Version() == meanVersion && work.coefficients.Owned().Version() == coefficientVersion,
                                  "read/auxiliary/reconstruction probe mutated its inputs");
            I local[2]{metrics.NumCell(), I(metrics.CellWorkPartition().Core().size())}, low[2], high[2];
            MPI_Allreduce(local, low, 2, MPI_INT64_T, MPI_MIN, mpi.comm);
            MPI_Allreduce(local, high, 2, MPI_INT64_T, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
                std::cout << "PLAN53_ACQUISITION " << J{{"trial", trial}, {"np", mpi.size}, {"width", width}, {"degree", 3}, {"owned_min", low[0]}, {"owned_max", high[0]}, {"core_min", low[1]}, {"core_max", high[1]}, {"input_generations_unchanged", true}, {"split_matches_unsplit", true}, {"phases", measurements}}.dump() << std::endl;
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
        AllocationProbe::active = false;
        std::cerr << "acquisition probe: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
