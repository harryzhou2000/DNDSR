// E6 frozen-operator assessment, opt-in and outside CTest. Each invocation
// retains only its selected storage mode until after the memory observations.
// Independent finite-difference/periodic/source oracles stay in test_Jacobian.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "CartesianFixture.hpp"
#include <sys/resource.h>
#include <fstream>
#include <iostream>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
using J = nlohmann::ordered_json;

namespace
{
    uint64_t ResidentBytes()
    {
        std::ifstream input("/proc/self/status");
        std::string line;
        while (std::getline(input, line))
            if (line.compare(0, 6, "VmRSS:") == 0)
                return std::stoull(line.substr(6)) * 1024;
        DNDS_check_throw_info(false, "Jacobian assessment requires Linux VmRSS");
        return 0;
    }

    uint64_t PeakResidentBytes()
    {
        rusage usage{};
        DNDS_check_throw_info(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
        return uint64_t(usage.ru_maxrss) * 1024;
    }

    void Run(int argc, char **argv)
    {
        DNDS_check_throw_info(argc == 5, "arguments: grid-width assembled|evaluated repetitions trials");
        const I width = std::stoll(argv[1]);
        const std::string mode = argv[2];
        const int repetitions = std::stoi(argv[3]), trials = std::stoi(argv[4]);
        const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
        DNDS_check_throw_info(width >= 4 && width <= 400 && width * width >= mpi.size &&
                                  (mode == "assembled" || mode == "evaluated") &&
                                  repetitions >= 1 && repetitions <= 20 && trials >= 1 && trials <= 4,
                              "Jacobian assessment arguments outside declared bounds");
        int provided = 0;
        MPI_Query_thread(&provided);
        auto mesh = CauchyTest::PeriodicQuads(mpi, width);
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .001;
        settings.implicit.collectiveChecks = false;
        settings.numerics.reconstruction = "variational";
        settings.numerics.limiter = "none";
        settings.numerics.positivity = false;
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 5;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto &metrics = fv.Metrics();
        auto state = metrics.BuildField(5), rhs = metrics.BuildField(5);
        auto action = metrics.BuildField(5), forward = metrics.BuildField(5), update = metrics.BuildField(5);
        R::Array<double> inverseTau(R::RowLayout::Dense(metrics.NumCell(), 1));
        {
            auto input = state.Owned().Write(), forcing = rhs.Owned().Write(), pseudo = inverseTau.Write();
            const auto u = input.View<R::UniformTableRows>(), b = forcing.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto point = metrics.GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .01 * std::sin(6.28 * point[0]),
                                                                .2 + .01 * std::cos(6.28 * point[1]), -.1, .05, 1});
                for (int j = 0; j < 5; ++j)
                {
                    u(iCell, j) = value[j];
                    b(iCell, j) = .01 * (j + 1) * std::cos(3 * point[0] + 2 * point[1]);
                }
                pseudo.View<R::FixedRows<1>>()(iCell, 0) = 1;
            }
        }
        auto halo = metrics.BindHalo(state);
        halo.Begin().Wait();
        const auto stateVersion = state.Owned().Version();
        C::AuxiliaryRequest request;
        request.meanThermodynamics = request.faceSpectralRadius = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(state, auxiliary);
        const auto storage = mode == "assembled" ? C::OffDiagonalStorage::Assembled : C::OffDiagonalStorage::SemiMatrixFree;
        const uint64_t before = ResidentBytes();
        auto problem = fv.BuildLinearProblem(state, C::DiagonalStorage::Entries, storage);
        C::LinearSolveSettings linear;
        linear.maxIterations = 5;
        linear.relativeTolerance = linear.absoluteTolerance = 0;
        C::LinearSolver solver(problem, state, [&](auto &field)
                               { return metrics.BindHalo(field); }, linear, {true, false});
        const auto prepare = [&]
        { fv.PrepareJacobian(state, auxiliary, inverseTau, *problem, {1000, 1, false}); };
        const auto pair = [&]
        { problem->ApplySGS(rhs, state, forward, update); };
        const auto measure = [&](auto &&operation, int count)
        {
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int iteration = 0; iteration < count; ++iteration)
                operation();
            const double elapsed = (MPI_Wtime() - start) / count;
            double maximum = 0;
            MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return maximum;
        };
        const double firstPreparation = measure(prepare, 1);
        problem->InvertDiagonal();
        pair();
        for (int trial = 0; trial < trials; ++trial)
        {
            J phases;
            phases["auxiliary"] = measure([&]
                                          { fv.PrepareAuxValues(state, auxiliary); }, repetitions);
            phases["prepare_warm"] = measure(prepare, repetitions);
            // Inversion is lazy: refresh its input outside this isolated timer.
            prepare();
            phases["inverse"] = measure([&]
                                        { problem->InvertDiagonal(); }, 1);
            phases["matrix_action"] = measure([&]
                                              { problem->Apply(state, action); }, repetitions);
            phases["sgs_pair_local"] = measure(pair, repetitions);
            for (int reuse : {1, 5, 20})
                phases["prepare_inverse_pairs_" + std::to_string(reuse)] = measure([&]
                                                                                   {
                    prepare();
                    problem->InvertDiagonal();
                    for (int use = 0; use < reuse; ++use)
                        pair(); }, repetitions);
            I completed = 0;
            phases["prepare_inverse_solve5"] = measure([&]
                                                       {
                prepare();
                problem->InvertDiagonal();
                completed += solver.Solve(rhs).iterations; }, repetitions);
            DNDS_check_throw_info(completed == 5 * I(repetitions), "communicated SGS solve changed its declared work count");
            DNDS_check_throw_info(state.Owned().Version() == stateVersion, "Jacobian assessment mutated frozen means");
            uint64_t memory[]{before, ResidentBytes(), PeakResidentBytes()};
            MPI_Allreduce(MPI_IN_PLACE, memory, 3, MPI_UINT64_T, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
                std::cout << "PLAN53_JACOBIAN " << J{{"np", mpi.size}, {"width", width}, {"degree", 3}, {"mode", mode}, {"trial", trial}, {"repetitions", repetitions}, {"provided", provided}, {"first_preparation", firstPreparation}, {"state_generation_unchanged", true}, {"completed_solve_iterations", completed}, {"rss_before_problem", memory[0]}, {"rss_after_work", memory[1]}, {"peak_rss", memory[2]}, {"phases", phases}}.dump() << std::endl;
        }
        // Allocate the other producer only after memory/timing observations.
        // Compare complete actions and SGS outputs, not just diagonal blocks.
        auto reference = fv.BuildLinearProblem(state, C::DiagonalStorage::Entries,
                                               mode == "assembled" ? C::OffDiagonalStorage::SemiMatrixFree : C::OffDiagonalStorage::Assembled);
        fv.PrepareJacobian(state, auxiliary, inverseTau, *reference, {1000, 1, false});
        reference->InvertDiagonal();
        auto expected = metrics.BuildField(5), referenceForward = metrics.BuildField(5);
        const auto equal = [&](const auto &a, const auto &b)
        {
            const auto x = a.Owned().Read(), y = b.Owned().Read();
            const auto u = x.template View<R::UniformTableRows>(), v = y.template View<R::UniformTableRows>();
            bool valid = true;
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    valid = valid && std::isfinite(u(iCell, j)) && std::isfinite(v(iCell, j)) &&
                            std::abs(u(iCell, j) - v(iCell, j)) <= 2e-12 * std::max({1., std::abs(u(iCell, j)), std::abs(v(iCell, j))});
            int all = valid;
            MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
            DNDS_check_throw_info(all, "assembled/evaluated full-field numerical oracle failed");
        };
        problem->Apply(state, action);
        reference->Apply(state, expected);
        equal(action, expected);
        pair();
        reference->ApplySGS(rhs, state, referenceForward, expected);
        equal(update, expected);
        if (mpi.rank == 0)
            std::cout << "PLAN53_JACOBIAN_VERIFIED " << J{{"action", true}, {"sgs", true}, {"state_generation_unchanged", state.Owned().Version() == stateVersion}}.dump() << std::endl;
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
        std::cerr << "Jacobian assessment: " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    DNDS::MPI::Finalize();
    return 0;
}
