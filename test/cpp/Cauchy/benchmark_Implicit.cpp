// Frozen-state CPU phase timings, separate from nonlinear convergence studies.
// Run on one allocated node, OMP1, with enough cells per MPI rank. Numerical
// oracle tests live in test_LinearProblem/test_Jacobian/test_FiniteVolume.
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "ControllerPhaseBenchmark.hpp"
#include "PolynomialMultigridBenchmark.hpp"
#include <iostream>

namespace C = DNDS::Cauchy;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(argc >= 7 && argc <= 9, "arguments: unit-square-mesh repetitions trials check-access overlap semi-matrix-free [extra-controller-phases] [polynomial-multigrid-only]");
        const int repetitions = std::stoi(argv[2]), trials = std::stoi(argv[3]);
        const bool checked = std::stoi(argv[4]), overlap = std::stoi(argv[5]), evaluated = std::stoi(argv[6]);
        const bool extra = argc >= 8 && std::stoi(argv[7]);
        const bool polynomial = argc == 9 && std::stoi(argv[8]);
        DNDS_check_throw_info(!polynomial || (!extra && !evaluated), "pMG cost profile uses PP-off assembled matrices; other phase modes remain separate");
        DNDS_check_throw_info(repetitions > 0 && trials > 0, "positive benchmark repetitions/trials required");
        const auto measure = [&](auto &&action, int count = 1)
        {
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            for (int i = 0; i < count; ++i)
                action();
            const double elapsed = (MPI_Wtime() - start) / count;
            double maximum = 0;
            MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            return maximum;
        };
        G::PartitionOptions partition;
        partition.metisSeed = 42;
        G::AutoAppendName2ID names;
        G::NativePeriodicImportOptions periodic;
        periodic.frames.translation[1].map() = G::tPoint{1, 0, 0};
        periodic.frames.translation[2].map() = G::tPoint{0, 1, 0};
        DNDS::ssp<G::UnstructuredMesh> mesh;
        const double import = measure([&]
                                      { mesh = G::ReadNativeMeshFromCGNS(mpi, 2, argv[1], partition, [&](const std::string &name)
                                                                         { return names(name); }, periodic); });
        const double topology = measure([&]
                                        { mesh->PrepareNativeTopology(); });
        C::Json input{{"defaults", {{"mesh", {{"dimension", 2}}}, {"equation", {{"transport", "NewtonianFourier"}, {"viscosity", .001}}}, {"numerics", {{"reconstruction", "variational"}, {"flux", "Roe"}, {"limiter", "none"}, {"faceFallback", false}, {"positivity", false}, {"viscousGradient", "polynomial"}, {"reconstructionIterations", 5}, {"reconstructionTolerance", 0}, {"cfv", {{"maxOrder", 3}, {"intOrder", 5}, {"SORInstead", false}, {"jacobiRelax", 1}, {"HQMWeights", true}}}}}, {"execution", {{"checkAccess", checked}, {"overlap", overlap}}}}}};
        // The timing switch controls both optional scheduling checks and the
        // composed operator's extra failure agreements, including Runtime halo
        // preflight in current code. Historical report pins kept that preflight on.
        input["defaults"]["implicit"]["collectiveChecks"] = checked;
        if (extra)
            input["defaults"]["numerics"]["positivity"] = true;
        // Uniform-square fixtures can have physical rather than periodic
        // labels. Declare every actual physical zone, identically on all ranks;
        // this metadata agreement is outside the measured numerical batches.
        std::vector<int> zones;
        {
            auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            for (I iCell = 0; iCell < view.NumCell(); ++iCell)
            {
                const auto faces = view.CellFaces(iCell);
                for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
                    if (view.CellFaceOther(iCell, faces[slot], slot) == DNDS::UnInitIndex)
                        zones.push_back(view.GetFaceZone(faces[slot]));
            }
        }
        std::sort(zones.begin(), zones.end());
        zones.erase(std::unique(zones.begin(), zones.end()), zones.end());
        const int zoneCount = int(zones.size());
        std::vector<int> counts(mpi.size), displacements(mpi.size, 0);
        MPI_Allgather(&zoneCount, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi.comm);
        for (int rank = 1; rank < mpi.size; ++rank)
            displacements[rank] = displacements[rank - 1] + counts[rank - 1];
        std::vector<int> globalZones(displacements.back() + counts.back());
        MPI_Allgatherv(zones.data(), zoneCount, MPI_INT, globalZones.data(), counts.data(), displacements.data(), MPI_INT, mpi.comm);
        for (auto zone : globalZones)
            input["defaults"]["boundaries"][std::to_string(zone)] = {{"kind", "farfield"}, {"primitive", {1, 1, 0, 0, 1}}};
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        const auto &settings = config.Domains().front().settings;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        std::optional<C::FiniteVolume> finiteVolume;
        const double preparation = measure([&]
                                           { finiteVolume.emplace(mesh, settings, equation); });
        auto &fv = *finiteVolume;
        const auto &metrics = fv.Metrics();
        auto state = metrics.BuildField(5), residual = metrics.BuildField(5), rawResidual = metrics.BuildField(5);
        auto action = metrics.BuildField(5), forward = metrics.BuildField(5), update = metrics.BuildField(5);
        const I owned = metrics.NumCell();
        I global = 0;
        MPI_Allreduce(&owned, &global, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
        {
            auto geometry = metrics.AcquireRead();
            const auto view = geometry.HostView();
            auto write = state.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            const double pi = std::acos(-1.);
            for (I iCell = 0; iCell < owned; ++iCell)
            {
                const auto point = view.GetCellBary(iCell);
                const C::FlowState primitive{1 + .01 * std::sin(2 * pi * point[0]) * std::cos(2 * pi * point[1]),
                                             1 + .01 * std::cos(2 * pi * point[1]), .01 * std::sin(2 * pi * point[0]), 0, 1};
                const auto value = equation.Gas().Conservative(primitive);
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = value[j];
            }
        }
        auto logicalState = C::EquationDofs::Coupled(equation.Schema(), state);
        auto logicalResidual = C::EquationDofs::Coupled(equation.Schema(), residual);
        auto logicalAction = C::EquationDofs::Coupled(equation.Schema(), action);
        auto logicalForward = C::EquationDofs::Coupled(equation.Schema(), forward);
        auto logicalUpdate = C::EquationDofs::Coupled(equation.Schema(), update);
        const auto ownedVersion = state.Owned().Version();
        C::AuxiliaryRequest request;
        request.meanThermodynamics = request.faceSpectralRadius = request.cellSpectralSum = true;
        auto workspace = fv.BuildWorkspace(logicalState);
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        R::EagerEngine engine(0, checked ? std::make_shared<R::AccessTrackingContext>() : nullptr);
        const auto stream = engine.CreateCPUStream();
        auto halo = metrics.BindHalo(state);
        R::Array<double> inversePseudoTime(R::RowLayout::Dense(owned, 1));
        {
            auto write = inversePseudoTime.Write();
            for (I iCell = 0; iCell < owned; ++iCell)
                write.View()(iCell, 0) = 1;
        }
        auto problem = fv.BuildLinearProblem(logicalState, C::DiagonalStorage::Entries,
                                             evaluated ? C::OffDiagonalStorage::SemiMatrixFree : C::OffDiagonalStorage::Assembled);
        C::LinearSolveSettings linear;
        linear.maxIterations = 5;
        linear.relativeTolerance = linear.absoluteTolerance = 0;
        C::EquationLinearSolver solver(problem, logicalState, [&](auto &field)
                                       { return metrics.BindHalo(field); }, linear, {overlap, checked});
        for (int trial = 0; trial < trials; ++trial)
        {
            if (polynomial)
            {
                auto variants = CauchyBenchmark::PolynomialPhases(mesh, fv, state, engine, stream, settings, checked, repetitions, measure);
                DNDS_check_throw_info(state.Owned().Version() == ownedVersion, "pMG cost profile changed its reference means");
                if (mpi.rank == 0)
                    std::cout << "CAUCHY_PMG_BENCH " << C::Json{{"cells", global}, {"np", mpi.size}, {"degree", 3}, {"trial", trial}, {"repetitions", repetitions}, {"checked", checked}, {"overlap", overlap}, {"import", import}, {"topology", topology}, {"fv_prepare", preparation}, {"mass", 1e5}, {"positivity", false}, {"owned_generation_preserved", true}, {"variants", std::move(variants)}}.dump() << std::endl;
                continue;
            }
            fv.Evaluate(logicalState, logicalResidual, *workspace, engine, stream);
            const double evaluate = measure([&]
                                            { fv.Evaluate(logicalState, logicalResidual, *workspace, engine, stream); }, repetitions);
            const double auxiliaryTime = measure([&]
                                                 { fv.PrepareAuxValues(logicalState, auxiliary); }, repetitions);
            fv.ValidateAuxValues(logicalState, auxiliary);
            double rawRHS = 0;
            {
                const auto rhs = fv.AcquireRHSOperator();
                const auto u = state.Owned().Read(), ug = state.Ghost().Read();
                const auto &coefficients = fv.FluxReconstruction(*workspace).CoupledField();
                const auto r = coefficients.Owned().Read(), rg = coefficients.Ghost().Read();
                const auto thermo = auxiliary.meanThermodynamics->Read();
                auto output = rawResidual.Owned().Write();
                C::RHSReadViews reads;
                reads.mean = {u.View<R::UniformTableRows>(), ug.View<R::UniformTableRows>()};
                reads.coefficients = {r.View<R::UniformTableRows>(), rg.View<R::UniformTableRows>()};
                reads.auxiliary.meanThermodynamics = thermo.View<R::UniformTableRows>();
                C::RHSLocalDiagnostics diagnostics;
                C::RHSWriteViews writes;
                writes.residual = output.View<R::UniformTableRows>();
                rawRHS = measure([&]
                                 { diagnostics.Merge(rhs.Apply(reads, writes)); }, repetitions);
                DNDS_check_throw_info(diagnostics.FaceFallbacks() == 0, "benchmark encountered an inadmissible reconstructed face");
            }
            double difference = 0;
            {
                const auto checkedRHS = residual.Owned().Read(), raw = rawResidual.Owned().Read();
                const auto a = checkedRHS.View<R::UniformTableRows>(), b = raw.View<R::UniformTableRows>();
                for (I iCell = 0; iCell < owned; ++iCell)
                    for (int j = 0; j < 5; ++j)
                    {
                        DNDS_check_throw_info(std::isfinite(a(iCell, j)) && std::isfinite(b(iCell, j)), "nonfinite benchmark residual");
                        difference = std::max(difference, std::abs(a(iCell, j) - b(iCell, j)));
                    }
            }
            MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(difference <= 1e-12, "raw RHS differs from its composed Evaluate result");
            const double jacobian = measure([&]
                                            { fv.PrepareJacobian(logicalState, auxiliary, inversePseudoTime, *problem, {1000, 1, false}); }, repetitions);
            // InvertDiagonal is lazy: time its first real inversion after the
            // last Jacobian fill, rather than averaging cached no-op calls.
            const double inverse = measure([&]
                                           { problem->InvertDiagonal(); });
            const double symbolic = measure([&]
                                            { problem->PrepareILUStructure(0, true); });
            const double factor = measure([&]
                                          { problem->FactorizeILU(); }, repetitions);
            const double matvec = measure([&]
                                          { problem->Apply(logicalState, logicalAction); }, repetitions);
            const double sgs = measure([&]
                                       { problem->ApplySGS(logicalResidual, logicalState, logicalForward, logicalUpdate); }, repetitions);
            const double ilu = measure([&]
                                       { problem->ApplyILU(logicalResidual, logicalForward, logicalUpdate); }, repetitions);
            I linearIterations = 0;
            const double solve = measure([&]
                                         { linearIterations += solver.Solve(logicalResidual).iterations; }, repetitions);
            DNDS_check_throw_info(linearIterations == I(5) * repetitions, "unexpected standalone SGS work count");
            const double transfer = measure([&]
                                            { halo.Begin().Wait(); }, repetitions);
            C::Json controllers;
            if (extra)
                controllers = CauchyBenchmark::ControllerPhases(fv, logicalState, *workspace, engine, stream, settings, checked, repetitions, measure);
            DNDS_check_throw_info(state.Owned().Version() == ownedVersion, "benchmark changed its frozen means");
            DNDS_check_throw_info(std::isfinite(A::NormGlobal(update)), "nonfinite local factor solve");
            if (mpi.rank == 0)
            {
                C::Json report{{"cells", global}, {"np", mpi.size}, {"degree", 3}, {"trial", trial}, {"repetitions", repetitions}, {"checked", checked}, {"overlap", overlap}, {"semi_matrix_free", evaluated}, {"import", import}, {"topology", topology}, {"fv_prepare", preparation}, {"evaluate", evaluate}, {"auxiliary", auxiliaryTime}, {"raw_rhs", rawRHS}, {"jacobian", jacobian}, {"inverse", inverse}, {"ilu_symbolic", symbolic}, {"ilu_factor", factor}, {"matrix_action", matvec}, {"sgs_pair", sgs}, {"ilu_solve", ilu}, {"sgs5_orchestration", solve}, {"mean_halo", transfer}, {"rhs_max_difference", difference}, {"linear_iterations_per_call", 5}, {"owned_generation_preserved", true}};
                report["extra_controller_phases"] = extra;
                if (extra)
                    report["controllers"] = std::move(controllers);
                std::cout << "CAUCHY_PHASE_BENCH " << report.dump() << std::endl;
            }
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Cauchy phase benchmark rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    MPI_Finalize();
}
