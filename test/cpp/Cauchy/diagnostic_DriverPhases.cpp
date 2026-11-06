// Broad production-process diagnosis on a real SA mesh. Internal timer counters
// are neither read nor used to attribute cost. See OperatorDiagnostic.md.
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Physics/StateInput.hpp"
#include "Euler/EulerSolver.hpp"
#include <functional>
#include <iostream>
#include <numeric>
#include "DiagnosticTiming.hpp"

namespace DNDS::Euler::DriverDiagnostic
{
    struct Actions
    {
        std::function<void()> reset, spatial, auxiliary, implicit, composed;
        ArrayDOFV<6> *residual = nullptr;
        ArrayDOFV<1> *pseudoStep = nullptr;
    };
    inline std::function<void(Actions &)> Run;
    inline int levels = 0, corrections = 1;
}
#include "ExtractedEulerDriver.hpp"
// Supplies the production workspace definition for diagnostic-only wall-input
// alignment, without changing the production wall-distance providers.
#include "ExtractedEquationPipeline.cpp"

namespace C = DNDS::Cauchy;
namespace E = DNDS::Euler;
namespace R = DNDS::Runtime;
using I = DNDS::index;

static void RunBatch(E::EulerSolver<E::NS_SA> &solver, C::DomainSettings settings,
                     E::DriverDiagnostic::Actions &legacy, int repetitions, int trials)
{
    const int levels = E::DriverDiagnostic::levels, corrections = E::DriverDiagnostic::corrections;
    auto mesh = solver.getMesh();
    const auto mpi = mesh->getMPI();
    I counts[]{mesh->NumCell(), mesh->NumCell(), mesh->NumCell()};
    MPI_Allreduce(MPI_IN_PLACE, counts, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, counts + 1, 1, MPI_INT64_T, MPI_MIN, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, counts + 2, 1, MPI_INT64_T, MPI_MAX, mpi.comm);
    DNDS_check_throw_info(counts[0] >= 50000 && counts[1] >= 1000,
                          "broad diagnosis requires >=50000 real SA cells and >=1000 owned cells/rank");
    const auto &oldConfig = solver.getConfiguration();
    DNDS_check_throw_info(!settings.execution.overlap && settings.execution.hostThreads == 0,
                          "broad timing partition currently requires synchronous CPU-inline execution");
    const auto &linear = oldConfig.linearSolverControl;
    DNDS_check_throw_info((linear.jacobiCode == 1 || linear.jacobiCode == 2) && (linear.gmresCode == 0 || linear.gmresCode == 1) &&
                              oldConfig.implicitReconstructionControl.nInternalRecStep == 1 &&
                              !oldConfig.limiterControl.useLimiter && oldConfig.vfvSettings.maxOrder == 3,
                          "broad batch requires a maintained P3 one-VR-sweep SGS/ILU/GMRES case");
    const auto &implicit = settings.implicit;
    // The inputs own every policy. Validate correspondence instead of silently
    // rewriting the native configuration from Euler defaults in this harness.
    DNDS_check_throw_info(implicit.pseudoCFL == oldConfig.implicitCFLControl.CFL && implicit.maxPseudoCFL == implicit.pseudoCFL &&
                              implicit.diagonal == (solver.getEval().settings.useScalarJacobian ? "entries" : "block"),
                          "explicit pseudo-time/diagonal controls differ");
    DNDS_check_throw_info(implicit.ilu.fill == linear.directPrecControl.iluCode && implicit.ilu.ordering == "rcm" &&
                              !implicit.ilu.coreFirst && linear.directPrecControl.orderingCode == 4,
                          "explicit factor controls differ");
    const auto verifyLinear = [&](const auto &old, const auto &actual)
    {
        const auto method = old.gmresCode ? "gmres" : old.jacobiCode == 1 ? "sgs"
                                                                          : "ilu";
        DNDS_check_throw_info(actual.linear.method == method && actual.sgs.exchangeBetweenSweeps == (old.jacobiCode == 1), "explicit linear/middle-exchange controls differ");
        if (old.gmresCode)
        {
            DNDS_check_throw_info(old.gmresCode == 1 && old.jacobiCode == 2 &&
                                      actual.linear.preconditioner == "ilu" && actual.ilu.preconditionerCorrections == 2 * (old.sgsIter + 1) &&
                                      actual.linear.iterations == linear.nGmresSpace * old.nGmresIter && actual.linear.subspace == linear.nGmresSpace &&
                                      actual.linear.relativeTolerance == 0 && actual.linear.absoluteTolerance == 0 && actual.linear.stopping == "residual" &&
                                      (old.gmresScale == 0 || old.gmresScale == 2) && actual.linear.componentScaling == (old.gmresScale == 2 ? "meanState" : "none"),
                                  "explicit GMRES budgets/preconditioner/metric differ");
        }
        else
            DNDS_check_throw_info(actual.linear.stopping == "fixedCount" && actual.linear.componentScaling == "none" &&
                                      actual.linear.iterations == (old.jacobiCode == 1 ? 1 : 2) * (old.sgsIter + 1),
                                  "explicit stationary correction counts differ");
    };
    verifyLinear(linear, implicit);
    const auto &pp = oldConfig.timeMarchControl;
    DNDS_check_throw_info(implicit.positivity.rhsFPPMode == (pp.rhsFPPMode == 0 ? "none" : pp.rhsFPPMode == 2 ? "dTau"
                                                                                                              : "rhs") &&
                              implicit.positivity.rhsFPPRelax == pp.rhsFPPRelax && implicit.positivity.rhsFPPScale == pp.rhsFPPScale &&
                              implicit.positivity.incrementRelax == pp.incrementPPRelax && settings.numerics.reconstructionIterations == 1 &&
                              settings.numerics.positivity == oldConfig.limiterControl.usePPRecLimiter,
                          "explicit PP/VR controls differ");
    const bool nonlinearController = levels || corrections > 1;
    if (nonlinearController)
    {
        DNDS_check_throw_info(implicit.iterations == corrections && implicit.multigrid.cycles == corrections &&
                                  implicit.multigrid.levels == levels && implicit.multigrid.preSmooth == 1 && implicit.multigrid.postSmooth == 0 &&
                                  implicit.multigrid.coarseSmooth == 4 && implicit.relativeTolerance == oldConfig.convergenceControl.rhsThresholdInternal &&
                                  implicit.normOrder == oldConfig.convergenceControl.normOrd && implicit.residualComponents == std::vector<std::string>{"rho"},
                              "explicit nonlinear/hierarchy controls differ");
        for (int level = 1; level <= levels; ++level)
        {
            const auto &coarse = linear.coarseGridLinearSolverControlList.at(std::to_string(level));
            DNDS_check_throw_info((coarse.jacobiCode == 1 || coarse.jacobiCode == 2) && coarse.gmresCode == 0,
                                  "coarse diagnosis requires stationary SGS or ILU");
            verifyLinear(coarse, implicit.ResolveCoarseLevel(level));
        }
    }
    // The borrowed Euler mesh has its evaluator's distance samples, but the
    // native FV prerequisite is the mesh-owned nodal provider. Domain setup
    // normally builds this before constructing its equation operator.
    if (!mesh->HasNodeWallDist())
        mesh->BuildNodeWallDist([&](auto zone)
                                {
            const auto found = settings.boundaries.find(std::to_string(zone));
            return found != settings.boundaries.end() && found->second.kind == "noSlipWall"; });
    // Keep the selected production overlap/check policy from the native input.
    // This driver does not replace the eager engine or bypass native wrappers.
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(6));
    auto initial = C::EquationDofOperations::NewVector(state);
    auto source = C::EquationDofOperations::NewVector(state);
    C::EquationDofOperations::Fill(source, 0.);
    {
        auto write = state.WriteOwned();
        const auto layout = write.ComponentLayout();
        const auto values = layout.View();
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (int j = 0; j < 6; ++j)
                values(iCell, j) = solver.getU()[iCell][j];
    }
    C::EquationDofOperations::Copy(initial, state);
    auto work = fv.BuildEquationWorkspace(state);
    double wallDifference = 0;
    {
        const auto read = work->owner->wallDistance->AcquireRead();
        const auto distances = read.HostView();
        auto &oldEval = solver.getEval();
        for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
        {
            const auto samples = distances.cells[iCell];
            DNDS_check_throw_info(I(samples.size()) == oldEval.dWall[iCell].size(), "wall quadrature shape differs");
            for (I iPoint = 0; iPoint < I(samples.size()); ++iPoint)
            {
                wallDifference = std::max(wallDifference, std::abs(samples[iPoint] - oldEval.dWall[iCell][iPoint]));
                if (!levels)
                    oldEval.dWall[iCell][iPoint] = samples[iPoint];
            }
        }
        for (I iFace = 0; iFace < mesh->NumFaceProc(); ++iFace)
        {
            double sum = 0;
            for (auto value : distances.faces[iFace])
                sum += value;
            if (!levels)
                oldEval.dWallFace[iFace] = sum / double(distances.faces[iFace].size());
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &wallDifference, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    R::EagerEngine engine;
    auto stream = engine.CreateCPUStream();
    C::EquationNonlinearSolver nonlinear(fv, state, work, engine, stream, settings,
                                         DNDS::Algebra::ComponentSelection(6, {0}));
    std::unique_ptr<C::PolynomialMultigrid> multigrid;
    if (levels)
        multigrid = std::make_unique<C::PolynomialMultigrid>(mesh, fv, state, nonlinear, engine, stream, settings);
    const auto reset = [&]
    {
        C::EquationDofOperations::Copy(state, initial);
        fv.ResetReconstruction(*work);
    };
    const auto composed = [&]
    {
        if (multigrid)
        {
            const auto report = multigrid->Solve(source, 0., 1., 0.);
            DNDS_check_throw_info(report.iterations == corrections, "PMG diagnostic did not perform its requested update count");
        }
        else if (nonlinearController)
        {
            const auto report = nonlinear.Solve(state, source, 0., 1., 0.);
            DNDS_check_throw_info(report.iterations == corrections, "diagnostic did not perform its requested update count");
        }
        else
            nonlinear.CorrectOnce(state, source, 0., 1., 0.);
    };
    auto rate = C::EquationDofOperations::NewVector(state);
    const auto compare = [&](bool spatialOutput)
    {
        const auto read = spatialOutput ? rate.ReadOwned() : state.ReadOwned();
        const auto layout = read.ComponentLayout();
        const auto values = layout.View();
        const auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        double sums[18]{};
        for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            for (int j = 0; j < 6; ++j)
            {
                const auto actual = values(iCell, j);
                const auto expected = spatialOutput ? (*legacy.residual)[iCell][j] : solver.getU()[iCell][j];
                DNDS_check_throw_info(std::isfinite(actual) && std::isfinite(expected), "nonfinite broad phase result");
                sums[j] += actual * actual;
                sums[6 + j] += expected * expected;
                sums[12 + j] += DNDS::sqr(actual - expected);
            }
        MPI_Allreduce(MPI_IN_PLACE, sums, 18, MPI_DOUBLE, MPI_SUM, mpi.comm);
        C::Json relative = C::Json::array(), nativeNorm = C::Json::array(), oldNorm = C::Json::array();
        for (int j = 0; j < 6; ++j)
        {
            relative.push_back(std::sqrt(sums[12 + j] / std::max(sums[6 + j], 1e-300)));
            nativeNorm.push_back(std::sqrt(sums[j]));
            oldNorm.push_back(std::sqrt(sums[6 + j]));
        }
        return C::Json{{"relative_l2_difference", relative}, {"native_l2", nativeNorm}, {"Euler_l2", oldNorm}};
    };
    C::Json resolved = settings;
    if (!mpi.rank)
        std::cout << "BROAD_WORKLOAD " << C::Json{{"cells", counts[0]}, {"np", mpi.size}, {"minimum_owned_cells", counts[1]}, {"maximum_owned_cells", counts[2]}, {"wall_distance_input", levels ? "actual-production-providers" : "frozen-native-mesh-samples"}, {"raw_wall_provider_difference", wallDifference}, {"native_resolved_settings", resolved}, {"euler_resolved_settings", C::Json(oldConfig)}, {"reconstruction_start", "zero each repetition"}, {"linear_corrections", settings.implicit.linear.iterations}, {"pMG_levels", levels}, {"nonlinear_corrections", corrections}, {"nonlinear_stopping_norm_in_scope", nonlinearController}}.dump() << std::endl;
    // Independent untimed spatial check before any state correction. This
    // comparison normalizes through the actual production EvaluateRate block.
    reset();
    nonlinear.EvaluateRate(state, rate, 0.);
    legacy.reset();
    legacy.auxiliary();
    legacy.spatial();
    const auto spatialOracle = compare(true);
    if (!levels)
        for (const auto &difference : spatialOracle["relative_l2_difference"])
            DNDS_check_throw_info(difference.get<DNDS::real>() < 1e-9, "production spatial residuals disagree");
    if (!mpi.rank)
        std::cout << "BROAD_SPATIAL_ORACLE " << spatialOracle.dump() << std::endl;

    reset();
    composed();
    legacy.reset();
    legacy.composed();
    const auto baseline = compare(false);
    const auto measure = [&](bool native)
    {
        std::array<double, 8> total{};
        std::array<double, 33> perLevel{};
        constexpr std::array<const char *, DiagnosticTiming::Sample::detailNumber> detailNames{
            "mean-validation", "defect-assembly", "pseudo-time-pp", "correction-local",
            "candidate-acceptance", "correction-statistics", "gradient", "reconstruction",
            "polynomial-pp", "face-rhs", "source-sampling", "volume-source",
            "candidate-validation", "candidate-agreement"};
        std::array<double, DiagnosticTiming::Sample::detailNumber * 3> details{};
        for (int repeat = 0; repeat < repetitions; ++repeat)
        {
            if (native)
                reset();
            else
                legacy.reset();
            MPI_Barrier(mpi.comm);
            DiagnosticTiming::Sample observation;
            DiagnosticTiming::active = &observation;
            const double start = MPI_Wtime();
            if (native)
                composed();
            else
                legacy.composed();
            const double elapsed = MPI_Wtime() - start;
            DiagnosticTiming::active = nullptr;
            DNDS_check_throw_info(observation.spatialCalls >= 1 && observation.auxiliaryCalls > 0,
                                  "broad production scopes did not observe one complete pipeline");
            const double spatial = observation.spatialInclusive - (native ? observation.auxiliary : 0.);
            const double correction = elapsed - observation.auxiliary - spatial;
            DNDS_check_throw_info(spatial >= 0 && correction >= 0, "overlapping/invalid synchronous timing scopes");
            total[0] += observation.auxiliary;
            total[1] += spatial;
            total[2] += correction;
            total[3] += elapsed;
            const double matrixLinear = std::accumulate(observation.levelMatrixLinear.begin(), observation.levelMatrixLinear.end(), 0.);
            total[4] += matrixLinear;
            total[5] += correction - matrixLinear;
            total[6] += std::accumulate(observation.levelMatrix.begin(), observation.levelMatrix.end(), 0.);
            total[7] += std::accumulate(observation.levelLinear.begin(), observation.levelLinear.end(), 0.);
            DNDS_check_throw_info(matrixLinear >= 0 && correction + 1e-9 >= matrixLinear, "matrix/linear scopes exceed implicit remainder");
            for (int level = 0; level < 3; ++level)
            {
                for (int detail = 0; detail < DiagnosticTiming::Sample::detailNumber; ++detail)
                    details[3 * detail + level] += observation.details[detail][level];
                perLevel[2 * level] += observation.levelAuxiliary[level];
                perLevel[2 * level + 1] += observation.levelSpatialInclusive[level] -
                                           (native ? observation.levelAuxiliary[level] : 0.);
                perLevel[6 + 2 * level] += observation.levelAuxiliaryCalls[level];
                perLevel[7 + 2 * level] += observation.levelSpatialCalls[level];
                perLevel[12 + level] += observation.levelMatrixLinear[level];
                perLevel[15 + level] += observation.levelMatrixLinearCalls[level];
                perLevel[18 + level] += observation.levelMatrix[level];
                perLevel[21 + level] += observation.levelLinear[level];
                perLevel[24 + level] += observation.levelAuxiliaryPrepare[level];
                perLevel[27 + level] += observation.levelAuxiliaryExchange[level];
                perLevel[30 + level] += observation.levelAuxiliaryCompute[level];
            }
        }
        for (auto &value : total)
            value /= repetitions;
        for (auto &value : perLevel)
            value /= repetitions;
        for (auto &value : details)
            value /= repetitions;
        std::vector<double> detailRecords(!mpi.rank ? details.size() * mpi.size : 0);
        MPI_Gather(details.data(), int(details.size()), MPI_DOUBLE, detailRecords.data(), int(details.size()), MPI_DOUBLE, 0, mpi.comm);
        if (!mpi.rank)
            std::cout << "BROAD_DETAIL_TIMES " << C::Json{{"solver", native ? "Cauchy" : "Euler"}, {"instrumented", native}, {"scope_names", detailNames}, {"levels", {"P3", "P1", "P0"}}, {"rank_rows", detailRecords}, {"nested_scopes", {{"mean-validation", "candidate-validation"}, {"candidate-validation", "candidate-acceptance"}, {"candidate-agreement", "candidate-acceptance"}, {"source-sampling", "volume-source"}}}, {"timing", "external-same-production-calls"}}.dump() << std::endl;
        std::array<double, 37> rankRecord{};
        std::copy_n(total.begin(), 4, rankRecord.begin());
        std::copy(perLevel.begin(), perLevel.end(), rankRecord.begin() + 4);
        std::vector<double> rankRecords(!mpi.rank ? 37 * mpi.size : 0);
        MPI_Gather(rankRecord.data(), 37, MPI_DOUBLE, rankRecords.data(), 37, MPI_DOUBLE, 0, mpi.comm);
        if (!mpi.rank)
            std::cout << "BROAD_RANK_TIMES " << C::Json{{"solver", native ? "Cauchy" : "Euler"}, {"columns", {"aux", "reconstructionRHS", "implicit", "total", "P3aux", "P3RHS", "P1aux", "P1RHS", "P0aux", "P0RHS", "P3auxScopes", "P3RHScalls", "P1auxScopes", "P1RHScalls", "P0auxScopes", "P0RHScalls", "P3matrixLinear", "P1matrixLinear", "P0matrixLinear", "P3matrixLinearCalls", "P1matrixLinearCalls", "P0matrixLinearCalls", "P3matrix", "P1matrix", "P0matrix", "P3linear", "P1linear", "P0linear", "P3auxPrepare", "P1auxPrepare", "P0auxPrepare", "P3auxExchange", "P1auxExchange", "P0auxExchange", "P3auxCompute", "P1auxCompute", "P0auxCompute"}}, {"rank_rows", rankRecords}}.dump() << std::endl;
        // These are maxima of local sums. The critical rank can differ across
        // blocks, so do not add reduced blocks to claim a composed interval.
        MPI_Allreduce(MPI_IN_PLACE, total.data(), 8, MPI_DOUBLE, MPI_MAX, mpi.comm);
        return total;
    };
    for (int trial = 0; trial < trials; ++trial)
    {
        std::array<double, 8> nativeSeconds{}, oldSeconds{};
        if (trial % 2)
            oldSeconds = measure(false), nativeSeconds = measure(true);
        else
            nativeSeconds = measure(true), oldSeconds = measure(false);
        const auto outputs = compare(false);
        DNDS_check_throw_info(outputs == baseline, "broad frozen-repeat output changed");
        const std::array<const char *, 8> phases{"auxiliary", "reconstruction-rhs", "implicit-correction", "composed-correction", "matrix-linear", "implicit-other", "matrix", "linear"};
        for (int iPhase = 0; iPhase < 8; ++iPhase)
            if (!mpi.rank)
                std::cout << "BROAD_OPERATOR " << C::Json{{"phase", phases[iPhase]}, {"trial", trial}, {"repetitions", repetitions}, {"native_seconds", nativeSeconds[iPhase]}, {"euler_seconds", oldSeconds[iPhase]}, {"outputs", outputs}, {"timing", "external-whole-call-max-rank"}, {"preserves_production_call_sequence", true}, {"Euler_ODE_arithmetic", "production-BDF1-Step"}, {"pMG_levels", levels}, {"nonlinear_corrections", corrections}, {"nonlinear_stopping_norm_in_scope", nonlinearController}, {"reset_and_prerequisites_timed", false}, {"internal_counters_used", false}}.dump() << std::endl;
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        DNDS_check_throw_info(argc == 5 || argc == 7, "arguments: Euler-config Cauchy-config repetitions trials [MG-levels nonlinear-corrections]");
        const int repetitions = std::stoi(argv[3]), trials = std::stoi(argv[4]);
        if (argc == 7)
        {
            E::DriverDiagnostic::levels = std::stoi(argv[5]);
            E::DriverDiagnostic::corrections = std::stoi(argv[6]);
        }
        DNDS_check_throw_info(repetitions > 0 && trials > 0, "positive batch counts required");
        DNDS_check_throw_info(E::DriverDiagnostic::levels >= 0 && E::DriverDiagnostic::levels <= 2 &&
                                  E::DriverDiagnostic::corrections > 0 && E::DriverDiagnostic::corrections <= 32,
                              "diagnostic supports zero/one/two MG levels and 1..32 corrections");
        C::Json input;
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            std::ifstream file(argv[2]);
            DNDS_check_throw_info(bool(file), "cannot open broad native input");
            input = C::Json::parse(file, nullptr, true, true); });
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        DNDS_check_throw_info(config.Domains().size() == 1, "broad diagnosis requires one domain");
        auto settings = config.Domains().front().settings;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        settings = C::ResolvePhysicalStateInputs(settings, equation, std::nullopt).settings;
        DNDS_check_throw_info(equation.HasRANS() && !equation.HasMixture(), "broad diagnosis requires perfect-gas SA");
        E::EulerSolver<E::NS_SA> solver(mpi);
        const std::string defaults = std::string(argv[1]) + ".generated-default.json";
        solver.ConfigureFromJson(defaults, false);
        solver.ConfigureFromJson(defaults, true, argv[1]);
        solver.ReadMeshAndInitialize();
        E::DriverDiagnostic::Run = [&](auto &actions)
        { RunBatch(solver, settings, actions, repetitions, trials); };
        solver.RunImplicitEuler();
        E::DriverDiagnostic::Run = {};
    }
    catch (const std::exception &error)
    {
        std::cerr << "broad diagnostic rank " << mpi.rank << ": " << error.what() << std::endl;
        MPI_Abort(mpi.comm, 1);
    }
    return DNDS::MPI::Finalize();
}
