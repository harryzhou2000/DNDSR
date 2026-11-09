#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "Cauchy/Implicit/NonlinearIteration.hpp"
#include "Cauchy/Implicit/LinearScaling.hpp"
#include "CartesianFixture.hpp"
#include <limits>
#include <cstring>
#include <sstream>
#include "Cauchy/Physics/ImplicitPositivity.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
TEST_CASE("Mean-state Krylov scaling uses volume averages and one common momentum scale")
{
    R::Array<DNDS::real> values(R::RowLayout::Dense(3, 6));
    const std::array<DNDS::real, 6> state{2, 6, 8, 0, 100, 12};
    {
        auto write = values.Write();
        for (DNDS::index iCell = 0; iCell < 3; ++iCell)
            for (DNDS::index j = 0; j < 6; ++j)
                write.View()(iCell, j) = state[j];
    }
    struct Metric
    {
        DNDS::real GetCellVol(DNDS::index iCell) const { return 1 + .25 * iCell; }
    };
    const auto read = values.Read();
    const auto scales = C::NonlinearDetail::MeanStateScales(read.View<R::UniformTableRows>(), Metric{}, 6, 0, mpi.comm);
    const std::vector<DNDS::real> expected{2, 10, 10, 10, 50, 12};
    REQUIRE(scales.size() == expected.size());
    for (DNDS::index j = 0; j < 6; ++j)
        CHECK(scales[j] == doctest::Approx(expected[j]).epsilon(1e-14));
}
TEST_CASE("Shared nonlinear controller observes coupled updates once and advances CFL once")
{
    C::ImplicitSettings settings;
    settings.iterations = 2;
    settings.minIterations = 2;
    settings.pseudoCFL = 1;
    settings.pseudoGrowth = 2;
    settings.maxPseudoCFL = 8;
    settings.absoluteTolerance = .3;
    settings.relativeTolerance = 0;
    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
    std::vector<int> visits;
    std::vector<double> cfl;
    double value = 1;
    const auto report = C::NonlinearDetail::Iterate(settings, schema, mpi.rank, 0, {}, C::NonlinearIterationReport{}, 0, [&]
                                                    { visits.push_back(2); }, [&](auto &norms)
                                                    { norms[0] = value; visits.push_back(3); }, []
                                                    { return std::make_pair(0, 0.); }, [&](auto &current)
                                                    {
            cfl.push_back(current.pseudoCFL);
            visits.push_back(0); // Intermediate correction + refresh.
            visits.push_back(1); // Endpoint correction + refresh.
            value *= .5; }, C::NonlinearDetail::ResidualObservation::AfterUpdate);
    CHECK(report.converged);
    CHECK(report.iterations == 2);
    CHECK(report.initialResidual[0] == .5);
    CHECK(report.residual[0] == .25);
    CHECK(visits == std::vector<int>{0, 1, 2, 3, 0, 1, 2, 3});
    CHECK(cfl == std::vector<double>{1, 2});
    CHECK(report.pseudoCFL == 2);
}
TEST_CASE("Cauchy nonlinear progress labels iterations and stage one-based")
{
    C::ImplicitSettings settings;
    settings.iterations = 2;
    settings.minIterations = 2;
    settings.consoleInterval = 1;
    settings.relativeTolerance = 1;
    settings.absoluteTolerance = 0;
    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
    std::ostringstream captured;
    auto *const previous = std::cout.rdbuf(captured.rdbuf());
    const auto report = C::NonlinearDetail::Iterate(settings, schema, mpi.rank, 0, {}, C::NonlinearIterationReport{}, 0, [] {}, [](auto &norms)
                                                    { norms[0] = 1; }, []
                                                    { return std::make_pair(0, 0.); }, [](auto &) {});
    std::cout.rdbuf(previous);
    CHECK(report.converged);
    if (mpi.rank == 0)
    {
        CHECK(captured.str().find("Cauchy nonlinear stage 1 iteration 1") != std::string::npos);
        CHECK(captured.str().find("Cauchy nonlinear stage 1 iteration 2") != std::string::npos);
        CHECK(captured.str().find("iteration 0") == std::string::npos);
    }
}
TEST_CASE("Implicit convergence retains initial and running references without hiding unselected components")
{
    CHECK(C::ImplicitSettings{}.residualReference == "runningMaximum");
    const auto schema = C::EquationSystem::Resolve(C::EquationSettings{}).Schema();
    for (const auto *reference : {"initial", "runningMaximum"})
    {
        C::ImplicitSettings settings;
        settings.relativeTolerance = .1;
        settings.absoluteTolerance = 0;
        settings.residualReference = reference;
        settings.Validate();
        C::ResidualConvergence policy(settings, schema);
        CHECK_FALSE(policy.Accepted());
        policy.Observe({2, 3, 0, 0, 0});
        policy.Observe({4, 1, 0, 0, 0});
        policy.Observe({.3, .2, 0, 0, 0});
        CHECK(policy.Initial() == C::FlowState{2, 3, 0, 0, 0});
        const bool running = settings.residualReference == "runningMaximum";
        CHECK(policy.Reference()[0] == (running ? 4 : 2));
        CHECK(policy.Accepted() == running);
        CHECK(policy.TargetRatio() == doctest::Approx(running ? .75 : 1.5));
        CHECK_THROWS(policy.Observe({0, -1, 0, 0, 0}));
    }
    C::ImplicitSettings exact;
    {
        auto allSettings = exact;
        allSettings.residualComponents.clear();
        auto extended = schema;
        extended.blocks.push_back({"rans", {{"rhoNuTilde", "", ""}}, {}});
        extended.blocks.push_back({"species", {{"rhoY:H2", "", ""}}, {}});
        C::BasicResidualConvergence<std::vector<double>> all(allSettings, extended, std::vector<double>(7, 0));
        all.Observe({0, 0, 0, 0, 0, 0, 1});
        CHECK_FALSE(all.Accepted()); // Empty selects the last added equation too.
        all.Observe({0, 0, 0, 0, 0, 0, 0});
        CHECK(all.Accepted());
        C::BasicResidualConvergence<std::vector<double>> density(exact, extended, std::vector<double>(7, 0));
        density.Observe({0, 0, 0, 0, 0, 0, 1});
        CHECK(density.Accepted()); // Default density stop does not select added equations.
        CHECK(density.Initial().back() == 1);
        allSettings.residualComponents = {"species.rhoY:H2"};
        C::BasicResidualConvergence<std::vector<double>> species(allSettings, extended, std::vector<double>(7, 0));
        species.Observe({0, 0, 0, 0, 0, 0, 1});
        CHECK_FALSE(species.Accepted());
        CHECK(extended.ResolveComponents({"rhoY:H2", "flow.rhou"}) == std::vector<DNDS::index>{6, 1});
    }
    exact.relativeTolerance = exact.absoluteTolerance = 0;
    exact.residualComponents = {"rho"};
    C::ResidualConvergence policy(exact, schema);
    policy.Observe({0, 100, 0, 0, 0});
    CHECK(policy.Accepted());
    CHECK(policy.TargetRatio() == 0);
    CHECK(policy.Initial()[1] == 100);
    policy.Observe({1e-300, 100, 0, 0, 0});
    CHECK_FALSE(policy.Accepted());
    CHECK(std::isfinite(policy.TargetRatio()));
    for (const std::vector<std::string> components : {std::vector<std::string>{"rho", "flow.rho"}, {"missing"}})
    {
        exact.residualComponents = components;
        CHECK_THROWS(schema.ResolveComponents(exact.residualComponents));
    }
    exact.residualComponents = {"rho"};
    exact.normOrder = 3;
    CHECK_THROWS(exact.Validate());
    exact.normOrder = 1;
    exact.residualReference = "unknown";
    CHECK_THROWS(exact.Validate());
}
TEST_CASE("Distributed implicit L1 L2 norms and selected-component stops use the actual manufactured defect")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4, 2);
    constexpr DNDS::real cellVolume = 4. / 16;
    DNDS::real expectedL1 = 0, expectedL2Squared = 0;
    for (DNDS::index iCell = 0; iCell < 16; ++iCell)
    {
        const DNDS::real defect = cellVolume * .1 * (iCell + 1);
        expectedL1 += defect;
        expectedL2Squared += defect * defect / cellVolume;
    }
    for (int order : {1, 2})
        for (bool densityOnly : {false, true})
        {
            C::DomainSettings settings;
            settings.numerics.reconstruction = "constant";
            settings.implicit.minIterations = 0;
            settings.implicit.normOrder = order;
            settings.implicit.relativeTolerance = .1;
            settings.implicit.absoluteTolerance = 1e-12;
            settings.implicit.residualReference = "runningMaximum";
            settings.implicit.residualComponents = densityOnly ? std::vector<std::string>{"rho"} : std::vector<std::string>{};
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fv(mesh, settings, equation);
            const auto layout = order == 1 ? R::RowLayout::ColMajor(0, 5) : R::RowLayout::TiledColMajor(0, 5, 4);
            auto state = fv.Metrics().BuildField(layout), source = fv.Metrics().BuildField(layout);
            fv.Initialize(state, {}, {1, 0, 0, 0, 1});
            const auto mean = equation.Gas().Conservative({1, 0, 0, 0, 1});
            {
                auto write = source.Owned().Write();
                const auto values = write.View<R::UniformTableRows>();
                for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                {
                    for (int j = 0; j < 5; ++j)
                        values(iCell, j) = 20 * mean[j];
                    values(iCell, 4) += .1 * (fv.Metrics().Cells().LocalToGlobal(iCell) + 1);
                }
            }
            R::EagerEngine engine;
            const auto stream = engine.CreateCPUStream();
            C::AuxiliaryRequest request;
            request.faceSpectralRadius = request.cellSpectralSum = true;
            C::NonlinearSolver solver(fv, state, fv.BuildWorkspace(state, request), engine, stream, settings);
            const auto version = state.Owned().Version();
            const auto report = solver.Solve(state, source, 20, 1, 0, 0, {0, 0});
            CHECK(report.iterations == 0);
            CHECK(report.converged == densityOnly);
            CHECK(report.residual[0] == 0);
            CHECK(report.residual[4] == doctest::Approx(order == 1 ? expectedL1 : std::sqrt(expectedL2Squared)).epsilon(1e-12));
            CHECK(report.initialResidual == report.referenceResidual);
            CHECK(report.residualTargetRatio == doctest::Approx(densityOnly ? 0 : 10));
            CHECK(state.Owned().Version() == version);
        }
}
TEST_CASE("One correction reuses the ordinary solve update without a stopping loop")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.implicit.iterations = 1;
    settings.implicit.minIterations = 1;
    settings.implicit.relativeTolerance = settings.implicit.absoluteTolerance = 0;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto reference = fv.Metrics().BuildField(5), single = fv.Metrics().BuildField(5), source = fv.Metrics().BuildField(5);
    fv.Initialize(reference, {}, {1, .1, 0, 0, 1});
    A::Copy(single, reference);
    A::Copy(source, reference);
    A::Scale(source, 20.);
    {
        auto write = source.Owned().Write();
        for (DNDS::index iCell = 0; iCell < source.Owned().RowCount(); ++iCell)
            write.View()(iCell, 0) += .01 * (fv.Metrics().Cells().LocalToGlobal(iCell) + 1);
    }
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    C::AuxiliaryRequest request;
    request.faceSpectralRadius = request.cellSpectralSum = true;
    C::NonlinearSolver full(fv, reference, fv.BuildWorkspace(reference, request), engine, stream, settings);
    C::NonlinearSolver one(fv, single, fv.BuildWorkspace(single, request), engine, stream, settings);
    const auto version = source.Owned().Version();
    const auto solved = full.Solve(reference, source, 20, 1, .1);
    C::NonlinearProfile profile;
    const auto corrected = one.CorrectOnce(single, source, 20, 1, .1, 0, {-1, 0, &profile});
    CHECK(corrected.iterations == 1);
    CHECK_FALSE(corrected.converged);
    CHECK(profile.residualEvaluations == 1);
    CHECK(corrected.linearIterations == solved.linearIterations);
    CHECK(source.Owned().Version() == version);
    const auto a = reference.Owned().Read(), b = single.Owned().Read();
    for (DNDS::index iCell = 0; iCell < reference.Owned().RowCount(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(a.View()(iCell, j) == b.View()(iCell, j));
}
TEST_CASE("Euler alpha reductions agree sparse triggers and reject mismatched policies or statistics")
{
    for (const auto scope : {C::AlphaReduction::Cell, C::AlphaReduction::MinimumLimited, C::AlphaReduction::MinimumAll})
    {
        const auto stats = C::AgreeAlphaStatistics(mpi.rank == 0 ? 2 : 0, mpi.rank == 0 ? .25 : 1, scope, mpi.comm);
        CHECK(stats.limited == 2);
        CHECK(stats.minimum == .25);
        CHECK(C::SelectAlpha(.75, stats.minimum, scope) == (scope == C::AlphaReduction::Cell ? .75 : .25));
        CHECK(C::SelectAlpha(1, stats.minimum, scope) == (scope == C::AlphaReduction::MinimumAll ? .25 : 1));
        const auto inactive = C::AgreeAlphaStatistics(0, 1, scope, mpi.comm);
        CHECK(inactive.limited == 0);
        CHECK(inactive.minimum == 1);
    }
    CHECK_THROWS(C::AgreeAlphaStatistics(mpi.rank == 0 ? -1 : 0, 1, C::AlphaReduction::Cell, mpi.comm));
    CHECK_THROWS(C::AgreeAlphaStatistics(0, mpi.rank == 0 ? .5 : 1, C::AlphaReduction::Cell, mpi.comm));
    CHECK_THROWS(C::AgreeAlphaStatistics(mpi.rank == 0 ? 1 : 0, mpi.rank == 0 ? std::numeric_limits<DNDS::real>::quiet_NaN() : 1, C::AlphaReduction::Cell, mpi.comm));
    if (mpi.size > 1)
        CHECK_THROWS(C::AgreeAlphaStatistics(0, 1, mpi.rank == 0 ? C::AlphaReduction::Cell : C::AlphaReduction::MinimumAll, mpi.comm));
}
TEST_CASE("Global Euler RHS alpha policies affect unmarked rows only for minimumAll in RHS and dTau styles")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const auto *mode : {"rhs", "dTau"})
        for (const auto *scope : {"cell", "minimumLimited", "minimumAll"})
        {
            C::DomainSettings settings;
            settings.numerics.reconstruction = "constant";
            settings.numerics.positivity = true;
            settings.implicit.iterations = 1;
            settings.implicit.minIterations = 0;
            settings.implicit.relativeTolerance = settings.implicit.absoluteTolerance = 0;
            settings.implicit.positivity.rhsFPPMode = mode;
            settings.implicit.positivity.rhsFPPReduction = scope;
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fv(mesh, settings, equation);
            const auto layout = std::string(scope) == "cell" ? R::RowLayout::Dense(0, 5) : std::string(scope) == "minimumLimited" ? R::RowLayout::ColMajor(0, 5)
                                                                                                                                  : R::RowLayout::TiledColMajor(0, 5, 8);
            auto state = fv.Metrics().BuildField(layout), source = fv.Metrics().BuildField(layout);
            fv.Initialize(state, {}, {1, 0, 0, 0, 1});
            {
                auto write = source.Owned().Write();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const bool bad = mpi.rank == 0 && iCell < 2;
                    write.View<R::UniformTableRows>()(iCell, 0) = 20 + (bad ? (iCell ? -1e6 : -1e8) : 1);
                    for (int j = 1; j < 5; ++j)
                        write.View<R::UniformTableRows>()(iCell, j) = j == 4 ? 50 : 0;
                }
            }
            R::EagerEngine engine;
            const auto stream = engine.CreateCPUStream();
            C::AuxiliaryRequest request;
            request.faceSpectralRadius = request.cellSpectralSum = true;
            C::NonlinearSolver solver(fv, state, fv.BuildWorkspace(state, request), engine, stream, settings);
            const auto report = solver.Solve(state, source, 20, 1, .1);
            CAPTURE(mode);
            CAPTURE(scope);
            CHECK(report.rhsLimited == 2); // Original triggers, not all scaled rows.
            CHECK(report.minimumRHSAlpha > 0);
            CHECK(report.minimumRHSAlpha < 1);
            CHECK(report.iterations == 1);
            CHECK(fv.Admissible(state));
            DNDS::real maximumGoodChange = 0;
            {
                auto read = state.Owned().Read();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    if (mpi.rank != 0 || iCell >= 2)
                        maximumGoodChange = std::max(maximumGoodChange, std::abs(read.View<R::UniformTableRows>()(iCell, 0) - 1));
            }
            MPI_Allreduce(MPI_IN_PLACE, &maximumGoodChange, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (std::string(scope) == "minimumAll")
                CHECK(maximumGoodChange < 1e-5);
            else
                CHECK(maximumGoodChange > 1e-3);
        }
}
TEST_CASE("Native implicit stage recovers a constant manufactured solution from distributed nonconstant means")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        C::DomainSettings settings;
        // This manufactured oracle requires every equation to meet its target;
        // do not silently inherit the production density-only default.
        settings.implicit.residualComponents.clear();
        settings.numerics.reconstruction = scenario == 2 ? "variational" : "constant";
        settings.numerics.cfv.maxOrder = scenario == 2 ? 3 : 1;
        settings.numerics.reconstructionIterations = 100;
        settings.numerics.reconstructionTolerance = 1e-11;
        settings.numerics.positivity = scenario == 2;
        settings.implicit.linear.preconditioner = scenario == 1 ? "ilu" : "sgs";
        settings.implicit.linear.method = scenario == 0 ? "sgs" : "gmres";
        settings.implicit.linear.relativeTolerance = 1e-4;
        settings.implicit.relativeTolerance = 1e-9;
        settings.implicit.absoluteTolerance = 1e-11;
        settings.implicit.minIterations = 0;
        settings.implicit.diagonal = scenario == 1 ? "block" : "entries";
        settings.implicit.roeDissipation = scenario == 1;
        settings.implicit.collectiveChecks = scenario != 1;
        settings.execution.overlap = scenario != 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto layout = scenario == 0 ? R::RowLayout::Dense(0, 5) : scenario == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                      : R::RowLayout::TiledColMajor(0, 5, 32);
        auto state = fv.Metrics().BuildField(layout), source = fv.Metrics().BuildField(layout);
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = request.cellSpectralSum = true;
        auto workspace = fv.BuildWorkspace(state, request);
        R::EagerEngine engine(scenario == 2 ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        const auto target = equation.Gas().Conservative({1.2, .3, -.1, .2, 1.1});
        {
            auto u = state.Owned().Write(), b = source.Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    u.View<R::UniformTableRows>()(iCell, j) = target[j] * (1 + .01 * std::sin(2 * iCell + mpi.rank + j));
                    b.View<R::UniformTableRows>()(iCell, j) = 20 * target[j];
                }
        }
        const auto sourceVersion = source.Owned().Version();
        C::NonlinearSolver solver(fv, state, workspace, engine, stream, settings);
        const auto report = solver.Solve(state, source, 20, .5, .1);
        CAPTURE(scenario);
        CAPTURE(report.iterations);
        CAPTURE(report.linearIterations);
        CHECK(report.converged);
        CHECK(report.iterations > 0);
        CHECK(report.linearIterations > 0);
        CHECK(source.Owned().Version() == sourceVersion);
        {
            const auto read = state.Owned().Read();
            const auto values = read.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(values(iCell, j) == doctest::Approx(target[j]).epsilon(2e-10));
        }
        for (int j = 0; j < 5; ++j)
            CHECK(report.residual[j] <= std::max(settings.implicit.absoluteTolerance, settings.implicit.relativeTolerance * report.initialResidual[j]));
        CHECK_THROWS(solver.Solve(state, state, 20, .5, .1));
        CHECK_THROWS(solver.Solve(state, source, -1, .5, .1));
        int eligible = state.Ghost().Storage().Bytes() >= source.Owned().Storage().Bytes() && source.Owned().Storage().Bytes() > 0;
        MPI_Allreduce(MPI_IN_PLACE, &eligible, 1, MPI_INT, MPI_MIN, mpi.comm);
        if (eligible)
        {
            auto aliased = source;
            aliased.Owned() = R::Array<double>::FromStorage(source.Owned().Layout(), state.Ghost().Storage().Subspan(0, source.Owned().Storage().Bytes()));
            const auto ghostGeneration = state.Ghost().Version();
            CHECK_THROWS(solver.Solve(state, aliased, 20, .5, .1));
            CHECK(state.Ghost().Version() == ghostGeneration);
        }
    }
}
TEST_CASE("Implicit stage cap reports unconverged and auxiliary requirements fail before iteration")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.implicit.iterations = 1;
    settings.implicit.relativeTolerance = 0;
    settings.implicit.absoluteTolerance = 0;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), source = fv.Metrics().BuildField(5);
    fv.Initialize(state, {}, {1, .1, 0, 0, 1});
    A::Fill(source, 0.);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    CHECK_THROWS(C::NonlinearSolver(fv, state, fv.BuildWorkspace(state), engine, stream, settings));
    C::AuxiliaryRequest request;
    request.faceSpectralRadius = request.cellSpectralSum = true;
    C::NonlinearSolver solver(fv, state, fv.BuildWorkspace(state, request), engine, stream, settings);
    const auto report = solver.Solve(state, source, 1, 1, .1);
    CHECK_FALSE(report.converged);
    CHECK(report.iterations == 1);
    CHECK(fv.Admissible(state));
}
TEST_CASE("Implicit bad corrections reject without poisoning state and nonlinear controls agree before allocation")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.implicit.minIterations = 0;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), source = fv.Metrics().BuildField(5);
    fv.Initialize(state, {}, {1, .1, 0, 0, 1});
    const auto target = equation.Gas().Conservative({1, .1, 0, 0, 1});
    {
        auto write = source.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = -20 * target[j];
    }
    C::AuxiliaryRequest request;
    request.faceSpectralRadius = request.cellSpectralSum = true;
    auto workspace = fv.BuildWorkspace(state, request);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    if (mpi.size > 1)
    {
        auto mismatched = settings;
        mismatched.implicit.positivity.compressionPower += mpi.rank;
        CHECK_THROWS(C::NonlinearSolver(fv, state, workspace, engine, stream, mismatched));
    }
    C::NonlinearSolver solver(fv, state, workspace, engine, stream, settings);
    const auto generation = state.Owned().Version();
    CHECK_THROWS(solver.Solve(state, source, 20, 1, .1));
    CHECK(state.Owned().IsValid());
    CHECK(state.Owned().Version() == generation);
    CHECK(fv.Admissible(state));
    {
        auto write = source.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = 20 * target[j];
    }
    CHECK(solver.Solve(state, source, 20, 1, .1).converged);
}
TEST_CASE("Subset correction masks before Euler PP and preserves frozen means exactly")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 1);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.positivity = true;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), increment = fv.Metrics().BuildField(5), source = fv.Metrics().BuildField(5);
    const auto initial = equation.Gas().Conservative({1, .3, -.1, .05, 1});
    fv.Initialize(state, {}, {1, .3, -.1, .05, 1});
    C::FlowState delta{};
    delta[4] = -2 * C::PositivityDetail::InternalEnergy(initial);
    {
        auto write = increment.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = j == 4 ? delta[4] : std::numeric_limits<double>::quiet_NaN();
    }
    C::AuxiliaryRequest channels;
    channels.faceSpectralRadius = channels.cellSpectralSum = true;
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    C::NonlinearSolver solver(fv, state, fv.BuildWorkspace(state, channels), engine, stream, settings);
    const auto alpha = C::FlowIncrementFraction(initial, delta, fv.ThermalConstraints(), settings.implicit.positivity.incrementRelax, 2, settings.implicit.positivity.compressionPower);
    CHECK(alpha > 0);
    CHECK(alpha < 1);
    auto limitedDelta = delta;
    for (auto &value : limitedDelta)
        value *= alpha;
    limitedDelta = C::FixFlowIncrement(initial, limitedDelta, fv.ThermalConstraints());
    const auto result = solver.ApplyCorrection(state, increment, 1, {4});
    CHECK(result.first == 1);
    CHECK(result.second == alpha);
    CHECK(fv.Admissible(state));
    {
        const auto read = state.Owned().Read();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                if (j == 4)
                    CHECK(read.View()(iCell, j) == initial[j] + limitedDelta[j]);
                else
                    CHECK(std::memcmp(&read.View()(iCell, j), &initial[j], sizeof(double)) == 0);
    }
    const auto generation = state.Owned().Version();
    for (const std::vector<DNDS::index> bad : {std::vector<DNDS::index>{0, 0}, {5}})
    {
        C::NonlinearSolveControl control;
        control.updateComponents = bad;
        CHECK_THROWS(solver.Solve(state, source, 1, 1, 0, 0, control));
        CHECK_THROWS(solver.ApplyCorrection(state, increment, 1, bad));
        CHECK(state.Owned().Version() == generation);
        auto invalid = settings.implicit;
        invalid.multigrid.updateComponents = bad;
        CHECK_THROWS(invalid.Validate());
    }
    if (mpi.size > 1)
    {
        C::NonlinearSolveControl mismatch;
        mismatch.updateComponents = mpi.rank == 0 ? std::vector<DNDS::index>{0} : std::vector<DNDS::index>{4};
        CHECK_THROWS(solver.Solve(state, source, 1, 1, 0, 0, mismatch));
        CHECK(state.Owned().Version() == generation);
    }
}
TEST_CASE("Optional nonlinear phase output preserves exact corrections and records early failures without extra collectives")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.positivity = true;
    settings.implicit.linear.method = "sgs";
    settings.implicit.linear.iterations = 2;
    settings.implicit.linear.relativeTolerance = settings.implicit.linear.absoluteTolerance = 0;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto plain = fv.Metrics().BuildField(5), profiled = fv.Metrics().BuildField(5), source = fv.Metrics().BuildField(5);
    fv.Initialize(plain, {}, {1, 0, 0, 0, 1});
    A::Copy(profiled, plain);
    {
        const auto read = plain.Owned().Read();
        auto write = source.Owned().Write();
        const auto u = read.View<R::UniformTableRows>();
        const auto b = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                b(iCell, j) = 20 * u(iCell, j) + (j == 4 ? .1 * (fv.Metrics().Cells().LocalToGlobal(iCell) + 1) : 0);
    }
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    C::AuxiliaryRequest request;
    request.faceSpectralRadius = request.cellSpectralSum = true;
    C::NonlinearSolver plainSolver(fv, plain, fv.BuildWorkspace(plain, request), engine, stream, settings);
    C::NonlinearSolver profiledSolver(fv, profiled, fv.BuildWorkspace(profiled, request), engine, stream, settings);
    const auto a = plainSolver.Solve(plain, source, 20, 1, 0, 0, {1});
    C::NonlinearProfile profile;
    profile.seconds.fill(-1);
    profile.residualEvaluations = 777;
    const auto b = profiledSolver.Solve(profiled, source, 20, 1, 0, 0, {1, 0, &profile});
    CHECK(a.iterations == 1);
    CHECK(b.iterations == a.iterations);
    CHECK(b.linearIterations == a.linearIterations);
    CHECK(b.residual == a.residual);
    CHECK(b.residualTargetRatio == a.residualTargetRatio);
    CHECK(b.converged == a.converged);
    CHECK(profile.residualEvaluations == 2);
    DNDS::real sections = 0;
    for (int phase = 0; phase < C::NonlinearProfile::Total; ++phase)
    {
        CHECK(profile.seconds[phase] > 0);
        sections += profile.seconds[phase];
    }
    CHECK(profile.seconds[C::NonlinearProfile::Total] >= sections);
    {
        const auto x = plain.Owned().Read(), y = profiled.Owned().Read();
        for (DNDS::index iCell = 0; iCell < plain.Owned().RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(x.View<R::UniformTableRows>()(iCell, j) == y.View<R::UniformTableRows>()(iCell, j));
    }
    CHECK_THROWS(profiledSolver.Solve(profiled, source, std::numeric_limits<double>::quiet_NaN(), 1, 0, 0, {1, 0, &profile}));
    CHECK(profile.residualEvaluations == 0);
    CHECK(profile.seconds[C::NonlinearProfile::Entry] > 0);
    CHECK(profile.seconds[C::NonlinearProfile::Total] >= profile.seconds[C::NonlinearProfile::Entry]);
    for (int phase = C::NonlinearProfile::Residual; phase < C::NonlinearProfile::Total; ++phase)
        CHECK(profile.seconds[phase] == 0);
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    MPI_Finalize();
    return result;
}
