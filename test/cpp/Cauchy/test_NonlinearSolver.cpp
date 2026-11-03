#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/PositivityReduction.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "CartesianFixture.hpp"
#include <limits>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
TEST_CASE("Implicit convergence retains initial and running references without hiding unselected components")
{
    for (const auto *reference : {"initial", "runningMaximum"})
    {
        C::ImplicitSettings settings;
        settings.relativeTolerance = .1;
        settings.absoluteTolerance = 0;
        settings.residualReference = reference;
        settings.Validate();
        C::ResidualConvergence policy(settings);
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
    exact.relativeTolerance = exact.absoluteTolerance = 0;
    exact.residualComponents = {0};
    C::ResidualConvergence policy(exact);
    policy.Observe({0, 100, 0, 0, 0});
    CHECK(policy.Accepted());
    CHECK(policy.TargetRatio() == 0);
    CHECK(policy.Initial()[1] == 100);
    policy.Observe({1e-300, 100, 0, 0, 0});
    CHECK_FALSE(policy.Accepted());
    CHECK(std::isfinite(policy.TargetRatio()));
    for (const std::vector<int> components : {std::vector<int>{}, {0, 0}, {-1}, {5}})
    {
        exact.residualComponents = components;
        CHECK_THROWS(exact.Validate());
    }
    exact.residualComponents = {0};
    exact.normOrder = 3;
    CHECK_THROWS(exact.Validate());
    exact.normOrder = 1;
    exact.residualReference = "unknown";
    CHECK_THROWS(exact.Validate());
}
TEST_CASE("Distributed implicit L1 L2 norms and selected-component stops use the actual manufactured defect")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
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
            if (densityOnly)
                settings.implicit.residualComponents = {0};
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
            CHECK(report.residual[4] == doctest::Approx(order == 1 ? .85 : std::sqrt(.935)).epsilon(1e-12));
            CHECK(report.initialResidual == report.referenceResidual);
            CHECK(report.residualTargetRatio == doctest::Approx(densityOnly ? 0 : 10));
            CHECK(state.Owned().Version() == version);
        }
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
            settings.implicit.positivity.rhsMode = mode;
            settings.implicit.positivity.rhsReduction = scope;
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
