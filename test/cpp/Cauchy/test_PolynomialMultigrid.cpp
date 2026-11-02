#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "CartesianFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
namespace
{
    C::DomainSettings Controls()
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 160;
        settings.numerics.reconstructionTolerance = 1e-12;
        settings.implicit.minIterations = 0;
        settings.implicit.relativeTolerance = 1e-9;
        settings.implicit.absoluteTolerance = 1e-11;
        settings.implicit.multigrid.levels = 2;
        settings.implicit.multigrid.coarseSmooth = 3;
        return settings;
    }
    void Fill(C::FiniteVolume const &fv, C::EquationSystem const &equation, DNDS::CFV::FVField &state, double perturbation)
    {
        auto write = state.Owned().Write();
        const auto values = write.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
        {
            const auto global = fv.Metrics().Cells().LocalToGlobal(iCell);
            const auto mean = equation.Gas().Conservative({1 + .05 * std::sin(global * .7), .2 + .02 * std::cos(global), -.1, .03, 1.1});
            for (int j = 0; j < 5; ++j)
                values(iCell, j) = mean[j] * (1 + perturbation * std::cos(global + j));
        }
    }
}
TEST_CASE("Polynomial FAS recovers the original P3 manufactured equation across layouts and distributed halos")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        auto settings = Controls();
        settings.implicit.multigrid.levels = scenario == 0 ? 1 : 2;
        settings.numerics.positivity = scenario == 2;
        settings.execution.overlap = scenario != 1;
        settings.implicit.collectiveChecks = scenario != 1;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fine(mesh, settings, equation);
        const auto layout = scenario == 0 ? R::RowLayout::Dense(0, 5, 8) : scenario == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                         : R::RowLayout::TiledColMajor(0, 5, 32);
        auto state = fine.Metrics().BuildField(layout), exact = fine.Metrics().BuildField(layout), source = fine.Metrics().BuildField(layout);
        C::AuxiliaryRequest channels;
        channels.faceSpectralRadius = channels.cellSpectralSum = true;
        auto workspace = fine.BuildWorkspace(state, channels);
        R::EagerEngine engine(scenario == 2 ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        C::NonlinearSolver smoother(fine, state, workspace, engine, stream, settings);
        Fill(fine, equation, state, 0);
        A::Copy(exact, state);
        smoother.EvaluateEquation(state, source, 30, .5, .1);
        const auto generation = source.Owned().Version();
        Fill(fine, equation, state, .01);
        C::PolynomialMultigrid solver(mesh, fine, state, smoother, engine, stream, settings);
        const auto report = solver.Solve(source, 30, .5, .1);
        CAPTURE(scenario);
        CAPTURE(report.multigridCycles);
        CAPTURE(report.residual[0]);
        CHECK(report.converged);
        CHECK(report.multigridCycles > 0);
        REQUIRE(report.levelCorrections.size() == size_t(settings.implicit.multigrid.levels + 1));
        CHECK(report.levelCorrections.back() > 0);
        CHECK(report.linearIterations > 0);
        CHECK(source.Owned().Version() == generation);
        const auto solutionRead = state.Owned().Read(), exactRead = exact.Owned().Read();
        const auto actual = solutionRead.View<R::UniformTableRows>(), expected = exactRead.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < actual.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(actual(iCell, j) == doctest::Approx(expected(iCell, j)).epsilon(5e-9));
    }
}
TEST_CASE("Polynomial FAS fixed points and preflight errors preserve fine means")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    auto settings = Controls();
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fine(mesh, settings, equation);
    auto state = fine.Metrics().BuildField(5), source = fine.Metrics().BuildField(5), wrong = fine.Metrics().BuildField(4);
    fine.Initialize(state, {}, {1, .2, -.1, 0, 1});
    A::Copy(source, state);
    A::Scale(source, 10.);
    C::AuxiliaryRequest channels;
    channels.faceSpectralRadius = channels.cellSpectralSum = true;
    auto workspace = fine.BuildWorkspace(state, channels);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    C::NonlinearSolver smoother(fine, state, workspace, engine, stream, settings);
    C::PolynomialMultigrid solver(mesh, fine, state, smoother, engine, stream, settings);
    const auto generation = state.Owned().Version();
    CHECK_THROWS(solver.Solve(state, 10, .5, 0));
    CHECK_THROWS(solver.Solve(wrong, 10, .5, 0));
    const auto report = solver.Solve(source, 10, .5, 0);
    CHECK(report.converged);
    CHECK(report.multigridCycles == 0);
    CHECK(state.Owned().Version() == generation);
    A::Fill(source, -1e6);
    CHECK_THROWS(solver.Solve(source, 10, .5, 0));
    CHECK(state.Owned().IsValid());
    CHECK(state.Owned().Version() == generation);
    // A callback failure invalidates the generic cycle. The physical wrapper
    // must rebuild its cycle scratch before the next trial, not silently reuse
    // the partially updated hierarchy. Fine means survived the rejected delta.
    A::Copy(source, state);
    A::Scale(source, 10.);
    Fill(fine, equation, state, .01);
    fine.ResetReconstruction(*workspace);
    CHECK(solver.Solve(source, 10, .5, 0).converged);
}
TEST_CASE("Cauchy pMG physical discard and commit retry retain the fine reconstruction transaction")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    auto settings = Controls();
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::TimeSettings time;
    time.method = "esdirk2";
    C::Domain reference("reference", mesh, settings, time), retry("retry", mesh, settings, time);
    Fill(reference.Discretization(), equation, reference.Flow(), 0);
    Fill(retry.Discretization(), equation, retry.Flow(), 0);
    reference.Advance(.001);
    retry.PrepareStep(.001);
    retry.DiscardStep();
    retry.PrepareStep(.001);
    {
        std::optional<R::Array<double>::ReadLease> retained;
        if (mpi.rank == 0)
            retained.emplace(retry.Flow().Owned().Read());
        CHECK_THROWS(retry.CommitStep());
    }
    retry.CommitStep();
    CHECK(retry.Time() == .001);
    CHECK(retry.Epoch().acceptedStep == 1);
    const auto first = reference.Flow().Owned().Read(), second = retry.Flow().Owned().Read();
    const auto a = first.View<R::UniformTableRows>(), b = second.View<R::UniformTableRows>();
    for (DNDS::index iCell = 0; iCell < a.RowCount(); ++iCell)
        for (int j = 0; j < 5; ++j)
            CHECK(a(iCell, j) == b(iCell, j));
    REQUIRE_FALSE(retry.LastImplicitStages().empty());
    for (const auto &stage : retry.LastImplicitStages())
    {
        CHECK(stage.converged);
        CHECK(stage.multigridCycles > 0);
    }
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
