#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "CartesianFixture.hpp"
#include <cstring>

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
TEST_CASE("Finest FAS acceptance shares selected-component L1 L2 convergence with the stage solver")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int order : {1, 2})
    {
        auto settings = Controls();
        settings.implicit.normOrder = order;
        settings.implicit.residualComponents = {"rho"};
        settings.implicit.residualReference = "runningMaximum";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fine(mesh, settings, equation);
        auto state = fine.Metrics().BuildField(5), source = fine.Metrics().BuildField(5);
        fine.Initialize(state, {}, {1, 0, 0, 0, 1});
        const auto mean = equation.Gas().Conservative({1, 0, 0, 0, 1});
        {
            auto write = source.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = 20 * mean[j];
                values(iCell, 4) += .1 * (fine.Metrics().Cells().LocalToGlobal(iCell) + 1);
            }
        }
        C::AuxiliaryRequest channels;
        channels.faceSpectralRadius = channels.cellSpectralSum = true;
        auto work = fine.BuildWorkspace(state, channels);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        C::NonlinearSolver smoother(fine, state, work, engine, stream, settings);
        C::PolynomialMultigrid solver(mesh, fine, state, smoother, engine, stream, settings);
        const auto version = state.Owned().Version();
        const auto report = solver.Solve(source, 20, 1, 0);
        CHECK(report.converged);
        CHECK(report.multigridCycles == 0);
        CHECK(report.initialResidual == report.referenceResidual);
        CHECK(report.residual[4] == doctest::Approx(order == 1 ? .85 : std::sqrt(.935)).epsilon(1e-12));
        CHECK(report.residualTargetRatio < 1);
        CHECK(state.Owned().Version() == version);
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
        CHECK(report.iterations == report.multigridCycles);
        CHECK(report.levelCorrections.front() == report.multigridCycles *
                                                     (settings.implicit.multigrid.preSmooth + settings.implicit.multigrid.postSmooth));
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
TEST_CASE("Selected pMG recovers nonconstant principal equations while preserving frozen means and full residual reporting")
{
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        const auto mesh = CauchyTest::PeriodicQuads(mpi, scenario == 3 ? 1 : 4);
        auto settings = Controls();
        settings.implicit.multigrid.updateComponents = {4, 0};
        settings.numerics.positivity = scenario == 2;
        settings.execution.overlap = scenario != 1;
        settings.implicit.collectiveChecks = scenario != 1;
        settings.implicit.diagonal = scenario == 0 ? "entries" : "block";
        settings.implicit.roeDissipation = scenario != 0;
        settings.implicit.jacobianAction = scenario == 2 ? "semiMatrixFree" : "assembled";
        settings.implicit.residualComponents = scenario == 3 ? std::vector<std::string>{} : std::vector<std::string>{"rho", "rhoE"};
        if (scenario == 1)
        {
            settings.implicit.linear.method = "gmres";
            settings.implicit.linear.preconditioner = "ilu";
        }
        if (scenario == 3)
            settings.implicit.multigrid.cycles = 8;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fine(mesh, settings, equation);
        const auto layout = scenario == 0 ? R::RowLayout::Dense(0, 5, 8) : scenario == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                         : R::RowLayout::TiledColMajor(0, 5, 8);
        auto state = fine.Metrics().BuildField(layout), exact = fine.Metrics().BuildField(layout), source = fine.Metrics().BuildField(layout);
        auto value = fine.Metrics().BuildField(layout);
        C::AuxiliaryRequest channels;
        channels.faceSpectralRadius = channels.cellSpectralSum = true;
        auto workspace = fine.BuildWorkspace(state, channels);
        R::EagerEngine engine(scenario == 2 ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        C::NonlinearSolver smoother(fine, state, workspace, engine, stream, settings);
        Fill(fine, equation, state, 0);
        A::Copy(exact, state);
        smoother.EvaluateEquation(state, source, 30, .5, .1);
        {
            auto write = state.Owned().Write();
            const auto u = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
                for (int j : {0, 4})
                    u(iCell, j) *= 1 + .01 * std::cos(fine.Metrics().Cells().LocalToGlobal(iCell) + j);
        }
        if (scenario == 3)
        {
            // This frozen momentum equation is deliberately incompatible.
            // A subset solve may reduce its own equations, but must not claim
            // convergence of the configured full finest equation.
            auto write = source.Owned().Write();
            for (DNDS::index iCell = 0; iCell < DNDS::index(source.Space().OwnedRows()); ++iCell)
                write.View<R::UniformTableRows>()(iCell, 2) += .1;
        }
        const auto sourceVersion = source.Owned().Version();
        const auto measure = [&]
        {
            smoother.EvaluateEquation(state, value, 30, .5, .1);
            std::array<double, 6> norms{};
            {
                const auto read = value.Owned().Read(), bRead = source.Owned().Read();
                for (DNDS::index iCell = 0; iCell < DNDS::index(state.Space().OwnedRows()); ++iCell)
                {
                    const auto volume = fine.Metrics().GetCellVol(iCell);
                    norms[5] += volume;
                    for (int j = 0; j < 5; ++j)
                    {
                        const auto error = bRead.View<R::UniformTableRows>()(iCell, j) - read.View<R::UniformTableRows>()(iCell, j);
                        norms[j] += volume * error * error;
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, norms.data(), 6, MPI_DOUBLE, MPI_SUM, mpi.comm);
            C::FlowState result{};
            for (int j = 0; j < 5; ++j)
                result[j] = std::sqrt(norms[j] / norms[5]);
            return result;
        };
        // Compare the initial equation with the same cold VR guess on both
        // evaluations; otherwise finite VR stopping changes the operator by
        // a few tolerance units before any mean correction has occurred.
        fine.ResetReconstruction(*workspace);
        const auto initialNorm = measure();
        fine.ResetReconstruction(*workspace);
        C::PolynomialMultigrid solver(mesh, fine, state, smoother, engine, stream, settings);
        const auto report = solver.Solve(source, 30, .5, .1);
        CAPTURE(scenario);
        CAPTURE(report.multigridCycles);
        CAPTURE(report.residual);
        CHECK(report.multigridCycles > 0);
        CHECK(report.converged == (scenario != 3));
        CHECK(source.Owned().Version() == sourceVersion);
        for (int j = 0; j < 5; ++j)
            CHECK(std::abs(report.initialResidual[j] - initialNorm[j]) < 1e-12);
        if (scenario == 3)
        {
            CHECK(report.residual[2] == doctest::Approx(.1).epsilon(1e-10));
            auto subsetControls = settings.implicit;
            subsetControls.residualComponents = {"rho", "rhoE"};
            C::ResidualConvergence subset(subsetControls, equation.Schema());
            subset.Observe(report.initialResidual);
            subset.Observe(report.residual);
            CHECK(subset.Accepted());
        }
        {
            const auto read = state.Owned().Read(), reference = exact.Owned().Read();
            const auto u = read.View<R::UniformTableRows>(), expected = reference.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    if (j == 0 || j == 4)
                    {
                        if (scenario != 3)
                            CHECK(u(iCell, j) == doctest::Approx(expected(iCell, j)).epsilon(1e-8));
                    }
                    else
                        CHECK(std::memcmp(&u(iCell, j), &expected(iCell, j), sizeof(double)) == 0);
        }
        const auto finalNorm = measure();
        // Re-evaluation advances an already converged VR warm guess. Do not
        // demand relative precision below its absolute roundoff/tolerance
        // floor; the independent initial norms above are nontrivial, and the
        // incompatible frozen momentum equation below remains O(0.1).
        for (int j = 0; j < 5; ++j)
            CHECK(std::abs(report.residual[j] - finalNorm[j]) < 1e-12);
    }
}
TEST_CASE("Cauchy pMG physical discard and commit retry retain the fine reconstruction transaction")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const std::string method : {"esdirk2", "ditr"})
    {
        auto settings = Controls();
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::TimeSettings time;
        time.method = method;
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
        if (method == "ditr")
            CHECK(retry.LastCoupledTargetRatio() <= 1);
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
