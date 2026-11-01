#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CartesianFixture.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include <filesystem>
#include <unistd.h>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
TEST_CASE("Native Cauchy flow5 residual preserves constant state across layouts reconstruction and workers")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    for (const std::string method : {"constant", "gradient", "variational"})
        for (int layout = 0; layout < 3; ++layout)
        {
            C::DomainSettings settings;
            settings.mesh.dimension = 3;
            settings.boundaries["7"].kind = "farfield";
            settings.boundaries["7"].primitive = {1.2, .3, -.2, .4, 1.1};
            settings.numerics.reconstruction = method;
            settings.numerics.cfv.maxOrder = 1;
            settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
            const auto equation = C::EquationSystem::Resolve(settings.equation);
            C::FiniteVolume fv(mesh, settings, equation);
            const auto physical = layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                           : R::RowLayout::TiledColMajor(0, 5, 32);
            auto state = fv.Metrics().BuildField(physical), residual = fv.Metrics().BuildField(physical);
            const auto exact = equation.Gas().Conservative(settings.boundaries["7"].primitive);
            {
                auto write = state.Owned().Write();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        write.View<R::UniformTableRows>()(iCell, j) = exact[j];
            }
            auto workspace = fv.BuildWorkspace(state);
            for (int workers : {0, 2})
            {
                R::EagerEngine engine(workers, std::make_shared<R::AccessTrackingContext>());
                const auto stream = engine.CreateCPUStream();
                fv.Evaluate(state, residual, *workspace, engine, stream);
                engine.Synchronize();
                CHECK(A::NormGlobal(residual) < 2e-14);
            }
        }
}
TEST_CASE("Native viscous Cauchy residual reproduces linear Couette heat production")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 7);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .02;
    settings.boundaries["7"].kind = "extrapolate";
    settings.numerics.reconstruction = "gradient";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
    const double shear = .3;
    {
        auto write = state.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto center = fv.Metrics().GetCellCent(iCell);
            auto value = equation.Gas().Conservative({1, shear * center[1], 0, 0, 1});
            // Conservative cell average of u^2 includes the exact y variance.
            value[4] += .5 * shear * shear / (12 * 49);
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    auto work = fv.BuildWorkspace(state);
    auto inviscidSettings = settings;
    inviscidSettings.equation.transport = "inviscid";
    inviscidSettings.equation.viscosity = 0;
    C::FiniteVolume inviscid(mesh, inviscidSettings, C::EquationSystem::Resolve(inviscidSettings.equation));
    auto inviscidState = inviscid.Metrics().BuildField(5), inviscidResidual = inviscid.Metrics().BuildField(5);
    {
        auto source = state.Owned().Read();
        auto target = inviscidState.Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                target.View()(iCell, j) = source.View()(iCell, j);
    }
    auto inviscidWork = inviscid.BuildWorkspace(inviscidState);
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, residual, *work, engine, stream);
    inviscid.Evaluate(inviscidState, inviscidResidual, *inviscidWork, engine, stream);
    const auto read = residual.Owned().Read();
    const auto base = inviscidResidual.Owned().Read();
    DNDS::index tested = 0;
    // Interior cells have an independent exact gradient/stress oracle; exclude
    // two-cell boundary closure whose extrapolation does not prescribe shear.
    for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
    {
        const auto point = fv.Metrics().GetCellCent(iCell);
        if ((point.array() > .29).all() && (point.array() < .71).all())
        {
            ++tested;
            const double volume = fv.Metrics().GetCellVol(iCell);
            CHECK(std::abs((read.View()(iCell, 1) - base.View()(iCell, 1)) / volume) < 1e-12);
            CHECK((read.View()(iCell, 4) - base.View()(iCell, 4)) / volume == doctest::Approx(.02 * shear * shear).epsilon(2e-8));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &tested, 1, MPI_INT64_T, MPI_SUM, mpi.comm);
    CHECK(tested == 27);
}
TEST_CASE("Cauchy SSPRK3 preserves periodic constants and rolls back rejected trials")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    settings.initialPrimitive = {1.1, .3, -.2, .4, 1.2};
    settings.execution.hostThreads = 2;
    C::Domain domain("periodic", mesh, settings);
    const auto exact = domain.Equation().Gas().Conservative(settings.initialPrimitive);
    for (int i = 0; i < 3; ++i)
        domain.Advance(.001);
    CHECK(domain.Time() == doctest::Approx(.003));
    CHECK(domain.Epoch().acceptedStep == 3);
    CHECK(domain.LastFaceFallbacks() == 0);
    {
        const auto read = domain.Flow().Owned().Read();
        for (DNDS::index iCell = 0; iCell < domain.Discretization().Metrics().NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(read.View()(iCell, j) == doctest::Approx(exact[j]).epsilon(2e-13));
    }
    domain.PrepareStep(.001);
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(domain.Flow().Owned().Read());
    CHECK_THROWS(domain.CommitStep());
    CHECK(domain.Time() == doctest::Approx(.003));
    held.reset();
    domain.CommitStep();
    CHECK(domain.Epoch().acceptedStep == 4);
    CHECK_THROWS(domain.Advance(-1));
    CHECK(domain.Time() == doctest::Approx(.004));
}
TEST_CASE("Cauchy periodic density transport has independently measured second-order spatial residual")
{
    double previous = 0;
    constexpr double pi = 3.14159265358979323846;
    for (DNDS::index width : {8, 16, 32})
    {
        C::DomainSettings settings;
        settings.numerics.cfv.maxOrder = 1;
        settings.execution.hostThreads = 2;
        C::Domain domain("transport", CauchyTest::PeriodicQuads(mpi, width), settings);
        const auto &fv = domain.Discretization().Metrics();
        const double averageFactor = std::sin(pi / width) / (pi / width);
        {
            auto write = domain.Flow().Owned().Write();
            for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
            {
                const auto x = fv.GetCellCent(iCell);
                const double rho = 1 + .1 * averageFactor * averageFactor * std::sin(2 * pi * (x[0] + x[1]));
                const auto value = domain.Equation().Gas().Conservative({rho, .4, .2, .3, 1});
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = value[j];
            }
        }
        const auto &rhs = domain.Residual();
        auto read = rhs.Owned().Read();
        double error = 0, total = 0;
        for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
        {
            const auto x = fv.GetCellCent(iCell);
            const double volume = fv.GetCellVol(iCell);
            const double exact = -.1 * averageFactor * averageFactor * 2 * pi * .6 * std::cos(2 * pi * (x[0] + x[1]));
            error += volume * std::pow(read.View()(iCell, 0) / volume - exact, 2);
            total += read.View()(iCell, 0);
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
        error = std::sqrt(error);
        CHECK(std::abs(total) < 1e-12);
        CHECK(domain.LastFaceFallbacks() == 0);
        if (previous)
            CHECK(previous / error > 3.4);
        previous = error;
    }
}
TEST_CASE("Cauchy periodic transport SSPRK3 advances a smooth physical state and conserves mass")
{
    constexpr double pi = 3.14159265358979323846;
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    settings.execution.hostThreads = 2;
    constexpr DNDS::index width = 16;
    C::Domain domain("advection", CauchyTest::PeriodicQuads(mpi, width), settings);
    const auto &fv = domain.Discretization().Metrics();
    const double averageFactor = std::sin(pi / width) / (pi / width);
    {
        auto write = domain.Flow().Owned().Write();
        for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
        {
            const auto point = fv.GetCellCent(iCell);
            const double rho = 1 + .1 * averageFactor * averageFactor * std::sin(2 * pi * (point[0] + point[1]));
            const auto state = domain.Equation().Gas().Conservative({rho, .4, .2, .3, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = state[j];
        }
    }
    for (int step = 0; step < 10; ++step)
        domain.Advance(.001);
    double error = 0, mass = 0;
    const auto read = domain.Flow().Owned().Read();
    for (DNDS::index iCell = 0; iCell < fv.NumCell(); ++iCell)
    {
        const auto point = fv.GetCellCent(iCell);
        const double volume = fv.GetCellVol(iCell), exact = 1 + .1 * averageFactor * averageFactor * std::sin(2 * pi * (point[0] + point[1] - .6 * domain.Time()));
        error += volume * DNDS::sqr(read.View()(iCell, 0) - exact);
        mass += volume * read.View()(iCell, 0);
    }
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(std::sqrt(error) < 2e-4);
    CHECK(mass == doctest::Approx(1).epsilon(2e-13));
    CHECK(domain.Epoch().acceptedStep == 10);
    CHECK(domain.LastFaceFallbacks() == 0);
}
TEST_CASE("Every Cauchy Riemann selection is integrated into periodic SSPRK3")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const auto *name : {"rusanov", "Roe", "Roe_M1", "Roe_M2", "Roe_M3", "Roe_M4", "Roe_M5", "Roe_M6", "Roe_M7", "Roe_M8", "Roe_M9",
                             "HLLC", "HLLEP", "HLLEP_V1", "AUSMPWPlus", "AUSMPlusUp", "AUSMPlusUp2"})
    {
        CAPTURE(name);
        C::DomainSettings settings;
        settings.numerics.flux = name;
        settings.numerics.cfv.maxOrder = 1;
        settings.initialPrimitive = {1.2, .4, -.2, .3, 1.1};
        C::Domain domain("flow", mesh, settings);
        domain.Advance(.001);
        const auto exact = domain.Equation().Gas().Conservative(settings.initialPrimitive);
        const auto read = domain.Flow().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(read.View()(iCell, j) == doctest::Approx(exact[j]).epsilon(3e-13));
    }
}
TEST_CASE("Cauchy variational reconstruction with WBAP advances a nonconstant conservative field")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.limiter = "WBAP";
    settings.numerics.flux = "HLLEP";
    settings.numerics.cfv.maxOrder = 2;
    settings.execution.hostThreads = 2;
    C::Domain domain("limited", mesh, settings);
    {
        auto write = domain.Flow().Owned().Write();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            const auto point = domain.Discretization().Metrics().GetCellCent(iCell);
            const auto value = domain.Equation().Gas().Conservative({1 + .05 * std::sin(6.283185307179586 * point[0]), .3, .1, -.2, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    for (int step = 0; step < 3; ++step)
        domain.Advance(.001);
    double mass = 0;
    const auto read = domain.Flow().Owned().Read();
    for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
    {
        C::FlowState value;
        for (int j = 0; j < 5; ++j)
            value[j] = read.View()(iCell, j);
        CHECK(domain.Equation().Gas().Admissible(value));
        mass += value[0] * domain.Discretization().Metrics().GetCellVol(iCell);
    }
    MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(mass == doctest::Approx(1).epsilon(1e-12));
    CHECK(domain.LastFaceFallbacks() == 0);
    CHECK(domain.Epoch().acceptedStep == 3);
}
TEST_CASE("Cauchy physical execution includes ranks with empty owned cells")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 2);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    C::Domain domain("empty", mesh, settings);
    CHECK(domain.StableStep(.5) > 0);
    domain.Advance(.001);
    CHECK(domain.Epoch().acceptedStep == 1);
    CHECK(domain.Time() == .001);
    CHECK(A::NormGlobal(domain.Residual()) < 1e-13);
}
TEST_CASE("Cauchy rejects inadmissible trial collectively and retries without accepted-state corruption")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.cfv.maxOrder = 1;
    C::Domain domain("rejection", mesh, settings);
    {
        auto write = domain.Flow().Owned().Write();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
        {
            const auto x = domain.Discretization().Metrics().GetCellCent(iCell);
            const auto value = domain.Equation().Gas().Conservative({1, std::sin(6.283185307179586 * x[0]), 0, 0, 1});
            for (int j = 0; j < 5; ++j)
                write.View()(iCell, j) = value[j];
        }
    }
    const auto before = domain.State().Clone();
    CHECK_THROWS(domain.Advance(1e4));
    CHECK(domain.Time() == 0);
    CHECK(domain.Epoch().acceptedStep == 0);
    {
        auto saved = before.Block("flow").Owned().Read(), actual = domain.Flow().Owned().Read();
        for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View()(iCell, j) == saved.View()(iCell, j));
    }
    domain.Advance(.0001);
    CHECK(domain.Epoch().acceptedStep == 1);
    const auto &rhs = domain.Residual();
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(rhs.Owned().Read());
    CHECK_THROWS(domain.Residual());
    held.reset();
    CHECK_NOTHROW(domain.Residual());
}
TEST_CASE("Cauchy matching-partition restart preserves layouts continuation and guards mesh identity")
{
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("cauchy-restart-" + std::to_string(pid) + ".h5")).string();
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const std::string layout : {"aos", "soa", "aosoa32"})
    {
        C::DomainSettings settings;
        settings.numerics.cfv.maxOrder = 1;
        settings.execution.layout = layout;
        settings.initialPrimitive = {1.2, .3, -.2, .1, 1.1};
        C::Domain source("flow", mesh, settings), restored("flow", mesh, settings);
        const C::Json resolved{{"test", layout}};
        // Preserve a non-constant state, so a restart that silently restores
        // only the initial condition cannot pass continuation equivalence.
        {
            auto write = source.Flow().Owned().Write();
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                auto primitive = settings.initialPrimitive;
                primitive[0] += .01 * std::sin(6.283185307179586 * source.Discretization().Metrics().GetCellCent(iCell)[0]);
                const auto value = source.Equation().Gas().Conservative(primitive);
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = value[j];
            }
        }
        source.WriteCheckpoint(file, resolved);
        restored.ReadCheckpoint(file, resolved);
        CHECK(restored.Epoch().acceptedStep == 0);
        source.Advance(.001);
        source.WriteCheckpoint(file, resolved);
        CHECK_THROWS(restored.ReadCheckpoint(file, C::Json{{"test", "wrong"}}));
        CHECK(restored.Time() == 0);
        restored.ReadCheckpoint(file, resolved);
        CHECK(restored.Flow().Owned().Layout().Kind() == source.Flow().Owned().Layout().Kind());
        source.Advance(.001);
        restored.Advance(.001);
        CHECK(restored.Time() == source.Time());
        CHECK(restored.Epoch().acceptedStep == source.Epoch().acceptedStep);
        {
            auto left = source.Flow().Owned().Read(), right = restored.Flow().Owned().Read();
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(left.View()(iCell, j) == right.View()(iCell, j));
        }
        restored.PrepareStep(.001);
        restored.Residual();
        CHECK_THROWS(restored.CommitStep());
        const auto shifted = CauchyTest::PeriodicQuads(mpi, 4);
        shifted->ApplyAffineGeometryTransform(Eigen::Matrix3d::Identity(), DNDS::Geom::tPoint{.01, 0, 0});
        C::Domain different("flow", shifted, settings);
        CHECK_THROWS(different.ReadCheckpoint(file, resolved));
        CHECK(different.Time() == 0);
    }
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(file));
    MPI_Barrier(mpi.comm);
}
TEST_CASE("Cauchy real single-domain Simulation owns its accepted clock and restart epoch")
{
    const auto config = C::SimulationConfig::ResolveCollective({{"defaults", {{"numerics", {{"cfv", {{"maxOrder", 1}}}}}}}}, mpi.comm);
    C::Simulation simulation(config, mpi.comm, [](DNDS::index, const auto &, const auto &domainMPI)
                             { return CauchyTest::PeriodicQuads(domainMPI, 4); });
    simulation.Advance(.001);
    CHECK(simulation.Step() == 1);
    CHECK(simulation.Time() == simulation.GetDomain(0).Time());
    std::optional<R::Array<double>::ReadLease> held;
    if (mpi.rank == 0)
        held.emplace(simulation.GetDomain(0).Flow().Owned().Read());
    CHECK_THROWS(simulation.Advance(.001));
    CHECK(simulation.Step() == 1);
    held.reset();
    simulation.Advance(.001);
    CHECK(simulation.Step() == 2);
    int pid = int(getpid());
    MPI_Bcast(&pid, 1, MPI_INT, 0, mpi.comm);
    const auto file = (std::filesystem::temp_directory_path() / ("cauchy-simulation-" + std::to_string(pid) + ".h5")).string();
    simulation.GetDomain(0).WriteCheckpoint(file, config.Root());
    simulation.Advance(.001);
    simulation.ReadCheckpoint(file);
    CHECK(simulation.Step() == 2);
    CHECK(simulation.Time() == .002);
    simulation.Advance(.001);
    CHECK(simulation.Step() == 3);
    CHECK(simulation.Time() == .003);
    if (mpi.rank == 0)
        CHECK(std::filesystem::remove(file));
    MPI_Barrier(mpi.comm);
}
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
