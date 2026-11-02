#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "Euler/Gas.hpp"
#include "CartesianFixture.hpp"
#include <limits>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
namespace
{
    R::RowLayout Layout(int scenario)
    {
        return scenario == 0 ? R::RowLayout::Dense(0, 5) : scenario == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                         : R::RowLayout::TiledColMajor(0, 5, 4);
    }
    // Independent Euler matrix oracle: actual legacy Gas scheme-one solve,
    // density relaxation first, then both affine sensible-energy constraints.
    double LegacyAlpha(const C::FlowState &state, C::FlowState increment, const C::ThermalBounds &bounds, double relax, int power)
    {
        using Vector = Eigen::Matrix<double, 5, 1>;
        const Vector u = Eigen::Map<const Vector>(state.data());
        Vector delta = Eigen::Map<const Vector>(increment.data());
        const double density = delta[0] < 0 ? std::min(1., relax * (u[0] - bounds.densityFloor) / (-delta[0] * (1 + DNDS::smallReal))) : 1;
        delta *= density;
        const Vector next = u + delta;
        const auto energy = [](const Vector &value)
        { return value[4] - .5 * value.segment<3>(1).squaredNorm() / value[0]; };
        double alpha = 1;
        for (int constraint = 0; constraint < 2; ++constraint)
        {
            const double baseOld = constraint ? u[0] * bounds.specificEnergyAtMinimumTemperature : 0;
            const double baseNew = constraint ? next[0] * bounds.specificEnergyAtMinimumTemperature : 0;
            const double floor = constraint ? 0 : bounds.energyDensityFloor;
            const double oldEnergy = energy(u) - baseOld, newEnergy = energy(next) - baseNew;
            const double target = newEnergy < oldEnergy ? floor + (oldEnergy - floor) * (1 - relax) : floor;
            if (newEnergy < target)
                alpha = std::min(alpha, DNDS::Euler::Gas::IdealGasGetCompressionRatioPressure<3, 1, 5>(u, delta, target, baseOld, baseNew));
        }
        alpha *= density;
        return alpha < 1 ? std::pow(alpha, power) * (1 - 1e-5) : 1;
    }
}
TEST_CASE("Physical timestep PP agrees with Euler global compress-one bounds across layouts and active halos")
{
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        const auto mesh = CauchyTest::PeriodicQuads(mpi, scenario == 2 ? 2 : 4);
        C::DomainSettings settings;
        settings.numerics.temperatureFloor = .02;
        settings.numerics.pressureFloor = .008;
        C::FiniteVolume fv(mesh, settings, C::EquationSystem::Resolve(settings.equation));
        auto state = fv.Metrics().BuildField(Layout(scenario)), residual = fv.Metrics().BuildField(Layout(scenario));
        fv.Initialize(state, {}, {1, 0, 0, 0, 1});
        const C::FlowState mean{1, 0, 0, 0, 2.5};
        const std::array<C::FlowState, 4> deltas{{{-5, 0, 0, 0, 0}, {0, 0, 0, 0, -20}, {0, 20, 0, 0, 0}, {.25, 0, 0, 0, 0}}};
        const double dt = .05, scale = 1.5, relax = .8;
        {
            auto write = residual.Owned().Write();
            const auto out = write.View<R::UniformTableRows>();
            for (DNDS::index iCell = 0; iCell < out.RowCount(); ++iCell)
            {
                const auto global = fv.Metrics().Cells().LocalToGlobal(iCell);
                const auto &delta = deltas[global < 3 ? global : 3];
                for (int j = 0; j < 5; ++j)
                    out(iCell, j) = delta[j] * fv.Metrics().GetCellVol(iCell) / (dt * scale);
            }
        }
        auto binding = fv.Metrics().BindHalo(state);
        for (int power : {1, 2})
        {
            const auto meanVersion = state.Owned().Version(), rhsVersion = residual.Owned().Version();
            auto operation = binding.Begin();
            const auto result = fv.LimitPhysicalStep(state, residual, dt, relax, scale, power);
            operation.Wait();
            double expected = 1;
            for (const auto &delta : deltas)
                expected = std::min(expected, LegacyAlpha(mean, delta, fv.ThermalConstraints(), relax, power));
            CHECK(result.limitedCells == 3);
            CHECK(result.minimumAlpha == doctest::Approx(expected).epsilon(2e-13));
            CHECK(result.dt == doctest::Approx(dt * expected).epsilon(2e-13));
            CHECK_FALSE(result.growthLimited);
            CHECK(state.Owned().Version() == meanVersion);
            CHECK(residual.Owned().Version() == rhsVersion);
        }
        CHECK_THROWS(fv.LimitPhysicalStep(state, state, dt, relax, scale));
        CHECK_THROWS(fv.LimitPhysicalStep(state, residual, dt, 0, scale));
        CHECK_THROWS(fv.LimitPhysicalStep(state, residual, dt, relax, 0));
        CHECK_THROWS(fv.LimitPhysicalStep(state, residual, dt, relax, scale, 0));
        CHECK_THROWS(fv.LimitPhysicalStep(state, residual, mpi.rank == 0 ? std::numeric_limits<double>::quiet_NaN() : dt, relax, scale));
        if (mpi.size > 1)
            CHECK_THROWS(fv.LimitPhysicalStep(state, residual, mpi.rank == 0 ? 2 * dt : dt, relax, scale));
        {
            auto write = residual.Owned().Write();
            for (DNDS::index iCell = 0; iCell < residual.Owned().RowCount(); ++iCell)
                if (fv.Metrics().Cells().LocalToGlobal(iCell) == 0)
                    write.View()(iCell, 4) = std::numeric_limits<double>::infinity();
        }
        CHECK_THROWS(fv.LimitPhysicalStep(state, residual, dt, relax, scale));
        CHECK(state.Owned().IsValid());
        A::Fill(residual, 0.);
        const auto unchanged = fv.LimitPhysicalStep(state, residual, dt, relax, scale);
        CHECK(unchanged.dt == dt);
        CHECK(unchanged.minimumAlpha == 1);
        CHECK(unchanged.limitedCells == 0);
        {
            auto u = state.Owned().Write(), r = residual.Owned().Write();
            for (DNDS::index iCell = 0; iCell < state.Owned().RowCount(); ++iCell)
            {
                u.View<R::UniformTableRows>()(iCell, 0) = fv.ThermalConstraints().densityFloor;
                r.View<R::UniformTableRows>()(iCell, 0) = -1;
            }
        }
        const auto impossible = fv.LimitPhysicalStep(state, residual, dt, relax, scale);
        CHECK(impossible.dt == 0);
        CHECK(impossible.minimumAlpha == 0);
    }
}
TEST_CASE("Typed mean residual skips VR and filtering without modifying the normal warm guess")
{
    for (int scenario = 0; scenario < 3; ++scenario)
    {
        const auto mesh = CauchyTest::PeriodicQuads(mpi, scenario == 2 ? 2 : 4);
        C::DomainSettings settings;
        settings.execution.overlap = scenario != 1;
        settings.numerics.reconstruction = "variational";
        settings.numerics.limiter = "WBAP";
        settings.numerics.positivity = true;
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.flux = scenario == 1 ? "AUSMPWPlus" : "Roe";
        if (scenario == 2)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .02;
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(Layout(scenario)), residual = fv.Metrics().BuildField(Layout(scenario));
        {
            auto write = state.Owned().Write();
            for (DNDS::index iCell = 0; iCell < state.Owned().RowCount(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .1 * std::sin(6.283185307179586 * point[0]), .3, -.1, .05, 1 + .05 * std::cos(6.283185307179586 * point[1])});
                for (int j = 0; j < 5; ++j)
                    write.View<R::UniformTableRows>()(iCell, j) = value[j];
            }
        }
        auto regular = fv.BuildWorkspace(state);
        const auto &guess = fv.ReconstructionGuess(*regular).Owned();
        R::Array<double> seed(guess.Layout());
        {
            auto write = seed.Write();
            for (DNDS::index iCell = 0; iCell < seed.RowCount(); ++iCell)
                for (DNDS::index j = 0; j < seed.RowSize(iCell); ++j)
                    write.View<R::UniformTableRows>()(iCell, j) = .01;
        }
        fv.RestoreReconstructionGuess(*regular, seed);
        const auto guessVersion = guess.Version(), meanVersion = state.Owned().Version();
        auto work = fv.BuildMeanWorkspace(state);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        fv.EvaluateMean(state, residual, *work, engine, stream);
        CHECK(guess.Version() == guessVersion);
        CHECK(state.Owned().Version() == meanVersion);
        CHECK(fv.LastReconstructionIterations(*regular) == 0);
        auto referenceSettings = settings;
        referenceSettings.numerics.reconstruction = "constant";
        referenceSettings.numerics.limiter = "none";
        referenceSettings.numerics.positivity = false;
        referenceSettings.equation.transport = "inviscid";
        referenceSettings.equation.viscosity = 0;
        C::FiniteVolume reference(mesh, referenceSettings, C::EquationSystem::Resolve(referenceSettings.equation));
        auto referenceState = reference.Metrics().BuildField(Layout(scenario)), expected = reference.Metrics().BuildField(Layout(scenario));
        {
            const auto read = state.Owned().Read();
            auto write = referenceState.Owned().Write();
            for (DNDS::index iCell = 0; iCell < state.Owned().RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    write.View<R::UniformTableRows>()(iCell, j) = read.View<R::UniformTableRows>()(iCell, j);
        }
        auto referenceWork = reference.BuildWorkspace(referenceState);
        reference.Evaluate(referenceState, expected, *referenceWork, engine, stream);
        if (equation.Viscous())
        {
            // The real viscous residual differs on this nonconstant thermal
            // field, so matching the independent inviscid operator is active.
            auto viscousSettings = referenceSettings;
            viscousSettings.equation = settings.equation;
            C::FiniteVolume viscous(mesh, viscousSettings, equation);
            auto values = viscous.Metrics().BuildField(Layout(scenario)), output = viscous.Metrics().BuildField(Layout(scenario));
            {
                const auto read = state.Owned().Read();
                auto write = values.Owned().Write();
                for (DNDS::index iCell = 0; iCell < values.Owned().RowCount(); ++iCell)
                    for (int j = 0; j < 5; ++j)
                        write.View<R::UniformTableRows>()(iCell, j) = read.View<R::UniformTableRows>()(iCell, j);
            }
            auto viscousWork = viscous.BuildWorkspace(values);
            viscous.Evaluate(values, output, *viscousWork, engine, stream);
            double difference = 0;
            {
                const auto meanRHS = residual.Owned().Read(), viscousRHS = output.Owned().Read();
                for (DNDS::index iCell = 0; iCell < output.Owned().RowCount(); ++iCell)
                    difference += std::abs(meanRHS.View<R::UniformTableRows>()(iCell, 4) - viscousRHS.View<R::UniformTableRows>()(iCell, 4));
            }
            MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
            CHECK(difference > 1e-4);
        }
        const auto actual = residual.Owned().Read(), oracle = expected.Owned().Read(), rec = guess.Read();
        for (DNDS::index iCell = 0; iCell < state.Owned().RowCount(); ++iCell)
        {
            for (int j = 0; j < 5; ++j)
                CHECK(actual.View<R::UniformTableRows>()(iCell, j) == oracle.View<R::UniformTableRows>()(iCell, j));
            for (DNDS::index j = 0; j < guess.RowSize(iCell); ++j)
                CHECK(rec.View<R::UniformTableRows>()(iCell, j) == .01);
        }
        if (mpi.size > 1)
        {
            CHECK_THROWS([&]
                         {
                if (mpi.rank == 0) fv.BuildMeanWorkspace(state);
                else fv.BuildWorkspace(state); }());
            C::AuxiliaryRequest different;
            different.cellWaveJump = mpi.rank == 0;
            CHECK_THROWS(fv.BuildWorkspace(state, different));
        }
    }
}
TEST_CASE("Domain timestep growth follows committed dt through prepare discard and blocked commit")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (const std::string method : {"ssprk3", "bdf2"})
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.implicit.minIterations = 0;
        C::TimeSettings time;
        time.method = method;
        time.positivity.enabled = true;
        C::Domain domain("step", mesh, settings, time);
        CHECK(domain.PreviousTimeStep() == 0);
        CHECK(domain.LimitPhysicalStep(.01).dt == .01);
        domain.Advance(.001);
        const auto version = domain.Flow().Owned().Version();
        const auto limit = domain.LimitPhysicalStep(.1);
        CHECK(limit.dt == .002);
        CHECK(limit.minimumAlpha == 1);
        CHECK(limit.limitedCells == 0);
        CHECK(limit.growthLimited);
        CHECK(domain.Flow().Owned().Version() == version);
        CHECK(domain.Time() == .001);
        domain.PrepareStep(.004);
        domain.DiscardStep();
        CHECK(domain.PreviousTimeStep() == .001);
        CHECK(domain.LimitPhysicalStep(.1).dt == .002);
        domain.PrepareStep(.003);
        std::optional<R::Array<double>::ReadLease> read;
        if (mpi.rank == 0)
            read.emplace(domain.Flow().Owned().Read());
        CHECK_THROWS(domain.CommitStep());
        CHECK(domain.PreviousTimeStep() == .001);
        read.reset();
        domain.CommitStep();
        CHECK(domain.PreviousTimeStep() == .003);
        CHECK(domain.LimitPhysicalStep(.1).dt == .006);
        CHECK_THROWS(domain.LimitPhysicalStep(mpi.rank == 0 ? -1 : .1));
        if (mpi.size > 1)
            CHECK_THROWS(domain.LimitPhysicalStep(mpi.rank == 0 ? .1 : .2));
    }
}
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const auto result = context.run();
    MPI_Finalize();
    return result;
}
