#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Euler/Gas.hpp"
#include "CartesianFixture.hpp"
#include <limits>
#include <iomanip>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
static DNDS::MPIInfo mpi;
namespace
{
    void CheckFlatMeanIntegral(std::array<double, 2> error, const char *label)
    {
        // These fixtures have flat faces and cellwise constant integrands.
        // A zero-polynomial trial retains its parent's quadrature, whereas an
        // independent constant FV operator now uses midpoint. Their sums are
        // equal mathematically, but the floating-point operation counts differ.
        MPI_Allreduce(MPI_IN_PLACE, error.data(), 2, MPI_DOUBLE, MPI_MAX, mpi.comm);
        if (!mpi.rank)
            std::cout << "MEAN_QUADRATURE_COMPARISON " << label << " max_difference=" << std::setprecision(17) << error[0]
                      << " max_magnitude=" << error[1] << std::endl;
        CHECK((std::isfinite(error[0]) && std::isfinite(error[1]) &&
               error[0] <= 128 * std::numeric_limits<double>::epsilon() * std::max(1., error[1])));
    }
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
#ifdef DNDS_USE_CANTERA
TEST_CASE("Equation mean trial and thermal dt preserve all block layouts and the ordinary warm guess")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    C::DomainSettings settings;
    settings.equation.eos = "idealGasMixture";
    settings.equation.transport = "NewtonianFourier";
    settings.equation.source = "chemical";
    settings.equation.mixture.mechanismFile = "h2o2.yaml";
    settings.equation.mixture.rho0 = 2;
    settings.equation.mixture.U0 = 200;
    settings.equation.mixture.L0 = .04;
    settings.equation.mixture.T0 = 300;
    settings.equation.rans.model = "SA";
    settings.equation.rans.viscosityReference = .01;
    settings.numerics.faceFallback = false;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.limiter = "WBAP";
    settings.numerics.characteristicLimiter = false;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.temperatureFloor = 1;
    settings.numerics.densityFloor = .1;
    const auto setup = C::EquationSetup::Resolve(settings.equation);
    const auto &equation = setup.equation;
    C::FiniteVolume fv(mesh, settings, equation);
    A::DofBundle fields(fv.Metrics().Space());
    fields.Add("flow", fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 5, {8, 1})));
    fields.Add("rans", fv.Metrics().BuildField(1));
    fields.Add("species", fv.Metrics().BuildField(R::RowLayout::TiledColMajor(0, equation.Mixture().SpeciesNumber() - 1, 4)));
    auto state = C::EquationDofs::Separate(equation.Schema(), std::move(fields));
    fv.InitializeConstant(state, {1.3, .1, -.2, .05, 1}, {{"H2", .03}, {"O2", .23}}, 4, setup.workers, {{"rhoNuTilde", .02}});
    {
        auto evaluator = equation.MakePointEvaluator(setup.workers->Acquire(), {});
        auto fractions = equation.IndependentMassFractions({{"H2", .03}, {"O2", .23}});
        auto write = state.WriteOwned();
        const auto flow = write.Block("flow"), species = write.Block("species");
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto center = fv.Metrics().GetCellCent(iCell);
            const double rho = 1.3 + .05 * std::sin(6.283185307179586 * center[0]);
            const double temperature = 4 + .1 * std::cos(6.283185307179586 * center[1]);
            auto rhoY = evaluator.SpeciesInput();
            for (int k = 0; k < rhoY.nSpecies; ++k)
                species(iCell, k) = rhoY[k] = rho * fractions[k];
            const auto value = evaluator.FromPrimitive({rho, .1, -.2, .05, 1}, rhoY, temperature);
            for (int j = 0; j < 5; ++j)
                flow(iCell, j) = value[j];
        }
    }
    auto rhs = C::EquationDofOperations::NewVector(state);
    auto regular = fv.BuildEquationWorkspace(state, setup.workers);
    const auto &guess = fv.ReconstructionGuess(*regular);
    auto seed = C::EquationDofOperations::NewVector(guess);
    C::EquationDofOperations::Fill(seed, .0001);
    fv.RestoreReconstructionGuess(*regular, seed);
    const C::EquationDofOperations::Binding before(state), warm(guess);
    auto meanWork = fv.BuildEquationMeanWorkspace(state, setup.workers);
    R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
    const auto stream = engine.CreateCPUStream();
    fv.EvaluateMean(state, rhs, *meanWork, engine, stream);
    CHECK_NOTHROW(before.Check(state, true));
    CHECK_NOTHROW(warm.Check(guess, true));
    // Compare the mean trial against a separate inviscid constant operator,
    // retaining the same material and chemistry. SA adds no isotropic stress.
    auto referenceSettings = settings;
    referenceSettings.equation.rans.model = "none";
    referenceSettings.equation.transport = "inviscid";
    referenceSettings.numerics.reconstruction = "constant";
    referenceSettings.numerics.limiter = "none";
    const auto referenceEquation = C::EquationSystem::FromIdealGasMixture(equation.Mixture(), false);
    C::FiniteVolume reference(mesh, referenceSettings, referenceEquation);
    auto referenceState = C::EquationDofs::Coupled(referenceEquation.Schema(), reference.Metrics().BuildField(4 + equation.Mixture().SpeciesNumber()));
    {
        const auto read = state.ReadOwned();
        auto write = referenceState.WriteOwned();
        for (const auto *name : {"flow", "species"})
        {
            const auto source = read.Block(name);
            const auto destination = write.Block(name);
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (DNDS::index j = 0; j < source.ComponentNumber(); ++j)
                    destination(iCell, j) = source(iCell, j);
        }
    }
    auto expected = C::EquationDofOperations::NewVector(referenceState);
    auto referenceWork = reference.BuildEquationWorkspace(referenceState, setup.workers);
    reference.Evaluate(referenceState, expected, *referenceWork, engine, stream);
    {
        const auto a = rhs.ReadOwned(), b = expected.ReadOwned();
        std::array<double, 2> error{};
        for (const auto *name : {"flow", "species"})
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (DNDS::index j = 0; j < a.Block(name).ComponentNumber(); ++j)
                {
                    const double left = a.Block(name)(iCell, j), right = b.Block(name)(iCell, j);
                    const double difference = std::abs(left - right);
                    error[0] = std::isfinite(difference) ? std::max(error[0], difference) : std::numeric_limits<double>::infinity();
                    error[1] = std::max({error[1], std::abs(left), std::abs(right)});
                }
        CheckFlatMeanIntegral(error, "mixture");
    }
    fv.InitializeConstant(state, {1.3, .1, -.2, .05, 1}, {{"H2", .03}, {"O2", .23}}, 4, setup.workers, {{"rhoNuTilde", .02}});
    const C::EquationDofOperations::Binding boundBefore(state);
    const C::EquationThermalPolicy policy(equation, .1, 1, 0, 1e-12, setup.workers->Acquire());
    const auto thermal = policy.View();
    std::vector<double> old(size_t(state.ComponentNumber())), delta(old.size());
    {
        const auto read = state.ReadOwned();
        if (fv.Metrics().NumCell())
        {
            const auto layout = read.ComponentLayout();
            const auto values = layout.View();
            for (DNDS::index j = 0; j < state.ComponentNumber(); ++j)
                old[j] = values(0, j);
        }
    }
    // This fixture has at least one owner per tested rank. All states are
    // identical; known residual increments isolate density, energy, kinetic
    // energy and a RANS-only change that must not limit the physical timestep.
    REQUIRE(fv.Metrics().NumCell() > 0);
    const double dt = .05, scale = 1.5, relax = .8;
    double minimum = 1;
    const auto referenceEnergy = [&](const auto &value, int c)
    {
        const auto *energy = thermal.speciesEnergies[c];
        double result = value[0] * energy[thermal.independent];
        for (DNDS::index k = 0; k < thermal.independent; ++k)
            result += value[thermal.speciesFirst + k] * (energy[k] - energy[thermal.independent]);
        return result;
    };
    {
        auto write = rhs.WriteOwned();
        const auto outputLayout = write.ComponentLayout();
        const auto output = outputLayout.View();
        for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto pattern = fv.Metrics().Cells().LocalToGlobal(iCell) % 4;
            std::fill(delta.begin(), delta.end(), 0);
            if (pattern == 0)
                for (DNDS::index j = 0; j < state.ComponentNumber(); ++j)
                    delta[j] = -2 * old[j];
            if (pattern == 1)
                delta[4] = -20 * (old[4] - referenceEnergy(old, 1));
            if (pattern == 2)
                delta[1] = 30;
            if (pattern == 3)
                delta[5] = -1e20; // SA, after flow5, outside thermal criteria.
            for (DNDS::index j = 0; j < state.ComponentNumber(); ++j)
                output(iCell, j) = delta[j] * fv.Metrics().GetCellVol(iCell) / (dt * scale);
            const double density = delta[0] < 0 ? std::min(1., relax * (old[0] - thermal.densityFloor) / (-delta[0] * (1 + DNDS::smallReal))) : 1;
            Eigen::Matrix<double, 5, 1> u, du;
            std::vector<double> next = old;
            for (DNDS::index j = 0; j < state.ComponentNumber(); ++j)
                next[j] += density * delta[j];
            for (int j = 0; j < 5; ++j)
            {
                u[j] = old[j];
                du[j] = density * delta[j];
            }
            double energy = 1;
            for (int c = 0; c < 2; ++c)
            {
                const double baseOld = referenceEnergy(old, c), baseNew = referenceEnergy(next, c);
                const double a = u[4] - .5 * u.segment<3>(1).squaredNorm() / u[0] - baseOld;
                const Eigen::Matrix<double, 5, 1> trial = u + du;
                const double b = trial[4] - .5 * trial.segment<3>(1).squaredNorm() / trial[0] - baseNew;
                const double target = b < a ? thermal.floors[c] + (a - thermal.floors[c]) * (1 - relax) : thermal.floors[c];
                if (b < target)
                    energy = std::min(energy, DNDS::Euler::Gas::IdealGasGetCompressionRatioPressure<3, 1, 5>(u, du, target, baseOld, baseNew));
            }
            const double fraction = density * energy;
            minimum = std::min(minimum, fraction < 1 ? fraction * (1 - 1e-5) : 1);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
    std::vector<R::TransferOperation> halo;
    for (DNDS::index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
        halo.push_back(fv.Metrics().BindHalo(state.PhysicalField(iField)).Begin());
    const auto limit = fv.LimitPhysicalStep(state, rhs, thermal, dt, relax, scale);
    for (auto &operation : halo)
        operation.Wait();
    CHECK(limit.minimumAlpha == doctest::Approx(minimum).epsilon(2e-12));
    CHECK(limit.dt == doctest::Approx(dt * minimum).epsilon(2e-12));
    CHECK(limit.limitedCells == 12);
    CHECK_NOTHROW(boundBefore.Check(state, true));
    CHECK_NOTHROW(warm.Check(guess, true));
    auto wrong = thermal;
    ++wrong.speciesFirst;
    CHECK_THROWS(fv.LimitPhysicalStep(state, rhs, wrong, dt, relax, scale));
    wrong = thermal;
    wrong.speciesEnergies[0] = nullptr;
    CHECK_THROWS(fv.LimitPhysicalStep(state, rhs, wrong, dt, relax, scale));
}
#endif

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
        std::array<double, 2> error{};
        for (DNDS::index iCell = 0; iCell < state.Owned().RowCount(); ++iCell)
        {
            for (int j = 0; j < 5; ++j)
            {
                const double left = actual.View<R::UniformTableRows>()(iCell, j), right = oracle.View<R::UniformTableRows>()(iCell, j);
                const double difference = std::abs(left - right);
                error[0] = std::isfinite(difference) ? std::max(error[0], difference) : std::numeric_limits<double>::infinity();
                error[1] = std::max({error[1], std::abs(left), std::abs(right)});
            }
            for (DNDS::index j = 0; j < guess.RowSize(iCell); ++j)
                CHECK(rec.View<R::UniformTableRows>()(iCell, j) == .01);
        }
        CheckFlatMeanIntegral(error, "flow5");
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
