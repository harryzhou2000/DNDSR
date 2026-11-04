#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Physics/EquationSource.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Euler/SourceTermContributor.hpp"
#include <Eigen/LU>
#include <cstdlib>
#include <numeric>
#include <iomanip>

namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
namespace P = DNDS::Physics;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    std::string Mechanism()
    {
        const auto *directory = std::getenv("DNDS_MECH_PATH");
        return directory ? std::string(directory) + "/h2o2.yaml" : "h2o2.yaml";
    }
    struct Fixture
    {
        const P::ReferenceScales scales{1.225, 340, 2, 288.15};
        P::Chemistry::ChemicalSource prototype{Mechanism(), "", scales.U0, scales.rho0};
        P::Thermodynamics::IdealGasMixture material{prototype, scales};
        C::EquationSystem equation;
        P::Chemistry::ChemicalWorkerPool workers{prototype, 1};
        const int species = prototype.nSpecies(), components = species + 4;
        explicit Fixture(bool viscous = false) : equation(C::EquationSystem::FromIdealGasMixture(material, viscous)) {}
        std::vector<double> Fractions(int sample) const
        {
            std::vector<double> fractions(species, 1e-6);
            fractions[0] = .027 + sample * 1e-5;
            fractions[3] = .22 - sample * 2e-5;
            fractions[5] = .02 + sample * 1e-5;
            fractions.back() = 0;
            fractions.back() = 1 - std::accumulate(fractions.begin(), fractions.end(), 0.);
            return fractions;
        }
        Eigen::VectorXd Point(int sample, double temperature = 1400)
        {
            const auto fractions = Fractions(sample);
            const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), species};
            const double rho = .8 + sample * .003;
            const double pressure = rho * scales.rho0 * prototype.mixtureR(Y) * temperature;
            Eigen::VectorXd u(components);
            u[0] = rho;
            u[1] = rho * .2;
            u[2] = rho * -.04;
            u[3] = rho * .03;
            u[4] = rho * prototype.mixtureIntEnergy(temperature, Y, pressure) / scales.SpecificEnergy() + .5 * u.segment<3>(1).squaredNorm() / rho;
            for (int k = 0; k < species - 1; ++k)
                u[5 + k] = rho * fractions[k];
            return u;
        }
        C::EquationDofs State(const A::DofSpace &space, int layout)
        {
            if (layout == 0)
                return C::EquationDofs::Coupled(equation.Schema(), A::DofField<double>(space, R::RowLayout::Dense(0, components, components + 3)));
            A::DofBundle fields(space);
            if (layout == 1)
            {
                fields.Add("flow", A::DofField<double>(space, R::RowLayout::ColMajor(0, 5, {16, 1})));
                fields.Add("species", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, species - 1, 32)));
                return C::EquationDofs::Separate(equation.Schema(), std::move(fields));
            }
            // Physical species-first storage must not change logical ordering.
            fields.Add("packed", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, components, 32)));
            return C::EquationDofs(equation.Schema(), std::move(fields), {{"flow", "packed", species - 1}, {"species", "packed", 0}});
        }
        void Fill(C::EquationDofs &state)
        {
            auto output = state.WriteOwned();
            const auto flow = output.Block("flow"), speciesValues = output.Block("species");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                const auto point = Point(int(iCell) + mpi.rank, 1300 + iCell * 2);
                for (int j = 0; j < 5; ++j)
                    flow(iCell, j) = point[j];
                for (int k = 0; k < species - 1; ++k)
                    speciesValues(iCell, k) = point[5 + k];
            }
        }
    };
}

TEST_CASE("Mixture source point matches the Euler source and all conservative derivative columns")
{
    Fixture fixture;
    auto evaluate = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {false, false, false, true, true});
    const auto u = fixture.Point(2);
    const auto acousticOnly = evaluate.Evaluate({u[0], u[1], u[2], u[3], u[4]}, {u.data() + 5, fixture.species - 1}, 0, {});
    CHECK(acousticOnly.chemicalSource.nSpecies == 0);
    CHECK(acousticOnly.sourceJacobian.data == nullptr);
    auto point = evaluate.Evaluate({u[0], u[1], u[2], u[3], u[4]}, {u.data() + 5, fixture.species - 1}, 900 / fixture.scales.T0);
    Eigen::MatrixXd derivative(fixture.species - 1, fixture.components);
    for (int k = 0; k < derivative.rows(); ++k)
        for (int j = 0; j < derivative.cols(); ++j)
            derivative(k, j) = point.sourceJacobian(k, j);

    using Traits = DNDS::Euler::EulerModelTraits<DNDS::Euler::NS_EX_3D>;
    auto legacyPool = std::make_shared<std::vector<P::Chemistry::ChemicalSource>>();
    legacyPool->push_back(std::move(*fixture.prototype.clone()));
    DNDS::Euler::EulerEvaluatorSettings<DNDS::Euler::NS_EX_3D>::IdealGasProperty settings;
    settings.rho0 = fixture.scales.rho0;
    settings.U0 = fixture.scales.U0;
    settings.L0 = fixture.scales.L0;
    settings.T0 = fixture.scales.T0;
    // Preserve two untouched RANS slots between energy and species in Euler.
    const int nVars = fixture.components + 2;
    DNDS::Euler::ChemicalContributor<DNDS::Euler::NS_EX_3D> legacy(legacyPool, settings, .7, nVars);
    Traits::TU oldU(nVars), source(nVars);
    oldU.head<5>() = u.head<5>();
    oldU.segment<2>(5).setConstant(.12);
    oldU.tail(fixture.species - 1) = u.tail(fixture.species - 1);
    Traits::TJacobianU jacobian(nVars, nVars);
    Traits::TDiffU gradient(3, nVars);
    gradient.setZero();
    DNDS::Euler::SourceCellAux aux;
    aux.T = point.thermo.temperature;
    aux.p = point.thermo.pressure;
    aux.pPhys = aux.p * fixture.scales.Pressure();
    aux.reactiveSplitCoupledScale = .6;
    source.setZero();
    jacobian.setZero();
    legacy.evaluate(source, jacobian, oldU, gradient, {}, aux, 0, 0, 2);
    CHECK(source.head<7>().isZero(0));
    CHECK(jacobian.topRows(7).isZero(0));
    CHECK(jacobian.middleCols(5, 2).isZero(0));
    for (int k = 0; k < fixture.species - 1; ++k)
    {
        CHECK(source[7 + k] == doctest::Approx(.42 * point.chemicalSource[k]).epsilon(5e-10).scale(1e-9));
        for (int j = 0; j < fixture.components; ++j)
            CHECK(jacobian(7 + k, j < 5 ? j : j + 2) == doctest::Approx(-.42 * derivative(k, j)).epsilon(5e-10).scale(1e-9));
    }
    // An independent conservative perturbation includes UV/composition chain
    // rules, not merely a fixed-temperature kinetics derivative.
    for (int j = 0; j < fixture.components; ++j)
    {
        CAPTURE(j);
        const double h = 1e-6 * std::max(std::abs(u[j]), 1e-3);
        auto sample = [&](double sign)
        {
            Eigen::VectorXd shifted = u;
            shifted[j] += sign * h;
            const auto value = evaluate.Evaluate({shifted[0], shifted[1], shifted[2], shifted[3], shifted[4]}, {shifted.data() + 5, fixture.species - 1});
            Eigen::VectorXd result(fixture.species - 1);
            for (int k = 0; k < result.size(); ++k)
                result[k] = value.chemicalSource[k];
            return result;
        };
        const Eigen::VectorXd finiteDifference = (sample(1) - sample(-1)) / (2 * h);
        CHECK((finiteDifference - derivative.col(j)).lpNorm<Eigen::Infinity>() <= 3e-5 * std::max(1., derivative.col(j).lpNorm<Eigen::Infinity>()));
    }
}

TEST_CASE("Owned chemical sources honor separate layouts, stale temperature guesses and ghost independence")
{
    Fixture fixture;
    const I rows = mpi.size > 1 && mpi.rank == mpi.size - 1 ? 0 : 35;
    A::DofSpace space(size_t(rows), 2, "local chemical sources");
    std::vector<double> canonicalMatrix(size_t(rows) * fixture.components * fixture.components);
    for (int layout = 0; layout < 3; ++layout)
    {
        CAPTURE(layout);
        auto state = fixture.State(space, layout), source = fixture.State(space, (layout + 1) % 3);
        fixture.Fill(state);
        C::DerivedFields oldTemperature(space, {C::DerivedQuantity::Temperature});
        C::RefreshDerivedFields(state, fixture.equation, oldTemperature, fixture.workers.Acquire());
        // Deliberately invalidate freshness without changing values.
        {
            auto write = state.WriteOwned();
        }
        CHECK_FALSE(oldTemperature.Current(state, fixture.equation));
        auto activeGhost = state.PhysicalField(0).Ghost().Write();
        R::Array<double> jacobian(R::RowLayout::ColMajor(rows, fixture.components * fixture.components, {16, 1}));
        C::EvaluateChemicalSource(fixture.equation, state, source, fixture.workers.Acquire(), &jacobian, .7, &oldTemperature);
        activeGhost.CancelUnmodified();
        const auto read = source.ReadOwned();
        const auto flow = read.Block("flow").owned, speciesValues = read.Block("species").owned;
        const auto matrixRead = jacobian.Read();
        const auto matrix = matrixRead.View<R::UniformTableRows>();
        const auto guessRead = oldTemperature.ReadTemperatureGuess(state, fixture.equation);
        const auto oldT = guessRead.Field(C::DerivedQuantity::Temperature);
        double relativeJacobianError = 0, observedWorst = 0, expectedWorst = 0, layoutError = 0;
        for (I iCell = 0; iCell < rows; ++iCell)
        {
            std::vector<double> fractions(fixture.species);
            const double temperature = 1300 + iCell * 2;
            const auto point = fixture.Point(int(iCell) + mpi.rank, temperature);
            P::Chemistry::RepairMassFractions(point[0], {point.data() + 5, fixture.species - 1}, {fractions.data(), fixture.species});
            const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), fixture.species};
            double speedSquared = 0;
            for (int d = 0; d < 3; ++d)
                speedSquared += DNDS::sqr(point[1 + d] / point[0]);
            // Compare derivatives at the same recovered conservative state,
            // rather than the analytic generator's rounded temperature. The
            // independent point test above also checks all columns by FD.
            const double recoveredT = fixture.prototype.temperatureFromUV((point[4] / point[0] - .5 * speedSquared) * fixture.scales.SpecificEnergy(),
                                                                          1 / (point[0] * fixture.scales.rho0), Y, oldT(iCell, 0) * fixture.scales.T0);
            CHECK(recoveredT == doctest::Approx(temperature).epsilon(1e-11));
            P::Thermodynamics::Properties thermo;
            const double pressure = point[0] * fixture.scales.rho0 * fixture.prototype.mixtureR(Y) * recoveredT;
            fixture.prototype.EvaluateProperties(recoveredT, pressure, Y, thermo);
            const double sourceT = thermo.temperature / fixture.scales.T0 * fixture.scales.T0;
            const double sourceP = thermo.pressure / fixture.scales.Pressure() * fixture.scales.Pressure();
            std::vector<double> rates(fixture.species), rawJacobian(fixture.species * fixture.components);
            P::Chemistry::JacobianBufferView expected{rawJacobian.data(), fixture.species, fixture.components, fixture.species};
            fixture.prototype.productionRatesAndJacobian(sourceT, sourceP, point[0], point[4], point[1], point[2], point[3], 4, Y, {rates.data(), fixture.species}, expected);
            for (int j = 0; j < 5; ++j)
            {
                CHECK(flow(iCell, j) == 0);
                for (int column = 0; column < fixture.components; ++column)
                    CHECK(matrix(iCell, j + fixture.components * column) == 0);
            }
            for (int k = 0; k < fixture.species - 1; ++k)
            {
                const double scaling = .7 * fixture.prototype.molecularWeights()[k] / fixture.scales.SourceRate();
                CHECK(speciesValues(iCell, k) == doctest::Approx(scaling * rates[k]).epsilon(5e-10).scale(1e-9));
                for (int j = 0; j < fixture.components; ++j)
                {
                    const double observed = matrix(iCell, 5 + k + fixture.components * j), reference = scaling * expected(k, j);
                    const double error = std::abs(observed - reference) / (1e-9 + std::max(std::abs(observed), std::abs(reference)));
                    if (error > relativeJacobianError)
                    {
                        relativeJacobianError = error;
                        observedWorst = observed;
                        expectedWorst = reference;
                    }
                }
            }
            for (int j = 0; j < fixture.components * fixture.components; ++j)
            {
                const auto offset = size_t(iCell) * fixture.components * fixture.components + j;
                if (layout == 0)
                    canonicalMatrix[offset] = matrix(iCell, j);
                else
                    layoutError = std::max(layoutError, std::abs(matrix(iCell, j) - canonicalMatrix[offset]));
            }
        }
        INFO(std::setprecision(17) << "worst observed=" << observedWorst << ", expected=" << expectedWorst);
        CAPTURE(relativeJacobianError);
        // Cantera's numerical rate derivatives vary under roundoff in the
        // independent UV/property recovery. The full conservative FD test
        // above supplies a second physical oracle. Layouts themselves must
        // produce exactly the same complete block, including zero flow rows.
        CHECK(relativeJacobianError < 1e-6);
        CHECK(layoutError == 0);
    }
}

TEST_CASE("Mixture primitive and conservative gradients include composition energy and all kinetic terms")
{
    namespace T = P::Thermodynamics;
    Fixture fixture;
    for (const double temperature : {400., 1400.})
    {
        CAPTURE(temperature);
        const auto u = fixture.Point(0, temperature);
        const C::FlowState flow{u[0], u[1], u[2], u[3], u[4]};
        const auto velocity = T::ConservativeVelocity<3>(flow);
        {
            auto materialPoint = fixture.material.Borrow(fixture.workers.Acquire());
            const auto rhoT = materialPoint.FromRhoT(flow[0], temperature / fixture.scales.T0, {u.data() + 5, fixture.species - 1});
            CHECK(T::ConservativeTotalEnergyDensity<3>(flow[0], velocity, rhoT.thermo.internalEnergy) == doctest::Approx(flow[4]).epsilon(2e-12));
            const auto rhoP = materialPoint.FromRhoP(flow[0], rhoT.thermo.pressure, {u.data() + 5, fixture.species - 1});
            CHECK(rhoP.thermo.temperature == doctest::Approx(temperature / fixture.scales.T0).epsilon(2e-12));
            CHECK(T::ConservativeSpecificInternalEnergy<3>(flow) == doctest::Approx(rhoP.thermo.internalEnergy).epsilon(2e-12));
        }
        if (temperature == 400)
            CHECK(T::ConservativeSpecificInternalEnergy<3>(flow) < 0);
        auto evaluate = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {false, false, true});
        const auto point = evaluate.Evaluate(flow, {u.data() + 5, fixture.species - 1});
        const C::FlowState direction{.002, -.001, .003, -.004, .007};
        std::vector<double> speciesDerivative(size_t(2 * (fixture.species - 1)), 777);
        for (int k = 0; k < fixture.species - 1; ++k)
            speciesDerivative[2 * k] = .0001 * (k - 4);
        const R::RowSpan<const double> dSpecies{speciesDerivative.data(), fixture.species - 1, 2};
        const auto derivative = C::PreparePrimitiveDerivative(flow, direction, point, dSpecies);
        constexpr double step = 1e-4;
        const auto sample = [&](double sign)
        {
            Eigen::VectorXd shifted = u;
            for (int j = 0; j < 5; ++j)
                shifted[j] += sign * step * direction[j];
            for (int k = 0; k < fixture.species - 1; ++k)
                shifted[5 + k] += sign * step * dSpecies[k];
            const auto property = evaluate.Evaluate({shifted[0], shifted[1], shifted[2], shifted[3], shifted[4]},
                                                    {shifted.data() + 5, fixture.species - 1}, temperature / fixture.scales.T0);
            return std::array<double, 6>{shifted[0], shifted[1] / shifted[0], shifted[2] / shifted[0], shifted[3] / shifted[0],
                                         property.thermo.temperature, property.thermo.pressure};
        };
        const auto plus = sample(1), minus = sample(-1);
        for (int j = 0; j < 5; ++j)
            CHECK(derivative.rhoVelocityTemperature[j] == doctest::Approx((plus[j] - minus[j]) / (2 * step)).epsilon(3e-6).scale(1e-6));
        CHECK(derivative.pressure == doctest::Approx((plus[5] - minus[5]) / (2 * step)).epsilon(3e-6).scale(1e-6));
    }
}

TEST_CASE("Mixture viscous transport matches Euler heat and species enthalpy flux with effective coefficients")
{
    Fixture fixture(true);
    const auto u = fixture.Point(1);
    const C::FlowState flow{u[0], u[1], u[2], u[3], u[4]};
    auto evaluator = fixture.equation.MakePointEvaluator(fixture.workers.Acquire(), {true, true, true});
    const auto point = evaluator.Evaluate(flow, {u.data() + 5, fixture.species - 1});
    C::FlowGradient gradient;
    Eigen::MatrixXd speciesGradient(fixture.species - 1, 3);
    Eigen::Matrix<double, 3, Eigen::Dynamic> oldGradient(3, fixture.components);
    for (int d = 0; d < 3; ++d)
    {
        for (int j = 0; j < 5; ++j)
            oldGradient(d, j) = gradient[j][d] = .002 * (j + 1) * (d + 1);
        for (int k = 0; k < fixture.species - 1; ++k)
            oldGradient(d, 5 + k) = speciesGradient(k, d) = .0001 * (k - 4) * (d + 1);
    }
    auto legacyPool = std::make_shared<std::vector<P::Chemistry::ChemicalSource>>();
    legacyPool->push_back(std::move(*fixture.prototype.clone()));
    DNDS::Euler::EulerEvaluatorSettings<DNDS::Euler::NS_EX_3D>::IdealGasProperty settings;
    settings.rho0 = fixture.scales.rho0;
    settings.U0 = fixture.scales.U0;
    settings.L0 = fixture.scales.L0;
    settings.T0 = fixture.scales.T0;
    DNDS::Euler::PhysicsProperties<DNDS::Euler::NS_EX_3D> legacy(settings);
    legacy.setChemicalSourcePool(legacyPool);
    const auto thermal = legacy.conservativeThermal(u, point.thermo.temperature, 1e-12);
    Eigen::Matrix<double, 3, Eigen::Dynamic> primitiveGradient;
    DNDS::Euler::Gas::GradientCons2Prim_IdealGas<3>(u, oldGradient, primitiveGradient, thermal.gammaEq, legacy.mixtureBaseInternalRhoESpecies());
    C::EquationTransportScratch scratch(fixture.species);
    const C::Vector3 normal{2. / 3, -1. / 3, 2. / 3};
    for (bool wall : {false, true})
    {
        CAPTURE(wall);
        C::EquationTransportOptions options;
        options.adiabatic = options.impermeable = wall;
        options.turbulentViscosity = .3 * point.transport.viscosity;
        options.turbulentConductivity = .2 * point.transport.conductivity;
        options.turbulentSpeciesDiffusivity = 1e-7;
        std::vector<double> output(size_t(2 * (fixture.species - 1)), 777);
        R::RowSpan<double> speciesFlux{output.data(), fixture.species - 1, 2};
        const P::Chemistry::ConstSpeciesBufferView rawRhoY{u.data() + 5, fixture.species - 1};
        const auto flux = C::PrepareViscousFlux(flow, gradient, point, rawRhoY, speciesGradient, normal, speciesFlux, scratch, options);
        // Legacy flow slots overwrite, but species diffusion is additive;
        // its production caller zeroes those slots before invoking the helper.
        Eigen::VectorXd expected = Eigen::VectorXd::Zero(fixture.components);
        DNDS::Euler::PhysicsProperties<DNDS::Euler::NS_EX_3D>::MixtureAveragedDiffusionBuffers buffers;
        const Eigen::Vector3d oldNormal(normal[0], normal[1], normal[2]);
        const double totalViscosity = point.transport.viscosity + options.turbulentViscosity;
        legacy.viscousFluxIdealGas(thermal.T, thermal.p, u, oldGradient, primitiveGradient, oldNormal, wall, wall,
                                   thermal.gammaEq, thermal.gamma, totalViscosity, options.turbulentViscosity / totalViscosity, false,
                                   point.transport.conductivity + options.turbulentConductivity, point.thermo.cp,
                                   options.turbulentSpeciesDiffusivity, buffers, expected);
        for (int j = 0; j < 5; ++j)
            CHECK(flux[j] == doctest::Approx(expected[j]).epsilon(3e-10).scale(1e-20));
        for (int k = 0; k < fixture.species - 1; ++k)
        {
            CHECK(speciesFlux[k] == doctest::Approx(expected[5 + k]).epsilon(3e-10).scale(1e-20));
            CHECK(output[2 * k + 1] == 777);
            if (wall)
                CHECK(speciesFlux[k] == 0);
        }
        if (!wall)
        {
            // Independently include the dependent species to test total mass
            // balance, rather than defining its flux as minus the output sum.
            double sumRaw = 0, sumGradient = 0, sumFractions = 0;
            for (int k = 0; k < fixture.species - 1; ++k)
            {
                const double Y = u[5 + k] / u[0];
                double normalDerivative = 0;
                for (int d = 0; d < 3; ++d)
                    normalDerivative += (speciesGradient(k, d) - Y * gradient[0][d]) / u[0] * normal[d];
                sumRaw -= u[0] * (point.diffusivity[k] + options.turbulentSpeciesDiffusivity) * normalDerivative;
                sumGradient += normalDerivative;
                sumFractions += Y;
            }
            const double lastRaw = u[0] * (point.diffusivity[fixture.species - 1] + options.turbulentSpeciesDiffusivity) * sumGradient;
            sumRaw += lastRaw;
            const double lastFlux = -(lastRaw - (1 - sumFractions) * sumRaw);
            double total = lastFlux, scale = std::abs(lastFlux);
            for (int k = 0; k < fixture.species - 1; ++k)
            {
                total += speciesFlux[k];
                scale += std::abs(speciesFlux[k]);
            }
            CHECK(scale > 0);
            CHECK(std::abs(total) < 1e-12 * scale);
        }
    }
}

TEST_CASE("Chemical source rejects output reservations and Jacobian aliases before changing values")
{
    Fixture fixture;
    A::DofSpace space(2, 0, "chemical reservation failures");
    auto state = C::EquationDofs::Coupled(fixture.equation.Schema(), A::DofField<double>(space,
                                                                                         R::RowLayout::Dense(0, fixture.components, fixture.components * fixture.components)));
    auto source = fixture.State(space, 1);
    fixture.Fill(state);
    C::EquationDofOperations::Fill(source, 17);
    R::Array<double> jacobian(R::RowLayout::Dense(2, fixture.components * fixture.components));
    const auto jacobianVersion = jacobian.Version(), sourceVersion = source.PhysicalField(0).Owned().Version();
    {
        const auto occupied = source.PhysicalField(1).Owned().Read();
        CHECK_THROWS(C::EvaluateChemicalSource(fixture.equation, state, source, fixture.workers.Acquire(), &jacobian));
    }
    CHECK(jacobian.Version() == jacobianVersion);
    CHECK(source.PhysicalField(0).Owned().Version() == sourceVersion);
    auto aliasedJacobian = R::Array<double>::FromStorage(R::RowLayout::Dense(2, fixture.components * fixture.components), state.PhysicalField(0).Owned().Storage());
    CHECK_THROWS(C::EvaluateChemicalSource(fixture.equation, state, source, fixture.workers.Acquire(), &aliasedJacobian));
    const auto read = source.ReadOwned();
    CHECK(read.Block("flow")(0, 0) == 17);
    CHECK(read.Block("species")(1, 3) == 17);
}

TEST_CASE("Actual conservative chemical sources advance a separate-field implicit reactor toward Cantera")
{
    Fixture fixture;
    const A::DofSpace space(1, 0, "homogeneous chemical stage");
    const auto initial = fixture.Point(0);
    constexpr double totalTime = 1e-5;
    const auto set = [&](C::EquationDofs &state, const Eigen::VectorXd &u)
    {
        auto write = state.WriteOwned();
        const auto flow = write.Block("flow"), species = write.Block("species");
        for (int j = 0; j < 5; ++j)
            flow(0, j) = u[j];
        for (int k = 0; k < fixture.species - 1; ++k)
            species(0, k) = u[5 + k];
    };
    const auto get = [&](const C::EquationDofs &state)
    {
        auto read = state.ReadOwned();
        const auto flow = read.Block("flow").owned, species = read.Block("species").owned;
        Eigen::VectorXd result(fixture.components);
        for (int j = 0; j < 5; ++j)
            result[j] = flow(0, j);
        for (int k = 0; k < fixture.species - 1; ++k)
            result[5 + k] = species(0, k);
        return result;
    };
    const auto run = [&](int steps)
    {
        auto accepted = fixture.State(space, 1), trial = fixture.State(space, 2), rates = fixture.State(space, 0);
        set(accepted, initial);
        R::Array<double> jacobian(R::RowLayout::Dense(1, fixture.components * fixture.components));
        C::EquationTimeStepper stepper(accepted, "bdf1");
        const auto evaluate = [&](auto &state, auto &rate, double, int)
        { C::EvaluateChemicalSource(fixture.equation, state, rate, fixture.workers.Acquire()); };
        const auto solve = [&](C::EquationDofs &state, const C::EquationDofs &source, double mass, double scale, double, int)
        {
            const auto right = get(source);
            for (int iteration = 0; iteration < 15; ++iteration)
            {
                C::EvaluateChemicalSource(fixture.equation, state, rates, fixture.workers.Acquire(), &jacobian);
                const auto u = get(state), rate = get(rates);
                Eigen::MatrixXd matrix = mass * Eigen::MatrixXd::Identity(fixture.components, fixture.components);
                {
                    const auto read = jacobian.Read();
                    const auto view = read.View<R::UniformTableRows>();
                    for (int j = 0; j < fixture.components; ++j)
                        for (int i = 0; i < fixture.components; ++i)
                            matrix(i, j) -= scale * view(0, i + fixture.components * j);
                }
                const Eigen::VectorXd delta = matrix.partialPivLu().solve(right - mass * u + scale * rate);
                if (!delta.allFinite())
                    return false;
                set(state, u + delta);
                if (delta.lpNorm<Eigen::Infinity>() < 1e-13 * std::max(1., u.lpNorm<Eigen::Infinity>()))
                    return true;
            }
            return false;
        };
        const double dt = totalTime / steps;
        for (int step = 0; step < steps; ++step)
        {
            REQUIRE(stepper.Prepare(accepted, trial, step * dt, dt, evaluate, solve));
            C::EquationDofOperations::Copy(accepted, trial);
            stepper.Commit();
        }
        return get(accepted);
    };
    auto fractions = fixture.Fractions(0);
    double referenceTemperature = 1400;
    fixture.prototype.advanceConstVolume(referenceTemperature, initial[0] * fixture.scales.rho0,
                                         {fractions.data(), fixture.species}, 1, totalTime * fixture.scales.Time(), 1e-11, 1e-20);
    Eigen::VectorXd expected = initial;
    for (int k = 0; k < fixture.species - 1; ++k)
        expected[5 + k] = initial[0] * fractions[k];
    const auto coarse = run(4), fine = run(8);
    const double coarseError = (coarse - expected).lpNorm<Eigen::Infinity>(), fineError = (fine - expected).lpNorm<Eigen::Infinity>();
    CAPTURE(coarseError);
    CAPTURE(fineError);
    CAPTURE(referenceTemperature);
    CHECK((fine.head<5>() - initial.head<5>()).isZero(1e-13));
    CHECK((fine - initial).lpNorm<Eigen::Infinity>() > 1e-9);
    CHECK(coarseError > 1e-12);
    CHECK(fineError < .8 * coarseError);
    CHECK(fineError < 1e-5);
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
