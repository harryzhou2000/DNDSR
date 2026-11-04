#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Implicit/LinearSolver.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationSource.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
#include <cstdlib>
#include <numeric>
#include <limits>

namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
namespace R = DNDS::Runtime;
namespace P = DNDS::Physics;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    void Compare(const C::EquationDofs &a, const C::EquationDofs &b, double tolerance)
    {
        auto x = a.ReadOwned(), y = b.ReadOwned();
        for (I iBlock = 0; iBlock < a.BlockNumber(); ++iBlock)
        {
            const auto u = x.Block(iBlock), v = y.Block(iBlock);
            for (I iCell = 0; iCell < I(a.Space().OwnedRows()); ++iCell)
                for (I j = 0; j < u.ComponentNumber(); ++j)
                    CHECK(u(iCell, j) == doctest::Approx(v(iCell, j)).epsilon(tolerance).scale(1));
        }
    }
    void ConservativeIntegral(const C::EquationDofs &residual)
    {
        auto read = residual.ReadOwned();
        for (I iBlock = 0; iBlock < residual.BlockNumber(); ++iBlock)
        {
            const auto u = read.Block(iBlock);
            for (I j = 0; j < u.ComponentNumber(); ++j)
            {
                double integral = 0, absolute = 0;
                for (I iCell = 0; iCell < u.owned.RowCount(); ++iCell)
                {
                    integral += u(iCell, j);
                    absolute += std::abs(u(iCell, j));
                }
                MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
                MPI_Allreduce(MPI_IN_PLACE, &absolute, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
                CHECK(std::abs(integral) < 1e-10 * (1 + absolute));
            }
        }
    }
    void CompareAuxiliary(const C::AuxiliaryValues &a, const C::AuxiliaryValues &b)
    {
        const auto arrays = [](const auto &v)
        {
            return std::array<const R::Array<double> *, 7>{v.meanThermodynamics ? &*v.meanThermodynamics : nullptr,
                                                           v.faceSpectralRadius ? &*v.faceSpectralRadius : nullptr, v.faceWaveJump ? &*v.faceWaveJump : nullptr,
                                                           v.facePressureMinimum ? &*v.facePressureMinimum : nullptr, v.cellSpectralSum ? &v.cellSpectralSum->Owned() : nullptr,
                                                           v.cellWaveJump ? &v.cellWaveJump->Owned() : nullptr, v.cellPressureMinimum ? &v.cellPressureMinimum->Owned() : nullptr};
        };
        const auto x = arrays(a), y = arrays(b);
        for (int k = 0; k < 7; ++k)
            if (x[k] && y[k])
            {
                const auto xr = x[k]->Read(), yr = y[k]->Read();
                const auto u = xr.View<R::UniformTableRows>(), v = yr.View<R::UniformTableRows>();
                REQUIRE(u.RowCount() == v.RowCount());
                for (I iRow = 0; iRow < u.RowCount(); ++iRow)
                    for (I j = 0; j < u.RowSize(iRow); ++j)
                        CHECK(u(iRow, j) == doctest::Approx(v(iRow, j)).epsilon(2e-11).scale(1));
            }
    }
}

TEST_CASE("Equation FV uses the established P3/gradient/WBAP flow5 kernels across CPU launchers and layouts")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    for (int variant = 0; variant < 3; ++variant)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = variant == 0 ? "gradient" : "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 5;
        settings.numerics.limiter = variant == 2 ? "WBAP" : "none";
        settings.numerics.cfv.SORInstead = variant == 2;
        settings.numerics.characteristicLimiter = false;
        settings.numerics.faceFallback = false;
        settings.numerics.viscousGradient = "polynomial";
        if (variant == 1)
        {
            settings.equation.transport = "NewtonianFourier";
            settings.equation.viscosity = .002;
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        const auto shape = variant == 0 ? R::RowLayout::Dense(0, 5) : variant == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                   : R::RowLayout::TiledColMajor(0, 5, 32);
        auto raw = fv.Metrics().BuildField(shape);
        {
            auto write = raw.Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                const auto state = equation.Gas().Conservative({1 + .01 * std::sin(6.283185307179586 * center[0]),
                                                                .3, -.1 + .01 * std::cos(6.283185307179586 * center[1]), .2, 1});
                for (int j = 0; j < 5; ++j)
                    values(iCell, j) = state[j];
            }
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(shape));
        A::Copy(state.CoupledField(), raw);
        auto expected = fv.Metrics().BuildField(shape);
        auto result = C::EquationDofOperations::NewVector(state);
        auto rawWork = fv.BuildWorkspace(raw);
        auto work = fv.BuildEquationWorkspace(state);
        R::EagerEngine engine(variant == 0 ? 0 : 2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(raw, expected, *rawWork, engine, stream);
        fv.Evaluate(state, result, *work, engine, stream);
        Compare(result, C::EquationDofs::Coupled(equation.Schema(), expected), 2e-12);
        ConservativeIntegral(result);
        if (variant == 1)
        {
            const auto &coefficients = fv.ReconstructionGuess(*work);
            auto snapshot = C::EquationDofOperations::NewVector(coefficients);
            auto replacement = C::EquationDofOperations::NewVector(coefficients);
            C::EquationDofOperations::Copy(snapshot, coefficients);
            C::EquationDofOperations::Fill(replacement, .125);
            std::optional<C::EquationDofs::ReadScope> retained;
            if (mpi.rank == 0)
                retained.emplace(coefficients.ReadOwned());
            CHECK_THROWS(fv.RestoreReconstructionGuess(*work, replacement));
            retained.reset();
            Compare(coefficients, snapshot, 1e-15);
            if (mpi.rank == 0)
            {
                auto write = replacement.WriteOwned();
                write.Block(0)(0, 0) = std::numeric_limits<double>::quiet_NaN();
            }
            CHECK_THROWS(fv.RestoreReconstructionGuess(*work, replacement));
            Compare(coefficients, snapshot, 1e-15);
            C::EquationDofOperations::Fill(replacement, .125);
            fv.RestoreReconstructionGuess(*work, replacement);
            Compare(coefficients, replacement, 1e-15);
        }
    }
}

TEST_CASE("Logical nonlinear flow5 retains native finite-sweep stage mathematics")
{
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = scenario ? "variational" : "constant";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 8;
        settings.numerics.reconstructionTolerance = scenario ? 1e-8 : 0;
        settings.numerics.faceFallback = false;
        settings.implicit.linear.method = scenario ? "gmres" : "sgs";
        settings.implicit.linear.relativeTolerance = 1e-5;
        settings.implicit.diagonal = scenario ? "block" : "entries";
        settings.execution.overlap = scenario;
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto raw = fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 5, {8, 1}));
        auto source = fv.Metrics().BuildField(5);
        const auto target = equation.Gas().Conservative({1.2, .3, -.1, .2, 1.1});
        {
            auto u = raw.Owned().Write(), b = source.Owned().Write();
            for (I iCell = 0; iCell < raw.Owned().RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    u.View()(iCell, j) = target[j] * (1 + .01 * std::sin(2 * iCell + mpi.rank + j));
                    b.View()(iCell, j) = 20 * target[j];
                }
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(R::RowLayout::TiledColMajor(0, 5, 8)));
        A::Copy(state.CoupledField(), raw);
        auto right = C::EquationDofs::Coupled(equation.Schema(), source);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = request.cellSpectralSum = true;
        C::NonlinearSolver reference(fv, raw, fv.BuildWorkspace(raw, request), engine, stream, settings);
        C::EquationNonlinearSolver solver(fv, state, fv.BuildEquationWorkspace(state), engine, stream, settings, A::ComponentSelection(5));
        const auto expected = reference.Solve(raw, source, 20, .5, .1, 0, {2});
        const auto actual = solver.Solve(state, right, 20, .5, .1, 0, {2});
        CHECK(actual.iterations == expected.iterations);
        CHECK(actual.linearIterations == expected.linearIterations);
        CHECK(actual.converged == expected.converged);
        CHECK(actual.reconstructionIterations == expected.reconstructionIterations);
        CHECK(actual.reconstructionRelativeUpdate == doctest::Approx(expected.reconstructionRelativeUpdate).epsilon(1e-10));
        for (int j = 0; j < 5; ++j)
            CHECK(actual.residual[j] == doctest::Approx(expected.residual[j]).epsilon(1e-10).scale(1e-10));
        Compare(state, C::EquationDofs::Coupled(equation.Schema(), raw), 1e-12);
    }
}

TEST_CASE("Equation wall values preserve internal energy and support empty owners on a physical 3D mesh")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .002;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.boundaries["7"].kind = scenario < 2 ? "noSlipWall" : scenario == 2 ? "farfield"
                                                                                    : "analytic";
        settings.boundaries["7"].primitive = {.9, -.3, .1, .2, 1.2};
        if (scenario >= 2)
        {
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
        }
        if (scenario == 1)
        {
            settings.boundaries["7"].thermal = "isothermal";
            settings.boundaries["7"].temperature = .7;
        }
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        auto raw = fv.Metrics().BuildField(5);
        {
            auto write = raw.Owned().Write();
            const auto u = equation.Gas().Conservative({1, .1, .2, -.1, 1});
            for (I iCell = 0; iCell < raw.Owned().RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    write.View()(iCell, j) = u[j];
        }
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        A::Copy(state.CoupledField(), raw);
        auto expected = fv.Metrics().BuildField(5);
        auto result = C::EquationDofOperations::NewVector(state);
        auto rawWork = fv.BuildWorkspace(raw);
        auto work = fv.BuildEquationWorkspace(state);
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(raw, expected, *rawWork, engine, stream);
        fv.Evaluate(state, result, *work, engine, stream);
        Compare(result, C::EquationDofs::Coupled(equation.Schema(), expected), 2e-12);
        C::AuxiliaryRequest request;
        request.meanThermodynamics = request.faceSpectralRadius = request.cellSpectralSum = true;
        auto originalAux = fv.BuildAuxiliaryValues(request), equationAux = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(raw, originalAux);
        fv.PrepareAuxValues(state, equationAux);
        CompareAuxiliary(originalAux, equationAux);
    }
}

#ifdef DNDS_USE_CANTERA
TEST_CASE("A single-species material uses empty independent composition and no chemical equation")
{
    const P::ReferenceScales scales{1, 340, 1, 300};
    P::Chemistry::ChemicalSource prototype(std::string(DNDS_TEST_SOURCE_ROOT) + "/test/cpp/Cauchy/argon.yaml", "argon", scales.U0, scales.rho0);
    P::Thermodynamics::IdealGasMixture material(prototype, scales);
    P::Chemistry::ChemicalWorkerPool workers(prototype, 1);
    const auto equation = C::EquationSystem::FromIdealGasMixture(material, false);
    REQUIRE(equation.Schema().blocks.size() == 1);
    double fraction = 0;
    prototype.massFractions(1, P::Chemistry::ConstSpeciesBufferView{}, {&fraction, 1});
    CHECK(fraction == 1);
    const P::Chemistry::ConstSpeciesBufferView fractions{&fraction, 1};
    const double energy = prototype.mixtureIntEnergy(300, fractions, prototype.mixtureR(fractions) * 300) / scales.SpecificEnergy();
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    for (const std::string source : {"none", "chemical"})
    {
        C::DomainSettings settings;
        settings.mesh.dimension = 3;
        settings.boundaries["7"].kind = "extrapolate";
        settings.equation.source = source;
        settings.numerics.reconstruction = "constant";
        settings.numerics.faceFallback = false;
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(5));
        {
            auto write = state.WriteOwned();
            const auto u = write.Block("flow");
            for (I iCell = 0; iCell < u.RowCount(); ++iCell)
            {
                u(iCell, 0) = 1;
                u(iCell, 4) = energy;
                for (int j = 1; j < 4; ++j)
                    u(iCell, j) = 0;
            }
        }
        auto residual = C::EquationDofOperations::NewVector(state);
        auto work = fv.BuildEquationWorkspace(state, workers);
        R::EagerEngine engine;
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, residual, *work, engine, stream);
        {
            const auto read = residual.ReadOwned();
            const auto r = read.Block("flow");
            for (I iCell = 0; iCell < r.owned.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j)
                    CHECK(std::abs(r(iCell, j)) < 1e-12);
        }
        R::Array<double> jacobian(R::RowLayout::Dense(I(state.Space().OwnedRows()), 25));
        C::EvaluateChemicalSource(equation, state, residual, workers.Acquire(), &jacobian);
        const auto read = residual.ReadOwned();
        const auto matrix = jacobian.Read();
        for (I iCell = 0; iCell < I(state.Space().OwnedRows()); ++iCell)
        {
            for (int j = 0; j < 5; ++j)
                CHECK(read.Block("flow")(iCell, j) == 0);
            for (int j = 0; j < 25; ++j)
                CHECK(matrix.View()(iCell, j) == 0);
        }
    }
}
namespace
{
    struct MixtureFixture
    {
        const P::ReferenceScales scales;
        P::Chemistry::ChemicalSource prototype;
        P::Thermodynamics::IdealGasMixture material;
        P::Chemistry::ChemicalWorkerPool workers;
        C::EquationSystem equation;
        int species;
        explicit MixtureFixture(double length = 2) : scales{1.225, 340, length, 288.15}, prototype((std::getenv("DNDS_MECH_PATH") ? std::string(std::getenv("DNDS_MECH_PATH")) + "/" : "") + "h2o2.yaml", "", scales.U0, scales.rho0),
                                                     material(prototype, scales), workers(prototype, 1), equation(C::EquationSystem::FromIdealGasMixture(material, true)), species(prototype.nSpecies()) {}
        C::EquationDofs State(const A::DofSpace &space, int layout)
        {
            if (!layout)
                return C::EquationDofs::Coupled(equation.Schema(), A::DofField<double>(space, R::RowLayout::Dense(0, species + 4)));
            A::DofBundle fields(space);
            if (layout == 1)
            {
                fields.Add("flow", A::DofField<double>(space, mpi.rank % 2 ? R::RowLayout::Dense(0, 5, 8) : R::RowLayout::ColMajor(0, 5, {16, 1})));
                fields.Add("species", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, species - 1, 32)));
                return C::EquationDofs::Separate(equation.Schema(), std::move(fields));
            }
            fields.Add("packed", A::DofField<double>(space, R::RowLayout::TiledColMajor(0, species + 4, 32)));
            return C::EquationDofs(equation.Schema(), std::move(fields), {{"flow", "packed", species - 1}, {"species", "packed", 0}});
        }
        void Fill(C::EquationDofs &state, const DNDS::CFV::FiniteVolumePlan &metrics, bool constant)
        {
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow"), rhoY = write.Block("species");
            std::vector<double> Y(size_t(species), 1e-6);
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                const auto center = metrics.GetCellCent(iCell);
                const double wave = constant ? 0 : .01 * std::sin(6.283185307179586 * center[0]) * std::cos(6.283185307179586 * center[1]);
                Y[0] = .027 + wave * .05;
                Y[3] = .22 - wave * .07;
                Y[5] = .02 + wave * .02;
                Y.back() = 0;
                Y.back() = 1 - std::accumulate(Y.begin(), Y.end(), 0.);
                const double rho = .8 + wave, temperature = 1400 + 50 * wave;
                const P::Chemistry::ConstSpeciesBufferView composition{Y.data(), species};
                const double pressure = rho * scales.rho0 * prototype.mixtureR(composition) * temperature;
                flow(iCell, 0) = rho;
                flow(iCell, 1) = rho * .2;
                flow(iCell, 2) = rho * -.04;
                flow(iCell, 3) = rho * .03;
                flow(iCell, 4) = rho * prototype.mixtureIntEnergy(temperature, composition, pressure) / scales.SpecificEnergy() + .5 * rho * (.04 + .0016 + .0009);
                for (int k = 0; k < species - 1; ++k)
                    rhoY(iCell, k) = rho * Y[k];
            }
        }
    };
}

TEST_CASE("Distributed prescribed mixture boundaries match independent nonconstant FV fluxes across layouts and launchers")
{
    MixtureFixture mixture;
    const auto equation = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    const auto mesh = CFVTest::CartesianHexes(mpi, 4);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    auto &boundary = settings.boundaries["7"];
    boundary.kind = "analytic";
    std::vector<double> fractions(size_t(mixture.species), 0);
    fractions[0] = .04;
    fractions[3] = .21;
    fractions[5] = .03;
    fractions.back() = .72;
    const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), mixture.species};
    constexpr double density = .9, temperature = 1800;
    const double pressureSI = density * mixture.scales.rho0 * mixture.prototype.mixtureR(Y) * temperature;
    boundary.primitive = {density, -.2, .07, -.03, pressureSI / mixture.scales.Pressure()};
    for (int k = 0; k < mixture.species - 1; ++k)
        boundary.massFractions[mixture.material.SpeciesNames()[k]] = fractions[k];
    std::vector<double> prescribed(size_t(mixture.species + 4), 0);
    prescribed[0] = density;
    for (int d = 0; d < 3; ++d)
        prescribed[1 + d] = density * boundary.primitive[1 + d];
    prescribed[4] = density * mixture.prototype.mixtureIntEnergy(temperature, Y, pressureSI) / mixture.scales.SpecificEnergy() +
                    .5 * density * (.2 * .2 + .07 * .07 + .03 * .03);
    for (int k = 0; k < mixture.species - 1; ++k)
        prescribed[5 + k] = density * fractions[k];
    std::optional<C::EquationDofs> baseline;
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        settings.execution.overlap = scenario;
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = mixture.State(fv.Metrics().Space(), scenario ? 2 : 1);
        fv.InitializeConstant(state, boundary.primitive, boundary.massFractions, temperature / mixture.scales.T0, mixture.workers);
        {
            const auto initial = state.ReadOwned();
            const auto flow = initial.Block("flow"), species = initial.Block("species");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    CHECK(flow(iCell, j) == doctest::Approx(prescribed[j]).epsilon(2e-12));
                for (int k = 0; k < mixture.species - 1; ++k)
                    CHECK(species(iCell, k) == doctest::Approx(prescribed[5 + k]).epsilon(2e-12));
            }
        }
        mixture.Fill(state, fv.Metrics(), false);
        auto result = C::EquationDofOperations::NewVector(state);
        auto work = fv.BuildEquationWorkspace(state, mixture.workers);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, result, *work, engine, stream);
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = request.cellSpectralSum = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(state, auxiliary, 0, mixture.workers);
        // Independent EOS backend and explicit face integration: no Cauchy
        // boundary, point producer or Riemann flux is used by this oracle.
        auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
        auto metricRead = fv.Metrics().AcquireRead();
        const auto topology = topologyRead.HostView();
        const auto metric = metricRead.HostView();
        const auto values = state.ReadWithGhost(), residual = result.ReadOwned();
        const auto flow = values.Block("flow"), species = values.Block("species");
        const auto outFlow = residual.Block("flow"), outSpecies = residual.Block("species");
        std::vector<double> left(prescribed.size()), right(prescribed.size()), expected(prescribed.size());
        const auto gather = [&](I iCell, auto &point)
        {
            for (int j = 0; j < 5; ++j)
                point[j] = flow(iCell, j);
            for (int k = 0; k < mixture.species - 1; ++k)
                point[5 + k] = species(iCell, k);
        };
        const auto properties = [&](const auto &point)
        {
            double sum = 0, kinetic = 0;
            for (int k = 0; k < mixture.species - 1; ++k)
            {
                fractions[k] = point[5 + k] / point[0];
                sum += fractions[k];
            }
            fractions.back() = 1 - sum;
            for (int d = 1; d <= 3; ++d)
                kinetic += .5 * point[d] * point[d] / point[0];
            const double T = mixture.prototype.temperatureFromUV((point[4] - kinetic) / point[0] * mixture.scales.SpecificEnergy(),
                                                                 1 / (point[0] * mixture.scales.rho0), Y, 1400, 1e-12);
            const double p = point[0] * mixture.scales.rho0 * mixture.prototype.mixtureR(Y) * T;
            P::Thermodynamics::Properties thermo;
            mixture.prototype.EvaluateProperties(T, p, Y, thermo);
            return std::array<double, 2>{p / mixture.scales.Pressure(), thermo.soundSpeed / mixture.scales.U0};
        };
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            std::fill(expected.begin(), expected.end(), 0);
            gather(iCell, left);
            const auto lp = properties(left);
            const auto faces = topology.CellFaces(iCell);
            for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
            {
                const auto face = faces[slot], other = topology.CellFaceOther(iCell, face, slot);
                const int side = topology.CellIsFaceBack(iCell, face, slot) ? 0 : 1;
                if (other == DNDS::UnInitIndex)
                    right = prescribed;
                else
                    gather(other, right);
                const auto rp = properties(right);
                for (I q = metric.FaceQuadratureBegin(face); q < metric.FaceQuadratureEnd(face); ++q)
                {
                    const auto normal = ((side ? -1 : 1) * metric.GetFaceNorm(q)).eval();
                    double ul = 0, ur = 0;
                    for (int d = 0; d < 3; ++d)
                    {
                        ul += left[1 + d] / left[0] * normal[d];
                        ur += right[1 + d] / right[0] * normal[d];
                    }
                    const double alpha = std::max(std::abs(ul) + lp[1], std::abs(ur) + rp[1]);
                    for (I j = 0; j < I(expected.size()); ++j)
                    {
                        double fl = left[j] * ul, fr = right[j] * ur;
                        if (j >= 1 && j <= 3)
                        {
                            fl += lp[0] * normal[j - 1];
                            fr += rp[0] * normal[j - 1];
                        }
                        if (j == 4)
                        {
                            fl += lp[0] * ul;
                            fr += rp[0] * ur;
                        }
                        expected[j] -= metric.GetFaceWeight(q) * .5 * (fl + fr - alpha * (right[j] - left[j]));
                    }
                }
            }
            for (int j = 0; j < 5; ++j)
                CHECK(outFlow(iCell, j) == doctest::Approx(expected[j]).epsilon(2e-10).scale(1e-10));
            for (int k = 0; k < mixture.species - 1; ++k)
                CHECK(outSpecies(iCell, k) == doctest::Approx(expected[5 + k]).epsilon(2e-10).scale(1e-10));
        }
        if (baseline)
            Compare(result, *baseline, 2e-12);
        else
            baseline = result;
    }
}

TEST_CASE("Mixture isothermal wall conducts the independent Fourier flux without species leakage")
{
    MixtureFixture mixture;
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    settings.numerics.viscousGradient = "polynomial";
    auto &wall = settings.boundaries["7"];
    wall.kind = "noSlipWall";
    wall.thermal = "isothermal";
    wall.temperature = 600 / mixture.scales.T0;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1), result = C::EquationDofOperations::NewVector(state);
    std::vector<double> Y(size_t(mixture.species), 0);
    Y[0] = .04;
    Y[3] = .21;
    Y[5] = .03;
    Y.back() = .72;
    std::map<std::string, double> composition;
    for (int k = 0; k < mixture.species; ++k)
        composition[mixture.material.SpeciesNames()[k]] = Y[k];
    fv.InitializeConstant(state, {.8, 0, 0, 0, 1}, composition, 1400 / mixture.scales.T0, mixture.workers);
    auto work = fv.BuildEquationWorkspace(state, mixture.workers);
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, result, *work, engine, stream);
    const P::Chemistry::ConstSpeciesBufferView fractions{Y.data(), mixture.species};
    const double r1 = .8, r2 = r1 * 1400 / 600;
    const double p = r1 * mixture.scales.rho0 * mixture.prototype.mixtureR(fractions) * 1400;
    const double e1 = mixture.prototype.mixtureIntEnergy(1400, fractions, p) / mixture.scales.SpecificEnergy();
    const double e2 = mixture.prototype.mixtureIntEnergy(600, fractions, p) / mixture.scales.SpecificEnergy();
    const double rho = .5 * (r1 + r2), energy = .5 * (r1 * e1 + r2 * e2) / rho;
    const double temperature = mixture.prototype.temperatureFromUV(energy * mixture.scales.SpecificEnergy(), 1 / (rho * mixture.scales.rho0), fractions, 1000, 1e-12);
    P::Thermodynamics::Properties thermo;
    P::Transport::Properties transport;
    mixture.prototype.EvaluateProperties(temperature, rho * mixture.scales.rho0 * mixture.prototype.mixtureR(fractions) * temperature, fractions, thermo, &transport);
    // The unit cube's reflected cell-centroid separation is one; composition
    // is constant, so the independent full-energy chain rule has no dY term.
    const double gradient = (r2 * e2 - r1 * e1 - energy * (r2 - r1)) / (rho * thermo.cv / mixture.scales.HeatCapacity());
    const double expected = 6 * transport.conductivity / mixture.scales.Conductivity() * gradient;
    CHECK(expected < -1e-8);
    double actual = 0;
    {
        const auto read = result.ReadOwned();
        const auto flow = read.Block("flow"), species = read.Block("species");
        for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
        {
            actual += flow(iCell, 4);
            CHECK(std::abs(flow(iCell, 0)) < 1e-12);
            for (int k = 0; k < mixture.species - 1; ++k)
                CHECK(std::abs(species(iCell, k)) < 1e-12);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &actual, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(actual == doctest::Approx(expected).epsilon(2e-10));
}

TEST_CASE("Configured mixture Domains advance every block through production time and publication paths")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    std::vector<double> fractions(size_t(mixture.species), 1e-6);
    fractions[0] = .027;
    fractions[3] = .22;
    fractions[5] = .02;
    fractions.back() = 0;
    fractions.back() = 1 - std::accumulate(fractions.begin(), fractions.end(), 0.);
    C::Json composition = C::Json::object();
    for (int k = 0; k < mixture.species; ++k)
        composition[mixture.material.SpeciesNames()[k]] = fractions[k];
    C::Json input{{"defaults", {{"equation", {{"eos", "idealGasMixture"}, {"source", "chemical"}, {"mixture", {{"mechanismFile", mixture.prototype.mechanismFile()}, {"rho0", mixture.scales.rho0}, {"U0", mixture.scales.U0}, {"L0", mixture.scales.L0}, {"T0", mixture.scales.T0}}}}}, {"initialPrimitive", {.8, .2, -.04, .03, 1}}, {"initialTemperature", 1400 / mixture.scales.T0}, {"initialMassFractions", composition}, {"execution", {{"layout", "soa"}, {"hostThreads", 2}}}, {"numerics", {{"reconstruction", "constant"}, {"faceFallback", false}}}, {"implicit", {{"diagonal", "block"}, {"pseudoCFL", 1e4}, {"maxPseudoCFL", 1e8}, {"relativeTolerance", 1e-10}, {"absoluteTolerance", 1e-8}, {"residualComponents", C::Json::array()}, {"linear", {{"method", "gmres"}, {"preconditioner", "ilu"}, {"relativeTolerance", 1e-10}, {"absoluteTolerance", 1e-13}}}}}}}, {"time", {{"method", "bdf1"}}}};
    const auto startFractions = fractions;
    constexpr double totalTime = 1e-5;
    double temperature = 1400;
    mixture.prototype.advanceConstVolume(temperature, .8 * mixture.scales.rho0, {fractions.data(), mixture.species}, 1,
                                         totalTime * mixture.scales.Time(), 1e-12, 1e-18);
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        const int steps = scenario == 0 ? 8 : 4;
        input["time"]["method"] = scenario == 0 ? "ssprk3" : scenario == 1 ? "bdf1"
                                                         : scenario == 2   ? "esdirk4"
                                                                           : "ditr";
        if (scenario >= 3)
        {
            // Reaction-dominated production profile: use the requested U2R1.
            // U2R2 remains covered by the independent public time-method oracle
            // and nonreacting Domain tests; it need not converge this stiff
            // reactor within the same pseudo-time budget.
            input["time"]["ditr"] = C::Json{{"variant", "U2R1"}};
        }
        if (scenario == 3)
        {
            auto names = C::Json::array();
            for (int k = 0; k < mixture.species - 1; ++k)
                names.push_back("rhoY:" + mixture.material.SpeciesNames()[k]);
            input["defaults"]["implicit"]["residualComponents"] = std::move(names);
            // In this homogeneous reactor density cannot stop chemistry.
            // DITR observes after one pair of corrections. Density-only
            // acceptance need not imply convergence of the species equations.
            auto densityInput = input;
            densityInput["defaults"]["implicit"].erase("residualComponents");
            const auto densityConfig = C::SimulationConfig::ResolveCollective(densityInput, mpi.comm);
            C::Domain densityOnly("density-only", mesh, densityConfig.Domains()[0].settings, densityConfig.Time());
            const auto unchanged = densityOnly.Dofs().Clone();
            densityOnly.Advance(totalTime / steps);
            CHECK(densityOnly.LastCoupledIterations() == 1);
            CHECK(densityOnly.LastCoupledTargetRatio() <= 1);
            REQUIRE(densityOnly.LastImplicitStages().size() == 1);
            CHECK(densityOnly.LastImplicitStages().front().converged);
            const auto after = densityOnly.Dofs().ReadOwned(), before = unchanged.ReadOwned();
            const auto flowAfter = after.Block("flow"), flowBefore = before.Block("flow");
            for (I iCell = 0; iCell < flowAfter.owned.RowCount(); ++iCell)
                for (I j = 0; j < flowAfter.ComponentNumber(); ++j)
                    CHECK(flowAfter(iCell, j) == doctest::Approx(flowBefore(iCell, j)).epsilon(1e-11));
        }
        input["defaults"]["execution"]["layout"] = scenario == 0 ? "aos" : scenario == 1 ? "soa"
                                                                                         : "aosoa32";
        input["defaults"]["execution"]["overlap"] = bool(scenario);
        input["defaults"]["execution"]["hostThreads"] = scenario ? 2 : 0;
        input["defaults"]["numerics"]["positivity"] = scenario != 0;
        if (scenario == 2)
        {
            input["defaults"]["numerics"]["reconstruction"] = "variational";
            input["defaults"]["numerics"]["cfv"] = C::Json{{"maxOrder", 3}};
            input["defaults"]["numerics"]["limiter"] = "WBAP";
            input["defaults"]["numerics"]["characteristicLimiter"] = false;
        }
        const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
        C::Domain domain("mixture", mesh, config.Domains()[0].settings, config.Time());
        REQUIRE(domain.Dofs().PhysicalFieldNumber() == 2);
        const auto initial = domain.Dofs().Clone();
        const auto dt = domain.StableStep(.5);
        CHECK(std::isfinite(dt));
        CHECK(dt > 0);
        CHECK(domain.ResidualState().ComponentNumber() == mixture.species + 4);
        int firstStep = 0;
        if (scenario == 1)
        {
            domain.PrepareStep(totalTime / steps);
            const C::EquationDofOperations::Binding before(domain.Dofs());
            {
                const auto location = domain.Dofs().LocateBlock("species");
                const auto held = domain.Dofs().PhysicalField(location.physicalField).Owned().Read();
                CHECK_THROWS(domain.CommitStep()); // A species-only view must block all publication.
                CHECK_NOTHROW(before.Check(domain.Dofs(), true));
                CHECK(domain.Time() == 0);
                CHECK(domain.Epoch().acceptedStep == 0);
            }
            domain.CommitStep();
            firstStep = 1;
        }
        for (int step = firstStep; step < steps; ++step)
            domain.Advance(totalTime / steps);
        CHECK(domain.Epoch().acceptedStep == uint64_t(steps));
        CHECK(domain.Time() == doctest::Approx(totalTime).epsilon(2e-15));
        if (scenario)
        {
            REQUIRE_FALSE(domain.LastImplicitStages().empty());
            for (const auto &report : domain.LastImplicitStages())
            {
                INFO("method=" << config.Time().method << " variant=" << config.Time().ditr.variant << " iterations=" << report.iterations << " target=" << report.residualTargetRatio
                               << " residual=" << C::Json(report.residual).dump() << " initial=" << C::Json(report.initialResidual).dump());
                CHECK(report.converged);
                CHECK(report.residual.size() == size_t(mixture.species + 4));
            }
            if (scenario >= 3)
            {
                REQUIRE(domain.LastImplicitStages().size() == 1);
                CHECK(domain.LastCoupledTargetRatio() <= 1);
                CHECK(domain.LastImplicitStages().front().iterations == domain.LastCoupledIterations());
                CHECK(domain.LastImplicitStages().front().residualTargetRatio == domain.LastCoupledTargetRatio());
            }
        }
        double error = 0, change = 0;
        {
            const auto read = domain.Dofs().ReadOwned(), before = initial.ReadOwned();
            const auto flow = read.Block("flow"), oldFlow = before.Block("flow"), species = read.Block("species");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    CHECK(flow(iCell, j) == doctest::Approx(oldFlow(iCell, j)).epsilon(2e-12).scale(1));
                for (int k = 0; k < mixture.species - 1; ++k)
                {
                    error = std::max(error, std::abs(species(iCell, k) - .8 * fractions[k]));
                    change = std::max(change, std::abs(species(iCell, k) - .8 * startFractions[k]));
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(change > 1e-7);
        CHECK(error < 3e-7);
        const auto diagnostics = domain.Diagnostics(config.Domains()[0].settings, {});
        CHECK(diagnostics[2] > 0);
        CHECK(diagnostics[3] > 0);
        CHECK(diagnostics[13] > 0);
    }
    input["defaults"]["numerics"]["reconstruction"] = "constant";
    input["defaults"]["numerics"]["limiter"] = "none";
    input["time"]["method"] = "bdf1";
    input["domains"] = C::Json::array({{{"name", "a"}}, {{"name", "b"}}});
    const auto config = C::SimulationConfig::ResolveCollective(input, mpi.comm);
    C::Simulation simulation(config, mpi.comm, [&](I, const auto &, const auto &)
                             { return mesh; });
    simulation.Advance(totalTime);
    for (I iDomain = 0; iDomain < 2; ++iDomain)
    {
        const auto &domain = simulation.GetDomain(iDomain);
        const auto read = domain.Dofs().ReadOwned();
        const auto species = read.Block("species");
        double change = 0;
        for (I iCell = 0; iCell < species.owned.RowCount(); ++iCell)
            for (int k = 0; k < mixture.species - 1; ++k)
                change = std::max(change, std::abs(species(iCell, k) - .8 * startFractions[k]));
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(change > 1e-7);
        CHECK(domain.Time() == simulation.Time());
    }
}

TEST_CASE("Physical mixture Jacobians match frozen-signal residual differences and solve through the logical driver")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 6);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.cfv.maxOrder = 1;
    settings.numerics.faceFallback = false;
    settings.equation.source = "chemical";
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1), direction = mixture.State(fv.Metrics().Space(), 2);
    mixture.Fill(state, fv.Metrics(), false);
    const I nv = state.ComponentNumber();
    R::Array<double> inverseTau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
    {
        const auto read = state.ReadOwned();
        const auto input = read.ComponentLayout();
        auto write = direction.WriteOwned();
        const auto output = write.ComponentLayout();
        auto pseudo = inverseTau.Write();
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto center = fv.Metrics().GetCellCent(iCell);
            for (I j = 0; j < nv; ++j)
                output.View()(iCell, j) = j < 5 ? .02 * (j + 1) * std::cos(2 * center[0] + center[1]) : .03 * input.View()(iCell, j) * std::cos(center[0] + 2 * center[1]);
            pseudo.View()(iCell, 0) = 2 + center[0];
        }
    }
    for (auto *field : {&state, &direction})
    {
        std::vector<R::TransferBinding> bindings;
        for (I iField = 0; iField < field->PhysicalFieldNumber(); ++iField)
            bindings.push_back(fv.Metrics().BindHalo(field->PhysicalField(iField)));
        std::vector<R::TransferOperation> active;
        for (auto &binding : bindings)
            active.push_back(binding.Begin());
        for (auto &operation : active)
            operation.Wait();
    }
    auto auxiliary = fv.BuildAuxiliaryValues({true, true});
    fv.PrepareAuxValues(state, auxiliary, 0, mixture.workers);
    C::JacobianPreparation controls{3, .7, false};
    auto assembled = fv.BuildLinearProblem(state, C::DiagonalStorage::Block);
    auto evaluated = fv.BuildLinearProblem(state, C::DiagonalStorage::Block, C::OffDiagonalStorage::SemiMatrixFree);
    fv.PrepareJacobian(state, auxiliary, inverseTau, *assembled, controls, 0, mixture.workers);
    fv.PrepareJacobian(state, auxiliary, inverseTau, *evaluated, controls, 0, mixture.workers);
    auto action = C::EquationDofOperations::NewVector(state), other = C::EquationDofOperations::NewVector(direction);
    assembled->Apply(direction, action);
    evaluated->Apply(direction, other);
    Compare(action, other, 2e-12);
    {
        const auto version = assembled->Diagonal().Version();
        const auto pin = assembled->OffDiagonal().Read();
        CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *assembled, controls, 0, mixture.workers));
        CHECK(assembled->Diagonal().Version() == version);
    }
    {
        auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
        const auto topology = topologyRead.HostView();
        auto geometry = fv.Metrics().AcquireRead();
        const auto metric = geometry.HostView();
        const auto meanRead = state.ReadWithGhost(), deltaRead = direction.ReadWithGhost();
        const auto mean = meanRead.ComponentLayout(), delta = deltaRead.ComponentLayout();
        const auto resultRead = action.ReadOwned();
        const auto result = resultRead.ComponentLayout();
        const auto faceRead = auxiliary.faceSpectralRadius->Read(), pseudo = inverseTau.Read();
        const auto radius = faceRead.View();
        auto evaluator = mixture.equation.MakePointEvaluator(mixture.workers.Acquire(), {false, false, false, true, false});
        const auto point = [&](I iCell, double h)
        {
            Eigen::VectorXd value(nv);
            for (I j = 0; j < nv; ++j)
                value[j] = mean.View()(iCell, j) + h * delta.View()(iCell, j);
            return value;
        };
        const auto physicalFlux = [&](const Eigen::VectorXd &value, const C::Vector3 &normal)
        {
            const C::FlowState flow{value[0], value[1], value[2], value[3], value[4]};
            const auto properties = evaluator.Evaluate(flow, {value.data() + 5, mixture.species - 1}, 0, {});
            const auto first = C::Riemann::PhysicalFlux({flow, properties.thermo.pressure}, normal);
            Eigen::VectorXd flux(nv);
            double speed = 0;
            for (int d = 0; d < 3; ++d)
                speed += flow[1 + d] / flow[0] * normal[d];
            for (I j = 0; j < nv; ++j)
                flux[j] = j < 5 ? first[j] : value[j] * speed;
            return flux;
        };
        const auto residual = [&](I iCell, double h)
        {
            const auto left = point(iCell, h);
            Eigen::VectorXd value = Eigen::VectorXd::Zero(nv);
            const auto source = evaluator.Evaluate({left[0], left[1], left[2], left[3], left[4]}, {left.data() + 5, mixture.species - 1}, 0, {false, false, false, true, false});
            for (int k = 0; k < mixture.species - 1; ++k)
                value[5 + k] = metric.GetCellVol(iCell) * source.chemicalSource[k];
            const auto faces = topology.CellFaces(iCell);
            for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
            {
                const auto iFace = faces[slot], neighbor = topology.CellFaceOther(iCell, iFace, slot);
                REQUIRE(neighbor != DNDS::UnInitIndex);
                const auto right = point(neighbor, h);
                const auto found = std::lower_bound(fv.AuxiliaryFaces().begin(), fv.AuxiliaryFaces().end(), iFace);
                REQUIRE(found != fv.AuxiliaryFaces().end());
                const auto iAux = found - fv.AuxiliaryFaces().begin();
                const int sign = topology.CellIsFaceBack(iCell, iFace, slot) ? 1 : -1;
                for (I q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                {
                    const DNDS::Geom::tPoint normal = sign * metric.GetFaceNorm(q);
                    const C::Vector3 n{normal[0], normal[1], normal[2]};
                    const auto a = physicalFlux(left, n), b = physicalFlux(right, n);
                    const double lambda = radius(iAux, 0) + radius(iAux, 1);
                    value.noalias() -= .5 * metric.GetFaceWeight(q) * (a + b - lambda * (right - left));
                }
            }
            return value;
        };
        constexpr double h = 2e-6;
        double signal = 0;
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const Eigen::VectorXd derivative = (residual(iCell, h) - residual(iCell, -h)) / (2 * h);
            Eigen::VectorXd expected(nv), actual(nv);
            for (I j = 0; j < nv; ++j)
            {
                expected[j] = metric.GetCellVol(iCell) * (pseudo.View()(iCell, 0) + controls.physicalMass) * delta.View()(iCell, j) - controls.residualScale * derivative[j];
                actual[j] = result.View()(iCell, j);
            }
            signal = std::max(signal, actual.lpNorm<Eigen::Infinity>());
            CHECK((actual - expected).lpNorm<Eigen::Infinity>() < 3e-5 * std::max(1., expected.lpNorm<Eigen::Infinity>()));
        }
        MPI_Allreduce(MPI_IN_PLACE, &signal, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(signal > .01);
    }
    C::LinearSolveSettings linear;
    linear.method = "gmres";
    linear.preconditioner = "ilu";
    linear.subspace = 20;
    linear.maxIterations = 100;
    linear.relativeTolerance = 1e-10;
    linear.absoluteTolerance = 1e-12;
    C::EquationLinearSolver solver(evaluated, direction, [&](auto &field)
                                   { return fv.Metrics().BindHalo(field); }, linear);
    const auto report = solver.Solve(action);
    CHECK(report.converged);
    CHECK(report.usable);
    Compare(direction, solver.Solution(), 1e-8);
    const auto generation = evaluated->OffDiagonalGeneration(), diagonalVersion = evaluated->Diagonal().Version();
    {
        auto write = state.WriteOwned();
        const auto species = write.Block("species");
        if (species.RowCount())
            species(0, 0) += 1e-8;
    }
    CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *evaluated, controls, 0, mixture.workers));
    CHECK(evaluated->OffDiagonalGeneration() == generation);
    CHECK(evaluated->Diagonal().Version() == diagonalVersion);
    evaluated->Apply(direction, other); // The already-frozen operator remains unchanged.
    Compare(action, other, 2e-12);
}

TEST_CASE("Actual mixture FV residual preserves separate/permuted fields and core overlap without chemistry in face queries")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 5;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.characteristicLimiter = false;
    settings.numerics.faceFallback = false;
    C::FiniteVolume baseline(mesh, settings, mixture.equation);
    auto referenceState = mixture.State(baseline.Metrics().Space(), 0);
    mixture.Fill(referenceState, baseline.Metrics(), false);
    auto expected = C::EquationDofOperations::NewVector(referenceState);
    auto baseWork = baseline.BuildEquationWorkspace(referenceState, mixture.workers);
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    baseline.Evaluate(referenceState, expected, *baseWork, engine, stream);
    ConservativeIntegral(expected);
    const C::AuxiliaryRequest all{true, true, true, true, true, true, true};
    auto referenceAuxiliary = baseline.BuildAuxiliaryValues(all);
    baseline.PrepareAuxValues(referenceState, referenceAuxiliary, .25, mixture.workers);
    for (int layout = 1; layout < 3; ++layout)
    {
        settings.execution.overlap = layout == 1;
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto state = mixture.State(fv.Metrics().Space(), layout);
        mixture.Fill(state, fv.Metrics(), false);
        auto result = C::EquationDofOperations::NewVector(state);
        auto work = fv.BuildEquationWorkspace(state, mixture.workers);
        const auto before = state.PhysicalField(0).Owned().Version();
        fv.Evaluate(state, result, *work, engine, stream);
        Compare(expected, result, 5e-11);
        CHECK(state.PhysicalField(0).Owned().Version() == before);
        auto request = all;
        if (layout == 1)
            request.derived.enabled[int(C::DerivedQuantity::Temperature)] = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        std::vector<R::TransferBinding> bindings;
        std::vector<R::TransferOperation> active;
        for (I iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
            bindings.push_back(fv.Metrics().BindHalo(state.PhysicalField(iField)));
        for (auto &binding : bindings)
            active.push_back(binding.Begin());
        auto preparation = fv.BeginAuxValues(state, auxiliary, .25, mixture.workers);
        preparation.Apply(DNDS::CFV::ReconstructionRegion::Core);
        for (auto &operation : active)
            operation.Wait();
        preparation.Apply(DNDS::CFV::ReconstructionRegion::HaloDependent);
        preparation.Finish(auxiliary);
        fv.ValidateAuxValues(state, auxiliary, .25);
        CompareAuxiliary(referenceAuxiliary, auxiliary);
        // One late physical output conflict must reject on all ranks before
        // a field halo is posted or an earlier output generation is published.
        const auto version = result.PhysicalField(0).Owned().Version();
        std::optional<R::Array<double>::ReadLease> pin;
        if (mpi.rank == 0)
            pin.emplace(result.PhysicalField(result.PhysicalFieldNumber() - 1).Owned().Read());
        CHECK_THROWS(fv.Evaluate(state, result, *work, engine, stream));
        CHECK(result.PhysicalField(0).Owned().Version() == version);
        // A species-only mutation invalidates the property publication too.
        {
            const auto location = state.LocateBlock("species");
            auto write = state.PhysicalField(location.physicalField).Owned().Write();
            const auto values = write.View<R::UniformTableRows>();
            if (values.RowCount())
                values(0, location.firstComponent) += 1e-8;
        }
        CHECK_THROWS(fv.ValidateAuxValues(state, auxiliary, .25));
    }
    if (mpi.size > 1)
    {
        auto inconsistent = mixture.State(baseline.Metrics().Space(), mpi.rank == 0 ? 0 : 1);
        CHECK_THROWS(baseline.BuildEquationWorkspace(inconsistent, mixture.workers));
    }
}

TEST_CASE("Mixture volume quadrature adds the conservative chemical rate and leaves full energy unchanged")
{
    MixtureFixture mixture;
    C::DomainSettings settings;
    settings.equation.source = "chemical";
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.cacheDiffBase = false;
    settings.numerics.reconstructionIterations = 2;
    settings.numerics.viscousGradient = "polynomial";
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    settings.numerics.faceFallback = false;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1);
    mixture.Fill(state, fv.Metrics(), true);
    auto result = C::EquationDofOperations::NewVector(state);
    auto work = fv.BuildEquationWorkspace(state, mixture.workers);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, result, *work, engine, stream);
    auto evaluator = mixture.equation.MakePointEvaluator(mixture.workers.Acquire(), {false, false, false, true, false});
    const auto read = state.ReadOwned(), residual = result.ReadOwned();
    const auto flow = read.Block("flow"), species = read.Block("species"), rhsFlow = residual.Block("flow"), rhsSpecies = residual.Block("species");
    for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
    {
        C::FlowState u{};
        for (int j = 0; j < 5; ++j)
            u[j] = flow(iCell, j);
        auto input = evaluator.SpeciesInput();
        for (int k = 0; k < mixture.species - 1; ++k)
            input[k] = species(iCell, k);
        const auto point = evaluator.Evaluate(u, input);
        const double volume = fv.Metrics().GetCellVol(iCell);
        bool active = false;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            CHECK(rhsSpecies(iCell, k) == doctest::Approx(volume * point.chemicalSource[k]).epsilon(1e-8).scale(1e-8));
            active |= std::abs(point.chemicalSource[k]) > 1e-7;
        }
        CHECK(active);
        for (int j = 0; j < 5; ++j)
            CHECK(std::abs(rhsFlow(iCell, j)) < 1e-10);
    }
}

TEST_CASE("Nonconstant assembled mixture diffusion matches an independent face integration with direct backend coefficients")
{
    MixtureFixture mixture(2e-4); // Resolve a substantial code-unit diffusion signal.
    constexpr I width = 4;
    constexpr double h = 1. / width;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, width);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 5;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.faceFallback = false;
    const auto inviscid = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    C::FiniteVolume noDiffusion(mesh, settings, inviscid), diffusion(mesh, settings, mixture.equation);
    auto a = C::EquationDofs::Coupled(inviscid.Schema(), noDiffusion.Metrics().BuildField(mixture.species + 4));
    auto b = C::EquationDofs::Coupled(mixture.equation.Schema(), diffusion.Metrics().BuildField(mixture.species + 4));
    mixture.Fill(a, noDiffusion.Metrics(), false);
    mixture.Fill(b, diffusion.Metrics(), false);
    auto wa = noDiffusion.BuildEquationWorkspace(a, mixture.workers), wb = diffusion.BuildEquationWorkspace(b, mixture.workers);
    auto ra = C::EquationDofOperations::NewVector(a), rb = C::EquationDofOperations::NewVector(b);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    noDiffusion.Evaluate(a, ra, *wa, engine, stream);
    diffusion.Evaluate(b, rb, *wb, engine, stream);
    struct FacePoint
    {
        I cell;
        DNDS::Geom::tPoint normal;
        double weight;
    };
    std::vector<FacePoint> entries;
    std::vector<I> queryCells;
    std::vector<DNDS::Geom::tPoint> queryPoints;
    auto meshRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
    const auto topology = meshRead.HostView();
    auto metricsRead = diffusion.Metrics().AcquireRead();
    const auto metric = metricsRead.HostView();
    for (I iCell = 0; iCell < diffusion.Metrics().NumCell(); ++iCell)
    {
        const auto faces = topology.CellFaces(iCell);
        for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
        {
            const I face = faces[slot], other = topology.CellFaceOther(iCell, face, slot);
            const int side = topology.CellIsFaceBack(iCell, face, slot) ? 0 : 1;
            REQUIRE(other >= 0);
            for (I q = metric.FaceQuadratureBegin(face); q < metric.FaceQuadratureEnd(face); ++q)
            {
                entries.push_back({iCell, (side ? -1 : 1) * metric.GetFaceNorm(q), metric.GetFaceWeight(q)});
                queryCells.push_back(iCell);
                queryCells.push_back(other);
                queryPoints.push_back(DNDS::CFV::Detail::FacePointInCell(topology, face, side, metric.GetFacePoint(q)));
                queryPoints.push_back(DNDS::CFV::Detail::FacePointInCell(topology, face, 1 - side, metric.GetFacePoint(q)));
            }
        }
    }
    const I count = I(queryCells.size()), components = mixture.species + 4;
    R::Array<I> cells(R::RowLayout::Dense(count, 1));
    R::Array<double> points(R::RowLayout::Dense(count, 3));
    {
        auto cw = cells.Write();
        auto pw = points.Write();
        for (I q = 0; q < count; ++q)
        {
            cw.View()(q, 0) = queryCells[q];
            for (int d = 0; d < 3; ++d)
                pw.View()(q, d) = queryPoints[q][d];
        }
    }
    DNDS::CFV::PolynomialSamplingPlan sampling(diffusion.Reconstruction(), cells, points);
    std::vector<R::Array<double>> samples;
    for (int derivative = 0; derivative < 3; ++derivative)
    {
        samples.emplace_back(R::RowLayout::Dense(count, components));
        sampling.Evaluate(b.PhysicalField(0), diffusion.FluxReconstruction(*wb).PhysicalField(0), samples.back(), derivative);
    }
    const auto valueRead = samples[0].Read(), dxRead = samples[1].Read(), dyRead = samples[2].Read();
    const auto values = valueRead.View<R::UniformTableRows>();
    const R::ArrayView<const double, R::AccessBoundsChecks, R::UniformTableRows> derivatives[2]{dxRead.View<R::UniformTableRows>(), dyRead.View<R::UniformTableRows>()};
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(diffusion.Metrics().NumCell(), mixture.species - 1);
    std::vector<double> face(size_t(components), 0), grad(size_t(components), 0), Y(size_t(mixture.species)), D(size_t(mixture.species)), J(size_t(mixture.species));
    for (I entry = 0; entry < I(entries.size()); ++entry)
    {
        const auto &quadrature = entries[entry];
        const I left = 2 * entry, right = left + 1;
        for (I j = 0; j < components; ++j)
        {
            face[j] = .5 * (values(left, j) + values(right, j));
            grad[j] = (values(right, j) - values(left, j)) / h;
            for (int d = 0; d < 2; ++d)
                grad[j] += .5 * (derivatives[d](left, j) + derivatives[d](right, j)) * quadrature.normal[d];
        }
        double sumY = 0, kinetic = 0;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            Y[k] = face[5 + k] / face[0];
            sumY += Y[k];
        }
        Y.back() = 1 - sumY;
        for (int d = 1; d <= 3; ++d)
            kinetic += .5 * face[d] * face[d] / face[0];
        const P::Chemistry::ConstSpeciesBufferView fractions{Y.data(), mixture.species};
        const double T = mixture.prototype.temperatureFromUV((face[4] - kinetic) / face[0] * mixture.scales.SpecificEnergy(), 1 / (face[0] * mixture.scales.rho0), fractions, 1400, 1e-12);
        const double p = face[0] * mixture.scales.rho0 * mixture.prototype.mixtureR(fractions) * T;
        P::Thermodynamics::Properties thermo;
        P::Transport::Properties transport;
        mixture.prototype.EvaluateProperties(T, p, fractions, thermo, &transport, {D.data(), mixture.species});
        double sumGradient = 0, sumJ = 0;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            const double gradient = (grad[5 + k] - Y[k] * grad[0]) / face[0];
            J[k] = -face[0] * D[k] / mixture.scales.Diffusivity() * gradient;
            sumGradient += gradient;
            sumJ += J[k];
        }
        J.back() = face[0] * D.back() / mixture.scales.Diffusivity() * sumGradient;
        sumJ += J.back();
        for (int k = 0; k < mixture.species - 1; ++k)
            expected(quadrature.cell, k) += quadrature.weight * (-J[k] + Y[k] * sumJ);
    }
    const auto ar = ra.ReadOwned(), br = rb.ReadOwned();
    const auto av = ar.Block("species"), bv = br.Block("species");
    double signal = 0;
    for (I iCell = 0; iCell < expected.rows(); ++iCell)
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            CHECK(bv(iCell, k) - av(iCell, k) == doctest::Approx(expected(iCell, k)).epsilon(2e-8).scale(1e-9));
            signal = std::max(signal, std::abs(expected(iCell, k)));
        }
    MPI_Allreduce(MPI_IN_PLACE, &signal, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    CHECK(signal > 1e-8); // Dropped/reversed diffusion cannot pass via a tiny signal.
}

TEST_CASE("Nonuniform chemical volume sampling agrees with independent Cartesian Gauss quadrature and direct backend rates")
{
    MixtureFixture mixture;
    const auto equation = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    constexpr I width = 4;
    constexpr double h = 1. / width;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, width);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.intOrder = 5;
    settings.numerics.reconstructionIterations = 0; // Prescribed nonconstant polynomial.
    settings.numerics.faceFallback = false;
    C::FiniteVolume withoutSource(mesh, settings, equation);
    settings.equation.source = "chemical";
    C::FiniteVolume withSource(mesh, settings, equation);
    auto a = C::EquationDofs::Coupled(equation.Schema(), withoutSource.Metrics().BuildField(mixture.species + 4));
    auto b = C::EquationDofs::Coupled(equation.Schema(), withSource.Metrics().BuildField(mixture.species + 4));
    mixture.Fill(a, withoutSource.Metrics(), true);
    mixture.Fill(b, withSource.Metrics(), true);
    auto wa = withoutSource.BuildEquationWorkspace(a, mixture.workers), wb = withSource.BuildEquationWorkspace(b, mixture.workers);
    for (const auto &[fv, work] : std::vector<std::pair<const C::FiniteVolume *, std::shared_ptr<C::FiniteVolume::EquationWorkspace>>>{{&withoutSource, wa}, {&withSource, wb}})
    {
        auto coefficients = fv->ReconstructionGuess(*work);
        C::EquationDofOperations::Fill(coefficients, 0);
        auto write = coefficients.WriteOwned();
        const auto flow = write.Block("flow");
        const I modes = fv->Reconstruction().NRecDOF();
        for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            flow(iCell, 4 * modes) = .4 * (1 + .25 * std::sin(6.283185307179586 * fv->Metrics().GetCellCent(iCell)[0]));
    }
    auto ra = C::EquationDofOperations::NewVector(a), rb = C::EquationDofOperations::NewVector(b);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    withoutSource.Evaluate(a, ra, *wa, engine, stream);
    withSource.Evaluate(b, rb, *wb, engine, stream);
    const I rows = withSource.Metrics().NumCell(), count = rows * 9;
    auto auxiliaryA = withoutSource.BuildAuxiliaryValues({false, true}), auxiliaryB = withSource.BuildAuxiliaryValues({false, true});
    withoutSource.PrepareAuxValues(a, auxiliaryA, 0, mixture.workers);
    withSource.PrepareAuxValues(b, auxiliaryB, 0, mixture.workers);
    R::Array<double> inverseTau(R::RowLayout::Dense(rows, 1));
    {
        auto write = inverseTau.Write();
        for (I iCell = 0; iCell < rows; ++iCell)
            write.View()(iCell, 0) = 0;
    }
    auto matrixA = withoutSource.BuildLinearProblem(a, C::DiagonalStorage::Block), matrixB = withSource.BuildLinearProblem(b, C::DiagonalStorage::Block);
    C::JacobianPreparation controls;
    withoutSource.PrepareJacobian(a, auxiliaryA, inverseTau, *matrixA, controls, 0, mixture.workers);
    const auto unpreparedVersion = matrixB->Diagonal().Version();
    CHECK_THROWS(withSource.PrepareJacobian(b, auxiliaryB, inverseTau, *matrixB, controls, 0, mixture.workers));
    CHECK(matrixB->Diagonal().Version() == unpreparedVersion);
    withSource.PrepareJacobian(b, auxiliaryB, inverseTau, *matrixB, controls, 0, mixture.workers, wb.get());

    R::Array<I> cells(R::RowLayout::Dense(count, 1));
    R::Array<double> points(R::RowLayout::Dense(count, 3));
    const double location[3]{-std::sqrt(3. / 5), 0, std::sqrt(3. / 5)}, weights[3]{5. / 9, 8. / 9, 5. / 9};
    {
        auto cw = cells.Write();
        auto pw = points.Write();
        for (I iCell = 0; iCell < rows; ++iCell)
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                {
                    const I q = iCell * 9 + j * 3 + i;
                    const auto center = withSource.Metrics().GetCellCent(iCell);
                    cw.View()(q, 0) = iCell;
                    pw.View()(q, 0) = center[0] + h * .5 * location[i];
                    pw.View()(q, 1) = center[1] + h * .5 * location[j];
                    pw.View()(q, 2) = 0;
                }
    }
    DNDS::CFV::PolynomialSamplingPlan sampling(withSource.Reconstruction(), cells, points, false);
    R::Array<double> sampled(R::RowLayout::Dense(count, mixture.species + 4));
    sampling.Evaluate(b.PhysicalField(0), withSource.FluxReconstruction(*wb).PhysicalField(0), sampled);
    const auto sampledRead = sampled.Read(), meanRead = b.PhysicalField(0).Owned().Read();
    const auto values = sampledRead.View<R::UniformTableRows>(), means = meanRead.View<R::UniformTableRows>();
    const auto lhs = ra.ReadOwned(), rhs = rb.ReadOwned();
    const auto x = lhs.Block("species"), y = rhs.Block("species");
    std::vector<double> fractions(size_t(mixture.species)), rates(size_t(mixture.species)), expected(size_t(mixture.species - 1));
    const int nv = mixture.species + 4;
    std::vector<double> rawJacobian(size_t(mixture.species * nv));
    Eigen::MatrixXd expectedJacobian(nv, nv);
    const auto rateAt = [&](const auto &input, I row)
    {
        double sum = 0, kinetic = 0;
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            fractions[k] = input(row, 5 + k) / input(row, 0);
            sum += fractions[k];
        }
        fractions.back() = 1 - sum;
        for (int d = 1; d <= 3; ++d)
            kinetic += .5 * input(row, d) * input(row, d) / input(row, 0);
        const P::Chemistry::ConstSpeciesBufferView Y{fractions.data(), mixture.species};
        const double T = mixture.prototype.temperatureFromUV((input(row, 4) - kinetic) / input(row, 0) * mixture.scales.SpecificEnergy(), 1 / (input(row, 0) * mixture.scales.rho0), Y, 1400, 1e-12);
        const double p = input(row, 0) * mixture.scales.rho0 * mixture.prototype.mixtureR(Y) * T;
        mixture.prototype.productionRatesAndJacobian(T, p, input(row, 0), input(row, 4), input(row, 1), input(row, 2), input(row, 3), 4, Y,
                                                     {rates.data(), mixture.species}, {rawJacobian.data(), mixture.species, nv, mixture.species});
        for (int k = 0; k < mixture.species - 1; ++k)
            rates[k] *= mixture.prototype.molecularWeights()[k] / mixture.scales.SourceRate();
    };
    double signal[4]{};
    {
        const auto da = matrixA->Diagonal().Read(), db = matrixB->Diagonal().Read();
        for (I iCell = 0; iCell < rows; ++iCell)
        {
            std::fill(expected.begin(), expected.end(), 0);
            expectedJacobian.setZero();
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                {
                    rateAt(values, iCell * 9 + j * 3 + i);
                    for (int k = 0; k < mixture.species - 1; ++k)
                    {
                        expected[k] += h * h * .25 * weights[i] * weights[j] * rates[k];
                        for (int column = 0; column < nv; ++column)
                            expectedJacobian(5 + k, column) += h * h * .25 * weights[i] * weights[j] * mixture.prototype.molecularWeights()[k] / mixture.scales.SourceRate() * rawJacobian[k + mixture.species * column];
                    }
                }
            rateAt(means, iCell);
            for (int column = 0; column < nv; ++column)
                for (int row = 0; row < nv; ++row)
                {
                    const double observed = db.View()(iCell, row + nv * column) - da.View()(iCell, row + nv * column);
                    CHECK(observed == doctest::Approx(-expectedJacobian(row, column)).epsilon(1e-6).scale(1e-8));
                    signal[2] = std::max(signal[2], std::abs(expectedJacobian(row, column)));
                    if (row >= 5)
                    {
                        const double meanOnly = h * h * mixture.prototype.molecularWeights()[row - 5] / mixture.scales.SourceRate() * rawJacobian[row - 5 + mixture.species * column];
                        signal[3] = std::max(signal[3], std::abs(expectedJacobian(row, column) - meanOnly));
                    }
                }
            for (int k = 0; k < mixture.species - 1; ++k)
            {
                const double observed = y(iCell, k) - x(iCell, k);
                CHECK(observed == doctest::Approx(expected[k]).epsilon(1e-8).scale(1e-9));
                signal[0] = std::max(signal[0], std::abs(expected[k]));
                signal[1] = std::max(signal[1], std::abs(expected[k] - h * h * rates[k]));
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, signal, 4, MPI_DOUBLE, MPI_MAX, mpi.comm);
    CHECK(signal[0] > 1e-10);
    CHECK(signal[1] > 1e-6 * signal[0]); // A mean/centroid-only source must fail this gate.
    CHECK(signal[2] > 1e-8);
    CHECK(signal[3] > 1e-6 * signal[2]); // The same guard applies to the source Jacobian.
    const auto version = matrixB->Diagonal().Version();
    {
        auto changed = withSource.FluxReconstruction(*wb).PhysicalField(0).Owned();
        auto write = changed.Write();
        const auto coefficients = write.View<R::UniformTableRows>();
        if (coefficients.RowCount())
            coefficients(0, 4 * withSource.Reconstruction().NRecDOF()) += 1e-8;
    }
    CHECK_THROWS(withSource.PrepareJacobian(b, auxiliaryB, inverseTau, *matrixB, controls, 0, mixture.workers, wb.get()));
    CHECK(matrixB->Diagonal().Version() == version);
}
TEST_CASE("Equation P3 positivity protects actual EOS samples and preserves unlimited guesses in split and permuted layouts")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 4;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.numerics.positivity = true;
        settings.numerics.temperatureFloor = 900 / mixture.scales.T0;
        settings.numerics.positivityCompressToMean = !scenario;
        settings.numerics.positivityVolumePoints = !scenario;
        settings.equation.source = "chemical"; // Its quadrature is mandatory even when the optional volume grid is off.
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto state = mixture.State(fv.Metrics().Space(), scenario + 1);
        mixture.Fill(state, fv.Metrics(), true);
        auto work = fv.BuildEquationWorkspace(state, mixture.workers);
        auto guess = fv.ReconstructionGuess(*work);
        const int modes = fv.Reconstruction().NRecDOF();
        {
            auto write = guess.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
            {
                flow(iCell, 4 * modes) = scenario ? .002 : 300;
                if (scenario)
                {
                    flow(iCell, 4 * modes + 2) = -300;
                    flow(iCell, 4 * modes + 5) = 100;
                }
            }
        }
        const C::EquationDofOperations::Binding meansBefore(state), guessBefore(guess);
        fv.LimitReconstructionPP(state, *work);
        CHECK_NOTHROW(meansBefore.Check(state, true));
        CHECK_NOTHROW(guessBefore.Check(guess, true));
        const auto statistics = fv.PositivityReconstructionStats(*work);
        CHECK(statistics.first > 0);
        CHECK(statistics.second == 0);
        {
            const auto read = fv.FluxReconstruction(*work).ReadOwned();
            const auto flow = read.Block("flow");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                CHECK(flow(iCell, 4 * modes) == (scenario ? .002 : 0));
                for (int mode = 2; mode < modes; ++mode)
                    CHECK(flow(iCell, 4 * modes + mode) == 0);
            }
        }
        // Independently query centers and the chemical integration points,
        // then recover T with the actual EOS, not the limiter's bound checker.
        const I count = fv.Metrics().NumCell() * 9;
        R::Array<I> cells(R::RowLayout::Dense(count, 1));
        R::Array<double> points(R::RowLayout::Dense(count, 3));
        {
            auto cw = cells.Write();
            auto pw = points.Write();
            const auto c = cw.View();
            const auto p = pw.View();
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto center = fv.Metrics().GetCellCent(iCell);
                for (int k = 0; k < 9; ++k)
                {
                    const I q = 9 * iCell + k;
                    c(q, 0) = iCell;
                    p(q, 0) = center[0] + (k % 3 - 1) * .25 * .5 * std::sqrt(.6);
                    p(q, 1) = center[1] + (k / 3 - 1) * .25 * .5 * std::sqrt(.6);
                    p(q, 2) = center[2];
                }
            }
        }
        DNDS::CFV::PolynomialSamplingPlan sampler(fv.Reconstruction(), cells, points, false);
        std::vector<R::Array<double>> values;
        for (I f = 0; f < state.PhysicalFieldNumber(); ++f)
        {
            values.emplace_back(R::RowLayout::Dense(count, I(state.PhysicalField(f).Width())));
            sampler.Evaluate(state.PhysicalField(f), fv.FluxReconstruction(*work).PhysicalField(f), values.back());
        }
        std::vector<R::Array<double>::ReadLease> leases;
        for (const auto &value : values)
            leases.emplace_back(value.Read());
        const auto fl = state.LocateBlock("flow"), sp = state.LocateBlock("species");
        const auto flow = leases[fl.physicalField].View(), species = leases[sp.physicalField].View();
        {
            auto evaluator = mixture.equation.MakePointEvaluator(mixture.workers.Acquire(), {});
            auto rhoY = evaluator.SpeciesInput();
            for (I q = 0; q < count; ++q)
            {
                C::FlowState u;
                for (int j = 0; j < 5; ++j)
                    u[j] = flow(q, fl.firstComponent + j);
                for (int k = 0; k < mixture.species - 1; ++k)
                    rhoY[k] = species(q, sp.firstComponent + k);
                CHECK(evaluator.Evaluate(u, rhoY).thermo.temperature > settings.numerics.temperatureFloor);
            }
        }
        // Run the full P3 PP/source orchestration from a cold uniform guess.
        // A uniform periodic state has zero flux divergence, leaving V*S(U).
        C::EquationDofOperations::Fill(guess, 0);
        auto residual = C::EquationDofOperations::NewVector(state), rates = C::EquationDofOperations::NewVector(state);
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        fv.Evaluate(state, residual, *work, engine, stream);
        C::EvaluateChemicalSource(mixture.equation, state, rates, mixture.workers.Acquire());
        const auto rr = residual.ReadOwned(), sr = rates.ReadOwned();
        const auto rl = rr.ComponentLayout(), sl = sr.ComponentLayout();
        const auto r = rl.View(), source = sl.View();
        for (I iCell = 0; iCell < r.RowCount(); ++iCell)
            for (I j = 0; j < r.ComponentNumber(); ++j)
                CHECK(r(iCell, j) == doctest::Approx(fv.Metrics().GetCellVol(iCell) * source(iCell, j)).epsilon(1e-10).scale(1));
        CHECK_NOTHROW(meansBefore.Check(state, true));
    }
}
TEST_CASE("Equation workspace agrees PP controls before binding conditional halo fields")
{
    if (mpi.size < 2)
        return;
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.numerics.reconstruction = "constant";
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.faceFallback = false;
    settings.numerics.positivity = mpi.rank == 0;
    C::FiniteVolume fv(mesh, settings, mixture.equation);
    auto state = mixture.State(fv.Metrics().Space(), 1);
    mixture.Fill(state, fv.Metrics(), true);
    CHECK_THROWS_WITH(fv.BuildEquationWorkspace(state, mixture.workers), doctest::Contains("operation controls"));
}
TEST_CASE("Equation implicit PP compresses RHS or pseudo-time and repairs species without changing flow")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = "constant";
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.numerics.positivity = true;
        settings.numerics.temperatureFloor = 300 / mixture.scales.T0;
        settings.implicit.iterations = 1;
        settings.implicit.relativeTolerance = settings.implicit.absoluteTolerance = 0;
        settings.implicit.positivity.rhsMode = scenario ? "dTau" : "rhs";
        settings.implicit.positivity.rhsReduction = scenario ? "minimumAll" : "cell";
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto state = mixture.State(fv.Metrics().Space(), scenario + 1);
        mixture.Fill(state, fv.Metrics(), true);
        auto source = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Copy(source, state);
        C::EquationDofOperations::Scale(source, 20);
        {
            auto write = source.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                if (fv.Metrics().Cells().LocalToGlobal(iCell) == 0)
                    flow(iCell, 0) = -1e8;
        }
        R::EagerEngine engine(2);
        const auto stream = engine.CreateCPUStream();
        C::EquationNonlinearSolver solver(fv, state, fv.BuildEquationWorkspace(state, mixture.workers), engine, stream,
                                          settings, A::ComponentSelection(state.ComponentNumber()), mixture.workers);
        const auto report = solver.Solve(state, source, 20, 1, .1);
        CHECK(report.iterations == 1);
        CHECK_FALSE(report.converged);
        CHECK(report.rhsLimited == 1);
        CHECK(report.minimumRHSAlpha > 0);
        CHECK(report.minimumRHSAlpha < 1);
        auto old = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Copy(old, state);
        auto increment = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Fill(increment, 0);
        {
            auto write = increment.WriteOwned();
            const auto species = write.Block("species");
            for (I iCell = 0; iCell < species.RowCount(); ++iCell)
                species(iCell, 0) = -100;
        }
        CHECK_NOTHROW(solver.ApplyCorrection(state, increment));
        {
            const auto read = state.ReadOwned(), before = old.ReadOwned();
            const auto flow = read.Block("flow"), prior = before.Block("flow"), species = read.Block("species");
            for (I iCell = 0; iCell < flow.owned.RowCount(); ++iCell)
            {
                for (int j = 0; j < 5; ++j)
                    CHECK(flow(iCell, j) == prior(iCell, j));
                CHECK(species(iCell, 0) >= 1e-30);
                double sum = 0;
                for (int k = 0; k < mixture.species - 1; ++k)
                    sum += species(iCell, k);
                CHECK(sum <= flow(iCell, 0));
            }
        }
        // A masked flow correction may not silently repair frozen species.
        // Remove kinetic-energy protection: a density-only trial can then
        // reach the simplex boundary while its thermal energy stays safe.
        {
            auto write = state.WriteOwned();
            const auto flow = write.Block("flow");
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                for (int d = 1; d <= 3; ++d)
                    flow(iCell, d) = 0;
        }
        C::EquationDofOperations::Fill(increment, 0);
        {
            const auto read = state.ReadOwned();
            const auto mean = read.Block("flow");
            auto write = increment.WriteOwned();
            const auto flow = write.Block("flow");
            // The power-two Euler compression makes an arbitrarily huge
            // increment almost vanish. A 1.01*rho decrement instead reaches
            // about .01*rho while retaining safe internal energy here.
            for (I iCell = 0; iCell < flow.RowCount(); ++iCell)
                flow(iCell, 0) = -1.01 * mean(iCell, 0);
        }
        const C::EquationDofOperations::Binding before(state);
        CHECK_THROWS(solver.ApplyCorrection(state, increment, 1, {0}));
        CHECK_NOTHROW(before.Check(state, true));
    }
}
TEST_CASE("Coupled mixture nonlinear stage recovers a manufactured target through separate or permuted arrays")
{
    MixtureFixture mixture;
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    for (int scenario = 0; scenario < 2; ++scenario)
    {
        C::DomainSettings settings;
        settings.numerics.reconstruction = scenario ? "variational" : "constant";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 40;
        settings.numerics.reconstructionTolerance = 1e-9;
        settings.numerics.viscousGradient = "polynomial";
        settings.numerics.faceFallback = false;
        settings.implicit.requireConvergedReconstruction = scenario;
        settings.implicit.linear.method = "gmres";
        settings.implicit.linear.preconditioner = scenario ? "sgs" : "ilu";
        settings.implicit.linear.relativeTolerance = 1e-5;
        settings.implicit.relativeTolerance = 1e-8;
        settings.implicit.absoluteTolerance = 1e-10;
        settings.implicit.diagonal = "block";
        settings.implicit.jacobianAction = scenario ? "semiMatrixFree" : "assembled";
        settings.implicit.collectiveChecks = scenario;
        settings.execution.haloFreshnessChecks = true; // Umbrella off/on both run the same halo protocol.
        settings.execution.overlap = scenario;
        C::FiniteVolume fv(mesh, settings, mixture.equation);
        auto target = mixture.State(fv.Metrics().Space(), 0), state = mixture.State(fv.Metrics().Space(), scenario + 1);
        mixture.Fill(target, fv.Metrics(), true);
        C::EquationDofOperations::Copy(state, target);
        {
            auto write = state.WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto u = layout.View();
            for (I iCell = 0; iCell < u.RowCount(); ++iCell)
                for (I j = 0; j < u.ComponentNumber(); ++j)
                    u(iCell, j) *= 1 + .001 * std::sin(2 * iCell + mpi.rank + j);
        }
        auto source = C::EquationDofOperations::NewVector(target);
        C::EquationDofOperations::Copy(source, target);
        C::EquationDofOperations::Scale(source, 20);
        R::EagerEngine engine(scenario ? 2 : 0, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        auto work = fv.BuildEquationWorkspace(state, mixture.workers);
        C::EquationNonlinearSolver solver(fv, state, work, engine, stream, settings, A::ComponentSelection(state.ComponentNumber()), mixture.workers);
        const auto report = solver.Solve(state, source, 20, .5, .1);
        CAPTURE(scenario);
        CAPTURE(report.iterations);
        CAPTURE(report.residualTargetRatio);
        REQUIRE(report.converged);
        CHECK(report.linearIterations > 0);
        CHECK(report.residual.size() == size_t(mixture.species + 4));
        CHECK(report.residual.back() < 1e-9);
        CHECK(solver.ReconstructionConverged());
        Compare(state, target, 2e-9);
        // A rejected negative species update must not mutate any physical field.
        auto increment = C::EquationDofOperations::NewVector(state);
        C::EquationDofOperations::Fill(increment, 0);
        {
            auto write = increment.WriteOwned();
            const auto species = write.Block("species");
            for (I iCell = 0; iCell < species.RowCount(); ++iCell)
                species(iCell, 0) = -100;
        }
        const C::EquationDofOperations::Binding before(state);
        CHECK_THROWS(solver.ApplyCorrection(state, increment));
        CHECK_NOTHROW(before.Check(state, true));
        CHECK_THROWS(solver.Solve(state, state, 20, .5, .1));
    }
}

TEST_CASE("Production equation nonlinear stages advance a distributed reacting FV state toward Cantera")
{
    MixtureFixture mixture;
    const auto equation = C::EquationSystem::FromIdealGasMixture(mixture.material, false);
    const auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    C::DomainSettings settings;
    settings.equation.source = "chemical";
    settings.numerics.reconstruction = "constant";
    settings.numerics.faceFallback = false;
    settings.implicit.linear.method = "gmres";
    settings.implicit.linear.preconditioner = "ilu";
    settings.implicit.linear.relativeTolerance = 1e-10;
    settings.implicit.linear.absoluteTolerance = 1e-13;
    settings.implicit.diagonal = "block";
    settings.implicit.pseudoCFL = 1e4;
    settings.implicit.maxPseudoCFL = 1e8;
    settings.implicit.relativeTolerance = 1e-10;
    settings.implicit.absoluteTolerance = 1e-8;
    C::FiniteVolume fv(mesh, settings, equation);
    auto initial = mixture.State(fv.Metrics().Space(), 0);
    mixture.Fill(initial, fv.Metrics(), true);
    std::vector<double> expected(size_t(mixture.species + 4)), fractions(size_t(mixture.species), 1e-6);
    {
        // Every rank uses the same analytic initial point, including any empty
        // owner. Do not infer reference composition from a local ghost row.
        fractions[0] = .027;
        fractions[3] = .22;
        fractions[5] = .02;
        fractions.back() = 0;
        fractions.back() = 1 - std::accumulate(fractions.begin(), fractions.end(), 0.);
        const double rho = .8;
        const P::Chemistry::ConstSpeciesBufferView y{fractions.data(), mixture.species};
        const double p = rho * mixture.scales.rho0 * mixture.prototype.mixtureR(y) * 1400;
        expected[0] = rho;
        expected[1] = rho * .2;
        expected[2] = rho * -.04;
        expected[3] = rho * .03;
        expected[4] = rho * mixture.prototype.mixtureIntEnergy(1400, y, p) / mixture.scales.SpecificEnergy() + .5 * rho * (.04 + .0016 + .0009);
    }
    constexpr double totalTime = 1e-5;
    double temperature = 1400;
    mixture.prototype.advanceConstVolume(temperature, expected[0] * mixture.scales.rho0, {fractions.data(), mixture.species}, 1,
                                         totalTime * mixture.scales.Time(), 1e-11, 1e-20);
    for (int k = 0; k < mixture.species - 1; ++k)
        expected[5 + k] = expected[0] * fractions[k];
    const auto run = [&](int steps)
    {
        auto accepted = mixture.State(fv.Metrics().Space(), 2), trial = mixture.State(fv.Metrics().Space(), 1);
        C::EquationDofOperations::Copy(accepted, initial);
        C::EquationDofOperations::Copy(trial, initial);
        R::EagerEngine engine(2, std::make_shared<R::AccessTrackingContext>());
        const auto stream = engine.CreateCPUStream();
        auto work = fv.BuildEquationWorkspace(trial, mixture.workers);
        C::EquationNonlinearSolver nonlinear(fv, trial, work, engine, stream, settings, A::ComponentSelection(trial.ComponentNumber()), mixture.workers);
        C::EquationTimeStepper stepper(accepted, "bdf1");
        const auto evaluate = [&](auto &state, auto &rate, double time, int)
        {
            fv.Evaluate(state, rate, *work, engine, stream, time);
            const auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto write = rate.WriteOwned();
            const auto layout = write.ComponentLayout();
            const auto values = layout.View();
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (I j = 0; j < values.ComponentNumber(); ++j)
                    values(iCell, j) /= metric.GetCellVol(iCell);
        };
        const auto solve = [&](auto &state, const auto &source, double mass, double scale, double time, int stage)
        {
            const auto report = nonlinear.Solve(state, source, mass, scale, time, stage);
            CHECK(report.converged);
            CHECK(report.residual.size() == expected.size());
            return report.converged;
        };
        for (int step = 0; step < steps; ++step)
        {
            REQUIRE(stepper.Prepare(accepted, trial, step * totalTime / steps, totalTime / steps, evaluate, solve));
            C::EquationDofOperations::Copy(accepted, trial);
            stepper.Commit();
        }
        double error = 0, change = 0;
        const auto read = accepted.ReadOwned(), old = initial.ReadOwned();
        const auto layout = read.ComponentLayout(), oldLayout = old.ComponentLayout();
        const auto u = layout.View(), before = oldLayout.View();
        for (I iCell = 0; iCell < u.RowCount(); ++iCell)
            for (I j = 0; j < u.ComponentNumber(); ++j)
            {
                error = std::max(error, std::abs(u(iCell, j) - expected[j]));
                change = std::max(change, std::abs(u(iCell, j) - before(iCell, j)));
                if (j < 5)
                    CHECK(u(iCell, j) == doctest::Approx(before(iCell, j)).epsilon(1e-12));
            }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
        CHECK(change > 1e-9);
        return error;
    };
    const double coarse = run(4), fine = run(8);
    CAPTURE(coarse);
    CAPTURE(fine);
    CHECK(coarse > 1e-12);
    CHECK(fine < .8 * coarse);
    CHECK(fine < 1e-5);
}

#endif

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.setWorld();
    const int result = doctest::Context(argc, argv).run();
    MPI_Finalize();
    return result;
}
