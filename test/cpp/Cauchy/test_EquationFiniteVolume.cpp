#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include <cstdlib>
#include <numeric>

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
    }
}

TEST_CASE("Equation wall values preserve internal energy and support empty owners on a physical 3D mesh")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 1);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.equation.transport = "NewtonianFourier";
    settings.equation.viscosity = .002;
    settings.numerics.viscousGradient = "polynomial";
    settings.numerics.faceFallback = false;
    settings.boundaries["7"].kind = "noSlipWall";
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
}

#ifdef DNDS_USE_CANTERA
namespace
{
    struct MixtureFixture
    {
        const P::ReferenceScales scales{1.225, 340, 2, 288.15};
        P::Chemistry::ChemicalSource prototype;
        P::Thermodynamics::IdealGasMixture material;
        P::Chemistry::ChemicalWorkerPool workers;
        C::EquationSystem equation;
        int species;
        MixtureFixture() : prototype((std::getenv("DNDS_MECH_PATH") ? std::string(std::getenv("DNDS_MECH_PATH")) + "/" : "") + "h2o2.yaml", "", scales.U0, scales.rho0),
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
        // One late physical output conflict must reject on all ranks before
        // a field halo is posted or an earlier output generation is published.
        const auto version = result.PhysicalField(0).Owned().Version();
        std::optional<R::Array<double>::ReadLease> pin;
        if (mpi.rank == 0)
            pin.emplace(result.PhysicalField(result.PhysicalFieldNumber() - 1).Owned().Read());
        CHECK_THROWS(fv.Evaluate(state, result, *work, engine, stream));
        CHECK(result.PhysicalField(0).Owned().Version() == version);
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
