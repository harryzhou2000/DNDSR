#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationSource.hpp"
#include "CartesianFixture.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Math/PeriodicFrame.hpp"
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
        mixture.prototype.productionRates(T, p, Y, {rates.data(), mixture.species});
        for (int k = 0; k < mixture.species - 1; ++k)
            rates[k] *= mixture.prototype.molecularWeights()[k] / mixture.scales.SourceRate();
    };
    double signal[2]{};
    for (I iCell = 0; iCell < rows; ++iCell)
    {
        std::fill(expected.begin(), expected.end(), 0);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
            {
                rateAt(values, iCell * 9 + j * 3 + i);
                for (int k = 0; k < mixture.species - 1; ++k)
                    expected[k] += h * h * .25 * weights[i] * weights[j] * rates[k];
            }
        rateAt(means, iCell);
        for (int k = 0; k < mixture.species - 1; ++k)
        {
            const double observed = y(iCell, k) - x(iCell, k);
            CHECK(observed == doctest::Approx(expected[k]).epsilon(1e-8).scale(1e-9));
            signal[0] = std::max(signal[0], std::abs(expected[k]));
            signal[1] = std::max(signal[1], std::abs(expected[k] - h * h * rates[k]));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, signal, 2, MPI_DOUBLE, MPI_MAX, mpi.comm);
    CHECK(signal[0] > 1e-10);
    CHECK(signal[1] > 1e-6 * signal[0]); // A mean/centroid-only source must fail this gate.
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
