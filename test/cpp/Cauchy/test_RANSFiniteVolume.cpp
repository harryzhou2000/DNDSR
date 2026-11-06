#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/Physics/EquationSetup.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Core/Simulation.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CartesianFixture.hpp"
#include "Geom/Quadratures/Quadrature.hpp"
#include <limits>

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
namespace A = DNDS::Algebra;
using I = DNDS::index;
static DNDS::MPIInfo mpi;
namespace
{
    C::DomainSettings Settings()
    {
        C::DomainSettings settings;
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .001;
        settings.numerics.reconstruction = "constant";
        settings.numerics.limiter = "none";
        settings.numerics.faceFallback = false;
        settings.numerics.characteristicLimiter = false;
        settings.numerics.viscousGradient = "polynomial";
        settings.boundaries["7"].kind = "extrapolate";
        return settings;
    }
    C::EquationSystem Equation(const C::DomainSettings &settings)
    {
        DNDS::Physics::RANS::ModelSettings model;
        model.model = "SA";
        model.viscosityReference = .01;
        return C::EquationSystem::Resolve(settings.equation).WithRANS(model);
    }
    void Fill(C::EquationDofs &state, const C::FiniteVolume &fv, bool nonconstant, double omega = .1)
    {
        auto write = state.WriteOwned();
        const auto flow = write.Block("flow"), sa = write.Block("rans");
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const auto center = fv.Metrics().GetCellCent(iCell);
            const double rho = nonconstant ? 1 + .03 * std::sin(6.283185307179586 * center[0]) : 1;
            const auto value = fv.Equation().Gas().Conservative({rho, nonconstant ? .1 : 0., nonconstant ? .05 * std::cos(6.283185307179586 * center[1]) : 0., 0, 1});
            for (int j = 0; j < 5; ++j)
                flow(iCell, j) = value[j];
            sa(iCell, 0) = nonconstant ? .02 * (1 + .1 * std::cos(6.283185307179586 * center[0])) : .02;
            if (sa.ComponentNumber() == 2)
                sa(iCell, 1) = omega;
        }
    }
    C::EquationDofs State(const C::FiniteVolume &fv, bool split)
    {
        const I components = fv.Equation().RANS().ComponentNumber();
        A::DofBundle fields(fv.Metrics().Space());
        std::vector<C::EquationDofBinding> bindings;
        if (split)
        {
            fields.Add("flow", fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 5, {16, 1})));
            fields.Add("rans", fv.Metrics().BuildField(R::RowLayout::TiledColMajor(0, components, 32)));
            bindings = {{"flow", "flow", 0}, {"rans", "rans", 0}};
        }
        else
        {
            fields.Add("coupled", fv.Metrics().BuildField(5 + components));
            bindings = {{"flow", "coupled", components}, {"rans", "coupled", 0}};
        }
        return {fv.Equation().Schema(), std::move(fields), std::move(bindings)};
    }
}

TEST_CASE("Explicit RANS repairs scalars separately from the uncompressed flow5 update")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    for (int workers : {0, 2})
    {
        auto settings = Settings();
        settings.boundaries.clear();
        settings.equation.rans.model = "KOmegaWilcox";
        settings.initialPrimitive = {1, 0, 0, 0, 1};
        settings.initialRANS = {{"rhoK", .02}, {"rhoOmega", 100}};
        settings.numerics.ransState.referenceConservative = {{"rho", 1}, {"rhoK", .02}, {"rhoOmega", 100}};
        settings.numerics.positivity = true;
        settings.execution.hostThreads = workers;
        settings.execution.layout = workers ? "soa" : "aos";
        settings.execution.fieldGrouping = workers ? "coupled" : "equationBlocks";
        settings.execution.overlap = bool(workers);
        const auto config = C::SimulationConfig::ResolveCollective({{"defaults", C::Json(settings)}}, mpi.comm);
        const auto factory = [&](auto, const auto &, const auto &)
        { return mesh; };
        C::Simulation simulation(config, mpi.comm, factory);
        auto &domain = simulation.GetDomain(0);
        const auto original = domain.Dofs().Clone();
        double criticalDt = std::numeric_limits<double>::infinity();
        {
            const auto &rhs = domain.ResidualState();
            const auto initial = original.ReadOwned(), residual = rhs.ReadOwned();
            const auto values = initial.Block("rans"), rate = residual.Block("rans");
            for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                for (I j = 0; j < values.ComponentNumber(); ++j)
                    if (rate(iCell, j) < 0)
                        criticalDt = std::min(criticalDt, -values(iCell, j) * domain.Discretization().Metrics().GetCellVol(iCell) / rate(iCell, j));
        }
        MPI_Allreduce(MPI_IN_PLACE, &criticalDt, 1, MPI_DOUBLE, MPI_MIN, mpi.comm);
        REQUIRE(std::isfinite(criticalDt));
        REQUIRE(criticalDt > 0);
        // The raw first stage has a negative RANS scalar. Its independent
        // scalar repair is allowed, but must not compress any flow component.
        const double dt = 2 * criticalDt;
        simulation.Advance(dt);
        CHECK(simulation.Time() == dt);
        CHECK(simulation.Step() == 1);
        CHECK(domain.PreviousTimeStep() == dt);
        // Independent Shu-Osher assembly over the public spatial operator:
        // raw flow5 plus the separate Euler exponential scalar repair.
        auto oracle = original.Clone();
        auto rhs = C::EquationDofOperations::NewVector(oracle);
        const auto &fv = domain.Discretization();
        auto work = fv.BuildEquationWorkspace(oracle);
        R::EagerEngine engine(workers);
        const auto stream = engine.CreateCPUStream();
        const double history[3]{0, .75, 1. / 3}, weight[3]{1, .25, 2. / 3}, stageTime[3]{0, dt, .5 * dt};
        int repaired = 0;
        for (int stage = 0; stage < 3; ++stage)
        {
            fv.Evaluate(oracle, rhs, *work, engine, stream, stageTime[stage]);
            const auto base = original.ReadOwned(), residual = rhs.ReadOwned();
            auto output = oracle.WriteOwned();
            for (I iBlock = 0; iBlock < oracle.BlockNumber(); ++iBlock)
            {
                const auto initial = base.Block(iBlock), rate = residual.Block(iBlock);
                const auto values = output.Block(iBlock);
                for (I iCell = 0; iCell < values.RowCount(); ++iCell)
                    for (I j = 0; j < values.ComponentNumber(); ++j)
                    {
                        const double old = values(iCell, j);
                        double next = history[stage] * initial(iCell, j) + weight[stage] *
                                                                               (values(iCell, j) + dt * rate(iCell, j) / fv.Metrics().GetCellVol(iCell));
                        if (oracle.Schema().blocks[iBlock].name == "rans")
                        {
                            const double change = next - old;
                            if (old + change < 0)
                            {
                                next = old + (old * std::exp(change / (old + 1e-6)) - old);
                                ++repaired;
                            }
                            else
                                next = old + change;
                            if (j == 1)
                                next = std::max(next, 1.); // .01 * prescribed rhoOmega.
                        }
                        values(iCell, j) = next;
                    }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &repaired, 1, MPI_INT, MPI_SUM, mpi.comm);
        CHECK(repaired > 0);
        const auto result = domain.Dofs().ReadOwned(), expected = oracle.ReadOwned(), before = original.ReadOwned();
        for (I iBlock = 0; iBlock < oracle.BlockNumber(); ++iBlock)
        {
            const auto actual = result.Block(iBlock), raw = expected.Block(iBlock);
            for (I iCell = 0; iCell < actual.RowCount(); ++iCell)
                for (I j = 0; j < actual.ComponentNumber(); ++j)
                {
                    CHECK(actual(iCell, j) == raw(iCell, j));
                    if (oracle.Schema().blocks[iBlock].name == "flow")
                        CHECK(actual(iCell, j) == doctest::Approx(before.Block(iBlock)(iCell, j)).epsilon(2e-13).scale(1));
                }
        }
    }
}

TEST_CASE("RANS wall force uses the completed RHS momentum flux with the physical pressure sign")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    auto settings = Settings();
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "noSlipWall";
    const auto equation = Equation(settings);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = State(fv, true);
    auto result = C::EquationDofOperations::NewVector(state);
    {
        auto write = state.WriteOwned();
        const auto flow = write.Block("flow"), sa = write.Block("rans");
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            const double pressure = 1 + .2 * fv.Metrics().GetCellCent(iCell)[0];
            for (int j = 0; j < 5; ++j)
                flow(iCell, j) = j == 0 ? 1 : j == 4 ? pressure / .4
                                                     : 0;
            sa(iCell, 0) = .02;
        }
    }
    auto work = fv.BuildEquationWorkspace(state);
    CHECK_THROWS(fv.LastBoundaryForce(*work, {7}));
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, result, *work, engine, stream);
    const auto force = fv.LastBoundaryForce(*work, {7});
    // Piecewise-constant pressure on the two x-walls is 1.05 / 1.15.
    // Each wall has area one, and the normal points out of the fluid.
    CHECK(force[0] == doctest::Approx(.1).epsilon(2e-12));
    CHECK(std::abs(force[1]) < 1e-12);
    CHECK(std::abs(force[2]) < 1e-12);
    auto read = result.ReadOwned();
    const auto momentum = read.Block("flow");
    std::array<double, 3> integral{};
    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        for (int d = 0; d < 3; ++d)
            integral[d] += momentum(iCell, 1 + d);
    MPI_Allreduce(MPI_IN_PLACE, integral.data(), 3, MPI_DOUBLE, MPI_SUM, mpi.comm);
    for (int d = 0; d < 3; ++d)
        CHECK(integral[d] == doctest::Approx(-force[d]).epsilon(2e-12).scale(1));
}

TEST_CASE("RANS FV integrates quadrature sources and their positive implicit destruction diagonal")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    {
        // Fully degenerate cube inertia must select global axes exactly,
        // without amplifying quadrature roundoff into a rotated hMax.
        const auto metrics = DNDS::CFV::FiniteVolumePlan::ConstructMetrics(mesh);
        const auto lengths = metrics.BuildCellMaxLenScale();
        const auto read = lengths.Read();
        for (I iCell = 0; iCell < metrics.NumCellProc(); ++iCell)
            CHECK(read.View<R::FixedRows<1>>()(iCell, 0) == doctest::Approx(.5).epsilon(2e-12));
    }
    // Distinct principal inertias give an independent, well-conditioned
    // nonunit hMax oracle, in addition to the degenerate cube above.
    const Eigen::Vector3d side{1, 2, 3};
    mesh->TransformCoords([&](const auto &point) -> DNDS::Geom::tPoint
                          { return point.cwiseProduct(side); });
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    for (const auto *reconstruction : {"constant", "variational"})
        for (const auto *name : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
        {
            CAPTURE(name);
            CAPTURE(reconstruction);
            auto settings = Settings();
            settings.mesh.dimension = 3;
            settings.numerics.reconstruction = reconstruction;
            settings.numerics.cfv.maxOrder = 1;
            settings.numerics.reconstructionIterations = 1;
            auto modelSettings = Equation(settings).RANS().Settings();
            modelSettings.model = name;
            if (std::string(name) == "SA" || std::string(name) == "KOmegaSST")
            {
                modelSettings.desScale = .01;
                modelSettings.desMode = 3;
            }
            const auto equation = C::EquationSystem::Resolve(settings.equation).WithRANS(modelSettings);
            const int components = equation.RANS().ComponentNumber(), width = 5 + components;
            C::FiniteVolume fv(mesh, settings, equation);
            const auto lengths = fv.Metrics().BuildCellMaxLenScale();
            {
                const auto read = lengths.Read();
                const auto values = read.View<R::FixedRows<1>>();
                for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
                    CHECK(values(iCell, 0) == doctest::Approx(1.5).epsilon(2e-12));
            }
            auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(width));
            auto result = C::EquationDofOperations::NewVector(state);
            // SST's F2 shielding must leave some off-wall quadrature points in
            // DES mode, otherwise a wrong mesh length would be invisible here.
            const double omega = std::string(name) == "KOmegaSST" ? 20 : .1;
            Fill(state, fv, false, omega);
            auto work = fv.BuildEquationWorkspace(state);
            R::EagerEngine engine;
            const auto stream = engine.CreateCPUStream();
            fv.Evaluate(state, result, *work, engine, stream);
            auto aux = fv.BuildAuxiliaryValues({true, true});
            auto uncachedAux = fv.BuildAuxiliaryValues({false, true});
            fv.PrepareCorrectionAux(state, *work, uncachedAux, engine, stream, 0);
            fv.PrepareCorrectionAux(state, *work, aux, engine, stream, 0);
            {
                const auto a = aux.faceSpectralRadius->Read(), b = uncachedAux.faceSpectralRadius->Read();
                const auto cached = a.View<R::UniformTableRows>(), direct = b.View<R::UniformTableRows>();
                for (I iFace = 0; iFace < cached.RowCount(); ++iFace)
                    for (I j = 0; j < cached.Width(); ++j)
                        CHECK(cached(iFace, j) == direct(iFace, j));
            }
            R::Array<double> tau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
            {
                auto write = tau.Write();
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    write.View()(iCell, 0) = 2;
            }
            auto matrix = fv.BuildLinearProblem(state, C::DiagonalStorage::Block);
            C::JacobianPreparation controls{3, .7, false};
            for (int refresh = 0; refresh < 2; ++refresh)
            {
                const double meanSA = refresh ? .03 : .02;
                if (refresh)
                {
                    {
                        auto write = state.WriteOwned();
                        const auto sa = write.Block("rans");
                        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                            sa(iCell, 0) = meanSA;
                    }
                    // Correction preparation must update destruction from the new mean,
                    // while retaining the polynomial and leaving the prior RHS untouched.
                    fv.PrepareCorrectionAux(state, *work, aux, engine, stream, 0);
                }
                fv.PrepareJacobian(state, aux, tau, *matrix, controls, 0, {}, work.get());
                const auto rhsRead = result.ReadOwned();
                const auto diagonalRead = matrix->Diagonal().Read();
                const auto rhs = rhsRead.Block("rans"), flowRHS = rhsRead.Block("flow");
                const auto diagonal = diagonalRead.View();
                const int sourceOrder = std::string(reconstruction) == "constant" ? 1 : settings.numerics.cfv.intOrder;
                CHECK(settings.numerics.RHSIntegrationOrder() == sourceOrder);
                const DNDS::Geom::Elem::Quadrature quadrature(DNDS::Geom::Elem::Element{DNDS::Geom::Elem::Hex8}, sourceOrder);
                int lengthChangesSource = 0;
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    std::array<double, 2> expected{}, expectedDiagonal{};
                    const auto center = fv.Metrics().GetCellCent(iCell);
                    for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                    {
                        const auto param = std::get<0>(quadrature.GetQuadraturePointInfo(iQ));
                        const auto point = (center + .25 * side.cwiseProduct(param)).eval();
                        double wall = .5;
                        for (int d = 0; d < 3; ++d)
                            wall *= 1 - std::abs(2 * point[d] / side[d] - 1);
                        DNDS::Physics::RANS::Point value;
                        value.state << 1, 0, 0, 0, 2.5, .02, components == 2 ? omega : 0;
                        value.wallDistance = std::max(settings.numerics.wallDistanceFloor, wall);
                        value.cellLength = 1.5; // Independent maximum edge length, not a unit placeholder.
                        value.molecularViscosity = .001;
                        const double weight = quadrature.GetWeight(iQ) * 6 / 64.;
                        const auto source = equation.RANS().Source(value);
                        if (modelSettings.desScale == .01)
                        {
                            auto wrong = value;
                            wrong.cellLength = 1;
                            const auto oldPlaceholder = equation.RANS().Source(wrong);
                            lengthChangesSource |= std::abs(source[0] - oldPlaceholder[0]) > 1e-10;
                        }
                        value.state[5] = meanSA;
                        const auto destruction = equation.RANS().ImplicitSourceDiagonal(value);
                        for (int j = 0; j < components; ++j)
                        {
                            expected[j] += weight * source[j];
                            expectedDiagonal[j] += weight * destruction[j];
                        }
                    }
                    for (int j = 0; j < components; ++j)
                    {
                        CHECK(rhs(iCell, j) == doctest::Approx(expected[j]).epsilon(1e-10).scale(1e-12));
                        CHECK(diagonal(iCell, (5 + j) * (width + 1)) - diagonal(iCell, 0) == doctest::Approx(.7 * expectedDiagonal[j]).epsilon(1e-9).scale(1e-12));
                    }
                    for (int j = 0; j < 5; ++j)
                        CHECK(std::abs(flowRHS(iCell, j)) < 1e-12);
                }
                // At the midpoint SST shielding can hide the cell-length change;
                // the full-quadrature variant retains the off-wall sensitivity
                // oracle. SA is sensitive in both variants.
                if (modelSettings.desScale == .01 && (std::string(name) == "SA" || sourceOrder > 1))
                {
                    MPI_Allreduce(MPI_IN_PLACE, &lengthChangesSource, 1, MPI_INT, MPI_MAX, mpi.comm);
                    CHECK(lengthChangesSource == 1);
                }
                const auto auxRead = aux.meanThermodynamics->Read();
                DNDS::Physics::RANS::Point point;
                point.state[0] = 1;
                point.state[5] = meanSA;
                point.state[6] = components == 2 ? omega : 0;
                point.wallDistance = .5 / 8;
                point.molecularViscosity = .001;
                point.cp = 3.5;
                const auto turbulent = equation.RANS().Transport(point);
                const double expectedDiffusion = std::max(4. / 3. * (.001 + turbulent.viscosity),
                                                          (.001 * 3.5 / settings.equation.prandtl + turbulent.conductivity) / 2.5);
                for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
                    CHECK(auxRead.View()(iCell, 2) == doctest::Approx(expectedDiffusion).epsilon(1e-12));
            }
        }
}

#ifdef USE_FIX_ZERO_SA_NUT_AT_WALL
TEST_CASE("Zero-SA wall convection retains the interior SA viscous jump")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    auto settings = Settings();
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "noSlipWall";
    // Uniform explicit distance isolates wall diffusion from SA source
    // geometry. All nodal distances are <= .5, so every source point uses 1.
    settings.numerics.wallDistanceFloor = 1;
    const auto equation = Equation(settings);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = State(fv, true);
    Fill(state, fv, false);
    auto result = C::EquationDofOperations::NewVector(state);
    auto work = fv.BuildEquationWorkspace(state);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, result, *work, engine, stream);

    DNDS::Physics::RANS::Point point;
    point.state << 1, 0, 0, 0, 2.5, .02, 0;
    point.gradient.setZero();
    point.molecularViscosity = .001;
    point.gamma = 1.4;
    point.cp = 3.5;
    point.wallDistance = 1;
    point.cellLength = .5;
    const double source = equation.RANS().Source(point)[0];
    // Each of the eight cubes has three exterior walls: A=1/4, V=1/8,
    // reflected-centroid separation=1/2. Convection sees SA=0 on both sides,
    // but Euler's viscosity uses the original interior .02 against exterior 0.
    const double wallDiffusion = -3 * (.001 / DNDS::Physics::RANS::SA::sigma) * .02 * (.25 / .5);
    const auto read = result.ReadOwned(), mean = state.ReadOwned();
    const auto sa = read.Block("rans"), original = mean.Block("rans");
    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
    {
        CHECK(sa(iCell, 0) == doctest::Approx(source * .125 + wallDiffusion).epsilon(2e-11).scale(1e-8));
        CHECK(original(iCell, 0) == .02);
    }
}
#endif

TEST_CASE("SA boundary samples use wall reflection and frame-relative incoming flow")
{
    auto settings = Settings();
    settings.boundaries["7"].kind = "noSlipWall";
    settings.boundaries["8"].kind = "farfield";
    settings.boundaries["8"].ransConservative = {{"rhoNuTilde", .07}};
    const auto equation = Equation(settings);
    const C::BoundarySystem boundaries(settings.boundaries, equation);
    auto evaluator = equation.MakePointEvaluator({}, {});
    C::FiniteVolumeDetail::Sample sample{};
    sample.normal = {1, 0, 0};
    sample.ownPoint.setZero();
    C::EquationFVDetail::PointScratch point(6);
    const auto reset = [&]
    {
        const auto flow = equation.Gas().Conservative({1, .1, 0, 0, 1});
        for (int j = 0; j < 5; ++j)
            point.values[j] = flow[j];
        point.values[5] = .02;
    };
    C::RigidFrame frame;
    sample.boundary = boundaries.At(7);
    reset();
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, true, {&equation.RANS(), 5});
    CHECK(point.values[5] == -.02);
    reset();
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, false, {&equation.RANS(), 5});
    CHECK(point.values[5] == 0);
    sample.boundary = boundaries.At(8);
    reset();
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, true, {&equation.RANS(), 5});
    CHECK(point.values[5] == .02);
    // The same absolute flow becomes inflow when the boundary moves faster.
    frame.moving = true;
    frame.centerVelocity = {.2, 0, 0};
    reset();
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, true, {&equation.RANS(), 5});
    CHECK(point.values[5] == .07);
    settings.boundaries["8"].ransConservative.clear();
    CHECK_THROWS(C::BoundarySystem(settings.boundaries, equation));
}

TEST_CASE("Two-equation boundary points use mean wall distance and the correct thermal viscosity")
{
    int materials = 1;
#ifdef DNDS_USE_CANTERA
    materials = 2;
#endif
    for (int material = 0; material < materials; ++material)
        for (const auto *name : {"KOmegaWilcox", "KOmegaSST", "RealizableKE"})
        {
            auto settings = Settings();
            settings.equation.rans.model = name;
            if (material)
            {
                settings.equation.eos = "idealGasMixture";
                settings.equation.viscosity = 0;
                settings.equation.mixture.mechanismFile = "h2o2.yaml";
            }
            auto setup = C::EquationSetup::Resolve(settings.equation);
            const auto &equation = setup.equation;
            const auto &model = equation.RANS();
            auto evaluator = equation.MakePointEvaluator(setup.workers ? setup.workers->Acquire() : DNDS::Physics::Chemistry::ChemicalWorkerPool::Lease{}, {true});
            const int ns = material ? equation.Mixture().SpeciesNumber() - 1 : 0;
            C::EquationFVDetail::PointScratch mean(7 + ns), inside(7 + ns);
            const auto fractions = material ? equation.IndependentMassFractions({{"H2", .03}, {"O2", .23}}) : std::vector<double>{};
            const auto fill = [&](auto &point, double rho, double temperature)
            {
                for (int k = 0; k < ns; ++k)
                    point.values[7 + k] = rho * fractions[k];
                const auto flow = evaluator.FromPrimitive({rho, .1, .02, 0, 1}, point.Species(7, ns), temperature);
                for (int j = 0; j < 5; ++j)
                    point.values[j] = flow[j];
            };
            fill(mean, 2, 2);
            fill(inside, 1.5, 3);
            mean.values[5] = .07;
            mean.values[6] = 1.1;
            inside.values[5] = .03;
            inside.values[6] = .4;
            const double distance = .17;
            C::FiniteVolumeDetail::Sample sample{};
            sample.normal = {1, 0, 0};
            sample.ownPoint.setZero();
            C::RigidFrame frame;
            for (bool isothermal : {false, true})
            {
                settings.boundaries["7"].kind = "noSlipWall";
                settings.boundaries["7"].thermal = isothermal ? "isothermal" : "adiabatic";
                settings.boundaries["7"].temperature = 4;
                sample.boundary = C::BoundarySystem(settings.boundaries, equation).At(7);
                auto viscosityPoint = isothermal ? inside : mean;
                if (isothermal)
                {
                    // Independent property oracle at p_inside,T_wall,Y_inside.
                    const auto thermal = evaluator.Evaluate(inside.Flow(0), inside.Species(7, ns), 0, {}).thermo;
                    const double rhoWall = thermal.pressure / (thermal.gasConstant * 4);
                    fill(viscosityPoint, rhoWall, 4);
                }
                const double mu = evaluator.Evaluate(viscosityPoint.Flow(0), viscosityPoint.Species(7, ns), 0, {true}).transport.viscosity;
                if (material && isothermal)
                    CHECK(std::abs(mu - evaluator.Evaluate(mean.Flow(0), mean.Species(7, ns), 0, {true}).transport.viscosity) > mu * .1);
                const double target = model.Kind() == DNDS::Physics::RANS::ModelKind::RealizableKE
                                          ? 2 * mu / 2 * (.07 / 2) / (distance * distance) * 1.5
                                          : 800 * mu / (distance * distance);
                for (bool exterior : {false, true})
                {
                    auto point = inside;
                    C::EquationFVDetail::BoundaryValue(point, 0, 7, ns, sample, evaluator, frame, 0, exterior,
                                                       {&model, 5, &mean, distance});
                    const double densityRatio = isothermal ? point.values[0] / inside.values[0] : 1;
                    CHECK(point.values[5] == doctest::Approx(exterior ? -.03 * densityRatio : 0));
                    CHECK(point.values[6] == doctest::Approx(exterior ? 2 * target - .4 : target).epsilon(2e-11));
                    CHECK(mean.values[0] == 2);
                    for (int k = 0; k < ns; ++k)
                        CHECK(point.values[7 + k] / point.values[0] == doctest::Approx(fractions[k]));
                }
            }
            auto point = inside;
            CHECK_THROWS(C::EquationFVDetail::BoundaryValue(point, 0, 7, ns, sample, evaluator, frame, 0, true, {&model, 5}));
            CHECK(point.values == inside.values);
            settings.boundaries["7"].kind = "farfield";
            settings.boundaries["7"].massFractions = material ? std::map<std::string, double>{{"H2", .03}, {"O2", .23}} : std::map<std::string, double>{};
            settings.boundaries["7"].ransConservative = {{model.ComponentNames()[0], .11}, {model.ComponentNames()[1], 2.3}};
            sample.boundary = C::BoundarySystem(settings.boundaries, equation).At(7);
            frame.moving = true;
            frame.centerVelocity = {.2, 0, 0};
            C::EquationFVDetail::BoundaryValue(point, 0, 7, ns, sample, evaluator, frame, 0, true, {&model, 5});
            CHECK(point.values[5] == .11);
            CHECK(point.values[6] == 2.3);
        }
}

TEST_CASE("RANS P3 residual and WBAP preserve split layout and core halo scheduling semantics")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    for (const auto *model : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
    {
        std::vector<double> reference;
        for (int variant = 0; variant < 2; ++variant)
        {
            auto settings = Settings();
            settings.boundaries.clear();
            settings.execution.overlap = variant == 1;
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
            settings.numerics.reconstructionIterations = 4;
            settings.numerics.flux = "Roe_M1";
            settings.numerics.limiter = "WBAP";
            settings.numerics.viscousLimited = true;
            auto modelSettings = Equation(settings).RANS().Settings();
            modelSettings.model = model;
            modelSettings.useQCR = true;
            auto configured = settings;
            configured.equation.rans = modelSettings;
            CHECK_NOTHROW(C::SimulationConfig::ResolveCollective({{"defaults", C::Json(configured)}}, mpi.comm));
            const auto equation = C::EquationSystem::Resolve(settings.equation).WithRANS(modelSettings);
            C::FiniteVolume fv(mesh, settings, equation);
            auto state = State(fv, variant);
            Fill(state, fv, true);
            auto result = C::EquationDofOperations::NewVector(state);
            auto work = fv.BuildEquationWorkspace(state);
            R::EagerEngine engine(variant ? 2 : 0);
            const auto stream = engine.CreateCPUStream();
            fv.Evaluate(state, result, *work, engine, stream);
            const auto read = result.ReadOwned();
            I position = 0;
            double magnitude = 0;
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (const auto *block : {"flow", "rans"})
                {
                    const auto values = read.Block(block);
                    for (I j = 0; j < values.ComponentNumber(); ++j, ++position)
                    {
                        CHECK(std::isfinite(values(iCell, j)));
                        magnitude += std::abs(values(iCell, j));
                        if (!variant)
                            reference.push_back(values(iCell, j));
                        else
                            CHECK(values(iCell, j) == doctest::Approx(reference[size_t(position)]).epsilon(2e-11).scale(1e-12));
                    }
                }
            MPI_Allreduce(MPI_IN_PLACE, &magnitude, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
            CHECK(magnitude > 1e-5);
            // Periodic flow has no boundary flux or flow5 volume source here.
            // Opposite incidences must cancel even with strided fields, WBAP,
            // a nonzero polynomial and core/dependent face sharing.
            std::array<double, 5> balance{}, absolute{};
            const auto flow = read.Block("flow");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                for (int j = 0; j < 5; ++j)
                {
                    balance[j] += flow(iCell, j);
                    absolute[j] += std::abs(flow(iCell, j));
                }
            MPI_Allreduce(MPI_IN_PLACE, balance.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
            MPI_Allreduce(MPI_IN_PLACE, absolute.data(), 5, MPI_DOUBLE, MPI_SUM, mpi.comm);
            for (int j = 0; j < 5; ++j)
                CHECK(std::abs(balance[j]) < 2e-11 * (1 + absolute[j]));
        }
    }
}

TEST_CASE("Explicit mean gradients exchange current values across layouts and CPU launchers")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    constexpr double pi = 3.14159265358979323846, h = 1. / 8.;
    for (int variant = 0; variant < 2; ++variant)
    {
        auto settings = Settings();
        settings.boundaries.clear();
        settings.execution.overlap = variant == 1;
        settings.numerics.gradient = "leastSquares";
        const auto equation = Equation(settings);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = State(fv, variant == 1);
        auto gradients = fv.BuildMeanGradients(state);
        auto aux = fv.BuildAuxiliaryValues({true, true, false, false, true});
        R::EagerEngine engine(variant ? 2 : 0);
        const auto stream = engine.CreateCPUStream();
        for (int refresh = 0; refresh < 2; ++refresh)
        {
            const double amplitude = refresh ? .2 : .1;
            {
                auto write = state.WriteOwned();
                const auto flow = write.Block("flow"), sa = write.Block("rans");
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const auto center = fv.Metrics().GetCellCent(iCell);
                    for (I j = 0; j < 5; ++j)
                        flow(iCell, j) = 0;
                    flow(iCell, 0) = 1;
                    flow(iCell, 2) = amplitude * std::sin(2 * pi * center[0]);
                    flow(iCell, 4) = 3;
                    sa(iCell, 0) = .02;
                }
            }
            fv.PrepareAuxValues(state, aux, gradients, engine, stream);
            // Independent discrete centered-difference oracle on this torus;
            // checks both owned and received gradients, including block offsets.
            const auto location = state.LocateBlock(0);
            const auto &field = gradients.fields[location.physicalField];
            const auto owned = field.Owned().Read(), ghost = field.Ghost().Read();
            const C::RHSFieldView values{owned.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
            for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
            {
                const double x = fv.Metrics().GetCellCent(iCell)[0];
                const double expected = amplitude * std::cos(2 * pi * x) * std::sin(2 * pi * h) / h;
                for (I j = 0; j < 5; ++j)
                    for (I d = 0; d < 2; ++d)
                        CHECK(values(iCell, (location.firstComponent + j) * 2 + d) ==
                              doctest::Approx(j == 2 && d == 0 ? expected : 0.).epsilon(2e-11).scale(1));
            }
            CHECK(std::isfinite(fv.StableStep(state, aux, .5)));
        }
        // A retained handle cannot silently supply gradients for a newer state.
        {
            auto write = state.WriteOwned();
            if (fv.Metrics().NumCell())
                write.Block("flow")(0, 2) += .01;
        }
        CHECK_THROWS(fv.PrepareAuxValues(state, aux, 0, {}, {&gradients}));
        fv.PrepareAuxValues(state, aux, gradients, engine, stream);
        {
            auto write = gradients.fields[0].Owned().Write();
            if (fv.Metrics().NumCell())
                write.View()(0, 0) += .01;
        }
        CHECK_THROWS(fv.ValidateAuxValues(state, aux));
    }
}

TEST_CASE("Two-equation auxiliary viscosity consumes current conservative mean gradients")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    for (const auto *name : {"KOmegaWilcox", "KOmegaSST", "RealizableKE"})
        for (int variant = 0; variant < 2; ++variant)
        {
            CAPTURE(name);
            CAPTURE(variant);
            auto settings = Settings();
            settings.mesh.dimension = 3;
            settings.execution.overlap = variant;
            settings.numerics.gradient = "leastSquares";
            auto modelSettings = Equation(settings).RANS().Settings();
            modelSettings.model = name;
            const auto equation = C::EquationSystem::Resolve(settings.equation).WithRANS(modelSettings);
            C::FiniteVolume fv(mesh, settings, equation);
            auto state = State(fv, variant);
            auto gradients = fv.BuildMeanGradients(state);
            auto aux = fv.BuildAuxiliaryValues({true, true, false, false, true});
            R::EagerEngine engine(variant ? 2 : 0);
            const auto stream = engine.CreateCPUStream();
            for (int refresh = 0; refresh < 2; ++refresh)
            {
                const double shear = refresh ? 2 : 1;
                {
                    auto write = state.WriteOwned();
                    const auto flow = write.Block("flow"), rans = write.Block("rans");
                    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    {
                        for (int j = 0; j < 5; ++j)
                            flow(iCell, j) = 0;
                        flow(iCell, 0) = 1;
                        flow(iCell, 2) = shear * fv.Metrics().GetCellCent(iCell)[0];
                        flow(iCell, 4) = 3;
                        rans(iCell, 0) = .02;
                        rans(iCell, 1) = 1e-4;
                    }
                }
                if (!refresh)
                    CHECK_THROWS(fv.PrepareAuxValues(state, aux));
                fv.PrepareAuxValues(state, aux, gradients, engine, stream);
                const auto read = aux.meanThermodynamics->Read();
                for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
                {
                    DNDS::Physics::RANS::Point point;
                    point.state << 1, 0, shear * fv.Metrics().GetCellCent(iCell)[0], 0, 3, .02, 1e-4;
                    // Two cells per axis with extrapolated boundary values: LS
                    // averages the interior slope and the zero boundary slope.
                    point.gradient(0, 2) = shear / 2;
                    // Hex P1 nodal wall values: one center node at .5, seven zero.
                    point.wallDistance = .5 / 8;
                    point.molecularViscosity = .001;
                    point.cp = 3.5;
                    const auto transport = equation.RANS().Transport(point);
                    const double expected = std::max(4. / 3. * (.001 + transport.viscosity),
                                                     (.001 * 3.5 / settings.equation.prandtl + transport.conductivity) / 2.5);
                    CHECK(read.View()(iCell, 2) == doctest::Approx(expected).epsilon(2e-11));
                    if (equation.RANS().Kind() != DNDS::Physics::RANS::ModelKind::KOmegaWilcox)
                    {
                        point.gradient.setZero();
                        CHECK(equation.RANS().Transport(point).viscosity > 2 * transport.viscosity);
                    }
                }
                CHECK(std::isfinite(fv.StableStep(state, aux, .5)));
            }
        }
}

TEST_CASE("RANS final thermal increment compression retains Euler whole-vector scaling")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    for (const auto *model : {"SA", "KOmegaSST"})
        for (int split = 0; split < 2; ++split)
        {
            auto settings = Settings();
            settings.boundaries.clear();
            settings.numerics.positivity = true;
            auto modelSettings = Equation(settings).RANS().Settings();
            modelSettings.model = model;
            const auto equation = C::EquationSystem::Resolve(settings.equation).WithRANS(modelSettings);
            std::map<std::string, double> initial{{equation.RANS().ComponentNames()[0], .02}};
            if (equation.RANS().ComponentNumber() == 2)
                initial[equation.RANS().ComponentNames()[1]] = .1;
            settings.numerics.ransState.referenceConservative = initial;
            settings.numerics.ransState.referenceConservative["rho"] = 1;
            C::FiniteVolume fv(mesh, settings, equation);
            auto state = State(fv, split);
            fv.InitializeConstant(state, {1, 0, 0, 0, 1}, {}, 0, {}, initial);
            auto delta = C::EquationDofOperations::NewVector(state);
            C::EquationDofOperations::Fill(delta, 0);
            {
                auto write = delta.WriteOwned();
                const auto flow = write.Block("flow"), rans = write.Block("rans");
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    flow(iCell, 4) = -5;
                    for (I j = 0; j < rans.ComponentNumber(); ++j)
                        rans(iCell, j) = .001 * (j + 1);
                }
            }
            R::EagerEngine engine(split ? 2 : 0);
            const auto stream = engine.CreateCPUStream();
            C::EquationNonlinearSolver solver(fv, state, fv.BuildEquationWorkspace(state), engine, stream,
                                              settings, A::ComponentSelection(state.ComponentNumber()));
            const auto report = solver.ApplyCorrection(state, delta);
            REQUIRE(report.second < 1);
            const auto read = state.ReadOwned();
            const auto flow = read.Block("flow"), rans = read.Block("rans");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                // Infer the actual applied factor from the flow energy, not
                // the implementation's alpha helper or its diagnostic minimum.
                const double factor = (2.5 - flow(iCell, 4)) / 5;
                CHECK(factor > 0);
                CHECK(factor < 1);
                CHECK(flow(iCell, 0) == 1);
                for (I j = 0; j < rans.ComponentNumber(); ++j)
                    CHECK(rans(iCell, j) == doctest::Approx((j ? .1 : .02) + factor * .001 * (j + 1)).epsilon(1e-13));
            }
        }
}

TEST_CASE("P3 thermal reconstruction compression preserves admissible RANS linear coefficients")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    for (const auto *model : {"SA", "KOmegaSST"})
        for (int reference = 0; reference < 2; ++reference)
        {
            auto settings = Settings();
            settings.boundaries.clear();
            settings.numerics.reconstruction = "variational";
            settings.numerics.cfv.maxOrder = 3;
            settings.numerics.cfv.intOrder = 4;
            settings.numerics.positivity = true;
            settings.numerics.positivityCompressToMean = !reference;
            auto modelSettings = Equation(settings).RANS().Settings();
            modelSettings.model = model;
            const auto equation = C::EquationSystem::Resolve(settings.equation).WithRANS(modelSettings);
            std::map<std::string, double> initial{{equation.RANS().ComponentNames()[0], .02}};
            if (equation.RANS().ComponentNumber() == 2)
                initial[equation.RANS().ComponentNames()[1]] = .1;
            settings.numerics.ransState.referenceConservative = initial;
            settings.numerics.ransState.referenceConservative["rho"] = 1;
            C::FiniteVolume fv(mesh, settings, equation);
            auto state = State(fv, reference);
            fv.InitializeConstant(state, {1, 0, 0, 0, 1}, {}, 0, {}, initial);
            auto work = fv.BuildEquationWorkspace(state);
            auto guess = fv.ReconstructionGuess(*work);
            C::EquationDofOperations::Fill(guess, 0);
            const int modes = fv.Reconstruction().NRecDOF();
            {
                auto write = guess.WriteOwned();
                const auto flow = write.Block("flow"), rans = write.Block("rans");
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    flow(iCell, 4 * modes + (reference ? 2 : 0)) = -300;
                    for (int j = 0; j < equation.RANS().ComponentNumber(); ++j)
                        rans(iCell, j * modes) = .0002 * (j + 1);
                }
            }
            const C::EquationDofOperations::Binding unchangedMeans(state), unchangedGuess(guess);
            fv.LimitReconstructionPP(state, *work);
            CHECK_NOTHROW(unchangedMeans.Check(state, true));
            CHECK_NOTHROW(unchangedGuess.Check(guess, true));
            const auto stats = fv.PositivityReconstructionStats(*work);
            REQUIRE(stats.first > 0);
            REQUIRE(stats.second < 1);
            const auto read = fv.FluxReconstruction(*work).ReadOwned();
            const auto flow = read.Block("flow"), rans = read.Block("rans");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                CHECK(std::abs(flow(iCell, 4 * modes + (reference ? 2 : 0))) < 300);
                for (int j = 0; j < equation.RANS().ComponentNumber(); ++j)
                {
                    CHECK(rans(iCell, j * modes) == .0002 * (j + 1));
                    for (int mode = 1; mode < modes; ++mode)
                        CHECK(rans(iCell, j * modes + mode) == 0);
                }
            }
        }
}

TEST_CASE("Two-equation PP damps scalar coefficient pairs and rejects scalar-invalid reference polynomials")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    for (int reference = 0; reference < 2; ++reference)
    {
        auto settings = Settings();
        settings.boundaries.clear();
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.cfv.intOrder = 4;
        settings.numerics.positivity = true;
        settings.numerics.positivityCompressToMean = !reference;
        settings.numerics.ransState.referenceConservative = {{"rho", .5}, {"rhoK", .02}, {"rhoOmega", .1}};
        auto model = Equation(settings).RANS().Settings();
        model.model = "KOmegaSST";
        const auto equation = C::EquationSystem::Resolve(settings.equation).WithRANS(model);
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = State(fv, reference);
        fv.InitializeConstant(state, {1, 0, 0, 0, 1}, {}, 0, {}, {{"rhoK", .02}, {"rhoOmega", .1}});
        auto work = fv.BuildEquationWorkspace(state);
        auto guess = fv.ReconstructionGuess(*work);
        C::EquationDofOperations::Fill(guess, 0);
        const int modes = fv.Reconstruction().NRecDOF();
        const I cells = fv.Metrics().NumCell();
        {
            auto write = guess.WriteOwned();
            const auto flow = write.Block("flow"), rans = write.Block("rans");
            for (I iCell = 0; iCell < cells; ++iCell)
            {
                flow(iCell, 4 * modes) = .001;
                rans(iCell, 0) = 1;
                rans(iCell, modes) = .001;
            }
        }
        // On these width-1/4 Cartesian cells the four face midpoints include
        // the extrema of every linear polynomial; interior quadrature cannot
        // be more restrictive. Sample the intentionally linear oracle before
        // adding modes which the reference-order fallback must remove.
        R::Array<I> pointCells(R::RowLayout::Dense(4 * cells, 1));
        R::Array<double> points(R::RowLayout::Dense(4 * cells, 3));
        {
            auto cw = pointCells.Write();
            auto pw = points.Write();
            const auto c = cw.View();
            const auto p = pw.View();
            for (I iCell = 0; iCell < cells; ++iCell)
                for (int side = 0; side < 4; ++side)
                {
                    const auto center = fv.Metrics().GetCellCent(iCell);
                    const I q = 4 * iCell + side;
                    c(q, 0) = iCell;
                    for (int d = 0; d < 3; ++d)
                        p(q, d) = center[d] + (d == side / 2 ? (side % 2 ? .125 : -.125) : 0);
                }
        }
        const auto location = state.LocateBlock("rans");
        DNDS::CFV::PolynomialSamplingPlan sampler(fv.Reconstruction(), pointCells, points, false);
        R::Array<double> values(R::RowLayout::Dense(4 * cells, I(state.PhysicalField(location.physicalField).Width())));
        sampler.Evaluate(state.PhysicalField(location.physicalField), guess.PhysicalField(location.physicalField), values);
        std::vector<double> expected(size_t(cells), 1);
        {
            const auto read = values.Read();
            const auto v = read.View();
            for (I iCell = 0; iCell < cells; ++iCell)
            {
                double minimum = .02;
                for (int side = 0; side < 4; ++side)
                    minimum = std::min(minimum, v(4 * iCell + side, location.firstComponent));
                REQUIRE(minimum < 0);
                expected[iCell] = .9 * (.02 - DNDS::smallReal * .02 / .5) / (.02 - minimum + DNDS::verySmallReal);
            }
        }
        if (reference)
        {
            auto write = guess.WriteOwned();
            const auto flow = write.Block("flow"), rans = write.Block("rans");
            for (I iCell = 0; iCell < cells; ++iCell)
            {
                flow(iCell, 4 * modes + 2) = .002;
                rans(iCell, 2) = 10;
                rans(iCell, modes + 2) = .002;
            }
        }
        const C::EquationDofOperations::Binding meanBefore(state), guessBefore(guess);
        fv.LimitReconstructionPP(state, *work);
        CHECK_NOTHROW(meanBefore.Check(state, true));
        CHECK_NOTHROW(guessBefore.Check(guess, true));
        CHECK(fv.PositivityReconstructionStats(*work).first == 0); // Reports thermal beta only, as Euler.
        const auto read = fv.FluxReconstruction(*work).ReadOwned();
        const auto flow = read.Block("flow"), rans = read.Block("rans");
        for (I iCell = 0; iCell < cells; ++iCell)
        {
            CHECK(flow(iCell, 4 * modes) == .001);
            CHECK(rans(iCell, 0) == doctest::Approx(expected[iCell]).epsilon(2e-12));
            CHECK(rans(iCell, modes) == doctest::Approx(.001 * expected[iCell]).epsilon(2e-12));
            for (int mode = 2; mode < modes; ++mode)
            {
                CHECK(flow(iCell, 4 * modes + mode) == 0);
                CHECK(rans(iCell, mode) == 0);
                CHECK(rans(iCell, modes + mode) == 0);
            }
        }
    }
}

TEST_CASE("SA implicit updates repair scalars after flow compression without unfreezing masked components")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 4);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    for (int variant = 0; variant < 2; ++variant)
    {
        auto settings = Settings();
        settings.boundaries.clear();
        settings.numerics.positivity = variant;
        CAPTURE(variant);
        settings.numerics.ransState.saUpperLimit = .05;
        settings.numerics.ransState.referenceConservative = {{"rho", 1}, {"rhoNuTilde", .02}};
        C::FiniteVolume fv(mesh, settings, Equation(settings));
        auto state = State(fv, variant);
        // Rest state isolates density compression from the kinetic-energy
        // constraint when the proposed density crosses zero.
        fv.InitializeConstant(state, {1, 0, 0, 0, 1}, {}, 0, {}, {{"rhoNuTilde", .02}});
        auto delta = C::EquationDofOperations::NewVector(state);
        R::EagerEngine engine(variant ? 2 : 0);
        const auto stream = engine.CreateCPUStream();
        C::EquationNonlinearSolver solver(fv, state, fv.BuildEquationWorkspace(state), engine, stream, settings, A::ComponentSelection(6));
        C::EquationDofOperations::Fill(delta, 0);
        {
            auto write = delta.WriteOwned();
            const auto sa = write.Block("rans");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                sa(iCell, 0) = -.04;
        }
        solver.ApplyCorrection(state, delta);
        {
            const auto read = state.ReadOwned();
            const auto sa = read.Block("rans"), flow = read.Block("flow");
#ifdef USE_NS_SA_ALLOW_NEGATIVE_MEAN
            const double expected = -.02;
#else
            const double expected = std::max(1e-6, .02 * std::exp(-.04 / (.02 + 1e-6)));
#endif
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                CHECK(sa(iCell, 0) == doctest::Approx(expected));
                CHECK(flow(iCell, 0) == 1);
                CHECK(flow(iCell, 4) == doctest::Approx(2.5));
            }
        }
        C::EquationDofOperations::Fill(delta, 0);
        {
            auto write = delta.WriteOwned();
            const auto flow = write.Block("flow"), sa = write.Block("rans");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                flow(iCell, 0) = variant ? -1.01 : -.1;
                sa(iCell, 0) = 1;
            }
        }
        const auto report = solver.ApplyCorrection(state, delta);
        CAPTURE(report.second);
        if (variant)
            CHECK(report.second < 1);
        {
            const auto read = state.ReadOwned();
            const auto sa = read.Block("rans"), flow = read.Block("flow");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                CHECK(sa(iCell, 0) == doctest::Approx(.05 * flow(iCell, 0)));
        }
        // Density-only correction must not silently lower a frozen SA value.
        C::EquationDofOperations::Fill(delta, 0);
        {
            const auto read = state.ReadOwned();
            auto write = delta.WriteOwned();
            const auto flow = read.Block("flow");
            const auto change = write.Block("flow");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                change(iCell, 0) = -.01 * flow(iCell, 0);
        }
        const C::EquationDofOperations::Binding before(state);
        CHECK_THROWS(solver.ApplyCorrection(state, delta, 1, {0}));
        CHECK_NOTHROW(before.Check(state, true));
        C::EquationDofOperations::Fill(delta, 0);
        {
            auto write = delta.WriteOwned();
            const auto sa = write.Block("rans");
            for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                sa(iCell, 0) = -std::numeric_limits<double>::infinity();
        }
        CHECK_THROWS(solver.ApplyCorrection(state, delta));
        CHECK_NOTHROW(before.Check(state, true));
    }
}

TEST_CASE("RANS nonlinear stages recover a manufactured viscous source balance")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    for (const auto *model : {"SA", "KOmegaWilcox", "KOmegaSST", "RealizableKE"})
        for (int variant = 0; variant < 2; ++variant)
        {
            auto settings = Settings();
            settings.mesh.dimension = 3;
            settings.numerics.positivity = true;
            settings.execution.overlap = variant;
            settings.implicit.collectiveChecks = variant;
            settings.implicit.diagonal = "block";
            settings.implicit.jacobianAction = variant ? "semiMatrixFree" : "assembled";
            settings.implicit.linear.method = "gmres";
            settings.implicit.linear.preconditioner = variant ? "sgs" : "ilu";
            settings.implicit.linear.relativeTolerance = 1e-5;
            settings.implicit.absoluteTolerance = 1e-11;
            auto modelSettings = Equation(settings).RANS().Settings();
            modelSettings.model = model;
            const auto equation = C::EquationSystem::Resolve(settings.equation).WithRANS(modelSettings);
            const auto &rans = equation.RANS();
            std::map<std::string, double> initial{{rans.ComponentNames()[0], .02}};
            if (rans.ComponentNumber() == 2)
                initial[rans.ComponentNames()[1]] = .1;
            settings.numerics.ransState.referenceConservative = initial;
            settings.numerics.ransState.referenceConservative["rho"] = 1;
            C::FiniteVolume fv(mesh, settings, equation);
            auto target = State(fv, variant);
            fv.InitializeConstant(target, {1, 0, 0, 0, 1}, {}, 0, {}, initial);
            auto state = C::EquationDofOperations::NewVector(target);
            auto source = C::EquationDofOperations::NewVector(target);
            auto rhs = C::EquationDofOperations::NewVector(target);
            C::EquationDofOperations::Copy(state, target);
            R::EagerEngine engine(variant ? 2 : 0);
            const auto stream = engine.CreateCPUStream();
            auto targetWork = fv.BuildEquationWorkspace(target);
            fv.Evaluate(target, rhs, *targetWork, engine, stream);
            {
                const auto read = target.ReadOwned(), rhsRead = rhs.ReadOwned();
                auto write = source.WriteOwned(), stateWrite = state.WriteOwned();
                for (const auto *block : {"flow", "rans"})
                {
                    const auto u = read.Block(block), r = rhsRead.Block(block);
                    const auto b = write.Block(block), trial = stateWrite.Block(block);
                    for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                        for (I j = 0; j < u.ComponentNumber(); ++j)
                        {
                            b(iCell, j) = 20 * u(iCell, j) - r(iCell, j) / fv.Metrics().GetCellVol(iCell);
                            trial(iCell, j) *= 1 + .005 * std::sin(fv.Metrics().Cells().LocalToGlobal(iCell) + 2 * j + 1);
                        }
                }
            }
            C::EquationNonlinearSolver solver(fv, state, fv.BuildEquationWorkspace(state), engine, stream, settings, A::ComponentSelection(state.ComponentNumber()));
            const auto report = solver.Solve(state, source, 20, 1, 0);
            CAPTURE(variant);
            CAPTURE(model);
            CAPTURE(report.iterations);
            CAPTURE(report.residualTargetRatio);
            REQUIRE(report.converged);
            CHECK(report.linearIterations > 0);
            for (int j = 0; j < rans.ComponentNumber(); ++j)
                CHECK(report.residual[5 + j] < 1e-8);
            const auto actual = state.ReadOwned(), expected = target.ReadOwned();
            for (const auto *block : {"flow", "rans"})
            {
                const auto a = actual.Block(block), b = expected.Block(block);
                for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                    for (I j = 0; j < a.ComponentNumber(); ++j)
                        CHECK(std::abs(a(iCell, j) - b(iCell, j)) <= 2e-10 + 2e-8 * std::abs(b(iCell, j)));
            }
        }
}

TEST_CASE("SA cached corrections retain pre-RHS auxiliary publication across CPU launchers")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    for (int workers : {0, 2})
    {
        auto settings = Settings();
        settings.boundaries.clear();
        settings.execution.overlap = bool(workers);
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 1;
        settings.numerics.flux = "Roe_M1";
        settings.numerics.ransState.referenceConservative = {{"rho", 1}, {"rhoNuTilde", .02}};
        settings.implicit.pseudoCFL = .1;
        settings.implicit.linear.method = "ilu";
        settings.implicit.linear.stopping = "fixedCount";
        settings.implicit.linear.iterations = 2;
        settings.implicit.diagonal = "block";
        C::FiniteVolume fv(mesh, settings, Equation(settings));
        auto direct = State(fv, 0), cached = State(fv, workers ? 1 : 0);
        Fill(direct, fv, true);
        C::EquationDofOperations::Copy(cached, direct);
        auto source = C::EquationDofOperations::NewVector(direct);
        auto rate = C::EquationDofOperations::NewVector(cached);
        C::EquationDofOperations::Fill(source, 0.);
        R::EagerEngine engine(workers);
        const auto stream = engine.CreateCPUStream();
        C::EquationNonlinearSolver a(fv, direct, fv.BuildEquationWorkspace(direct), engine, stream, settings, A::ComponentSelection(6));
        C::EquationNonlinearSolver b(fv, cached, fv.BuildEquationWorkspace(cached), engine, stream, settings, A::ComponentSelection(6));
        CHECK_THROWS_WITH(b.PreparedAuxValues(), doctest::Contains("not been prepared"));
        b.EvaluateRate(cached, rate, .2);
        fv.ValidateAuxValues(cached, b.PreparedAuxValues(), .2);
        const auto meanVersion = b.PreparedAuxValues().meanThermodynamics->Version();
        const auto faceVersion = b.PreparedAuxValues().faceSpectralRadius->Version();
        const auto sumVersion = b.PreparedAuxValues().cellSpectralSum->Owned().Version();
        const auto oldOwnedVersion = cached.PhysicalField(0).Owned().Version();
        const auto fresh = a.CorrectOnce(direct, source, 0., 1., .2);
        const auto reused = b.CorrectOnce(cached, source, 0., 1., .2, 0, {}, &rate);
        CHECK(fresh.linearIterations == reused.linearIterations);
        CHECK(cached.PhysicalField(0).Owned().Version() > oldOwnedVersion);
        CHECK(b.PreparedAuxValues().meanThermodynamics->Version() == meanVersion);
        CHECK(b.PreparedAuxValues().faceSpectralRadius->Version() == faceVersion);
        CHECK(b.PreparedAuxValues().cellSpectralSum->Owned().Version() == sumVersion);
        const auto x = direct.ReadOwned(), y = cached.ReadOwned();
        const auto xl = x.ComponentLayout(), yl = y.ComponentLayout();
        const auto xv = xl.View(), yv = yl.View();
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            for (I j = 0; j < 6; ++j)
                CHECK(xv(iCell, j) == doctest::Approx(yv(iCell, j)).epsilon(2e-10).scale(1e-12));
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    int status;
    {
        doctest::Context context(argc, argv);
        status = context.run();
    }
    MPI_Finalize();
    return status;
}
