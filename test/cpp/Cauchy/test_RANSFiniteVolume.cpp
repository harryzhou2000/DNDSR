#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CartesianFixture.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

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
    void Fill(C::EquationDofs &state, const C::FiniteVolume &fv, bool nonconstant)
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
        }
    }
    C::EquationDofs State(const C::FiniteVolume &fv, bool split)
    {
        A::DofBundle fields(fv.Metrics().Space());
        std::vector<C::EquationDofBinding> bindings;
        if (split)
        {
            fields.Add("flow", fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 5, {16, 1})));
            fields.Add("rans", fv.Metrics().BuildField(R::RowLayout::TiledColMajor(0, 1, 32)));
            bindings = {{"flow", "flow", 0}, {"rans", "rans", 0}};
        }
        else
        {
            fields.Add("coupled", fv.Metrics().BuildField(6));
            bindings = {{"flow", "coupled", 1}, {"rans", "coupled", 0}};
        }
        return {fv.Equation().Schema(), std::move(fields), std::move(bindings)};
    }
}

TEST_CASE("SA FV integrates quadrature sources and their positive implicit destruction diagonal")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
    auto settings = Settings();
    settings.mesh.dimension = 3;
    const auto equation = Equation(settings);
    C::FiniteVolume fv(mesh, settings, equation);
    auto state = C::EquationDofs::Coupled(equation.Schema(), fv.Metrics().BuildField(6));
    auto result = C::EquationDofOperations::NewVector(state);
    Fill(state, fv, false);
    auto work = fv.BuildEquationWorkspace(state);
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    fv.Evaluate(state, result, *work, engine, stream);
    auto aux = fv.BuildAuxiliaryValues({true, true});
    fv.PrepareAuxValues(state, aux);
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
        const DNDS::Geom::Elem::Quadrature quadrature(DNDS::Geom::Elem::Element{DNDS::Geom::Elem::Hex8}, settings.numerics.cfv.intOrder);
        for (I iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
        {
            double expected = 0, expectedDiagonal = 0;
            const auto center = fv.Metrics().GetCellCent(iCell);
            for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
            {
                const auto param = std::get<0>(quadrature.GetQuadraturePointInfo(iQ));
                const auto point = (center + .25 * param).eval();
                double wall = .5;
                for (int d = 0; d < 3; ++d)
                    wall *= 1 - std::abs(2 * point[d] - 1);
                DNDS::Physics::RANS::Point value;
                value.state << 1, 0, 0, 0, 2.5, .02, 0;
                value.wallDistance = std::max(settings.numerics.wallDistanceFloor, wall);
                value.molecularViscosity = .001;
                const double weight = quadrature.GetWeight(iQ) / 64.;
                expected += weight * equation.RANS().Source(value)[0];
                value.state[5] = meanSA;
                expectedDiagonal += weight * equation.RANS().ImplicitSourceDiagonal(value)[0];
            }
            CHECK(rhs(iCell, 0) == doctest::Approx(expected).epsilon(1e-10).scale(1e-12));
            CHECK(diagonal(iCell, 35) - diagonal(iCell, 0) == doctest::Approx(.7 * expectedDiagonal).epsilon(1e-9).scale(1e-12));
            for (int j = 0; j < 5; ++j)
                CHECK(std::abs(flowRHS(iCell, j)) < 1e-12);
        }
        const auto auxRead = aux.meanThermodynamics->Read();
        DNDS::Physics::RANS::Point point;
        point.state[0] = 1;
        point.state[5] = meanSA;
        point.molecularViscosity = .001;
        point.cp = 3.5;
        const auto turbulent = equation.RANS().Transport(point);
        const double expectedDiffusion = std::max(4. / 3. * (.001 + turbulent.viscosity),
                                                  (.001 * 3.5 / settings.equation.prandtl + turbulent.conductivity) / 2.5);
        for (I iCell = 0; iCell < fv.Metrics().NumCellProc(); ++iCell)
            CHECK(auxRead.View()(iCell, 2) == doctest::Approx(expectedDiffusion).epsilon(1e-12));
    }
}

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
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, true, 5);
    CHECK(point.values[5] == -.02);
    reset();
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, false, 5);
    CHECK(point.values[5] == 0);
    sample.boundary = boundaries.At(8);
    reset();
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, true, 5);
    CHECK(point.values[5] == .02);
    // The same absolute flow becomes inflow when the boundary moves faster.
    frame.moving = true;
    frame.centerVelocity = {.2, 0, 0};
    reset();
    C::EquationFVDetail::BoundaryValue(point, 0, -1, 0, sample, evaluator, frame, 0, true, 5);
    CHECK(point.values[5] == .07);
    settings.boundaries["8"].ransConservative.clear();
    CHECK_THROWS(C::BoundarySystem(settings.boundaries, equation));
}

TEST_CASE("SA P3 residual and WBAP preserve split layout and core halo scheduling semantics")
{
    auto mesh = CauchyTest::PeriodicQuads(mpi, 8);
    mesh->BuildNodeWallDist([](auto)
                            { return false; });
    std::vector<double> reference;
    for (int variant = 0; variant < 2; ++variant)
    {
        auto settings = Settings();
        settings.boundaries.clear();
        settings.execution.overlap = variant == 1;
        settings.numerics.reconstruction = "variational";
        settings.numerics.cfv.maxOrder = 3;
        settings.numerics.reconstructionIterations = 4;
        settings.numerics.limiter = "WBAP";
        settings.numerics.viscousLimited = true;
        const auto equation = Equation(settings);
        C::FiniteVolume fv(mesh, settings, equation);
        A::DofBundle fields(fv.Metrics().Space());
        std::vector<C::EquationDofBinding> binding;
        if (variant == 0)
        {
            fields.Add("coupled", fv.Metrics().BuildField(6));
            binding = {{"flow", "coupled", 1}, {"rans", "coupled", 0}};
        }
        else
        {
            fields.Add("flow", fv.Metrics().BuildField(R::RowLayout::ColMajor(0, 5, {16, 1})));
            fields.Add("rans", fv.Metrics().BuildField(R::RowLayout::TiledColMajor(0, 1, 32)));
            binding = {{"flow", "flow", 0}, {"rans", "rans", 0}};
        }
        C::EquationDofs state(equation.Schema(), std::move(fields), std::move(binding));
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

TEST_CASE("SA nonlinear stage recovers a manufactured viscous source balance")
{
    auto mesh = CFVTest::CartesianHexes(mpi, 2);
    mesh->BuildNodeWallDist([](auto zone)
                            { return zone == 7; });
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
        C::FiniteVolume fv(mesh, settings, Equation(settings));
        auto target = State(fv, variant);
        fv.InitializeConstant(target, {1, 0, 0, 0, 1}, {}, 0, {}, {{"rhoNuTilde", .02}});
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
        C::EquationNonlinearSolver solver(fv, state, fv.BuildEquationWorkspace(state), engine, stream, settings, A::ComponentSelection(6));
        const auto report = solver.Solve(state, source, 20, 1, 0);
        CAPTURE(variant);
        CAPTURE(report.iterations);
        CAPTURE(report.residualTargetRatio);
        REQUIRE(report.converged);
        CHECK(report.linearIterations > 0);
        CHECK(report.residual[5] < 1e-8);
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
