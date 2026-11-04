#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Discretization/EquationFiniteVolumeInternal.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/Implicit/LinearProblem.hpp"
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
