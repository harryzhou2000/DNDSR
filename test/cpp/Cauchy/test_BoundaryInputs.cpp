#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include "../CFV/ReconstructionFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

TEST_CASE("VR wall targets use reflected exterior states in both flow and equation profiles")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.boundaries["7"].kind = "slipWall";
    settings.numerics.reconstruction = "variational";
    settings.numerics.faceFallback = false;
    settings.numerics.reconstructionIterations = 1;
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.cfv.boundaryDerivativeWeights = {1, 0, 0, 0};
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume fv(mesh, settings, equation);
    const auto plan = DNDS::CFV::VariationalReconstructionPlan::ConstructRecCoeff(fv.Metrics(), settings.numerics.cfv);
    auto state = fv.Metrics().BuildField(5), residual = fv.Metrics().BuildField(5);
    {
        auto write = state.Owned().Write();
        const auto u = write.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < u.RowCount(); ++iCell)
        {
            const auto x = fv.Metrics().GetCellCent(iCell);
            const auto value = equation.Gas().Conservative({1 + .01 * x[0], .02 + .01 * x[1], -.03, .01, 10});
            for (int j = 0; j < 5; ++j)
                u(iCell, j) = value[j];
        }
    }
    auto expected = plan.BuildWorkspace(5);
    DNDS::Algebra::Fill(expected.coefficients, 0.);
    auto boundary = plan.BuildBoundaryValues(5);
    {
        const auto read = state.Owned().Read();
        const auto u = read.View<R::UniformTableRows>();
        auto write = boundary.values.Write();
        const auto values = write.View<R::UniformTableRows>();
        for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto point = plan.GetBoundaryQuadrature(iPoint);
            double normalMomentum = 0;
            for (int d = 0; d < 3; ++d)
                normalMomentum += u(point.cell, 1 + d) * point.normal[d];
            // Independent stationary slip reflection: rho and rhoE unchanged,
            // m' = m - 2*(m.n)*n. A projected trace has only half this jump.
            for (int j = 0; j < 5; ++j)
                values(iPoint, j) = u(point.cell, j);
            for (int d = 0; d < 3; ++d)
                values(iPoint, 1 + d) -= 2 * normalMomentum * point.normal[d];
        }
    }
    fv.Metrics().BindHalo(state).Begin().Wait();
    fv.Metrics().BindHalo(expected.coefficients).Begin().Wait();
    plan.DoReconstructionIter(state, expected, boundary);
    const auto check = [&](const auto &coefficients)
    {
        const auto a = coefficients.Owned().Read(), b = expected.coefficients.Owned().Read();
        const auto actual = a.template View<R::UniformTableRows>(), oracle = b.View<R::UniformTableRows>();
        for (I iCell = 0; iCell < actual.RowCount(); ++iCell)
            for (I j = 0; j < actual.Width(); ++j)
                CHECK(actual(iCell, j) == doctest::Approx(oracle(iCell, j)).epsilon(1e-12).scale(1));
    };
    R::EagerEngine engine;
    const auto stream = engine.CreateCPUStream();
    auto workspace = fv.BuildWorkspace(state);
    fv.Evaluate(state, residual, *workspace, engine, stream);
    check(fv.ReconstructionGuess(*workspace));
    auto logical = C::EquationDofs::Coupled(equation.Schema(), state);
    auto output = C::EquationDofs::Coupled(equation.Schema(), residual);
    auto equationWorkspace = fv.BuildEquationWorkspace(logical);
    fv.Evaluate(logical, output, *equationWorkspace, engine, stream);
    check(fv.ReconstructionGuess(*equationWorkspace).CoupledField());
}

TEST_CASE("Published interface packets reproduce analytic P3 boundaries through auxiliary RHS and frozen Jacobian operators")
{
    const auto mesh = CFVTest::CartesianHexes(mpi, 3);
    C::DomainSettings settings;
    settings.mesh.dimension = 3;
    settings.motion.angularVelocity = {.1, -.2, .3};
    settings.motion.center = {.3, .4, .5};
    settings.initialPrimitive = {1, .2, -.1, .05, 1.1};
    settings.initialProfile.kind = "isentropicVortex";
    settings.initialProfile.strength = 1.2;
    settings.initialProfile.center = {.4, .5, 0};
    settings.boundaries["7"].kind = "analytic";
    settings.boundaries["7"].primitive = settings.initialPrimitive;
    settings.boundaries["7"].profile = settings.initialProfile;
    settings.numerics.reconstruction = "variational";
    settings.numerics.cfv.maxOrder = 3;
    settings.numerics.reconstructionIterations = 2;
    const auto equation = C::EquationSystem::Resolve(settings.equation);
    C::FiniteVolume reference(mesh, settings, equation);
    settings.boundaries["7"].kind = "interface";
    C::FiniteVolume coupled(mesh, settings, equation);
    auto a = reference.Metrics().BuildField(5), b = coupled.Metrics().BuildField(5);
    auto ra = reference.Metrics().BuildField(5), rb = coupled.Metrics().BuildField(5);
    reference.Initialize(a, settings.initialProfile, settings.initialPrimitive);
    coupled.Initialize(b, settings.initialProfile, settings.initialPrimitive);
    C::AuxiliaryRequest withDerived;
    withDerived.meanThermodynamics = true;
    withDerived.derived = {C::DerivedQuantity::Temperature};
    auto workA = reference.BuildWorkspace(a), workB = coupled.BuildWorkspace(b, withDerived);
    CHECK_NOTHROW(coupled.RequireWorkspaceChannels(*workB, withDerived));
    if (mpi.size > 1)
    {
        C::AuxiliaryRequest inconsistent;
        if (mpi.rank == 0)
            inconsistent.derived = {C::DerivedQuantity::Temperature};
        CHECK_THROWS(coupled.BuildWorkspace(b, inconsistent));
    }
    auto incoming = coupled.BuildBoundaryInputs();
    constexpr double time = .3;
    CHECK_THROWS(coupled.ValidateBoundaryInputs(incoming, time));
    if (!coupled.BoundaryPoints().empty())
        CHECK_THROWS(coupled.PublishBoundaryInputs(incoming, time));
    const auto frame = coupled.Motion().At(time);
    {
        auto write = incoming.flow.Write();
        const auto fill = [&](const auto &points, const auto &values)
        {
            for (I iPoint = 0; iPoint < I(points.size()); ++iPoint)
            {
                const auto state = frame.LocalState(C::ProfileValue(settings.initialProfile, settings.initialPrimitive, equation.Gas(), frame.WorldPoint(points[iPoint].point), time));
                for (int j = 0; j < 5; ++j)
                    values(iPoint, j) = state[j];
            }
        };
        fill(coupled.BoundaryPoints(), write.View<R::UniformTableRows>());
        // Quadrature values alone cannot publish a packet for centered
        // Jacobians: the donor must explicitly sample the face centers too.
    }
    if (!coupled.BoundaryFaces().empty())
        CHECK_THROWS(coupled.PublishBoundaryInputs(incoming, time));
    {
        auto write = incoming.faceFlow.Write();
        const auto values = write.View<R::UniformTableRows>();
        const auto &points = coupled.BoundaryFaces();
        for (I iFace = 0; iFace < I(points.size()); ++iFace)
        {
            CHECK(points[iFace].quadrature == -1);
            const auto state = frame.LocalState(C::ProfileValue(settings.initialProfile, settings.initialPrimitive, equation.Gas(), frame.WorldPoint(points[iFace].point), time));
            for (int j = 0; j < 5; ++j)
                values(iFace, j) = state[j];
        }
    }
    coupled.PublishBoundaryInputs(incoming, time, {0, 1, 2, 0, 0, 0});
    coupled.SetBoundaryInputs(*workB, incoming);
    R::EagerEngine engine(2);
    const auto stream = engine.CreateCPUStream();
    reference.Evaluate(a, ra, *workA, engine, stream, time);
    coupled.Evaluate(b, rb, *workB, engine, stream, time);
    const auto compare = [](const auto &left, const auto &right)
    {
        const auto ar = left.Owned().Read(), br = right.Owned().Read();
        const auto u = ar.template View<R::UniformTableRows>(), v = br.template View<R::UniformTableRows>();
        for (I iCell = 0; iCell < u.RowCount(); ++iCell)
            for (I j = 0; j < u.Width(); ++j)
                CHECK(u(iCell, j) == v(iCell, j));
    };
    compare(ra, rb);
    compare(reference.FluxReconstruction(*workA), coupled.FluxReconstruction(*workB));
    C::AuxiliaryRequest request;
    request.faceSpectralRadius = true;
    auto auxA = reference.BuildAuxiliaryValues(request), auxB = coupled.BuildAuxiliaryValues(request);
    reference.PrepareAuxValues(a, auxA, time);
    coupled.PrepareAuxValues(b, auxB, time, &incoming);
    R::Array<double> tau(R::RowLayout::Dense(reference.Metrics().NumCell(), 1));
    {
        auto write = tau.Write();
        for (I iCell = 0; iCell < tau.RowCount(); ++iCell)
            write.View()(iCell, 0) = 10;
    }
    auto ja = reference.BuildLinearProblem(a, C::DiagonalStorage::Block), jb = coupled.BuildLinearProblem(b, C::DiagonalStorage::Block);
    const C::JacobianPreparation controls{3, .7, true};
    reference.PrepareJacobian(a, auxA, tau, *ja, controls, time);
    coupled.PrepareJacobian(b, auxB, tau, *jb, controls, time);
    // Full physical field a/b is an independent nonconstant action input.
    ja->Apply(a, ra);
    jb->Apply(b, rb);
    compare(ra, rb);
    settings.numerics.cfv.maxOrder = 1;
    C::FiniteVolume coarse(mesh, settings, equation);
    const auto alias = coarse.AliasBoundaryInputs(incoming);
    CHECK(alias.flow.SharedStorage() == incoming.flow.SharedStorage());
    CHECK(alias.faceFlow.SharedStorage() == incoming.faceFlow.SharedStorage());
    CHECK_NOTHROW(coarse.ValidateBoundaryInputs(alias, time));
    CHECK(alias.Epoch().stage == 2);
    {
        auto write = incoming.faceFlow.Write(); // Either table's debug write invalidates publication.
    }
    CHECK_THROWS(coupled.ValidateAuxValues(b, auxB, time));
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
