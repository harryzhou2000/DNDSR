#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Physics/FlowProfile.hpp"
#include "../CFV/ReconstructionFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

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
        const auto values = write.View<R::UniformTableRows>();
        const auto &points = coupled.BoundaryPoints();
        for (I iPoint = 0; iPoint < I(points.size()); ++iPoint)
        {
            const auto state = frame.LocalState(C::ProfileValue(settings.initialProfile, settings.initialPrimitive, equation.Gas(), frame.WorldPoint(points[iPoint].point), time));
            for (int j = 0; j < 5; ++j)
                values(iPoint, j) = state[j];
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
    CHECK_NOTHROW(coarse.ValidateBoundaryInputs(alias, time));
    CHECK(alias.Epoch().stage == 2);
    {
        auto write = incoming.flow.Write(); // Even a debug write invalidates publication.
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
