#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "CartesianFixture.hpp"

namespace C = DNDS::Cauchy;
namespace R = DNDS::Runtime;
static DNDS::MPIInfo mpi;

TEST_CASE("Integrated approximate Jacobian matches frozen-spectral flux differences on periodic and physical boundaries")
{
    for (bool periodic : {false, true})
    {
        const auto mesh = periodic ? CauchyTest::PeriodicQuads(mpi, 6) : CFVTest::CartesianHexes(mpi, 3);
        C::DomainSettings settings;
        settings.mesh.dimension = periodic ? 2 : 3;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        if (!periodic)
            settings.boundaries["7"].kind = "slipWall";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::BoundarySystem boundaries(settings.boundaries, equation.Gas());
        C::FiniteVolume fv(mesh, settings, equation);
        auto state = fv.Metrics().BuildField(5), delta = fv.Metrics().BuildField(5), action = fv.Metrics().BuildField(5);
        R::Array<double> inverseTau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
        {
            auto write = state.Owned().Write(), direction = delta.Owned().Write(), pseudo = inverseTau.Write();
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto point = fv.Metrics().GetCellCent(iCell);
                const auto value = equation.Gas().Conservative({1 + .02 * point[0], .2 + .03 * point[1], -.1, .05, 1 + .01 * point[0]});
                for (int j = 0; j < 5; ++j)
                {
                    write.View()(iCell, j) = value[j];
                    direction.View()(iCell, j) = .03 * (j + 1) * std::cos(2 * point[0] + point[1]) + .02 * point[2];
                }
                pseudo.View()(iCell, 0) = 2 + point[0];
            }
        }
        auto meanBinding = fv.Metrics().BindHalo(state), directionBinding = fv.Metrics().BindHalo(delta);
        meanBinding.Begin().Wait();
        directionBinding.Begin().Wait();
        C::AuxiliaryRequest request;
        request.faceSpectralRadius = request.meanThermodynamics = true;
        auto auxiliary = fv.BuildAuxiliaryValues(request);
        fv.PrepareAuxValues(state, auxiliary);
        const auto stateVersion = state.Owned().Version();
        for (auto diagonal : {C::DiagonalStorage::Entries, C::DiagonalStorage::Block})
        {
            // The scalar Euler diagonal intentionally omits physical-BC
            // derivatives. Full block/Roe mode supplies them for this oracle.
            if (!periodic && diagonal == C::DiagonalStorage::Entries)
                continue;
            auto problem = fv.BuildLinearProblem(state, diagonal);
            C::JacobianPreparation controls;
            controls.physicalMass = 3;
            controls.residualScale = .7;
            controls.roeDissipation = !periodic;
            fv.PrepareJacobian(state, auxiliary, inverseTau, *problem, controls);
            problem->InvertDiagonal();
            problem->Apply(delta, action);
            auto geometry = fv.Metrics().AcquireRead();
            const auto metric = geometry.HostView();
            auto topologyRead = mesh->AcquireRead(DNDS::Geom::MeshReadRequirements::CellFlux());
            const auto topology = topologyRead.HostView();
            const auto own = state.Owned().Read(), ghost = state.Ghost().Read();
            const auto direction = delta.Owned().Read(), remoteDirection = delta.Ghost().Read();
            const C::RHSFieldView u{own.View<R::UniformTableRows>(), ghost.View<R::UniformTableRows>()};
            const C::RHSFieldView du{direction.View<R::UniformTableRows>(), remoteDirection.View<R::UniformTableRows>()};
            const auto faceRead = auxiliary.faceSpectralRadius->Read(), pseudo = inverseTau.Read(), result = action.Owned().Read();
            const auto radius = faceRead.View();
            const auto perturbed = [&](DNDS::index iCell, double h)
            {
                C::FlowState value;
                for (int j = 0; j < 5; ++j)
                    value[j] = u(iCell, j) + h * du(iCell, j);
                return value;
            };
            const auto residual = [&](DNDS::index iCell, double h)
            {
                C::FlowState value{};
                const auto left = perturbed(iCell, h);
                const auto faces = topology.CellFaces(iCell);
                for (DNDS::rowsize slot = 0; slot < DNDS::rowsize(faces.size()); ++slot)
                {
                    const auto iFace = faces[slot];
                    const auto found = std::lower_bound(fv.AuxiliaryFaces().begin(), fv.AuxiliaryFaces().end(), iFace);
                    REQUIRE(found != fv.AuxiliaryFaces().end());
                    const auto iAux = found - fv.AuxiliaryFaces().begin();
                    const double lambda = radius(iAux, 0) + radius(iAux, 1);
                    const auto other = topology.CellFaceOther(iCell, iFace, slot);
                    const int sign = topology.CellIsFaceBack(iCell, iFace, slot) ? 1 : -1;
                    for (DNDS::index q = metric.FaceQuadratureBegin(iFace); q < metric.FaceQuadratureEnd(iFace); ++q)
                    {
                        const DNDS::Geom::tPoint normal = sign * metric.GetFaceNorm(q);
                        const C::Vector3 n{normal[0], normal[1], normal[2]};
                        const auto point = metric.GetFacePoint(q);
                        const auto right = other == DNDS::UnInitIndex ? boundaries.At(topology.GetFaceZone(iFace)).Exterior(equation.Gas(), left, n, {point[0], point[1], point[2]}) : perturbed(other, h);
                        const auto a = C::Riemann::PhysicalFlux(equation.PrepareFluxState(left).flow, n);
                        const auto b = C::Riemann::PhysicalFlux(equation.PrepareFluxState(right).flow, n);
                        for (int j = 0; j < 5; ++j)
                            value[j] -= .5 * metric.GetFaceWeight(q) * (a[j] + b[j] - lambda * (right[j] - left[j]));
                    }
                }
                return value;
            };
            constexpr double h = 1e-6;
            for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
            {
                const auto plus = residual(iCell, h), minus = residual(iCell, -h);
                for (int j = 0; j < 5; ++j)
                {
                    const double expected = metric.GetCellVol(iCell) * (pseudo.View()(iCell, 0) + controls.physicalMass) * du(iCell, j) - controls.residualScale * (plus[j] - minus[j]) / (2 * h);
                    CHECK(result.View()(iCell, j) == doctest::Approx(expected).epsilon(8e-8).scale(1));
                }
            }
            CHECK(state.Owned().Version() == stateVersion);
        }
        auto problem = fv.BuildLinearProblem(state, C::DiagonalStorage::Entries);
        CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *problem, {0, 1, true}));
        CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *problem, {-1, 1, false}));
        const auto outputVersion = problem->Diagonal().Version();
        {
            auto writer = state.Owned().Write();
        }
        CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *problem));
        CHECK(problem->Diagonal().Version() == outputVersion);
    }
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
