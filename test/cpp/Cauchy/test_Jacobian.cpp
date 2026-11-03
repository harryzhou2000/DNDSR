#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "../CFV/ReconstructionFixture.hpp"
#include "../CFV/RotatedSectorFixture.hpp"
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
            for (auto mode : {C::OffDiagonalStorage::Assembled, C::OffDiagonalStorage::SemiMatrixFree})
            {
                // The scalar Euler diagonal intentionally omits physical-BC
                // derivatives. Full block/Roe mode supplies them for this oracle.
                if (!periodic && diagonal == C::DiagonalStorage::Entries)
                    continue;
                auto problem = fv.BuildLinearProblem(state, diagonal, mode);
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
namespace
{
    void EqualFields(const DNDS::CFV::FVField &a, const DNDS::CFV::FVField &b)
    {
        const auto x = a.Owned().Read(), y = b.Owned().Read();
        const auto u = x.View<R::UniformTableRows>(), v = y.View<R::UniformTableRows>();
        for (DNDS::index iCell = 0; iCell < u.RowCount(); ++iCell)
            for (int j = 0; j < 5; ++j)
                CHECK(u(iCell, j) == doctest::Approx(v(iCell, j)).epsilon(8e-13).scale(1));
    }
}
TEST_CASE("Frozen physical semi-matrix-free blocks agree with assembled action, split SGS and local ILU")
{
    for (int meshKind = 0; meshKind < 4; ++meshKind)
    {
        const auto mesh = meshKind == 0 ? CauchyTest::PeriodicQuads(mpi, 6) : meshKind == 1 ? CFVTest::CartesianHexes(mpi, 3)
                                                                          : meshKind == 2   ? CFVTest::Sector(mpi)
                                                                                            : CauchyTest::PeriodicQuads(mpi, 1);
        C::DomainSettings settings;
        settings.mesh.dimension = meshKind == 1 ? 3 : 2;
        settings.numerics.reconstruction = "constant";
        settings.numerics.cfv.maxOrder = 1;
        settings.numerics.flux = "Roe_M1";
        settings.equation.transport = "NewtonianFourier";
        settings.equation.viscosity = .03;
        if (meshKind == 1)
            settings.boundaries["7"].kind = "noSlipWall";
        if (meshKind == 2)
            settings.boundaries["12"].kind = settings.boundaries["13"].kind = "slipWall";
        const auto equation = C::EquationSystem::Resolve(settings.equation);
        C::FiniteVolume fv(mesh, settings, equation);
        for (int layout = 0; layout < 3; ++layout)
        {
            const auto shape = layout == 0 ? R::RowLayout::Dense(0, 5, 8) : layout == 1 ? R::RowLayout::ColMajor(0, 5, {8, 1})
                                                                                        : R::RowLayout::TiledColMajor(0, 5, 32);
            auto state = fv.Metrics().BuildField(shape), delta = fv.Metrics().BuildField(shape), rhs = fv.Metrics().BuildField(shape);
            auto assembled = fv.Metrics().BuildField(shape), evaluated = fv.Metrics().BuildField(shape), split = fv.Metrics().BuildField(shape);
            auto forwardA = fv.Metrics().BuildField(shape), forwardM = fv.Metrics().BuildField(shape);
            R::Array<double> inverseTau(R::RowLayout::Dense(fv.Metrics().NumCell(), 1));
            {
                auto u = state.Owned().Write(), x = delta.Owned().Write(), b = rhs.Owned().Write(), tau = inverseTau.Write();
                for (DNDS::index iCell = 0; iCell < fv.Metrics().NumCell(); ++iCell)
                {
                    const auto p = fv.Metrics().GetCellCent(iCell);
                    const auto mean = equation.Gas().Conservative({1 + .02 * p[0], .4 + .03 * p[1], -.2 + .01 * p[0], .1, 1 + .02 * p[1]});
                    for (int j = 0; j < 5; ++j)
                    {
                        u.View<R::UniformTableRows>()(iCell, j) = mean[j];
                        x.View<R::UniformTableRows>()(iCell, j) = .02 * (j + 1) * std::cos(2 * p[0] + p[1]) + .01 * p[2];
                        b.View<R::UniformTableRows>()(iCell, j) = .2 + .03 * p[0] + .04 * j;
                    }
                    tau.View()(iCell, 0) = 2 + p[0];
                }
            }
            auto meanHalo = fv.Metrics().BindHalo(state), deltaHalo = fv.Metrics().BindHalo(delta);
            auto forwardAHalo = fv.Metrics().BindHalo(forwardA), forwardMHalo = fv.Metrics().BindHalo(forwardM);
            meanHalo.Begin().Wait();
            deltaHalo.Begin().Wait();
            C::AuxiliaryRequest request;
            request.faceSpectralRadius = request.meanThermodynamics = request.cellWaveJump = request.cellPressureMinimum = true;
            auto auxiliary = fv.BuildAuxiliaryValues(request);
            fv.PrepareAuxValues(state, auxiliary);
            fv.Metrics().BindHalo(*auxiliary.cellWaveJump).Begin().Wait();
            fv.Metrics().BindHalo(*auxiliary.cellPressureMinimum).Begin().Wait();
            for (int approximation = 0; approximation < 3; ++approximation)
            {
                const auto mode = approximation == 0 ? C::DiagonalStorage::Entries : C::DiagonalStorage::Block;
                auto a = fv.BuildLinearProblem(state, mode);
                auto m = fv.BuildLinearProblem(state, mode, C::OffDiagonalStorage::SemiMatrixFree);
                CHECK_FALSE(m->HasAssembledOffDiagonal());
                CHECK_THROWS(m->OffDiagonal());
                CHECK_THROWS(m->ReadOffDiagonal());
                C::JacobianPreparation controls{3, .7, approximation == 2};
                fv.PrepareJacobian(state, auxiliary, inverseTau, *a, controls);
                fv.PrepareJacobian(state, auxiliary, inverseTau, *m, controls);
                a->InvertDiagonal();
                m->InvertDiagonal();
                a->Apply(delta, assembled);
                m->Apply(delta, evaluated);
                EqualFields(assembled, evaluated);
                {
                    const auto generation = m->OffDiagonalGeneration(), diagonalVersion = m->Diagonal().Version();
                    auto held = m->ReadOffDiagonal();
                    CHECK_THROWS(m->PublishOffDiagonalEvaluator(m->PreparedOffDiagonalEvaluator()));
                    CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *m, controls));
                    CHECK(m->OffDiagonalGeneration() == generation);
                    CHECK(m->Diagonal().Version() == diagonalVersion);
                }
                {
                    auto exchange = deltaHalo.Begin();
                    m->Apply(delta, split, C::LinearRegion::Core);
                    bool ghostColumn = false;
                    for (const auto &row : m->Columns())
                        for (auto column : row)
                            ghostColumn |= column >= fv.Metrics().NumCell();
                    if (ghostColumn)
                        CHECK_THROWS(m->Apply(delta, split, C::LinearRegion::HaloDependent));
                    exchange.Wait();
                    m->Apply(delta, split, C::LinearRegion::HaloDependent);
                }
                EqualFields(assembled, split);
                for (bool middle : {false, true})
                {
                    auto exchange = deltaHalo.Begin();
                    auto sa = a->BeginSGS(rhs, delta, forwardA, assembled, middle);
                    auto sm = m->BeginSGS(rhs, delta, forwardM, evaluated, middle);
                    sa.Lower(C::LinearRegion::Core);
                    sm.Lower(C::LinearRegion::Core);
                    exchange.Wait();
                    sa.Lower(C::LinearRegion::HaloDependent);
                    sm.Lower(C::LinearRegion::HaloDependent);
                    if (middle)
                    {
                        forwardAHalo.Begin().Wait();
                        forwardMHalo.Begin().Wait();
                    }
                    sa.Upper(C::LinearRegion::HaloDependent);
                    sm.Upper(C::LinearRegion::HaloDependent);
                    sa.Upper(C::LinearRegion::Core);
                    sm.Upper(C::LinearRegion::Core);
                    sa.Finish();
                    sm.Finish();
                    EqualFields(assembled, evaluated);
                }
                for (int fill : {0, 1})
                {
                    a->PrepareILUStructure(fill);
                    m->PrepareILUStructure(fill);
                    a->FactorizeILU();
                    m->FactorizeILU();
                    a->ApplyILU(rhs, forwardA, assembled);
                    m->ApplyILU(rhs, forwardM, evaluated);
                    EqualFields(assembled, evaluated);
                    const auto diagonalVersion = m->Diagonal().Version();
                    m->PublishOffDiagonalEvaluator(m->PreparedOffDiagonalEvaluator());
                    CHECK(m->Diagonal().Version() == diagonalVersion);
                    CHECK_FALSE(m->ILUFactorsCurrent());
                    CHECK_THROWS(m->ApplyILU(rhs, forwardM, evaluated));
                    m->FactorizeILU();
                    CHECK(m->ILUFactorsCurrent());
                }
                // A fresh snapshot may reuse geometry without changing action.
                a->Apply(delta, assembled);
                const auto old = m->PreparedOffDiagonalEvaluator();
                fv.PrepareJacobian(state, auxiliary, inverseTau, *m, controls);
                CHECK(old != m->PreparedOffDiagonalEvaluator());
                m->Apply(delta, evaluated);
                EqualFields(assembled, evaluated);
            }
            // Live mean mutation must not change an already prepared operator.
            auto frozen = fv.BuildLinearProblem(state, C::DiagonalStorage::Entries, C::OffDiagonalStorage::SemiMatrixFree);
            fv.PrepareJacobian(state, auxiliary, inverseTau, *frozen);
            frozen->Apply(delta, assembled);
            {
                auto u = state.Owned().Write();
                const auto values = u.View<R::UniformTableRows>();
                for (DNDS::index iCell = 0; iCell < values.RowCount(); ++iCell)
                    values(iCell, 1) += .1;
            }
            meanHalo.Begin().Wait();
            frozen->Apply(delta, evaluated);
            EqualFields(assembled, evaluated);
            CHECK_THROWS(fv.PrepareJacobian(state, auxiliary, inverseTau, *frozen)); // stale aux, before writes
        }
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
