#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/GradientPlan.hpp"
#include "CFV/Plans/PolynomialSamplingPlan.hpp"
#include "CFV/Math/TaylorBasis.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "Solver/Linear.hpp"
#include "../Geom/NativeMeshOverlapFixture.hpp"
#include "../Geom/NativePeriodicFixtures.hpp"

namespace C = DNDS::CFV;
namespace G = DNDS::Geom;
namespace R = DNDS::Runtime;
using I = DNDS::index;
static DNDS::MPIInfo mpi;

static std::shared_ptr<G::UnstructuredMesh> Sector()
{
    using NativePeriodicFixtures::Rows;
    auto nodes = G::EntitySpace::CreateOwned(G::EntityKind::Node, mpi, mpi.rank == mpi.size - 1 ? 4 : 0);
    const bool first = mpi.rank == 0, second = mpi.rank == (mpi.size == 1 ? 0 : 1);
    auto cells = nodes.CreateSibling(G::EntityKind::Cell, I(first) + I(second));
    auto boundaries = nodes.CreateSibling(G::EntityKind::Bnd, mpi.rank == mpi.size / 2 ? 6 : 0);
    std::vector<std::vector<I>> cones, bnd;
    std::vector<std::vector<G::NodePeriodicBits>> cellBits, bndBits;
    if (first)
    {
        cones.push_back({0, 1, 3, 2});
        cellBits.push_back({{0}, {0}, {0}, {0}});
    }
    if (second)
    {
        cones.push_back({2, 3, 1, 0});
        cellBits.push_back({{0}, {0}, {1}, {1}});
    }
    if (boundaries.NumOwned())
    {
        bnd = {{0, 1}, {1, 0}, {1, 3}, {3, 1}, {2, 0}, {0, 2}};
        bndBits = {{{0}, {0}}, {{1}, {1}}, {{0}, {0}}, {{0}, {1}}, {{0}, {0}}, {{1}, {0}}};
    }
    G::NativeMeshInput input{mpi, 2, cells, nodes, boundaries, Rows(cones), Rows(bnd),
                             R::Array<double>(R::RowLayout::Dense(nodes.NumOwned(), 3)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(cells.NumOwned(), 1)),
                             R::Array<G::ElemInfo>(R::RowLayout::Dense(boundaries.NumOwned(), 1))};
    input.periodic = true;
    input.cellNodesPbi = Rows(cellBits), input.boundaryNodesPbi = Rows(bndBits);
    input.periodicInfo.rotation[1].map() << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    input.periodicInfo.translation[1].map().setZero();
    {
        const double h = std::sqrt(0.5);
        const G::tPoint points[]{{1, 0, 0}, {2, 0, 0}, {h, h, 0}, {2 * h, 2 * h, 0}};
        auto coordinates = input.coordinates.Write();
        for (I iNode = 0; iNode < nodes.NumOwned(); ++iNode)
            for (int d = 0; d < 3; ++d)
                coordinates.View()(iNode, d) = points[iNode][d];
        auto c = input.cellInfo.Write(), b = input.boundaryInfo.Write();
        for (I iCell = 0; iCell < cells.NumOwned(); ++iCell)
            c.View()(iCell, 0) = {G::Elem::Quad4, G::INTERNAL_ZONE};
        const int zones[]{G::BC_ID_PERIODIC_1, G::BC_ID_PERIODIC_1_DONOR, 12, 12, 13, 13};
        for (I iBnd = 0; iBnd < boundaries.NumOwned(); ++iBnd)
            b.View()(iBnd, 0) = {G::Elem::Line2, zones[iBnd]};
    }
    auto mesh = G::UnstructuredMesh::FromNative(mpi, G::ConstructNativeMesh(std::move(input)));
    mesh->PrepareNativeTopology();
    return mesh;
}

TEST_CASE("Rotated periodic sector preserves vector linear fields across distributed ownership")
{
    auto mesh = Sector();
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 2;
    settings.boundaryDerivativeWeights = {1, 0, 0, 0};
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto work = plan.BuildWorkspace(4, {0});
    auto mean = metrics.BuildField(4);
    auto bc = plan.BuildBoundaryValues(4);
    auto gradient = metrics.BuildField(8);
    auto grad = C::GradientPlan::Prepare(*mesh, metrics);
    auto gradBC = grad.BuildBoundaryValues(4);
    {
        auto geometry = metrics.AcquireRead();
        auto u = mean.Owned().Write(), r = work.coefficients.Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto p = geometry.HostView().GetCellBary(iCell);
            u.View()(iCell, 0) = p[0], u.View()(iCell, 1) = p[1], u.View()(iCell, 2) = 0, u.View()(iCell, 3) = 3;
            const auto scale = geometry.HostView().cells(iCell, 7);
            r.View()(iCell, 0) = scale, r.View()(iCell, plan.NRecDOF() + 1) = scale;
        }
        auto b = bc.values.Write();
        for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto p = plan.GetBoundaryQuadrature(iPoint).point;
            b.View()(iPoint, 0) = p[0], b.View()(iPoint, 1) = p[1], b.View()(iPoint, 2) = 0, b.View()(iPoint, 3) = 3;
        }
        auto gb = gradBC.Write();
        for (I iPoint = 0; iPoint < grad.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto p = grad.GetBoundaryQuadrature(iPoint).point;
            gb.View()(iPoint, 0) = p[0], gb.View()(iPoint, 1) = p[1], gb.View()(iPoint, 2) = 0, gb.View()(iPoint, 3) = 3;
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(work.coefficients).Begin().Wait();
    plan.DoReconstructionIter(mean, work, bc);
    {
        auto geometry = metrics.AcquireRead();
        auto read = work.coefficients.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I component = 0; component < 4; ++component)
                for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                    CHECK(read.View()(iCell, component * plan.NRecDOF() + mode) == doctest::Approx(
                                                                                       component < 2 && mode == component ? geometry.HostView().cells(iCell, 7) : 0)
                                                                                       .epsilon(1e-10)
                                                                                       .scale(1));
    }
    grad.Evaluate(mean, gradBC, gradient, {0});
    auto read = gradient.Owned().Read();
    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (I component = 0; component < 4; ++component)
            for (I d = 0; d < 2; ++d)
                CHECK(read.View()(iCell, component * 2 + d) == doctest::Approx(component == d ? 1. : 0.).epsilon(1e-10).scale(1));
    CHECK_THROWS(plan.BuildWorkspace(4, {2}));
}

TEST_CASE("Axisymmetric metrics and all gradient methods obey radial linear moments at the axis")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 5, true);
    CHECK(metrics.Axisymmetric());
    double total = 0;
    {
        auto geometry = metrics.AcquireRead();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            total += geometry.HostView().GetCellVol(iCell);
    }
    double global = 0;
    MPI_Allreduce(&total, &global, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(global == doctest::Approx(.5).epsilon(1e-13));
    for (auto method : {C::GradientMethod::GreenGauss, C::GradientMethod::LeastSquares, C::GradientMethod::GGMP})
    {
        auto plan = C::GradientPlan::Prepare(*mesh, metrics, method);
        auto u = metrics.BuildField(1), gradient = metrics.BuildField(2);
        auto bc = plan.BuildBoundaryValues(1);
        {
            auto geometry = metrics.AcquireRead();
            auto write = u.Owned().Write(), b = bc.Write();
            for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                const auto p = geometry.HostView().GetCellBary(iCell);
                write.View()(iCell, 0) = 2 + 3 * p[0] - .7 * p[1];
            }
            for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
            {
                const auto p = plan.GetBoundaryQuadrature(iPoint).point;
                b.View()(iPoint, 0) = 2 + 3 * p[0] - .7 * p[1];
            }
        }
        metrics.BindHalo(u).Begin().Wait();
        plan.Evaluate(u, bc, gradient);
        auto read = gradient.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            CHECK(read.View()(iCell, 0) == doctest::Approx(3).epsilon(1e-10));
            CHECK(read.View()(iCell, 1) == doctest::Approx(-.7).epsilon(1e-10));
        }
    }
    C::ReconstructionSettings settings;
    settings.boundaryDerivativeWeights = {1, 0, 0, 0};
    auto reconstruction = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto mean = metrics.BuildField(1);
    auto work = reconstruction.BuildWorkspace(1);
    auto boundary = reconstruction.BuildBoundaryValues(1);
    {
        auto geometry = metrics.AcquireRead();
        auto u = mean.Owned().Write(), r = work.coefficients.Owned().Write(), b = boundary.values.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto p = geometry.HostView().GetCellBary(iCell);
            u.View()(iCell, 0) = 2 + 3 * p[0] - .7 * p[1];
            r.View()(iCell, 0) = 3 * geometry.HostView().cells(iCell, 7), r.View()(iCell, 1) = -.7 * geometry.HostView().cells(iCell, 7);
        }
        for (I iPoint = 0; iPoint < reconstruction.NumBoundaryQuadrature(); ++iPoint)
        {
            const auto p = reconstruction.GetBoundaryQuadrature(iPoint).point;
            b.View()(iPoint, 0) = 2 + 3 * p[0] - .7 * p[1];
        }
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(work.coefficients).Begin().Wait();
    reconstruction.DoReconstructionIter(mean, work, boundary);
    auto read = work.coefficients.Owned().Read();
    auto geometry = metrics.AcquireRead();
    for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (I mode = 0; mode < reconstruction.NRecDOF(); ++mode)
            CHECK(read.View()(iCell, mode) == doctest::Approx(mode == 0 ? 3 * geometry.HostView().cells(iCell, 7) : mode == 1 ? -.7 * geometry.HostView().cells(iCell, 7)
                                                                                                                              : 0)
                                                  .epsilon(1e-9)
                                                  .scale(1));
}

TEST_CASE("Raw reconstruction operators preserve symmetry and agree with a preconditioned Jacobi step")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.jacobiRelax = .61;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto work = plan.BuildWorkspace(1);
    auto y = metrics.BuildField(plan.NRecDOF()), ax = metrics.BuildField(plan.NRecDOF()), ay = metrics.BuildField(plan.NRecDOF());
    auto mean = metrics.BuildField(1), rhs = metrics.BuildField(plan.NRecDOF()), residual = metrics.BuildField(plan.NRecDOF()), delta = metrics.BuildField(plan.NRecDOF());
    auto boundary = plan.BuildBoundaryValues(1);
    {
        auto xWrite = work.coefficients.Owned().Write(), yWrite = y.Owned().Write(), u = mean.Owned().Write(), b = boundary.values.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto id = mesh->OriginalEntityId(G::EntityKind::Cell, iCell);
            u.View()(iCell, 0) = std::cos(.2 * id);
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
            {
                xWrite.View()(iCell, mode) = std::sin(.7 * id + .4 * mode);
                yWrite.View()(iCell, mode) = std::cos(.3 * id - .2 * mode);
            }
        }
        for (I iPoint = 0; iPoint < plan.NumBoundaryQuadrature(); ++iPoint)
            b.View()(iPoint, 0) = std::sin(.2 * iPoint);
    }
    metrics.BindHalo(mean).Begin().Wait();
    metrics.BindHalo(work.coefficients).Begin().Wait();
    metrics.BindHalo(y).Begin().Wait();
    plan.ApplyLinearOperator(work.coefficients, ax);
    plan.ApplyLinearOperator(y, ay);
    double local[4]{}, global[4]{};
    {
        auto x = work.coefficients.Owned().Read(), yr = y.Owned().Read(), a = ax.Owned().Read(), b = ay.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
            {
                local[0] += x.View()(iCell, mode) * b.View()(iCell, mode);
                local[1] += yr.View()(iCell, mode) * a.View()(iCell, mode);
                local[2] += x.View()(iCell, mode) * a.View()(iCell, mode);
                local[3] += yr.View()(iCell, mode) * b.View()(iCell, mode);
            }
    }
    MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, mpi.comm);
    CHECK(global[0] == doctest::Approx(global[1]).epsilon(1e-11).scale(1));
    CHECK(global[2] > 0);
    CHECK(global[3] > 0);
    plan.BuildRightHandSide(mean, boundary, rhs);
    plan.ComputeResidual(rhs, work.coefficients, residual);
    plan.ApplyLocalPreconditioner(residual, delta);
    plan.DoReconstructionIterTo(mean, work.coefficients, work, boundary);
    {
        auto old = work.coefficients.Owned().Read(), next = work.temporary.Owned().Read(), step = delta.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                CHECK(next.View()(iCell, mode) == doctest::Approx(old.View()(iCell, mode) + settings.jacobiRelax * step.View()(iCell, mode)).epsilon(1e-10).scale(1));
    }
    CHECK_THROWS(plan.ApplyLinearOperator(work.coefficients, work.coefficients));
    CHECK_THROWS(plan.ApplyLocalPreconditioner(rhs, rhs));
    auto filteredSettings = settings;
    filteredSettings.svdTolerance = 1; // Every reciprocal is zero, but raw A/B must survive.
    auto filtered = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, filteredSettings);
    filtered.ApplyLinearOperator(work.coefficients, ay);
    filtered.ApplyLocalPreconditioner(rhs, delta);
    {
        auto original = ax.Owned().Read(), raw = ay.Owned().Read(), zero = delta.Owned().Read();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
            {
                CHECK(raw.View()(iCell, mode) == original.View()(iCell, mode));
                CHECK(zero.View()(iCell, mode) == 0);
            }
    }
}

TEST_CASE("Point caches snapshot IDs and coordinates and sample first and second derivatives")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    auto mean = metrics.BuildField(1), coefficients = metrics.BuildField(plan.NRecDOF());
    R::Array<I> cells(R::RowLayout::Dense(metrics.NumCell(), 1));
    R::Array<double> points(R::RowLayout::Dense(metrics.NumCell(), 3)), result(R::RowLayout::Dense(metrics.NumCell(), 1));
    // f=x^2+2*y, using square-cell moments h^2/12 as an independent oracle.
    {
        auto geometry = metrics.AcquireRead();
        auto u = mean.Owned().Write(), r = coefficients.Owned().Write(), p = points.Write();
        auto c = cells.Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto center = geometry.HostView().GetCellCent(iCell);
            const auto scale = geometry.HostView().cells(iCell, 7);
            u.View()(iCell, 0) = center[0] * center[0] + .01 / 12 + 2 * center[1];
            r.View()(iCell, 0) = 2 * center[0] * scale;
            r.View()(iCell, 1) = 2 * scale;
            r.View()(iCell, 2) = scale * scale;
            c.View()(iCell, 0) = iCell;
            for (int d = 0; d < 3; ++d)
                p.View()(iCell, d) = center[d];
        }
    }
    C::PolynomialSamplingPlan sampler(plan, cells, points);
    if (metrics.NumCell())
    {
        auto geometry = metrics.AcquireRead();
        auto read = mesh->AcquireRead(G::MeshReadRequirements::CellFlux());
        const auto view = read.HostView();
        const auto iFace = view.CellFaces(0)[0];
        const int side = view.CellIsFaceBack(0, iFace, 0) ? 0 : 1;
        const auto basis = plan.GetCachedFaceBasis(iFace, side, 0);
        auto scope = plan.AcquireBasisRead();
        const auto cache = scope.HostView();
        const auto iPoint = cache.faceOffsets(iFace, 0);
        const double scale = geometry.HostView().cells(0, 7);
        CHECK(basis(1, 0) == doctest::Approx(1 / scale));
        CHECK(basis(2, 0) == 0);
        CHECK(basis(3, 2) == doctest::Approx(2 / (scale * scale)));
        for (int derivative = 0; derivative <= plan.NRecDOF(); ++derivative)
            for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                CHECK(cache.FaceDerivative(iPoint, side, derivative, mode) == basis(derivative, mode));
    }
    {
        auto c = cells.Write();
        auto p = points.Write();
        for (I iRow = 0; iRow < cells.RowCount(); ++iRow)
        {
            c.View()(iRow, 0) = -1;
            p.View()(iRow, 0) = 1e20;
        }
    }
    for (int derivative : {0, 1, 2, 3})
    {
        sampler.Evaluate(mean, coefficients, result, derivative);
        auto read = result.Read();
        auto geometry = metrics.AcquireRead();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            const auto p = geometry.HostView().GetCellCent(iCell);
            CHECK(read.View()(iCell, 0) == doctest::Approx(derivative == 0 ? p[0] * p[0] + 2 * p[1] : derivative == 1 ? 2 * p[0]
                                                                                                                      : 2)
                                               .epsilon(1e-11));
        }
    }
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Geometry);
        mutation.Commit();
    }
    CHECK_THROWS(sampler.Evaluate(mean, coefficients, result));
}

// Test-only adapter for the historical Krylov vector concept. Runtime handles
// are shallow owners; numerical assignment must explicitly copy values.
struct KrylovField
{
    std::optional<C::FVField> field;
    KrylovField() = default;
    KrylovField(KrylovField &&) = default;
    KrylovField(const KrylovField &other) { *this = other; }
    KrylovField &operator=(KrylovField &&) = default;
    KrylovField &operator=(const KrylovField &other)
    {
        if (this != &other)
        {
            if (!field)
                field.emplace(other.field->Space(), other.field->Width());
            DNDS::Algebra::Copy(*field, *other.field);
        }
        return *this;
    }
    void addTo(const KrylovField &other, double factor) { DNDS::Algebra::Axpy(*field, factor, *other.field); }
    void operator*=(double factor) { DNDS::Algebra::Scale(*field, factor); }
};

TEST_CASE("Existing GMRES consumes native FV operators and DNDS field algebra through an explicit vector adapter")
{
    auto mesh = G::OverlapTest::BuildNativeFixture(mpi, true).mesh;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.maxOrder = 1;
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    const auto initialize = [&](KrylovField &x)
    { x.field.emplace(metrics.BuildField(plan.NRecDOF())); };
    KrylovField exact, rhs, solution, check;
    initialize(exact);
    initialize(rhs);
    initialize(solution);
    initialize(check);
    {
        auto write = exact.field->Owned().Write();
        for (I iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (I mode = 0; mode < plan.NRecDOF(); ++mode)
                write.View()(iCell, mode) = std::sin(.37 * mesh->OriginalEntityId(G::EntityKind::Cell, iCell) + .2 * mode);
    }
    const auto apply = [&](KrylovField &x, KrylovField &y)
    {
        metrics.BindHalo(*x.field).Begin().Wait();
        plan.ApplyLinearOperator(*x.field, *y.field);
    };
    const auto precondition = [&](KrylovField &x, KrylovField &y)
    { plan.ApplyLocalPreconditioner(*x.field, *y.field); };
    const auto dot = [](KrylovField &x, KrylovField &y)
    { return DNDS::Algebra::DotGlobal(*x.field, *y.field); };
    apply(exact, rhs);
    DNDS::Linear::GMRES_LeftPreconditioned<KrylovField> gmres(24, initialize);
    gmres.solve(apply, precondition, dot, rhs, solution, 20, [](int, double residual, double initial)
                { return residual < 1e-11 * initial; });
    apply(solution, check);
    DNDS::Algebra::Axpy(*check.field, -1., *rhs.field);
    const auto relative = std::sqrt(dot(check, check) / dot(rhs, rhs));
    CHECK(relative < 1e-9); // Independent final residual; return status alone is insufficient.
    DNDS::Algebra::Axpy(*solution.field, -1., *exact.field);
    CHECK(std::sqrt(dot(solution, solution) / dot(exact, exact)) < 1e-8);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.setWorld();
    int result;
    {
        doctest::Context context(argc, argv);
        result = context.run();
    }
    MPI_Finalize();
    return result;
}
