#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"
#include "CFV/Plans/FiniteVolumePlan.hpp"
#include "CFV/VariationalReconstruction.hpp"
#include "CFV/VariationalReconstruction_LimiterProcedure.hxx"
#include "CFV/Plans/InterfaceSamplingPlan.hpp"
#include "Geom/IO/NativeMeshCGNS.hpp"
#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"
#include "../Geom/InterfaceFixture.hpp"

using namespace DNDS;
using namespace DNDS::Geom;
namespace R = DNDS::Runtime;
namespace C = DNDS::CFV;
static MPIInfo mpi;
static std::string MeshPath(const char *name)
{
    std::string path = __FILE__;
    for (int i = 0; i < 4; ++i)
        path = path.substr(0, path.find_last_of('/'));
    return path + "/data/mesh/" + name;
}
static ssp<UnstructuredMesh> Mesh()
{
    PartitionOptions partition;
    partition.metisSeed = 42;
    AutoAppendName2ID names;
    auto mesh = ReadNativeMeshFromCGNS(mpi, 2, MeshPath("Uniform_3x3_wall.cgns"), partition,
                                       [&](const std::string &name)
                                       { return names(name); });
    mesh->PrepareNativeTopology();
    return mesh;
}
static DNDS::real Polynomial(const tPoint &p, int degree)
{
    DNDS::real v = 1 + 2 * p[0] - .5 * p[1];
    if (degree >= 2)
        v += .3 * p[0] * p[0] + .2 * p[0] * p[1] - .1 * p[1] * p[1];
    if (degree >= 3)
        v += .02 * p[0] * p[0] * p[1] - .03 * p[1] * p[1] * p[1];
    return v;
}
TEST_CASE("Runtime FV matches legacy VR sweeps and reproduces cell-average polynomials")
{
    auto mesh = Mesh();
    for (int degree = 1; degree <= 3; ++degree)
        for (bool hqm : {false, true})
        {
            CAPTURE(degree);
            CAPTURE(hqm);
            C::ReconstructionSettings settings;
            settings.maxOrder = degree, settings.intOrder = 6, settings.HQMWeights = hqm;
            auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 6);
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
            auto u = metrics.BuildField(1);
            auto workspace = plan.BuildWorkspace(1);
            auto boundary = plan.BuildBoundaryValues(1);
            C::VariationalReconstruction<2> legacy(mpi, mesh);
            C::VRSettings old(2);
            nlohmann::ordered_json json;
            old.WriteIntoJson(json);
            json["maxOrder"] = degree, json["intOrder"] = 6, json["SORInstead"] = false, json["cacheDiffBase"] = true;
            if (hqm)
            {
                json["functionalSettings"]["dirWeightScheme"] = "HQM_OPT";
                json["functionalSettings"]["geomWeightScheme"] = "HQM_SD";
                json["functionalSettings"]["geomWeightBias"] = 1.;
            }
            legacy.parseSettings(json);
            legacy.ConstructMetrics();
            legacy.ConstructBaseAndWeight([](t_index, int order)
                                          { return order == 0 ? 1. : 0.; });
            legacy.ConstructRecCoeff();
            C::tUDof<1> oldMean;
            C::tURec<1> oldRec, oldNext;
            legacy.BuildUDof(oldMean, 1), legacy.BuildURec(oldRec, 1), legacy.BuildURec(oldNext, 1);
            oldRec.setConstant(0), oldNext.setConstant(0);
            {
                auto write = u.Owned().Write();
                auto values = write.View<R::FixedRows<1>>();
                auto read = mesh->AcquireRead(MeshReadRequirements::Primary());
                const auto view = read.HostView();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                {
                    Elem::Quadrature q(view.GetCellElement(iCell), 6);
                    tSmallCoords coords;
                    view.GetCoordsOnCell(iCell, coords);
                    DNDS::real average = 0;
                    q.Integration(average, [&](auto &v, int, const tPoint &, const auto &shape)
                                  { v = Polynomial(Elem::PPhysicsCoordD01Nj(coords, shape), degree) * Elem::CellJacobianDet(2, coords, shape); });
                    values(iCell, 0) = average / metrics.GetCellVol(iCell);
                    oldMean[iCell](0) = values(iCell, 0);
                }
            }
            {
                auto write = boundary.values.Write();
                for (DNDS::index i = 0; i < plan.NumBoundaryQuadrature(); ++i)
                    write.View<R::FixedRows<1>>()(i, 0) = Polynomial(plan.GetBoundaryQuadrature(i).point, degree);
            }
            auto meanHalo = metrics.BindHalo(u), recHalo = metrics.BindHalo(workspace.coefficients);
            meanHalo.Begin().Wait();
            oldMean.trans.startPersistentPull(), oldMean.trans.waitPersistentPull();
            C::VariationalReconstruction<2>::TFBoundary<1> bc = [degree](const auto &, const auto &, DNDS::index, DNDS::index, int, const tPoint &, const tPoint &p, t_index)
            { return Eigen::Vector<DNDS::real, 1>{Polynomial(p, degree)}; };
            for (int iteration = 0; iteration < 12; ++iteration)
            {
                recHalo.Begin().Wait();
                oldRec.trans.startPersistentPull(), oldRec.trans.waitPersistentPull();
                plan.DoReconstructionIter(u, workspace, boundary);
                legacy.DoReconstructionIter<1>(oldRec, oldNext, oldMean, bc, true);
                std::swap(oldRec, oldNext);
                auto read = workspace.coefficients.Owned().Read();
                const auto values = read.View<R::TableRows>();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                        CHECK(values(iCell, mode) == doctest::Approx(oldRec[iCell](mode, 0)).epsilon(2e-9).scale(1.));
            }
            for (int iteration = 0; iteration < 800; ++iteration)
            {
                recHalo.Begin().Wait();
                plan.DoReconstructionIter(u, workspace, boundary);
            }
            R::Array<DNDS::index> ids(R::RowLayout::Dense(mesh->NumCell(), 1));
            R::Array<DNDS::real> points(R::RowLayout::Dense(mesh->NumCell(), 3)), result(R::RowLayout::Dense(mesh->NumCell(), 1));
            {
                auto idWrite = ids.Write();
                auto pointWrite = points.Write();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                {
                    idWrite.View()(iCell, 0) = iCell;
                    const tPoint p = metrics.GetCellCent(iCell) + tPoint{.01, -.02, 0};
                    for (int d = 0; d < 3; ++d)
                        pointWrite.View()(iCell, d) = p[d];
                }
            }
            plan.Evaluate(u, workspace.coefficients, ids, points, result);
            auto pointRead = points.Read(), resultRead = result.Read();
            for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
            {
                const auto p = pointRead.View()[iCell];
                CHECK(resultRead.View()(iCell, 0) == doctest::Approx(Polynomial({p[0], p[1], p[2]}, degree)).epsilon(2e-7));
            }
            if (mpi.size == 1)
            {
                std::vector<DNDS::index> selected(size_t(mesh->NumBnd()));
                std::iota(selected.begin(), selected.end(), DNDS::index(0));
                auto patch = BoundaryPatch::Create(mesh, selected);
                R::Array<DNDS::index> queryIds(R::RowLayout::Dense(patch.Size(), 1));
                R::Array<double> queryPoints(R::RowLayout::Dense(patch.Size(), 3)), sampled(R::RowLayout::Dense(patch.Size(), 1));
                {
                    auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
                    const auto view = read.HostView();
                    auto write = queryPoints.Write();
                    auto id = queryIds.Write();
                    for (DNDS::index i = 0; i < patch.Size(); ++i)
                    {
                        const auto face = patch.Element(i).face;
                        const auto nodes = view.FaceNodes(face);
                        const tPoint p = .4 * view.Coord(nodes[0]) + .6 * view.Coord(nodes[1]);
                        for (int d = 0; d < 3; ++d)
                            write.View()(i, d) = p[d];
                        id.View()(i, 0) = 2 * patch.Size() - i;
                    }
                }
                auto correspondence = InterfacePointMap::Match(patch, queryPoints, queryIds);
                C::InterfaceSamplingPlan sampling(plan, correspondence);
                sampling.Evaluate(u, workspace.coefficients, sampled);
                auto read = sampled.Read();
                for (DNDS::index i = 0; i < patch.Size(); ++i)
                    CHECK(read.View()(i, 0) == doctest::Approx(Polynomial(correspondence.Get(i).donorPoint, degree)).epsilon(2e-7));
            }
        }
}
TEST_CASE("WBAP explicit five-component face transforms match legacy and preserve a passive sixth field")
{
    auto mesh = Mesh();
    C::ReconstructionSettings settings;
    settings.maxOrder = 2;
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, settings.intOrder);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
    auto work = plan.BuildWorkspace(6);
    auto characteristic = plan.BuildFaceTransform(0, 5);
    Eigen::Matrix<DNDS::real, 5, 5> L = Eigen::Matrix<DNDS::real, 5, 5>::Identity();
    L(0, 0) = .8, L(0, 1) = -.6, L(1, 0) = .6, L(1, 1) = .8;
    {
        auto left = characteristic.forward.Write(), right = characteristic.inverse.Write();
        for (DNDS::index iFace = 0; iFace < metrics.NumFaceProc(); ++iFace)
        {
            Eigen::Map<Eigen::MatrixXd>(left.View()[iFace].data(), 5, 5) = L;
            Eigen::Map<Eigen::MatrixXd>(right.View()[iFace].data(), 5, 5) = L.transpose();
        }
    }
    CHECK_THROWS(plan.DoLimiterWBAP_3Order(work, 2, &characteristic));
    plan.FinalizeFaceTransform(characteristic);
    C::VariationalReconstruction<2> legacy(mpi, mesh);
    C::VRSettings old(2);
    nlohmann::ordered_json json;
    old.WriteIntoJson(json);
    json["maxOrder"] = 2, json["intOrder"] = 5, json["cacheDiffBase"] = true;
    legacy.parseSettings(json);
    legacy.ConstructMetrics();
    legacy.ConstructBaseAndWeight([](t_index, int order)
                                  { return order == 0 ? 1. : 0.; });
    legacy.ConstructRecCoeff();
    C::tUDof<6> mean;
    C::tURec<6> rec, output, temporary;
    C::tScalarPair indicator;
    legacy.BuildUDof(mean, 6), legacy.BuildURec(rec, 6), legacy.BuildURec(output, 6), legacy.BuildURec(temporary, 6);
    legacy.BuildScalar(indicator);
    mean.setConstant(2);
    for (DNDS::index iCell = 0; iCell < mesh->NumCellProc(); ++iCell)
        indicator(iCell, 0) = 1;
    {
        auto write = work.coefficients.Owned().Write();
        for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            for (int v = 0; v < 6; ++v)
                for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                {
                    const auto id = mesh->OriginalEntityId(EntityKind::Cell, iCell);
                    const double value = std::sin(.7 * id + .31 * v + .19 * mode);
                    write.View()(iCell, v * plan.NRecDOF() + mode) = rec[iCell](mode, v) = value;
                }
    }
    auto halo = metrics.BindHalo(work.coefficients);
    halo.Begin().Wait();
    rec.trans.startPersistentPull(), rec.trans.waitPersistentPull();
    mean.trans.startPersistentPull(), mean.trans.waitPersistentPull();
    using Batch = C::VariationalReconstruction<2>::tLimitBatch<6>;
    C::VariationalReconstruction<2>::tFMEig<6> forward = [L](const auto &a, const auto &b, const tPoint &, const Eigen::Ref<Batch> &input) -> Batch
    {
        CHECK(a.isConstant(2));
        CHECK(b.isConstant(2));
        Batch result = input;
        result.leftCols(5) = input.leftCols(5) * L.transpose();
        return result;
    };
    C::VariationalReconstruction<2>::tFMEig<6> inverse = [L](const auto &, const auto &, const tPoint &, const Eigen::Ref<Batch> &input) -> Batch
    {
        Batch result = input;
        result.leftCols(5) = input.leftCols(5) * L;
        return result;
    };
    legacy.DoLimiterWBAP_3<6>(mean, rec, output, temporary, indicator, true, forward, inverse, true);
    for (int order = 2; order >= 1; --order)
    {
        plan.DoLimiterWBAP_3Order(work, order, &characteristic);
        halo.Begin().Wait();
    }
    auto read = work.coefficients.Owned().Read();
    for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        for (int v = 0; v < 6; ++v)
            for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                CHECK(read.View()(iCell, v * plan.NRecDOF() + mode) == doctest::Approx(output[iCell](mode, v)).epsilon(1e-9).scale(1));
    auto bad = characteristic;
    bad.first = 2;
    CHECK_THROWS(plan.DoLimiterWBAP_3Order(work, 2, &bad));
    {
        auto write = characteristic.inverse.Write();
        if (metrics.NumFaceProc())
            write.View()(0, 0) = 7;
    }
    if (metrics.NumFaceProc())
        CHECK_THROWS(plan.FinalizeFaceTransform(characteristic));
}

TEST_CASE("FV rejects wrong association and invalid shapes before writes")
{
    auto mesh = Mesh();
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    auto other = C::FiniteVolumePlan::ConstructMetrics(mesh);
    auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics);
    {
        auto read = metrics.AcquireRead();
        const auto view = read.HostView();
        for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
        {
            CHECK(view.GetCellVol(iCell) == metrics.GetCellVol(iCell));
            CHECK((view.GetCellCent(iCell) - metrics.GetCellCent(iCell)).norm() == 0);
        }
        for (DNDS::index iFace = 0; iFace < metrics.NumFaceProc(); ++iFace)
            for (DNDS::index i = view.FaceQuadratureBegin(iFace); i < view.FaceQuadratureEnd(iFace); ++i)
            {
                CHECK(view.GetFaceWeight(i) > 0);
                CHECK(view.GetFaceNorm(i).norm() == doctest::Approx(1));
            }
        CHECK_THROWS(mesh->versions.BeginMutation(R::Change::Geometry));
    }
    auto work = plan.BuildWorkspace(1);
    auto boundary = plan.BuildBoundaryValues(1);
    auto wrong = other.BuildField(1);
    CHECK_THROWS(plan.DoReconstructionIter(wrong, work, boundary));
    auto malformed = plan.BuildBoundaryValues(2);
    auto mean = metrics.BuildField(1);
    CHECK_THROWS(plan.DoReconstructionIter(mean, work, malformed));
    CHECK_THROWS(plan.DoLimiterWBAP_3Order(work, 0));
    auto invalid = plan.Settings();
    invalid.maxOrder = 4;
    CHECK_THROWS(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, invalid));
    // Same-size storage from another plan must not be accepted by row count.
    auto wrongWork = C::VariationalReconstructionPlan::ConstructRecCoeff(other).BuildWorkspace(1);
    CHECK_THROWS(plan.DoReconstructionIter(mean, wrongWork, boundary));
    {
        auto mutation = mesh->versions.BeginMutation(R::Change::Geometry);
        mutation.Commit();
    }
    CHECK_THROWS(plan.Validate());
}

TEST_CASE("Periodic quad and triangle reconstruction matches relaxed legacy Jacobi and ordered SOR")
{
    for (const char *name : {"IV10_2.cgns", "IV10U_10.cgns"})
        for (bool sor : {false, true})
        {
            CAPTURE(name);
            CAPTURE(sor);
            NativePeriodicImportOptions frames;
            frames.frames.translation[1].map() = tPoint{10, 0, 0};
            frames.frames.translation[2].map() = tPoint{0, 10, 0};
            PartitionOptions partition;
            partition.metisSeed = 42;
            AutoAppendName2ID names;
            auto mesh = ReadNativeMeshFromCGNS(mpi, 2, MeshPath(name), partition, [&](const std::string &zone)
                                               { return names(zone); }, frames);
            mesh->PrepareNativeTopology();
            C::ReconstructionSettings settings;
            settings.SORInstead = sor, settings.jacobiRelax = .73;
            auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, settings.intOrder);
            auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
            auto mean = metrics.BuildField(1);
            auto work = plan.BuildWorkspace(1);
            auto boundary = plan.BuildBoundaryValues(1);
            CHECK(plan.NumBoundaryQuadrature() == 0);
            C::VariationalReconstruction<2> legacy(mpi, mesh);
            C::VRSettings old(2);
            nlohmann::ordered_json json;
            old.WriteIntoJson(json);
            json["maxOrder"] = 2, json["intOrder"] = settings.intOrder;
            json["SORInstead"] = sor, json["jacobiRelax"] = settings.jacobiRelax;
            legacy.parseSettings(json);
            legacy.ConstructMetrics();
            legacy.ConstructBaseAndWeight();
            legacy.ConstructRecCoeff();
            legacy.SetPeriodicTransformations();
            C::tUDof<1> oldMean;
            C::tURec<1> oldRec, next;
            legacy.BuildUDof(oldMean, 1), legacy.BuildURec(oldRec, 1), legacy.BuildURec(next, 1);
            oldRec.setConstant(0), next.setConstant(0);
            {
                auto write = mean.Owned().Write();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    write.View()(iCell, 0) = oldMean[iCell](0) = std::sin(.71 * mesh->OriginalEntityId(EntityKind::Cell, iCell));
            }
            auto halo = metrics.BindHalo(work.coefficients);
            metrics.BindHalo(mean).Begin().Wait();
            oldMean.trans.startPersistentPull(), oldMean.trans.waitPersistentPull();
            C::VariationalReconstruction<2>::TFBoundary<1> bc = [](const auto &, const auto &, DNDS::index, DNDS::index, int, const tPoint &, const tPoint &, t_index)
            { throw std::runtime_error("fully periodic mesh must never call boundary handler"); return Eigen::Vector<DNDS::real,1>::Zero().eval(); };
            for (int iteration = 0; iteration < 12; ++iteration)
            {
                halo.Begin().Wait();
                oldRec.trans.startPersistentPull(), oldRec.trans.waitPersistentPull();
                plan.DoReconstructionIter(mean, work, boundary);
                legacy.DoReconstructionIter<1>(oldRec, next, oldMean, bc, true);
                // putIntoNew=true publishes next for both schedules; legacy
                // internally swaps its SOR working rows at the end of a sweep.
                std::swap(oldRec, next);
                auto read = work.coefficients.Owned().Read();
                for (DNDS::index iCell = 0; iCell < mesh->NumCell(); ++iCell)
                    for (int mode = 0; mode < plan.NRecDOF(); ++mode)
                        CHECK(read.View()(iCell, mode) == doctest::Approx(oldRec[iCell](mode, 0)).epsilon(2e-9).scale(1));
            }
        }
}

TEST_CASE("Three-dimensional FV reproduces cubic cell-average polynomials on tetrahedron and hexahedron with empty ranks")
{
    const auto function = [](const tPoint &p)
    { return 1 + .2 * p[0] - .3 * p[1] + .5 * p[2] + .1 * p[0] * p[1] * p[2] + .07 * p[2] * p[2] * p[2]; };
    for (bool hex : {false, true})
    {
        auto mesh = InterfaceFixture::SingleCell(mpi, hex);
        C::ReconstructionSettings settings;
        settings.maxOrder = 3, settings.intOrder = 6;
        auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh, 6);
        auto plan = C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings);
        auto mean = metrics.BuildField(1);
        auto work = plan.BuildWorkspace(1);
        auto boundary = plan.BuildBoundaryValues(1);
        {
            auto read = mesh->AcquireRead(MeshReadRequirements::CellFlux());
            const auto view = read.HostView();
            auto write = mean.Owned().Write();
            for (DNDS::index iCell = 0; iCell < metrics.NumCell(); ++iCell)
            {
                tSmallCoords coords;
                view.GetCoordsOnCell(iCell, coords);
                Elem::Quadrature quad(view.GetCellElement(iCell), 6);
                double integral = 0;
                quad.Integration(integral, [&](auto &value, int, const tPoint &, const auto &shape)
                                 { value = function(Elem::PPhysicsCoordD01Nj(coords, shape)) * Elem::CellJacobianDet(3, coords, shape); });
                write.View()(iCell, 0) = integral / metrics.GetCellVol(iCell);
            }
        }
        {
            auto write = boundary.values.Write();
            for (DNDS::index i = 0; i < plan.NumBoundaryQuadrature(); ++i)
                write.View()(i, 0) = function(plan.GetBoundaryQuadrature(i).point);
        }
        metrics.BindHalo(mean).Begin().Wait();
        metrics.BindHalo(work.coefficients).Begin().Wait();
        plan.DoReconstructionIter(mean, work, boundary);
        R::Array<DNDS::index> ids(R::RowLayout::Dense(metrics.NumCell(), 1));
        R::Array<double> points(R::RowLayout::Dense(metrics.NumCell(), 3)), result(R::RowLayout::Dense(metrics.NumCell(), 1));
        {
            auto indexWrite = ids.Write();
            auto pointWrite = points.Write();
            for (DNDS::index i = 0; i < metrics.NumCell(); ++i)
            {
                indexWrite.View()(i, 0) = i;
                pointWrite.View()(i, 0) = .2, pointWrite.View()(i, 1) = .1, pointWrite.View()(i, 2) = .3;
            }
        }
        plan.Evaluate(mean, work.coefficients, ids, points, result);
        auto read = result.Read();
        for (DNDS::index i = 0; i < metrics.NumCell(); ++i)
            CHECK(read.View()(i, 0) == doctest::Approx(function({.2, .1, .3})).epsilon(1e-9));
    }
}

TEST_CASE("Rank-deficient boundary functional and rank-local invalid settings fail collectively")
{
    auto mesh = InterfaceFixture::SingleCell(mpi, true);
    auto metrics = C::FiniteVolumePlan::ConstructMetrics(mesh);
    C::ReconstructionSettings settings;
    settings.boundaryDerivativeWeights = {0, 0, 0, 0};
    CHECK_THROWS(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings));
    settings = C::ReconstructionSettings{};
    if (mpi.rank == 0)
        settings.maxOrder = 4;
    CHECK_THROWS(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings));
    CHECK_NOTHROW(C::VariationalReconstructionPlan::ConstructRecCoeff(metrics));
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
