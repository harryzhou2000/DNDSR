"""Shared Runtime ownership and CPU reconstruction/interface integration.

Build and install all Python extensions before running, including under MPI.
The linear function is an independent exact FV solution on the quad fixture.
"""
import gc
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS, Geom, CFV
from DNDSR.DNDS import Runtime as R


def mesh():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    path = Path(__file__).resolve().parents[2] / "data/mesh/Uniform_3x3_wall.cgns"
    return Geom.with_native_topology(Geom.read_native_cgns(mpi, str(path), 2))


def function(point):
    return 1 + 2 * point[0] - .5 * point[1]


@pytest.mark.mpi
def test_reconstruction_shared_buffers_explicit_halo_and_point_sampling():
    grid = mesh()
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(grid)
    settings = CFV.ReconstructionSettings()
    settings.maxOrder = 1
    plan = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics, settings)
    diagnostics = plan.GetInverseDiagnostics()
    assert diagnostics.cells == MPI.COMM_WORLD.allreduce(metrics.NumCell())
    assert diagnostics.rankZeroCells == 0
    assert settings.svdAbsoluteTolerance == 0
    for i_cell in range(metrics.NumCell()):
        cell = plan.GetCellInverseDiagnostics(i_cell)
        assert cell.rank == plan.NRecDOF()
        assert np.isfinite(cell.rawCondition)
    mean = metrics.BuildField(1)
    work = plan.BuildWorkspace(1)
    bc = plan.BuildBoundaryValues(1)
    centers = np.array([metrics.GetCellCent(i) for i in range(metrics.NumCell())]).reshape(-1, 3)
    expected = np.array([function(p) for p in centers]).reshape(-1, 1)
    shared_mean = mean.owned
    shared_mean.assign(expected)
    bc.values.assign(np.array([function(plan.GetBoundaryQuadrature(i).point)
                              for i in range(plan.NumBoundaryQuadrature())]).reshape(-1, 1))
    # This is the existing DNDS algebra field and the same allocation, not a
    # second copy behind a numerical wrapper.
    assert CFV.FVField is R.Algebra.DoubleDofField
    assert shared_mean.identity == mean.owned.identity
    metrics.BindHalo(mean).begin().wait()
    halo = metrics.BindHalo(work.coefficients)
    for _ in range(180):
        halo.begin().wait()
        plan.DoReconstructionIter(mean, work, bc)
    ids = R.Array(R.RowLayout.dense(metrics.NumCell(), 1), np.int64)
    points = R.Array(R.RowLayout.dense(metrics.NumCell(), 3))
    output = R.Array(R.RowLayout.dense(metrics.NumCell(), 1))
    ids.assign(np.arange(metrics.NumCell(), dtype=np.int64).reshape(-1, 1))
    points.assign(centers + [.017, -.011, 0])
    plan.Evaluate(mean, work.coefficients, ids, points, output)
    np.testing.assert_allclose(output.numpy(), expected + 2*.017 + .5*.011, atol=1e-9, rtol=1e-9)
    # An active input writer is rejected before a numerical sweep can run.
    with mean.owned.write():
        with pytest.raises(RuntimeError):
            plan.DoReconstructionIter(mean, work, bc)
    coefficients = work.coefficients.owned
    snapshot = coefficients.numpy()
    del work, plan, metrics, mean, grid, halo
    gc.collect()
    np.testing.assert_array_equal(coefficients.numpy(), snapshot)


@pytest.mark.mpi
def test_two_domains_interface_sampling_and_transform_publication():
    donor = mesh()
    receiver = mesh()  # independent domain identity even for coincident geometry
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(donor)
    plan = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics)
    mean = metrics.BuildField(6)
    mean.owned.assign(np.tile(np.arange(6, dtype=np.float64), (metrics.NumCell(), 1)))
    work = plan.BuildWorkspace(6)
    metrics.BindHalo(mean).begin().wait()
    metrics.BindHalo(work.coefficients).begin().wait()
    transform = plan.BuildFaceTransform(0, 5)
    same = transform.forward.numpy()
    transform.forward.assign(same)  # even an equal-value write needs publication
    with pytest.raises(RuntimeError, match="finalize"):
        plan.DoLimiterWBAP_3Order(work, 2, transform)
    plan.FinalizeFaceTransform(transform)
    plan.DoLimiterWBAP_3Order(work, 2, transform)
    if MPI.COMM_WORLD.size != 1:
        return  # C++ MPI tests exercise the explicit distributed-search rejection
    donor_patch = Geom.BoundaryPatch.Create(donor, list(range(donor.NumBnd())))
    target_patch = Geom.BoundaryPatch.Create(receiver, list(reversed(range(receiver.NumBnd()))))
    query = []
    with receiver.acquire_read(faces=True) as read:
        for i in range(target_patch.Size()):
            record = target_patch.Element(i)
            nodes = read.face_nodes(record.face)
            query.append(.3 * read.coord(int(nodes[0])) + .7 * read.coord(int(nodes[1])))
    points = R.Array(R.RowLayout.dense(len(query), 3))
    ids = R.Array(R.RowLayout.dense(len(query), 1), np.int64)
    points.assign(np.array(query))
    ids.assign(np.arange(100, 100+len(query), dtype=np.int64).reshape(-1,1))
    mapping = Geom.InterfacePointMap.Match(donor_patch, points, ids)
    sample = CFV.InterfaceSamplingPlan(plan, mapping)
    result = R.Array(R.RowLayout.dense(len(query), 6))
    sample.Evaluate(mean, work.coefficients, result)
    np.testing.assert_allclose(result.numpy(), np.tile(np.arange(6), (len(query), 1)))
    wrong_metrics = CFV.FiniteVolumePlan.ConstructMetrics(receiver)
    wrong_plan = CFV.VariationalReconstructionPlan.ConstructRecCoeff(wrong_metrics)
    with pytest.raises(RuntimeError, match="different mesh"):
        CFV.InterfaceSamplingPlan(wrong_plan, mapping)


@pytest.mark.mpi
def test_distributed_interface_owner_sampling_and_query_order():
    donor = mesh()
    receiver = mesh()
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(donor)
    plan = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics)
    mean = metrics.BuildField(1)
    work = plan.BuildWorkspace(1)
    mean.owned.assign(np.full((metrics.NumCell(), 1), MPI.COMM_WORLD.rank + 1.0))
    metrics.BindHalo(mean).begin().wait()
    metrics.BindHalo(work.coefficients).begin().wait()
    patch = Geom.BoundaryPatch.Create(donor, list(range(donor.NumBnd())))
    target = Geom.BoundaryPatch.Create(receiver, list(range(receiver.NumBnd())))
    query = []
    with receiver.acquire_read(faces=True) as read:
        for i in range(target.Size()):
            nodes = read.face_nodes(target.Element(i).face)
            query.append(.37 * read.coord(int(nodes[0])) + .63 * read.coord(int(nodes[1])))
    points = R.Array(R.RowLayout.dense(len(query), 3))
    keys = R.Array(R.RowLayout.dense(len(query), 1), np.int64)
    points.assign(np.asarray(query).reshape(-1, 3))
    keys.assign((100 - 7 * np.arange(len(query), dtype=np.int64)).reshape(-1, 1))
    maps = [Geom.DistributedInterfacePointMap.Match(patch, points, keys, mode)
            for mode in (Geom.InterfaceRoutingMode.CoarseBoxes, Geom.InterfaceRoutingMode.AllGatherBoundary)]
    for mapping in maps:
        sampler = CFV.DistributedInterfaceSamplingPlan(plan, mapping)
        scratch = sampler.BuildWorkspace(1)
        output = R.Array(R.RowLayout.dense(len(query), 1))
        sampler.Evaluate(mean, work.coefficients, scratch, output)
        expected = np.array([mapping.Get(i).donorRank + 1.0 for i in range(len(query))]).reshape(-1, 1)
        np.testing.assert_array_equal(output.numpy(), expected)
    assert [maps[0].Get(i).geometry.donorOriginalId for i in range(len(query))] == [
        maps[1].Get(i).geometry.donorOriginalId for i in range(len(query))]


@pytest.mark.mpi
def test_standalone_fv_gradients_and_vr_conversion():
    grid = mesh()
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(grid)
    mean = metrics.BuildField(1)
    mean.owned.assign(np.asarray([function(metrics.GetCellCent(i)) for i in range(metrics.NumCell())]).reshape(-1, 1))
    metrics.BindHalo(mean).begin().wait()
    reconstruction = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics)
    for method in (CFV.GradientMethod.GreenGauss, CFV.GradientMethod.LeastSquares, CFV.GradientMethod.GGMP):
        plan = CFV.GradientPlan.Prepare(grid, metrics, method)
        boundary = plan.BuildBoundaryValues(1)
        boundary.assign(np.asarray([function(plan.GetBoundaryQuadrature(i).point)
                                    for i in range(plan.NumBoundaryQuadrature())]).reshape(-1, 1))
        gradient = metrics.BuildField(2)
        plan.Evaluate(mean, boundary, gradient)
        np.testing.assert_allclose(gradient.owned.numpy(), np.tile([2, -.5], (metrics.NumCell(), 1)), atol=1e-12)
        coefficients = metrics.BuildField(reconstruction.NRecDOF())
        plan.ConvertToCoefficients(gradient, reconstruction, coefficients)
        np.testing.assert_array_equal(coefficients.owned.numpy()[:, 2:], 0)
