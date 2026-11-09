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
def test_metric_options_match_keyword_adapter():
    grid = mesh()
    options = CFV.FiniteVolumeMetricOptions()
    options.integrationOrder = 5
    options.faceIntegrationOrder = 1
    options.collectivePreflight = False
    typed = CFV.FiniteVolumePlan.ConstructMetrics(grid, options)
    adapted = CFV.FiniteVolumePlan.ConstructMetrics(
        grid, 5, False, collective_preflight=False, faceIntOrder=1
    )
    assert typed.NumCellProc() == adapted.NumCellProc()
    assert typed.NumFaceProc() == adapted.NumFaceProc()
    for cell in range(typed.NumCell()):
        assert typed.GetCellVol(cell) == adapted.GetCellVol(cell)
        np.testing.assert_array_equal(typed.GetCellCent(cell), adapted.GetCellCent(cell))
    with pytest.raises(TypeError):
        CFV.FiniteVolumePlan.ConstructMetrics(grid, 5, False, False)


@pytest.mark.mpi
def test_prepared_capabilities_preserve_sweep_and_freeze_operation_contract():
    grid = mesh()
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(grid)
    settings = CFV.ReconstructionSettings()
    settings.maxOrder = 1
    capabilities = CFV.ReconstructionCapabilities.ForSweeps(False)
    full = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics, settings)
    lean = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics, settings, capabilities)
    capabilities.linearSystem = True  # publication owns a frozen copy
    assert not lean.Capabilities().linearSystem
    assert not lean.Capabilities().Supports(CFV.ReconstructionOperation.WBAP)
    omitted = {entry.name for entry in lean.StorageReport().tables if not entry.allocated}
    assert {"secondary", "localMatrix", "localInverse", "rawMatrixB", "rawVectorB", "rawBoundaryInjection"} <= omitted
    mean = metrics.BuildField(1)
    mean.owned.assign(np.array([function(metrics.GetCellCent(i)) for i in range(metrics.NumCell())]).reshape(-1, 1))
    metrics.BindHalo(mean).begin().wait()
    workspaces = []
    for plan in (full, lean):
        work = plan.BuildWorkspace(1)
        work.coefficients.owned.assign(np.full((metrics.NumCell(), plan.NRecDOF()), .17))
        boundary = plan.BuildBoundaryValues(1)
        boundary.values.assign(np.array([function(plan.GetBoundaryQuadrature(i).point)
                                        for i in range(plan.NumBoundaryQuadrature())]).reshape(-1, 1))
        metrics.BindHalo(work.coefficients).begin().wait()
        plan.DoReconstructionIter(mean, work, boundary)
        workspaces.append(work)
    np.testing.assert_array_equal(workspaces[0].coefficients.owned.numpy(), workspaces[1].coefficients.owned.numpy())
    output = metrics.BuildField(lean.NRecDOF())
    output.owned.assign(np.full((metrics.NumCell(), lean.NRecDOF()), -13.))
    with pytest.raises(RuntimeError, match="not prepared"):
        lean.ApplyLinearOperator(workspaces[1].coefficients, output)
    np.testing.assert_array_equal(output.owned.numpy(), np.full((metrics.NumCell(), lean.NRecDOF()), -13.))


@pytest.mark.mpi
@pytest.mark.parametrize("cache_enabled", [True, False])
def test_reconstruction_shared_buffers_explicit_halo_and_point_sampling(cache_enabled):
    grid = mesh()
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(grid)
    lengths = metrics.BuildCellMaxLenScale().numpy()
    assert lengths.shape == (metrics.NumCellProc(), 1)
    assert np.all(np.isfinite(lengths)) and np.all(lengths > 0)
    settings = CFV.ReconstructionSettings()
    assert settings.cacheDiffBaseOrder == 1
    settings.cacheDiffBase = cache_enabled
    settings.maxOrder = 1
    settings.boundaryDerivativeWeights = [1, 0, 0, 0]
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
    options = R.TransferOptions()
    assert options.preflight == R.TransferPreflight.Collective
    options.preflight = R.TransferPreflight.AbortOnFailure
    metrics.BindHalo(mean, options).begin().wait()
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
    cache = CFV.PolynomialSamplingPlan(plan, ids, points)
    points.assign(np.full((metrics.NumCell(), 3), 100.0))
    cache.Evaluate(mean, work.coefficients, output, 1)
    np.testing.assert_allclose(output.numpy(), 2, atol=1e-9)
    rhs = metrics.BuildField(plan.NRecDOF())
    residual = metrics.BuildField(plan.NRecDOF())
    correction = metrics.BuildField(plan.NRecDOF())
    halo.begin().wait()
    plan.BuildRightHandSide(mean, bc, rhs)
    plan.ComputeResidual(rhs, work.coefficients, residual)
    plan.ApplyLocalPreconditioner(residual, correction)
    np.testing.assert_allclose(correction.owned.numpy(), 0, atol=1e-9)
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
        return  # The distributed API is exercised separately below and in C++.
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
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    coupling = R.CommunicationContext(mpi)
    maps = [Geom.DistributedInterfacePointMap.MatchCoupled(coupling, patch, points, keys, Geom.InterfaceRoutingMode.CoarseBoxes),
            Geom.DistributedInterfacePointMap.Match(patch, points, keys, Geom.InterfaceRoutingMode.AllGatherBoundary)]
    for mapping in maps:
        sampler = CFV.DistributedInterfaceSamplingPlan(plan, mapping)
        scratch = sampler.BuildWorkspace(1)
        output = R.Array(R.RowLayout.dense(len(query), 1))
        assert mapping.HasLocalDonor()
        sampler.EvaluateCoupled(mean, work.coefficients, scratch, output)
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

@pytest.mark.parametrize("format", ["col", "tiled"])
@pytest.mark.mpi
def test_stride_aware_reconstruction_uses_shared_runtime_physical_fields(format):
    grid = mesh()
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(grid)
    settings = CFV.ReconstructionSettings()
    settings.maxOrder = 1
    settings.boundaryDerivativeWeights = [1, 0, 0, 0]
    plan = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics, settings)
    prototype = R.RowLayout.col_major(0, 2, 8, 1) if format == "col" else R.RowLayout.tiled_col_major(0, 2, 8)
    mean = metrics.BuildFieldWithLayout(R.RowLayout.col_major(0, 1, 8, 1))
    work = plan.BuildWorkspaceWithLayout(1, prototype)
    bc = plan.BuildBoundaryValues(1)
    centers = np.array([metrics.GetCellCent(i) for i in range(metrics.NumCell())]).reshape(-1, 3)
    mean.owned.assign(np.array([function(p) for p in centers]).reshape(-1, 1))
    bc.values.assign(np.array([function(plan.GetBoundaryQuadrature(i).point)
                              for i in range(plan.NumBoundaryQuadrature())]).reshape(-1, 1))
    metrics.BindHalo(mean).begin().wait()
    halo = metrics.BindHalo(work.coefficients)
    for _ in range(180):
        halo.begin().wait()
        plan.DoReconstructionIter(mean, work, bc)
    ids = R.Array(R.RowLayout.dense(metrics.NumCell(), 1), np.dtype("int64"))
    points = R.Array(R.RowLayout.dense(metrics.NumCell(), 3))
    values = R.Array(R.RowLayout.col_major(metrics.NumCell(), 1, 8, 1))
    ids.assign(np.arange(metrics.NumCell(), dtype=np.int64).reshape(-1, 1))
    points.assign(centers + [0.01, -0.02, 0])
    plan.Evaluate(mean, work.coefficients, ids, points, values)
    expected = np.array([function(p + [0.01, -0.02, 0]) for p in centers]).reshape(-1, 1)
    np.testing.assert_allclose(values.numpy(), expected, atol=1e-9, rtol=1e-9)
    assert work.coefficients.owned.layout.kind == prototype.kind


@pytest.mark.mpi
@pytest.mark.parametrize("backend", ["mpi", "nccl"])
@pytest.mark.parametrize("kernel_name", ["FP64", "FP64TensorCore", "FP64TensorCoreShared", "FP64TensorCoreAsync"])
def test_device_reconstruction_publication_layouts_and_python_shared_buffers(backend, kernel_name):
    if backend == "nccl" and not R.NCCLTransferProvider.available():
        pytest.skip("build with NCCL")
    grid = mesh()
    metrics = CFV.FiniteVolumePlan.ConstructMetrics(grid)
    settings = CFV.ReconstructionSettings()
    settings.maxOrder = 2
    plan = CFV.VariationalReconstructionPlan.ConstructRecCoeff(metrics, settings)
    device = R.Device.cuda(0)
    context = R.ExecutionContext(device)
    if not R.cuda_enabled:
        with pytest.raises(RuntimeError):
            CFV.ReconstructionDevicePlan.Prepare(plan, device, context)
        return
    resident = CFV.ReconstructionDevicePlan.Prepare(plan, device, context)
    kernel = getattr(CFV.ReconstructionDeviceKernel, kernel_name)
    if not resident.SupportsKernel(kernel):
        pytest.skip("requested FP64 tensor-core kernel unavailable on this device/build")
    options = R.TransferOptions()
    if backend == "nccl":
        mpi = DNDS.MPIInfo()
        mpi.setWorld()
        options.provider = R.NCCLTransferProvider(mpi, device)
    assert sum(item.bytes for item in resident.Buffers()) > 0
    components = 5
    width = components * plan.NRecDOF()
    cpu = plan.BuildWorkspace(components)
    gpu = resident.BuildWorkspace(components, R.RowLayout.tiled_col_major(0, width, 8))
    mean = metrics.BuildField(components)
    gpu_mean = resident.BuildField(R.RowLayout.col_major(0, components, 32, 1))
    values = 1 + .01 * np.arange(metrics.NumCell() * components).reshape(-1, components)
    mean.owned.assign(values)
    gpu_mean.owned.assign(values)
    coefficients = .02 + .001 * np.arange(metrics.NumCell() * width).reshape(-1, width)
    cpu.coefficients.owned.assign(coefficients)
    gpu.coefficients.owned.assign(coefficients)
    boundary = plan.BuildBoundaryValues(components)
    boundary.values.assign(np.ones((plan.NumBoundaryQuadrature(), components)))
    gpu_boundary = resident.PrepareBoundary(boundary)
    metrics.BindHalo(mean).begin().wait()
    resident.BindHalo(gpu_mean, R.TransferTransport.DevicePacked, options).begin(context).wait()
    cpu_halo, gpu_halo = metrics.BindHalo(cpu.coefficients), resident.BindHalo(gpu.coefficients, R.TransferTransport.DevicePacked, options)
    engine = R.EagerEngine()
    stream = engine.cuda_stream(0)
    comm = engine.cuda_stream(0)
    for _ in range(3):
        cpu_halo.begin().wait()
        plan.DoReconstructionIter(mean, cpu, boundary)
        if backend == "nccl":
            comm.wait_for(stream.record())
            gpu_halo.enqueue(engine, comm)
            resident.DoReconstructionIterTo(engine, stream, gpu_mean, gpu.coefficients, gpu, gpu_boundary,
                                            CFV.ReconstructionRegion.Core, kernel)
            stream.wait_for(comm.record())
            resident.DoReconstructionIterTo(engine, stream, gpu_mean, gpu.coefficients, gpu, gpu_boundary,
                                            CFV.ReconstructionRegion.HaloDependent, kernel)
        else:
            gpu_halo.begin(context).wait()
            resident.DoReconstructionIterTo(engine, stream, gpu_mean, gpu.coefficients, gpu, gpu_boundary, kernel=kernel)
        resident.CommitReconstruction(engine, stream, gpu)
        engine.synchronize()
        resident.CheckStatus(gpu)
        np.testing.assert_allclose(gpu.coefficients.owned.numpy(), cpu.coefficients.owned.numpy(), rtol=1e-9, atol=1e-10)
    for order in (2, 1):
        cpu_halo.begin().wait()
        gpu_halo.begin(context).wait()
        plan.DoLimiterWBAP_3Order(cpu, order)
        resident.DoLimiterWBAP_3Order(gpu, order)
        np.testing.assert_allclose(gpu.coefficients.owned.numpy(), cpu.coefficients.owned.numpy(), rtol=1e-8, atol=1e-9)
    retained = gpu.coefficients.owned
    expected = retained.numpy()
    del gpu, resident, gpu_halo
    gc.collect()
    np.testing.assert_array_equal(retained.numpy(), expected)
