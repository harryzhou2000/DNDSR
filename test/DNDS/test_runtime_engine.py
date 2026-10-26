"""Native eager orchestration; rebuild/install all bindings and the bridge first."""

import gc
import importlib.util
import os
import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R

if importlib.util.find_spec("runtime_array_bridge_test") is None:
    pytest.skip("build runtime_array_bridge_test and add build/test/cpp to PYTHONPATH", allow_module_level=True)
import runtime_array_bridge_test as bridge


@pytest.fixture(params=[("cpu", 0), ("cpu", 2), ("cuda", 2)])
def execution(request):
    kind, workers = request.param
    if kind == "cuda" and not R.cuda_enabled:
        pytest.skip("this build has no CUDA backend")
    engine = R.EagerEngine(workers)
    create = engine.cuda_stream if kind == "cuda" else engine.cpu_stream
    device = R.Device.cuda(0) if kind == "cuda" else R.Device.host()
    return engine, create(), create(), device


def test_checkpoint_requires_finished_stage_then_restores_shared_debug_snapshot(execution):
    engine, first, last, device = execution
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    comm = MPI.COMM_WORLD
    filename = comm.bcast(f"/tmp/dnds_stage_checkpoint_{os.getpid()}.h5" if comm.rank == 0 else None)
    values = R.Array(R.RowLayout.dense(37 + comm.rank, 5, 8), np.dtype("float64"), device)
    expected = np.arange(values.shape[0] * 5).reshape(values.shape) + comm.rank * 1000
    values[:] = expected
    retained = values[::2, ::-2]
    serializer = DNDS.Serializer.SerializerH5(mpi)
    serializer.OpenFile(filename, False)
    gate = bridge.hold(engine, first)
    try:
        bridge.scale_async(engine, first, values, 2)
        last.wait_for(first.record())
        bridge.scale_async(engine, last, values, 3)
        with pytest.raises(RuntimeError):
            retained.numpy()
        with pytest.raises(RuntimeError):
            values.edit_rows(exclusive=True)
        with pytest.raises(RuntimeError, match="lease|snapshot|staging|device"):
            values.write_serializer(serializer, "state", mpi)
    finally:
        gate.open()
    last.record().wait()
    engine.synchronize()
    if device.space == R.MemorySpace.CUDA:
        with pytest.raises(RuntimeError, match="staging|device"):
            values.write_serializer(serializer, "state", mpi)
    values.to(R.Device.host()).write_serializer(serializer, "state", mpi)
    serializer.CloseFile()
    values.reset(R.RowLayout.dense(1, 5), device)
    serializer.OpenFile(filename, True)
    restored = R.Array.read_serializer(serializer, "state", mpi).to(device)
    serializer.CloseFile()
    assert restored.layout.kind == R.RowLayoutKind.Table
    np.testing.assert_array_equal(restored[:].numpy(), expected * 6)
    np.testing.assert_array_equal(retained.numpy(), (expected * 6)[::2, ::-2])
    np.testing.assert_array_equal(values.numpy(), np.zeros((1, 5)))
    comm.Barrier()
    if comm.rank == 0:
        os.remove(filename)


def test_python_orchestrates_native_operators_and_retains_debug_views(execution):
    engine, first, last, device = execution
    source = R.Array(R.RowLayout.dense(257, 5, 8), np.dtype("float64"), device)
    expected = np.arange(257 * 5, dtype="float64").reshape(257, 5)
    source[:] = expected
    source_view = source[::3, ::-2]
    target = engine.allocate_like(first, source)
    result = engine.allocate_like(last, source)
    alias = target.alias()
    retained = target[::7, ::-2]
    assert bridge.shares_layout(source, target)
    gate = bridge.hold(engine, first)
    try:
        assert engine.copy(first, source, target) is None
        assert bridge.scale_async(engine, first, target, 2) is None
        last.wait_for(first.record())
        bridge.scale_async(engine, last, target, 0.5)
        engine.copy(last, target, result)
        engine.release_on(target, last)
        assert target.shape == (0, 0)
        with pytest.raises(RuntimeError, match="pending"):
            alias[0, 0]
        tail = last.record()
        assert not tail.ready()
    finally:
        gate.open()
    tail.wait()
    engine.synchronize()
    np.testing.assert_array_equal(result.numpy(), expected)
    np.testing.assert_array_equal(alias.numpy(), expected)
    engine.release_on(alias, last)
    engine.release_on(source, last)
    engine.synchronize()
    assert source.shape == (0, 0)
    np.testing.assert_array_equal(retained.numpy(), expected[::7, ::-2])
    np.testing.assert_array_equal(source_view.numpy(), expected[::3, ::-2])


@pytest.mark.parametrize("ragged", [False, True])
def test_async_pod_values_share_prepared_variable_metadata(execution, ragged):
    engine, stream, _, device = execution
    lengths = [i % 8 for i in range(193)]
    layout = R.RowLayout.ragged(lengths) if ragged else R.RowLayout.padded(lengths, 11)
    source = R.Array(layout, bridge.ClassMetadata, device)
    source[191, 6] = bridge.ClassMetadata(991, 1.25)
    result = engine.allocate_like(stream, source)
    assert result.element_type is bridge.ClassMetadata
    assert bridge.shares_layout(source, result)
    engine.copy(stream, source, result)
    stream.record().wait()
    assert result[191, 6].tag == 991
    assert result[191, 6].weight == 1.25
    assert result[0].shape == (0,)
    view = result[191]
    engine.release_on(result, stream)
    engine.synchronize()
    assert result.shape == (0, 0)
    assert view[6].tag == 991


def test_dynamic_table_allocation_and_rejected_release_preserve_ownership(execution):
    engine, stream, _, device = execution
    gate = bridge.hold(engine, stream)
    try:
        array = engine.allocate(stream, R.RowLayout.dense(131, 7, 9), np.dtype("int64"))
        if device.space == R.MemorySpace.CUDA:
            with pytest.raises(RuntimeError, match="pending"):
                array[0, 0]
            with pytest.raises(RuntimeError, match="metadata"):
                engine.allocate(stream, R.RowLayout.ragged([0, 3]), np.dtype("int64"))
        fence = stream.record()
    finally:
        gate.open()
    fence.wait()
    assert array[130, 6] == 0
    array[130, 6] = 73
    identity = array.identity
    other = R.EagerEngine()
    with pytest.raises(RuntimeError, match="foreign"):
        other.release_on(array, stream)
    assert array.identity == identity and array.shape == (131, 7)
    assert array[130, 6] == 73
    incompatible = R.Array(array.layout, np.dtype("float64"), device)
    with pytest.raises(RuntimeError, match="dtype"):
        engine.copy(stream, array, incompatible)
    engine.synchronize()


@pytest.mark.parametrize("kind, workers", [("cpu", 0), ("cpu", 2), ("cuda", 2)])
def test_recorded_completion_can_outlive_python_engine(kind, workers):
    if kind == "cuda" and not R.cuda_enabled:
        pytest.skip("this build has no CUDA backend")
    engine = R.EagerEngine(workers)
    stream = engine.cuda_stream() if kind == "cuda" else engine.cpu_stream()
    values = engine.allocate(stream, R.RowLayout.dense(17, 5, 8))
    event = stream.record()
    event.wait()
    engine.synchronize()
    del stream, engine
    gc.collect()
    assert event.ready()
    event.wait()
    assert values[16, 4] == 0
