"""Native API regressions; run only after building AND installing pybind modules."""

import gc
import os
import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R


def world():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    return mpi


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32, np.uint8,
                                  np.dtype([("id", "i4"), ("x", "f8")], align=True)])
def test_runtime_leases_and_snapshots(dtype):
    a = R.Array(R.RowLayout.dense(3, 2, 4), np.dtype(dtype))
    values = np.zeros((3, 2), dtype=dtype)
    if values.dtype.fields:
        values["id"] = 7
        values["x"] = 2.5
    else:
        values[:] = np.arange(6).reshape(3, 2)
    a.assign(values)
    np.testing.assert_array_equal(a.numpy(), values)
    alias = a.alias()
    assert alias.identity == a.identity
    with a.read() as read:
        with pytest.raises(RuntimeError):
            alias.write()
        snapshot = read.row(2)
        np.testing.assert_array_equal(snapshot, values[2])
        old_version = a.version
        a.reset(R.RowLayout.dense(1, 1))
        assert alias.version == old_version
        np.testing.assert_array_equal(read.row(2), values[2])
    with pytest.raises(RuntimeError):
        read.row(0)
    clone = alias.clone()
    assert clone.identity != alias.identity
    with alias.write() as write:
        write.assign_row(0, values[1])
    assert alias.version == old_version + 1
    np.testing.assert_array_equal(clone.numpy(), values)


def test_runtime_failed_write_and_pair_overlap():
    a = R.Array(R.RowLayout.dense(2, 1))
    b = R.Array(R.RowLayout.dense(1, 1))
    pair = R.ArrayPair(a, b)
    with b.write() as incoming:
        with a.read() as core:
            assert core[0, 0] == 0
            with pytest.raises(RuntimeError):
                pair.read()
        incoming[0, 0] = 19
    with pair.read() as ring:
        assert ring[2, 0] == 19
    with pytest.raises(ValueError):
        with a.write() as write:
            write[0, 0] = 1
            raise ValueError("partial operator failure")
    assert not a.valid
    with pytest.raises(RuntimeError):
        a.read()
    a.reset(a.layout)
    assert a.valid


@pytest.mark.parametrize("device", [R.Device.host(), R.Device.cuda(0)])
def test_runtime_direct_indexing_and_live_debug_views(device):
    if device.space == R.MemorySpace.CUDA and not R.cuda_enabled:
        pytest.skip("this Python build has no CUDA backend")
    a = R.Array(R.RowLayout.dense(4, 3, 5), np.dtype("float64"), device)
    a[:] = np.arange(12).reshape(4, 3)
    assert a[2, -1] == 8
    column = a[:, 1]
    column[1:3] = [-5, -6]
    assert a[1, 1] == -5
    a[0][2] = 30  # Row indexing returns a live view, not a detached NumPy row.
    assert a[0, 2] == 30
    reverse = a[::-1, ::-1]
    expected = a.numpy()[::-1, ::-1]
    np.testing.assert_array_equal(reverse.numpy(), expected)
    reverse[0, 0] = 99
    assert a[-1, -1] == 99
    snapshot = column.numpy()
    snapshot[:] = 123
    assert a[1, 1] == -5
    with a.read():
        with pytest.raises(RuntimeError):
            column[0] = 17
    old = a.alias()
    a.reset(R.RowLayout.dense(1, 1), device)
    column[0] = 13
    assert old[0, 1] == 13
    assert a[0, 0] == 0
    with old.write() as edit:
        edit[1, 0] = 18
        assert edit[1, 0] == 18


def test_runtime_ragged_debug_row_views():
    a = R.Array(R.RowLayout.ragged([2, 0, 3]), np.dtype("int32"))
    a[0] = [7, 8]
    row = a[-1]
    row[:] = [10, 11, 12]
    assert a[2, -1] == 12
    assert a[1].shape == (0,)
    with pytest.raises(IndexError):
        a[:, :]


@pytest.mark.mpi
@pytest.mark.parametrize("transport", [R.TransferTransport.Direct, R.TransferTransport.HostPacked])
@pytest.mark.parametrize("device", [R.Device.host(), R.Device.cuda(0)])
def test_runtime_ragged_mpi(transport, device):
    if device.space == R.MemorySpace.CUDA and not R.cuda_enabled:
        pytest.skip("this Python build has no CUDA backend")
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    context = R.ExecutionContext(device)
    a = R.Array(R.RowLayout.ragged([0, 2, 1]), np.dtype("int64"), device)
    with a.write() as write:
        write.assign_row(1, np.array([100 * rank, 100 * rank + 1], dtype="int64"))
        write[2, 0] = rank
    route = R.RowTransferPlan(world(), 3, 3, [R.RowRequest((rank + 1) % size, 1, 0),
                                             R.RowRequest(rank, 0, 1), R.RowRequest(rank, 2, 2)])
    b = R.Array(route.destination_layout(a.layout), a.dtype, device)
    payload = R.PayloadPlan(route, a.layout, b.layout, a.element_schema())
    binding = payload.bind(a, b, transport=transport)
    op = binding.begin(context)
    with a.read() as core:
        assert core[2, 0] == rank
        with pytest.raises(RuntimeError):
            b.read()
    op.wait_destination_ready()
    with b.read() as ready:
        assert ready[0, 1] == 100 * ((rank + 1) % size) + 1
    op.wait()
    binding.begin(context).wait()
    # The binding retains allocation identity through facade reset/destruction.
    old = b.alias()
    b.reset(R.RowLayout.dense(0, 0))
    del a
    gc.collect()
    binding.begin(context).wait()
    with old.read() as read:
        assert read[2, 0] == rank


@pytest.mark.mpi
def test_runtime_pod_typemap_and_custom_reducer():
    comm = MPI.COMM_WORLD
    dtype = np.dtype([("tag", "i4"), ("weight", "f8")], align=True)
    raw = MPI.Datatype.Create_struct([1, 1], [dtype.fields["tag"][1], dtype.fields["weight"][1]], [MPI.INT32_T, MPI.DOUBLE])
    native = raw.Create_resized(0, dtype.itemsize)
    native.Commit()
    schema = R.ElementSchema.from_mpi(native.py2f(), dtype.itemsize, "test.pod.v1")
    native.Free()
    raw.Free()
    a = R.Array(R.RowLayout.dense(1, 1), dtype)
    values = np.zeros((1, 1), dtype=dtype)
    values["tag"] = comm.rank + 1
    values["weight"] = 0.25
    a.assign(values)
    b = R.Array(R.RowLayout.dense(1, 1), dtype)
    route = R.RowTransferPlan(world(), 1, 1, [R.RowRequest(rank, 0, 0) for rank in range(comm.size)])
    payload = R.PayloadPlan(route, a.layout, b.layout, schema)
    if comm.size > 1:
        with pytest.raises(RuntimeError):
            payload.bind(a, b)

    def combine(dst, src):
        dst["tag"] += src["tag"]
        dst["weight"] += src["weight"]
        return dst

    reducer = R.Reducer.custom(schema, dtype, "test.pod.sum", combine, (0, 0), R.DestinationInit.Identity)
    payload.bind(a, b, reducer).begin().wait()
    result = b.numpy()
    assert result[0, 0]["tag"] == comm.size * (comm.size + 1) // 2
    assert result[0, 0]["weight"] == comm.size * 0.25


@pytest.mark.mpi
def test_runtime_builtin_reverse_and_reducer_failure():
    comm = MPI.COMM_WORLD
    a = R.Array(R.RowLayout.dense(1, 2), np.dtype("int32"))
    a.assign(np.array([[1, 3]], dtype="int32"))
    b = R.Array(R.RowLayout.dense(1, 2), a.dtype)
    route = R.RowTransferPlan(world(), 1, 1, [R.RowRequest(r, 0, 0) for r in range(comm.size)])
    payload = R.PayloadPlan(route, a.layout, b.layout, a.element_schema())
    for op, expected in [(R.Reduction.Sum, comm.size * 3), (R.Reduction.Min, 3),
                         (R.Reduction.Max, 3), (R.Reduction.BitOr, 3),
                         (R.Reduction.BitXor, 3 if comm.size % 2 else 0), (R.Reduction.BitAnd, 3)]:
        payload.bind(a, b, R.Reducer.builtin(a.dtype, op, R.DestinationInit.Identity)).begin().wait()
        assert b.numpy()[0, 1] == expected

    def fail(dst, src):
        raise ValueError("intentional reducer failure")

    reducer = R.Reducer.custom(a.element_schema(), a.dtype, "failing", fail, 0)
    operation = payload.bind(a, b, reducer).begin()
    with pytest.raises(ValueError, match="intentional"):
        operation.wait()
    operation.wait()
    assert not b.valid


@pytest.mark.mpi
@pytest.mark.parametrize("device", [R.Device.host(), R.Device.cuda(0)])
def test_runtime_parallel_checkpoint(device):
    if device.space == R.MemorySpace.CUDA and not R.cuda_enabled:
        pytest.skip("this Python build has no CUDA backend")
    comm = MPI.COMM_WORLD
    name = comm.bcast(f"/tmp/dnds_runtime_py_{os.getpid()}.h5" if comm.rank == 0 else None)
    a = R.Array(R.RowLayout.ragged([2, 0, 1] if comm.rank == 0 else []), np.dtype("float64"), device)
    if comm.rank == 0:
        with a.write() as write:
            write.assign_row(0, np.array([1.0, 2.0]))
            write[2, 0] = 5
    s = DNDS.Serializer.SerializerH5(world())
    s.OpenFile(name, False)
    a.write_serializer(s, "field", world())
    s.CloseFile()
    s.OpenFile(name, True)
    b = R.Array.read_serializer(s, "field", world(), device=device)
    assert b.layout.rows == a.layout.rows
    if comm.rank == 0:
        with b.read() as read:
            assert read[2, 0] == 5
    s.CloseFile()
    s.OpenFile(name, True)
    c = R.Array.read_serializer(s, "field", world(), partition=DNDS.Serializer.ArrayGlobalOffset(1, 2), device=device)
    with c.read() as read:
        assert read[0, 0] == 5
    s.CloseFile()
    comm.Barrier()
    if comm.rank == 0:
        os.remove(name)
