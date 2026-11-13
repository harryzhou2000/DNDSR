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


def test_allocation_names_share_storage_without_registry_retention():
    R.enable_allocation_diagnostics(True)
    try:
        value = R.Array(R.RowLayout.dense(17, 5))
        value.name = "test/mean"
        alias = value.alias()
        assert alias.name == "test/mean"
        alias.name = "test/alias"
        assert value.name == "test/alias"
        records = [a for a in R.allocation_snapshot().allocations
                   if "test/mean" in a.names]
        assert len(records) == 1
        assert records[0].bytes == 17 * 5 * 8
        identity = records[0].identity
        del value
        assert any(a.identity == identity for a in R.allocation_snapshot().allocations)
        del alias
        gc.collect()
        assert not any(a.identity == identity for a in R.allocation_snapshot().allocations)
    finally:
        R.enable_allocation_diagnostics(False)


def test_collective_checkpoint_rejects_one_rank_shape_edit_and_recovers():
    comm = MPI.COMM_WORLD
    mpi = world()
    values = R.Array(R.RowLayout.dense(3, 2))
    expected = np.arange(6, dtype=np.float64).reshape(3, 2) + 100 * comm.rank
    values.assign(expected)
    edit = values.edit_rows(exclusive=True) if comm.rank == 0 else None
    name = comm.bcast(f"/tmp/dnds_unpublished_io_{os.getpid()}.h5" if comm.rank == 0 else None)
    serializer = DNDS.Serializer.SerializerH5(mpi)
    serializer.OpenFile(name, False)
    with pytest.raises(RuntimeError, match="live published source"):
        values.write_serializer(serializer, "rejected", mpi)
    assert "rejected" not in serializer.ListCurrentPath()
    if edit is not None:
        edit.commit()
    values.write_serializer(serializer, "accepted", mpi)
    serializer.CloseFile()
    serializer.OpenFile(name, True)
    with pytest.raises(RuntimeError, match="valid registered dtype"):
        R.Array.read_serializer(serializer, "accepted", mpi,
                                dtype=np.dtype(object) if comm.rank == 0 else np.dtype(np.float64))
    restored = R.Array.read_serializer(serializer, "accepted", mpi)
    serializer.CloseFile()
    np.testing.assert_array_equal(restored.numpy(), expected)
    comm.Barrier()
    if comm.rank == 0:
        os.remove(name)


@pytest.mark.parametrize("source_type,bound_type", [
    (np.float64, np.int64),
    (np.dtype([("id", "i4"), ("flag", "i4")]), np.dtype([("x", "f8")])),
])
def test_payload_rejects_same_size_different_element_schema(source_type, bound_type):
    layout = R.RowLayout.dense(31, 7, 9)
    prototype = R.Array(layout, np.dtype(source_type))
    source = R.Array(layout, np.dtype(bound_type))
    target = R.Array(layout, np.dtype(bound_type))
    route = R.RowTransferPlan(world(), 31, 31, [R.RowRequest(MPI.COMM_WORLD.rank, 17, 3)])
    schema = prototype.element_schema()
    payload = R.PayloadPlan(route, layout, layout, schema)
    with pytest.raises(RuntimeError):
        payload.bind(source, target)
    with pytest.raises(RuntimeError):
        R.DynamicRowTransfer(source, route, schema)
    with pytest.raises(RuntimeError):
        R.DynamicRowTransfer(source if MPI.COMM_WORLD.rank == 0 else prototype, route, schema)
    with pytest.raises(RuntimeError):
        R.Reducer.custom(schema, np.dtype(bound_type), "bad.dtype", lambda a, b: a, 0)
    # A correctly typed binding remains usable after collective rejection.
    payload.bind(prototype, R.Array(layout, np.dtype(source_type))).begin().wait()


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
    epoch = alias.mutation_epoch
    with alias.write() as write:
        assert alias.mutation_epoch == epoch + 1
        assert alias.version == old_version
        write.assign_row(0, values[1])
    assert alias.version == old_version + 1
    assert alias.mutation_epoch == epoch + 1
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
def test_runtime_builtin_reverse_reductions():
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

    # Post-MPI callback failure is fatal, unlike setup validation. Its Python
    # adapter is exercised in an isolated child by test_runtime_transient.


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
    if device.space == R.MemorySpace.CUDA:
        with pytest.raises(RuntimeError, match="staging|device"):
            a.write_serializer(s, "field", world())
    a.to(R.Device.host()).write_serializer(s, "field", world())
    s.CloseFile()
    s.OpenFile(name, True)
    if device.space == R.MemorySpace.CUDA:
        with pytest.raises(RuntimeError, match="device|staging"):
            R.Array.read_serializer(s, "field", world(), device=device)
    b = R.Array.read_serializer(s, "field", world()).to(device)
    assert b.layout.row_count == a.layout.row_count
    assert b.layout.rows == b.layout.row_count
    if comm.rank == 0:
        with b.read() as read:
            assert read[2, 0] == 5
    s.CloseFile()
    s.OpenFile(name, True)
    c = R.Array.read_serializer(s, "field", world(), partition=DNDS.Serializer.ArrayGlobalOffset(1, 2)).to(device)
    with c.read() as read:
        assert read[0, 0] == 5
    s.CloseFile()
    comm.Barrier()
    if comm.rank == 0:
        os.remove(name)

@pytest.mark.parametrize("layout", ["col", "tiled"])
def test_physical_layout_debug_views_and_element_updates(layout):
    shape = R.RowLayout.col_major(33, 7, 32, 1) if layout == "col" else R.RowLayout.tiled_col_major(33, 7, 8)
    a = R.Array(shape, np.dtype("float64"))
    expected = np.arange(231, dtype=np.float64).reshape(33, 7)
    a.assign(expected)
    np.testing.assert_array_equal(a.numpy(), expected)
    with a.write() as access:
        access.assign_row(32, np.arange(7, dtype=np.float64) + 900)
        access[8, 3] = -21
    expected[32] = np.arange(7) + 900
    expected[8, 3] = -21
    np.testing.assert_array_equal(a[::3, ::-2].numpy(), expected[::3, ::-2])
    with a.read() as access:
        np.testing.assert_array_equal(access.row(32), expected[32])
