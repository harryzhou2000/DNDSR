"""Transient MPI, dynamic shapes and eager progress using native storage."""

import gc
import weakref
import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R


def channel():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    return R.CommunicationContext(mpi)


@pytest.mark.parametrize("cuda", [False, True])
def test_retained_binding_and_slice_follow_old_mesh_snapshot_after_reset(cuda):
    if cuda and not R.cuda_enabled:
        pytest.skip("this build has no CUDA backend")
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    peer = (rank + 1) % size
    device = R.Device.cuda(0) if cuda else R.Device.host()
    context = R.ExecutionContext(device)
    ctx = channel()
    rows, width = 43, 6
    source = R.Array(R.RowLayout.dense(rows, width, 9), np.dtype("float64"), device)
    old_values = np.arange(rows * width).reshape(rows, width) + rank * 10000
    source[:] = old_values
    destination = R.Array(R.RowLayout.dense(rows, width, 8), source.dtype, device)
    retained = destination[::3, ::-2]
    route = R.RowTransferPlan(ctx, rows, rows,
                             [R.RowRequest(peer, rows - 1 - i, i) for i in range(rows)])
    plan = R.PayloadPlan(route, source.layout, destination.layout, source.element_schema())
    transport = R.TransferTransport.HostPacked if cuda else R.TransferTransport.Direct
    binding = plan.bind(source, destination, transport=transport)
    source.reset(R.RowLayout.dense(7, width), device)
    destination.reset(R.RowLayout.dense(7, width), device)
    source[:] = -17
    destination[:] = -29
    with pytest.raises(RuntimeError, match="layout"):
        plan.bind(source, destination, transport=transport)
    # The retained binding is valid for its old allocations, not the replacements.
    binding.begin(context).wait()
    expected = (np.arange(rows * width).reshape(rows, width) + peer * 10000)[::-1]
    np.testing.assert_array_equal(retained.numpy(), expected[::3, ::-2])
    np.testing.assert_array_equal(source.numpy(), np.full((7, width), -17))
    np.testing.assert_array_equal(destination.numpy(), np.full((7, width), -29))
    del binding, plan
    gc.collect()
    np.testing.assert_array_equal(retained.numpy(), expected[::3, ::-2])


@pytest.mark.parametrize("mode", [R.TransferMode.Transient, R.TransferMode.Persistent])
@pytest.mark.parametrize("transport", [R.TransferTransport.Direct, R.TransferTransport.HostPacked])
def test_shared_context_known_route_and_tag_reuse(mode, transport):
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    rows = 193
    ctx = channel()
    send, receive = [[] for _ in range(size)], [[] for _ in range(size)]
    permutation = [(i * 17) % rows for i in range(rows)]
    send[(rank - 1) % size] = permutation
    receive[(rank + 1) % size] = list(range(rows))
    route = R.RowTransferPlan.known(ctx, rows, rows, send, receive)
    source = R.Array(R.RowLayout.dense(rows, 7, 8), np.dtype("int64"))
    source[:] = np.arange(rows * 7).reshape(rows, 7) + rank * 10000
    target = R.Array(R.RowLayout.dense(rows, 7, 9), source.dtype)
    plan = R.PayloadPlan(route, source.layout, target.layout, source.element_schema())
    options = R.TransferOptions(mode, ctx, 71)
    binding = plan.bind(source, target, transport=transport, options=options)
    with pytest.raises(RuntimeError, match="tag"):
        plan.bind(source, target, transport=transport, options=options)
    for _ in range(3):
        binding.begin().wait()
        expected = np.arange(rows * 7).reshape(rows, 7)[permutation] + ((rank + 1) % size) * 10000
        np.testing.assert_array_equal(target.numpy(), expected)
    del binding
    gc.collect()
    plan.bind(source, target, transport=transport, options=options).begin().wait()


def test_sender_ordering_returns_route_and_explicit_array_permutation():
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    rows, width = 193, 5
    ctx = channel()
    send, receive = [[] for _ in range(size)], [[] for _ in range(size)]
    send_order = [(i * 17) % rows for i in range(rows)]
    send[(rank - 1) % size] = send_order
    receive[(rank + 1) % size] = list(range(rows))
    route = R.RowTransferPlan.known(ctx, rows, rows, send, receive)
    ordered_route, old_to_new = route.reorder_source_by_peer()
    assert sorted(old_to_new) == list(range(rows))
    values = np.arange(rows * width).reshape(rows, width) + rank * 10000
    permuted = np.empty_like(values)
    permuted[old_to_new] = values
    source = R.Array(R.RowLayout.dense(rows, width, 7), np.dtype("int64"))
    source[:] = permuted
    target = R.Array(R.RowLayout.dense(rows, width, 8), source.dtype)
    plan = R.PayloadPlan(ordered_route, source.layout, target.layout, source.element_schema())
    plan.bind(source, target).begin().wait()
    expected = np.arange(rows * width).reshape(rows, width)[send_order] + ((rank + 1) % size) * 10000
    np.testing.assert_array_equal(target.numpy(), expected)
    incomplete = R.RowTransferPlan.known(ctx, rows + (rank == 0), rows, send, receive)
    with pytest.raises(RuntimeError, match="each source row exactly once"):
        incomplete.reorder_source_by_peer()


@pytest.mark.parametrize("cuda", [False, True])
def test_private_dynamic_csr_migration_and_retained_views(cuda):
    if cuda and not R.cuda_enabled:
        pytest.skip("this build has no CUDA backend")
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    device = R.Device.cuda(0) if cuda else R.Device.host()
    execution = R.ExecutionContext(device)
    ctx = channel()
    for epoch in range(3):
        rows = 67 + epoch * 2
        lengths = [(i + rank + epoch) % 9 for i in range(rows)]
        source = R.Array(R.RowLayout.padded(lengths, 11), np.dtype("int64"), device)
        for i, width in enumerate(lengths):
            source[i] = np.arange(width) + rank * 100000 + i * 100
        route = R.RowTransferPlan(ctx, rows, rows + 1,
                                 [R.RowRequest((rank + 1) % size, rows - i - 1, i) for i in range(rows)])
        operation = R.DynamicRowTransfer(source, route, source.element_schema(), device=device,
                                        transport=R.TransferTransport.HostPacked if cuda else R.TransferTransport.Direct,
                                        options=R.TransferOptions(R.TransferMode.Transient, ctx, 72), context=execution)
        with pytest.raises(RuntimeError):
            source.write()
        source.reset(R.RowLayout.dense(1, 1), device)
        result = operation.commit()
        with pytest.raises(RuntimeError, match="committed"):
            operation.commit()
        peer = (rank + 1) % size
        for i in range(rows):
            src = rows - i - 1
            width = (src + peer + epoch) % 9
            np.testing.assert_array_equal(result[i].numpy(), np.arange(width) + peer * 100000 + src * 100)
        assert result[rows].shape == (0,)
        retained = result[1]
        expected = retained.numpy()
        del result, operation
        gc.collect()
        np.testing.assert_array_equal(retained.numpy(), expected)


def test_python_reducer_lifetime_through_eager_mpi_progress():
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    ctx = channel()
    source = R.Array(R.RowLayout.dense(2, 7, 8), np.dtype("int64"))
    source[:] = rank + 1
    target = R.Array(R.RowLayout.dense(1, 7, 9), source.dtype)
    route = R.RowTransferPlan(ctx, 2, 1, [R.RowRequest(rank, 0, 0), R.RowRequest((rank + 1) % size, 1, 0)])
    reducer = R.Reducer.custom(source.element_schema(), source.dtype, "python.progress.sum", lambda a, b: a + b,
                              0, R.DestinationInit.Identity)
    binding = R.PayloadPlan(route, source.layout, target.layout, source.element_schema()).bind(
        source, target, reducer=reducer, options=R.TransferOptions(R.TransferMode.Transient, ctx, 73))
    progress = R.MPIProgress()
    operation = binding.begin_scheduled()
    done = progress.import_operation(operation)
    del operation, binding, reducer
    gc.collect()
    # Event.wait releases the GIL; reducer invocation and final destruction must reacquire it.
    done.wait()
    progress.drain()
    np.testing.assert_array_equal(target.numpy(), np.full((1, 7), rank + 1 + (rank + 1) % size + 1))


@pytest.mark.parametrize("mode", [R.TransferMode.Transient, R.TransferMode.Persistent])
def test_failed_transfer_stays_failed_after_progress_import(mode):
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    ctx = channel()
    source = R.Array(R.RowLayout.dense(47, 6, 8), np.dtype("float64"))
    target = R.Array(R.RowLayout.dense(47, 6, 9), source.dtype)
    requests = [R.RowRequest((rank + 1) % size, 46 - i, i) for i in range(47)]
    route = R.RowTransferPlan(ctx, 47, 47, requests)
    calls = []

    def fail(a, b):
        calls.append(1)
        raise ValueError("audit Python reducer failure")

    reducer = R.Reducer.custom(source.element_schema(), source.dtype, "audit.python-failure", fail,
                              0, R.DestinationInit.Identity)
    binding = R.PayloadPlan(route, source.layout, target.layout, source.element_schema()).bind(
        source, target, reducer=reducer, options=R.TransferOptions(mode, ctx, 74))
    operation = binding.begin_scheduled()
    for observe in (operation.wait_destination_ready, operation.wait, operation.test):
        with pytest.raises(ValueError, match="audit Python reducer failure"):
            observe()
    progress = R.MPIProgress()
    event = progress.import_operation(operation)
    del operation, binding, reducer
    for observe in (progress.drain, event.wait, event.ready):
        with pytest.raises(ValueError, match="audit Python reducer failure"):
            observe()
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="failed allocation"):
        target.numpy()


def test_failed_callback_does_not_retain_its_frame_and_transfer():
    def run():
        rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
        ctx = channel()
        source = R.Array(R.RowLayout.dense(31, 6, 8), np.dtype("float64"))
        target = R.Array(source.layout, source.dtype)
        route = R.RowTransferPlan(ctx, 31, 31,
                                 [R.RowRequest((rank + 1) % size, 30 - i, i) for i in range(31)])

        def fail(a, b):
            raise ValueError("callback lifetime audit", 17)

        retained = weakref.ref(fail)
        reducer = R.Reducer.custom(source.element_schema(), source.dtype, "audit.python-lifetime", fail, 0)
        binding = R.PayloadPlan(route, source.layout, target.layout, source.element_schema()).bind(
            source, target, reducer=reducer)
        operation = binding.begin()
        for _ in range(2):
            with pytest.raises(ValueError) as error:
                operation.wait()
            assert error.value.args == ("callback lifetime audit", 17)
        return retained

    callback = run()
    gc.collect()
    assert callback() is None
