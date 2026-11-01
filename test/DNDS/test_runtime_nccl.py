"""Optional NCCL provider; rebuild and install bindings before running.

Run distributed GPU cases with one local rank per GPU. The eager caller supplies
all producer/transfer/consumer edges; enqueue does not infer dependencies.
"""

import gc
import numpy as np
import pytest
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R


def mpi_info():
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    return mpi


def test_nccl_capability_is_explicit():
    if not R.NCCLTransferProvider.available():
        with pytest.raises(RuntimeError, match="NCCL.*unavailable"):
            R.NCCLTransferProvider(mpi_info(), R.Device.cuda(0))


@pytest.mark.mpi
@pytest.mark.parametrize("packed", [False, True])
def test_nccl_queued_copies_and_retained_shared_arrays(packed):
    if not R.cuda_enabled or not R.NCCLTransferProvider.available():
        pytest.skip("build with CUDA and NCCL")
    mpi = mpi_info()
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    rows, width = 37, 5
    device = R.Device.cuda(0)
    shape = R.RowLayout.col_major(rows, width, 32, 1) if packed else R.RowLayout.dense(rows, width)
    source = R.Array(shape, np.dtype("float64"), device)
    produced = R.Array(shape, source.dtype, device)
    destination = R.Array(R.RowLayout.tiled_col_major(rows, width, 8) if packed else shape, source.dtype, device)
    checkpoint = R.Array(destination.layout, source.dtype, device)
    values = np.arange(rows * width, dtype=np.float64).reshape(rows, width) + rank * 10000
    source.assign(values)
    mapping = list(reversed(range(rows))) if packed else list(range(rows))
    route = R.RowTransferPlan(R.CommunicationContext(mpi), rows, rows,
                             [R.RowRequest((rank + 1) % size, mapping[i], i) for i in range(rows)])
    payload = R.PayloadPlan(route, produced.layout, destination.layout, source.element_schema())
    provider = R.NCCLTransferProvider(mpi, device)
    options = R.TransferOptions(provider=provider)
    binding = payload.bind(produced, destination,
                           transport=R.TransferTransport.DevicePacked if packed else R.TransferTransport.Direct,
                           options=options)
    engine = R.EagerEngine()
    producer, comm, consumer = (engine.cuda_stream(0) for _ in range(3))
    engine.copy(producer, source, produced)
    comm.wait_for(producer.record())
    binding.enqueue(engine, comm)
    consumer.wait_for(comm.record())
    engine.copy(consumer, destination, checkpoint)
    retained = checkpoint[::3, ::-2]
    del binding, payload, route, source, produced, destination, checkpoint
    gc.collect()
    engine.synchronize()
    provider.check_async_error()
    expected = (np.arange(rows * width, dtype=np.float64).reshape(rows, width) + ((rank + 1) % size) * 10000)[mapping]
    np.testing.assert_array_equal(retained.numpy(), expected[::3, ::-2])
