"""Load the freshly installed DNDS component, without importing other modules."""

import gc
import numpy as np
from mpi4py import MPI
import dnds_pybind11 as D


def run():
    mpi = D.MPIInfo()
    mpi.setWorld()
    R = D.Runtime
    default = R.CommunicationContext(mpi)
    assert default.collective_preflight is True
    context = R.CommunicationContext(mpi, collective_preflight=False)
    assert context.collective_preflight is False
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    peer = (rank + 1) % size
    source = R.Array(R.RowLayout.dense(4, 2, 4), np.dtype("float64"), R.Device.host())
    destination = R.Array(R.RowLayout.dense(3, 2, 3), source.dtype, R.Device.host())
    with source.write() as write:
        for i in range(4):
            for j in range(2):
                write[i, j] = float(i * 2 + j + rank * 100)
    order = [3, 1, 0]
    route = R.RowTransferPlan(context, 4, 3, [R.RowRequest(peer, row, i) for i, row in enumerate(order)])
    payload = R.PayloadPlan(route, source.layout, destination.layout, source.element_schema())
    binding = payload.bind(source, destination)
    binding.begin().wait()
    np.testing.assert_array_equal(destination.numpy(), (np.arange(8).reshape(4, 2) + peer * 100)[order])
    print(f"rank={rank} Python3.12 preflight=False padded-stride transfer passed", flush=True)


error, _ = D.MPI.Init_thread([])
assert error == 0
run()
gc.collect()
D.MPI.Finalize()
