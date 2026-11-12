"""Isolated fatal Python reducer; launched only by the bounded subprocess test."""

import sys
import numpy as np
from mpi4py import MPI
from DNDSR import DNDS
from DNDSR.DNDS import Runtime as R


def main():
    assert MPI.COMM_WORLD.size == 1
    mpi = DNDS.MPIInfo()
    mpi.setWorld()
    context = R.CommunicationContext(mpi)
    source = R.Array(R.RowLayout.dense(47, 6, 8), np.dtype("float64"))
    source.assign(np.ones((47, 6)))
    target = R.Array(R.RowLayout.dense(47, 6, 9), source.dtype)
    route = R.RowTransferPlan(context, 47, 47,
                              [R.RowRequest(0, 46 - i, i) for i in range(47)])

    def fail(a, b):
        print("injected Python reducer reached", file=sys.stderr, flush=True)
        raise ValueError("audit Python reducer failure")

    reducer = R.Reducer.custom(source.element_schema(), source.dtype, "audit.python-failure", fail,
                              0, R.DestinationInit.Identity)
    persistent = sys.argv[1] == "persistent"
    mode = R.TransferMode.Persistent if persistent else R.TransferMode.Transient
    binding = R.PayloadPlan(route, source.layout, target.layout, source.element_schema()).bind(
        source, target, reducer=reducer, options=R.TransferOptions(mode, context, 74))
    if persistent:
        progress = R.MPIProgress()
        operation = binding.begin_scheduled()
        event = progress.import_operation(operation)
        event.wait()
    else:
        binding.begin().wait()


if __name__ == "__main__":
    main()
