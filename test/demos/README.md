# Runtime Array integration mock

`test_runtime_solver.py` drives native C++/CUDA stencils through the same eager
engine interface. It exercises owned/ghost separation, direct MPI datatypes,
global norms, dynamic CSR migration with permutation, explicit rebinding, and
parallel HDF5 restart. The reference is an independently evolved global NumPy
field. This is an integration correctness test, not a fluid model or benchmark.

Build/install the four production bindings and `runtime_array_bridge_test` first.
Set `PYTHONPATH` to `python:<build>/test/cpp`. Run with pytest normally for CPU.
For CUDA MPI on the local two-GPU machine:

```sh
OMP_NUM_THREADS=2 PYTHONPATH="$PWD/python:$PWD/build-refactor-cuda/test/cpp" \
  mpirun -np 2 venv-interop/bin/python test/demos/one_rank_per_gpu.py \
  venv-interop/bin/python -m pytest test/demos/test_runtime_solver.py --with-mpi -q
```

The wrapper assigns distinct GPU UUIDs before either CUDA or MPI is imported.
The test checks those assignments and CUDA-aware MPI capability. One rank per
GPU is required. The engine still uses logical device 0 within each process.
GPU runs on a CPU-only build skip explicitly; verify `Runtime.cuda_enabled` before
accepting GPU coverage. The same wrapper can run native MPI executables.

There is no DNDS host packing in the default Direct path. MPI may internally
pack or stage noncontiguous device datatypes. This test does not measure GPUDirect
or speedup. Request mode (persistent/transient) is independent of transport.
Ragged tests reuse a persistent binding over three exchanges; the evolving mock
rebinds after buffer swaps. Collision tests require an explicit reduction.

The mock's core operand cannot see ghost storage. Its MPI request and core
submission coexist; actual time overlap is not inferred from a short wall-clock
run. Deterministically gated overlap tests in `test_EagerEngine.cpp`,
`test_RuntimeTransient.cpp`, and `test_CUDAExecution.cu` separately verify actual
concurrent progress and missing-edge behavior without inducing a hardware race.

Optional Torch tests live in `test/DNDS/test_runtime_dlpack.py`. Install only into
a separate checkout-local venv; no Torch headers/libraries are linked by DNDS.
Set `DNDS_REQUIRE_TORCH=1` to make absence a failure. The tested requirements and
disk measurements are in the completion validation record.
