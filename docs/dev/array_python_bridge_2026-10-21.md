# Shared ArrayStorage and Python POD bridge validation

Worktree: `/home/harry/projects/DNDSR-array-geom`.
Branch: `codex/refactor-dnds-array-geom`.
This checkpoint follows `5df2a5e8114b20735f109b9b9ee98b972844468e`.
The main `/home/harry/projects/DNDSR` checkout remains unchanged.

## Corrected behavior

The initial runtime Python wrapper owned a byte Array with separate logical
metadata. It could not generically export a solver's existing typed Array.
Native `Array<T,L>` and the single Python Array class now share the exact
`ArrayStorage`, including logical metadata and CUDA CSR offsets. The generic
binding header compiles conversion glue at application signatures without
per-width Python Array classes or duplicated method registrations.

Slices, nested views, class element proxies and Python access objects retain
the original shared record across parent reset/destruction. Views hold no
permanent access lease. Class values and NumPy results are independent
snapshots. Weak-pointer tests verify record/layout retention and eventual
release after the last Python owner is destroyed. CUDA tests also check
shared device-offset identity and updates from a real C++ CUDA kernel.

Bound POD classes register one codec per type. Class-value reads/writes and
tracked field/method edits work on CPU and CUDA; optional structured NumPy
metadata is independent of class binding. Class schema identity also works
through direct MPI row transfers and parallel HDF5 checkpoints. The existing
serializer and legacy Array interfaces remain in use.

See [the ownership guide](../architecture/ArrayPythonOwnership.md) for class
and object diagrams of the previous wrapper, shared-storage design and POD
editing, along with API examples and lifetime contracts.

## Executed checks

GCC 13.3, Open MPI 5.0.9, Python 3.12, CUDA 13.2, architecture 80, one A100
80 GB PCIe. CPU and CUDA have separate build directories. All four production
Python modules were rebuilt and installed for each tested profile. The bridge
is a separately compiled test extension, imported alongside the corresponding
production modules. Every MPI rank used device 0 in CUDA tests.

| Check | Result | Evidence |
|---|---|---|
| CPU native Array, row transfers, I/O and Geom reorder | 13/13 CTest entries passed; MPI np=1,2,4,8 | [Log](array_python_bridge_validation_2026-10-21/cpu-ctest.txt) |
| CUDA native Array/kernel, row transfers, I/O and Geom reorder | 14/14 CTest entries passed; MPI np=1,2,4,8 | [Log](array_python_bridge_validation_2026-10-21/cuda-ctest.txt) |
| CPU runtime and bridge Python tests | 25 passed, 14 CUDA-only skips per rank, np=1,2,4 | [np1](array_python_bridge_validation_2026-10-21/cpu-python-np1.txt), [np2](array_python_bridge_validation_2026-10-21/cpu-python-np2.txt), [np4](array_python_bridge_validation_2026-10-21/cpu-python-np4.txt) |
| CUDA runtime and bridge Python tests | 39 passed per rank, zero skips, np=1,2,4 | [np1](array_python_bridge_validation_2026-10-21/cuda-python-np1.txt), [np2](array_python_bridge_validation_2026-10-21/cuda-python-np2.txt), [np4](array_python_bridge_validation_2026-10-21/cuda-python-np4.txt) |
| CPU Python DNDS/Geom suite including bridge | 43 passed, 21 skipped | [Log](array_python_bridge_validation_2026-10-21/cpu-python-regression.txt) |
| CUDA Python DNDS/Geom suite including bridge | 54 passed, 10 skipped | [Log](array_python_bridge_validation_2026-10-21/cuda-python-regression.txt) |

Python counts are per process, not summed over ranks. The compatibility suites
omit `--with-mpi`, so their MPI-marked cases skip; those cases execute in the
explicit rank matrix. CPU suites additionally skip CUDA-only cases. Logs have
terminal colors, trailing whitespace and MPI's NUL banner byte removed.

Before committing this checkpoint, both profiles were rebuilt, installed and
rerun with the same CTest/Python rank matrix and compatibility suites. All
counts above reproduced. The rerun selected physical GPU 1 using
`CUDA_VISIBLE_DEVICES=1` (logical device 0 for every CUDA rank); it did not
exercise multi-GPU communication. See the compact
[precommit record](array_python_bridge_validation_2026-10-21/precommit.txt).
The upcoming stream-ordered engine and async allocator require additional
tests; these results cover the current bridge and runtime only.

During validation, the new mixed C++/CUDA test extension initially failed to
import with `undefined symbol: fatbinData`. Disabling pybind11's optional host
LTO/strip extras for that target fixed the device-link issue. The Python test
now skips only an absent optional extension; a present but unloadable extension
fails collection. Final CUDA runs imported and executed every bridge test.

## Reproduction

Use the existing dependency paths recorded in the initial refactor report.
Run from this worktree, with the selected Open MPI on PATH. The two profiles
install into the same Python package; finish all Python testing for one
profile before installing the other. Do not reuse Python processes across
installations.

```sh
# Select build-refactor for CPU, build-refactor-cuda for CUDA.
build_dir=build-refactor-cuda
cmake -S . -B "$build_dir"
cmake --build "$build_dir" --target runtime_array_bridge_test \
  dnds_test_runtime_array dnds_test_runtime_row_transfer \
  dnds_test_runtime_array_io geom_test_mesh_reorder \
  dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j8
if [ "$build_dir" = build-refactor-cuda ]; then
  cmake --build "$build_dir" --target dnds_test_runtime_cuda -j8
fi
cmake --install "$build_dir" --component py

ctest --test-dir "$build_dir" --output-on-failure \
  -R '^(dnds_runtime_array$|dnds_runtime_cuda$|dnds_runtime_(row_transfer|array_io)|geom_mesh_reorder)'

export PYTHONPATH="$PWD/python:$PWD/$build_dir/test/cpp"
/home/harry/projects/DNDSR/venv/bin/python -c 'import runtime_array_bridge_test'
for ranks in 1 2 4; do
  mpirun -np "$ranks" /home/harry/projects/DNDSR/venv/bin/python -m pytest \
    test/DNDS/test_runtime_array.py test/DNDS/test_runtime_array_bridge.py \
    --with-mpi -q || exit
done
/home/harry/projects/DNDSR/venv/bin/python -m pytest test/DNDS test/Geom -q
```

## Limits

This is shared ownership and checked operation access, not general tensor
slicing, automatic differentiation or writable NumPy aliasing. Bound POD reads
are class-value snapshots; use assignment or `element()` for writeback.
Snapshots and debug GPU access copy data; only Array handle conversion shares
storage without copying. A converted native argument is a shallow handle;
explicit owner setters are required to replace a solver field.

The native layout supports uniform-width tables, optional row stride and CSR.
It does not implement the legacy variable-length/uniform-capacity padded
layout. FixedRows validates width; it does not add a fixed-width kernel view.
This correction changes no legacy Array interfaces or Euler algorithms.
No multi-node/multi-GPU throughput, hot-path timing or general leak-detector
result is claimed. The complete prior solver compatibility matrix is in the
original phase report; this follow-up reruns the affected native/Geom paths
and Python DNDS/Geom suites.
