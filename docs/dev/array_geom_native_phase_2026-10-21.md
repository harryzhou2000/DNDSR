# Native Array phase: implementation and validation

Worktree: `/home/harry/projects/DNDSR-array-geom`.
Branch: `codex/refactor-dnds-array-geom`.
Base: `03891f8b5869017de355468d93294e3eb25f0feb` (`dev/harry`).
The main checkout remains separate. This report records implementation and
validation on the refactor branch; its source revision is tracked in Git.

The [architecture and usage guide](../architecture/ArrayRuntime.md) is the
authoritative interface description. The
[initial foundation report](array_geom_refactor_2026-10-20.md) records dependency
paths, independent build configuration and fixture isolation. Its original
prototype limits are superseded by the native interfaces described here.

The original phase was committed as `5df2a5e8114b20735f109b9b9ee98b972844468e`.
The subsequent [C++/Python sharing correction](array_python_bridge_2026-10-21.md)
adds the generic typed bridge, common ArrayStorage and bound-POD class access.
Its separate validation record covers that follow-up source; the tables below
record the original phase and its precommit checks.

## Implemented scope

- Native uniform/strided tables, fixed-width constraints and CSR `Runtime::Array`, retained layouts, explicit
  host/CUDA placement, typed element access and shallow handles/explicit clones.
- Whole-allocation RAII access, separate owned/ghost controllers, value
  publication on completion, failed-write invalidation, and unchecked kernel
  views with an optional checked development policy.
- Shared RowTransferPlan, retained ElementSchema, compiled PayloadPlan,
  persistent TransferBinding and operation-owned requests/leases. Direct MPI
  datatype transfer is the default. HostPacked is an explicit fallback.
- POD datatype extent handling, adjacent-segment coalescing, ragged width
  exchange, collision rejection, receiver-side fan-in, CPU/CUDA arithmetic and
  integer bitwise reducers, and custom reducer callbacks.
- Matching-rank and direct-partition checkpoint reads over the existing
  serializer, including empty ranks, changed MPI size, POD schemas and CUDA
  staging. C++ and Python share the native archive codec.
- Python direct indexing, synchronized GPU indexing, limited live row/column
  views, NumPy snapshots, scoped leases, transfers, reductions, serializers and
  ArrayPair. Views retain the allocation through handle reset.
- Native Geom Array registration using the existing exact permutation
  correspondence, including ragged rows. Existing Euler and legacy Array
  interfaces remain available. The prototype Exchange facade delegates to the
  new implementation.

## Validation record

CPU and CUDA builds use GCC 13.3.0, Open MPI 5.0.9, Python 3.12 and the existing
read-only external installation. CUDA uses toolkit 13.2, architecture 80 and
NVIDIA A100 80 GB PCIe. MPI cases use np=1,2,4,8 on this host. Device MPI cases
share device 0; they do not establish communication across eight GPUs or nodes.
Both profiles freshly built and installed all four Python modules before tests.

| Check | Result | Evidence |
|---|---|---|
| CPU DNDS and Geom, including native Array/MPI/I/O and registered field migration | 83/83 CTest entries passed | [Log](array_geom_validation_2026-10-21/cpu-dnds-geom.txt) |
| CPU CFV, Euler and Solver compatibility | 30/30 CTest entries passed | [Log](array_geom_validation_2026-10-21/cpu-cfv-euler-solver.txt) |
| Native CUDA kernels, MPI and checkpoints | 9/9 CTest entries passed; actual CUDA-aware direct MPI ran | [Summary](array_geom_validation_2026-10-21/cuda-native.txt), [case details](array_geom_validation_2026-10-21/cuda-native-details.txt) |
| Existing CFV CUDA device transfer | 1/1 CTest entry passed | [Log](array_geom_validation_2026-10-21/cuda-cfv.txt) |
| Native CPU Python, np=1,2,4,8 | 14 passed and 4 CUDA-only skips per rank | [np1](array_geom_validation_2026-10-21/cpu-python-np1.txt), [np2](array_geom_validation_2026-10-21/cpu-python-np2.txt), [np4](array_geom_validation_2026-10-21/cpu-python-np4.txt), [np8](array_geom_validation_2026-10-21/cpu-python-np8.txt) |
| Native CUDA Python, np=1,2,4,8 | 18 passed per rank; zero skips | [np1](array_geom_validation_2026-10-21/cuda-python-np1.txt), [np2](array_geom_validation_2026-10-21/cuda-python-np2.txt), [np4](array_geom_validation_2026-10-21/cuda-python-np4.txt), [np8](array_geom_validation_2026-10-21/cuda-python-np8.txt) |
| Python DNDS/Geom compatibility, CPU | 33 passed, 10 skipped | [Log](array_geom_validation_2026-10-21/cpu-python-regression.txt) |
| Python DNDS/Geom compatibility, CUDA | 35 passed, 8 skipped | [Log](array_geom_validation_2026-10-21/cuda-python-regression.txt) |
| Plan provenance | 23 topics and traceability rows preserved; pinned source and all 29 referenced hashes match | [Log](array_geom_validation_2026-10-21/plan-validation.txt) |

The Python compatibility runs omit `--with-mpi`, so eight MPI-marked native
cases skip there; the explicit multi-rank runs above cover them. The CPU run
also skips native CUDA indexing and the legacy CUDA ArrayDOF case. Counts are
pytest cases per process, not an inflated sum over ranks. Logs remove terminal
color escapes, trailing whitespace and NUL bytes from MPI version banners;
results are unchanged.

| Required behavior | Executed coverage |
|---|---|
| Shared aliases and RAII publication, move/lifetime/failure control | test_Runtime.cpp, test_RuntimeArray.cpp, test_Runtime_CUDA.cu; Python lease/failure/view cases |
| No per-element lease checks; owned reads while ghosts receive | Plain unchecked ArrayView implementation; split ArrayPair and active MPI lease tests. No timing claim. |
| Arbitrary row shapes, typed POD extent, direct default, explicit packing | test_RuntimeRowTransfer.cpp plus Python ragged and POD tests; CUDA direct and host-staged transfers |
| Collisions, reverse reductions, extensible callbacks | Copy rejection; CPU/CUDA arithmetic and integer bitwise cases; POD custom combine; failed reducer draining and invalidation |
| Matching-rank, direct-partition and changed communicator reads | test_RuntimeArrayIO.cpp; Python CPU/CUDA parallel checkpoint cases; empty ranks and POD schema rejection |
| CPU and synchronized GPU debugging, live slices | Python direct index/view cases on both devices, negative/strided selections, alias retention after reset, active-lease conflicts |
| Gradual legacy/Geom integration | Common host allocation bridge; unchanged legacy API regressions; distributed native ragged-field registration in test_MeshReorder.cpp |

The worktree's final installed Python profile is CUDA. The architecture guide's
CPU/GPU indexing example was executed against that installation successfully.
The main checkout remains clean on dev/harry at the recorded base; the two
generated CGNS fixture hashes still match the initial report. Validation was
completed before source publication. The standalone plan repository has a
separate Git history; its pending documentation changes are not part of this
source commit.

Before committing, the repository hook formatted eight C++ files. A comparison
confirmed that only whitespace changed. Fresh targeted rebuilds passed four
CPU entries (runtime, row transfer, I/O and Geom reorder), two CUDA MPI entries,
and all 18 native Python cases on each of two ranks after rebuilding/installing
all four Python modules. See the [CPU](array_geom_validation_2026-10-21/precommit-cpu.txt),
[CUDA](array_geom_validation_2026-10-21/precommit-cuda.txt) and
[Python](array_geom_validation_2026-10-21/precommit-python.txt) logs.

## Reproduction

Use the dependency setup from the initial foundation report. CPU and CUDA
builds have independent build directories and link the existing external
installation read-only. Both install into this worktree's Python package;
install only the intended build before importing Python. Do not mix installed
CPU and CUDA modules or reuse imports across installations.

```sh
cmake --build build-refactor --target dnds_unit_tests geom_unit_tests \
  cfv_unit_tests euler_unit_tests solver_unit_tests -j8
ctest --test-dir build-refactor --output-on-failure \
  -R '^(dnds_|geom_|cfv_|euler_|solver_)'

cmake --build build-refactor --target dnds_pybind11 geom_pybind11 \
  cfv_pybind11 eulerP_pybind11 -j8
cmake --install build-refactor --component py
PYTHONPATH="$PWD/python" /home/harry/projects/DNDSR/venv/bin/python \
  -m pytest test/DNDS/test_runtime_array.py -q
PYTHONPATH="$PWD/python" mpirun -np 4 \
  /home/harry/projects/DNDSR/venv/bin/python -m pytest test/DNDS/test_runtime_array.py --with-mpi -q

cmake --build build-refactor-cuda --target dnds_test_runtime_cuda \
  dnds_test_runtime_row_transfer dnds_test_runtime_array_io -j8
ctest --test-dir build-refactor-cuda --output-on-failure \
  -R '^dnds_runtime_(cuda|row_transfer|array_io)'
cmake --build build-refactor-cuda --target dnds_pybind11 geom_pybind11 \
  cfv_pybind11 eulerP_pybind11 -j8
cmake --install build-refactor-cuda --component py
PYTHONPATH="$PWD/python" /home/harry/projects/DNDSR/venv/bin/python \
  -m pytest test/DNDS/test_runtime_array.py -q

# Run the complete native matrix after installing the selected profile.
for ranks in 1 2 4 8; do
  PYTHONPATH="$PWD/python" mpirun -np "$ranks" \
    /home/harry/projects/DNDSR/venv/bin/python -m pytest \
    test/DNDS/test_runtime_array.py --with-mpi -q || exit
done

cmake --build build-refactor-cuda --target cfv_test_device_transferable -j8
ctest --test-dir build-refactor-cuda -R '^cfv_device_transferable' --output-on-failure
```

## Scope boundaries

This completes an Array foundation phase, not the complete solver/Geom roadmap.
Legacy raw pointer writes and all legacy mesh mutators are not automatically
tracked. Solver-specific stencil proof/binding, comprehensive Euler/CFV field
registration, atomic multi-field mesh publication, tree meshes, overset search,
AMG, DLPack/autograd and accelerator backends beyond CUDA remain later work.

FixedRows validates the width policy; the common kernel view currently stores
runtime width/stride. No fixed-width specialization speedup, zero submission
overhead, automatic MPI progress, or full solver overlap measurement is claimed.

The transport supports caller-provided valid non-overlapping POD typemaps
bounded by their element extent. Schema identity asserts matching MPI type
signatures; arbitrary datatype overlap/signature reflection is not implemented.
HostPacked uses pageable staging and MPI int-sized packing counts. Native row
metadata is size_t, while the reused CSR checkpoint adapter retains its guarded
int32 byte-row limit. No large-allocation benchmark or multi-node GPU transport
performance claim is made.

Checkpoint bytes require compatible ABI/endianness; POD schema names should be
explicit. Reopen a serializer read session to request another partition of the
same dataset, matching the existing shared-row-start contract. Python live
views support basic rectangular slicing, not fancy indexing or general tensor
semantics. NumPy outputs are snapshots; GPU debugging access synchronizes.
