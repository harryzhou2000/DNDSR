# Runtime Array audit, 2026-10-22

Checkpoint `963cfeb59550047781ed02dfec14764913198fb3` was committed and pushed
to `origin/codex/refactor-dnds-array-geom` before this audit. Remote and local
tips were checked. The audit changes described here form the next checkpoint,
committed under the user's subsequent explicit commit-and-push instruction.

## Scope and conclusion

The added framework has distinct storage, shape, execution and communication
roles. Legacy DNDS::Array still functions. Runtime::Array is a shallow typed
handle; Python has one dynamic facade sharing the same storage, independent
of the C++ fixed-width template policy. Geom integration remains optional.
The new path uses the existing Serializer and MPI libraries. No PETSc, Torch,
Legion or additional execution dependency was introduced.

The review traced ownership through normal completion, rejected submission,
failed execution, moved handles, retained views, exclusive shape editing,
MPI completion and CUDA ordered release. A reproduced transfer failure defect
was fixed, together with the Python exception replay/lifetime handling required
by retained asynchronous failures. Additional
tests cover device stencil overlap, numeric reducer dispatch and uneven
checkpoint partitions. This is a focused framework audit, not a proof of every
possible concurrent program or a solver accuracy study.

## Feature and ownership summary

| Layer | Responsibility | Ownership/access rule |
|---|---|---|
| AllocationResource / Buffer | Placement, byte ranges, allocator/import policy, generations | Shared allocation identity; final contextual decrement calls a unique provider |
| MetadataBuffer / ArrayLayout | Immutable offsets/sizes and prepared device mirrors | Independent fields may share shape buffers; views retain their metadata |
| ArrayStorage / Array | Values plus shape; typed views and fixed-width validation | Shallow C++ and Python handles share storage; reset detaches |
| RowBuilder / ExclusiveRows | Incremental CSR/padded setup and publication | Private per-row edits followed by join/Commit; unique exclusive edits block sharing; abandonment invalidates |
| EagerEngine / StreamLauncher | CPU inline/workers and native CUDA orchestration | Void kernel submission, explicit prefix events, caller-provided joins |
| ScheduledAccess | Submission reservations and completion publication | Immediate access conflicts fail; bare views have no per-element tracking |
| RowTransferPlan / PayloadPlan | Row relations and byte/typemap layout | Routes shared across SoA fields; POD extent is the element stride |
| TransferBinding / TransferOperation | Bound addresses, request policy, active requests | Direct datatypes by default; explicit staging/reducers for collisions |
| CommunicationContext / MPIProgress | Tag namespace and portable host progress | Matched collective setup; MPI progression stays on the permitted thread |
| Python / ArrayIO | Debug indexing, native interop and checkpoints | Native ownership retained by views; explicit device sync; Serializer stages logical rows |

Detailed class/object diagrams are in
[Python ownership](../architecture/ArrayPythonOwnership.md),
[allocation lifetime](../architecture/ArrayAllocationLifetime.md),
[row editing](../architecture/ArrayRowEditing.md),
[execution](../architecture/EagerExecution.md) and
[MPI](../architecture/ArrayTransientMPI.md).

| Contract reviewed | Existing regression suites | Audit complement |
|---|---|---|
| Shared/native/foreign ownership, allocation release | test_Runtime, test_RuntimeArray, test_Runtime_CUDA, test_runtime_array_bridge.py | Rebuild and run unchanged lifetime regressions |
| Metadata sharing, parallel CSR build, exclusive/detached edits | test_RuntimeRowBuilder, test_runtime_row_builder.py | Review publication/abandonment and retained snapshots |
| Eager ordering, cleanup, reservations, copies | test_EagerEngine, test_CUDAExecution, test_runtime_engine.py | CUDA owned-only stencil with pending ghost writer |
| Routing, typemaps, collisions, policy, dynamic rows | test_RuntimeRowTransfer, test_RuntimeTransient, test_runtime_transient.py | Sticky failed completion, progress import and full numeric CUDA dispatch |
| Serializer, matching/direct partition reads | test_RuntimeArrayIO, test_runtime_array.py, native Python bridge | Uneven field bundle and reversed direct partitions |
| Legacy compatibility | all DNDS native tests, Geom reorder, Python DNDS/Geom | Fresh builds and reinstalls in CPU and CUDA profiles |

## Numbered findings

1. **Fixed: a failed transfer lost its failure when observed again.**
   In the checkpoint's `RowTransfer.cpp`, the catch path in
   `WaitDestinationReady` drained requests, poisoned the destination and set
   `started=false`; subsequent Wait/Test treated that state as success.
   Importing it into MPIProgress therefore allowed dependent computation.
   The two-rank reproducer failed 56 of 96 assertions per rank across
   persistent/transient, direct/packed and immediate/scheduled combinations.
   TransferOperation now retains its first exception, rethrows it from all
   observation methods and carries it through moves. Replacing or destroying
   a drained failed operation remains safe and never invokes its reducer again.
   [Before](array_audit_validation_2026-10-22/sticky-failure-before.txt) and
   [after](array_audit_validation_2026-10-22/sticky-failure-after.txt) outputs
   preserve the reproduction. The after run passed 96/96 assertions per rank.
   C++ and Python regressions cover eager progress import, repeated observation,
   output invalidation and callback lifetime.
   Current anchors: `src/DNDS/Runtime/RowTransfer.cpp:601` (stored exception),
   `:746` (destination observation), `:832` (full observation), `:858` (polling),
   and `test/cpp/DNDS/test_RuntimeTransient.cpp:55` (regression matrix).
   The older repeated-Wait tests expected swallowed errors; those expectations
   were updated. The first CPU matrix with that outdated expectation is retained
   as `cpu-ctest-old-expectation.txt`, rather than being relabeled a passing run.

2. **Fixed during validation: repeatable Python errors must not retain frames.**
   Python validation exposed pybind11's single-use exception restoration:
   repeatedly reporting a retained `error_already_set` aborted on the second
   observation. An intermediate replay adapter preserved live tracebacks but
   retained the callback frame and transfer after all public owners left scope;
   `python-failure-lifetime-before.txt` records that regression. The callback
   boundary now retains type and arguments with GIL-safe ownership and creates
   a Python error per observation. It keeps the original traceback as C++
   diagnostic text rather than live frames. The original abort log is
   `python-repeated-error-before.txt`. This binding adaptation accompanies the
   native failure-retention correction. The lifetime test verifies both repeated
   multi-argument ValueError propagation and callback collection after scope exit.
   Custom exception constructors must be reconstructible from their `args`;
   arbitrary extra attributes and live callback tracebacks are not replayed.

3. **Coverage complemented: production-shaped CUDA overlap and reductions.**
   The existing engine tests exercised multistream ownership but did not test
   a CUDA owned-only random CSR stencil while ghost storage was independently
   write-pending. The new case uses 257 six-equation rows, physical stride eight,
   19 ghost rows and variable adjacency including empty rows. It checks core
   completion before opening the ghost gate, then explicitly joins ring work.
   Device adjacency storage is released after its last stencil use. New reducer
   tests exercise all ten numeric dispatch types, both initialization policies,
   all applicable operations, 129/257-component rows, overflow inputs, untouched
   rows and padding. They invoke the device reducer without requiring device
   MPI, so disabling an MPI accelerator cannot silently remove this coverage.

4. **Coverage complemented: uneven checkpoint redistribution.**
   The existing tests establish matching-rank reads, direct reads and changed
   communicator sizes. The new case adds a six-equation flow field plus variable
   adjacency, 67+2*rank rows, an empty rank, empty rows, and reversed direct
   partitions. Every logical value/order is checked after both matching and
   direct reads. Physical padding is deliberately absent from checkpoints.

## Reviewed contracts and practical limits

The two access paths are intentionally different policies over the same
allocation controller: immediate leases exclude conflicting accesses, whereas
scheduled reservations permit caller-ordered eager work. CPU and CUDA launchers
share orchestration, but CUDA issue is not device completion. Explicit Record
boundaries retire reservations; kernels do not allocate completion events.

Independent aliases do not establish ordering. Before a potentially final
ReleaseOn, the caller must join every independent use on the chosen release
stream. Ordinary destruction is conservative. Host DMA and foreign providers
without ordered-release support retain owners until actual completion.
Long sequences without an explicit Record/Drain retain command descriptors.

The audit did not introduce schedule proofs, graph capture, cross-device
copies, multirank failure recovery or stream-native MPI. Portable MPI remains
host-driven. Transient requests avoid persistent request creation, but route
discovery, typemap compilation and Begin preflight still contain synchronous
collectives and O(rank-count) tables. Large-scale sparse rendezvous and packed
GPU transport optimization remain separate work.

Published metadata is immutable; detached edits publish replacement metadata.
Exclusive setup requires uniqueness and joined row workers. Concurrent mutation
of the same public handle and escaped raw pointers are outside this contract.
New GPU variable metadata is prepared during setup; allocate_like shares it.
Python indexing synchronizes for debugging; NumPy results are snapshots.
Device descriptors and allocation/launcher providers leave extension points,
but the currently implemented placements are CPU and CUDA. Adding another
accelerator still requires its placement checks, copy and execution providers.

The tests use one host and one visible A100, including multiple MPI ranks on
that GPU. They do not establish multinode/multi-GPU transport, overlap speedup,
hardware-failure recovery or CFD convergence. The existing OpenMPI finalization
sanitizer issue and dependency-only control remain documented in the
[previous phase report](array_runtime_phase_2026-10-21.md).

## Validation

| Final profile | Result |
|---|---|
| CPU native plus legacy/Geom reorder, MPI np=1/2/4/8 | 58/58 CTest entries passed |
| CUDA native plus legacy/Geom reorder, MPI np=1/2/4/8 | 64/64 CTest entries passed |
| CPU Python Runtime/bridge/builder/engine/transfer, np=1/2/4 | 51 passed and 21 CUDA-only skips per process |
| CUDA Python Runtime/bridge/builder/engine/transfer, np=1/2/4 | 72 passed, zero skips per process |
| CPU Python DNDS/Geom regression | 69 passed, 28 skips |
| CUDA Python DNDS/Geom regression | 87 passed, 10 MPI-only skips |
| CUDA engine memcheck | 8 cases, 5435 assertions, zero errors |
| CUDA numeric reducer memcheck | 10 selected type cases, 85320 assertions, zero errors |
| CUDA staged MPI transient memcheck, np=2 | 6 cases per rank, 20041/20031 assertions, zero errors on both ranks |

The [evidence directory](array_audit_validation_2026-10-22/) contains failed
reproducers, final results, build/install logs, source hashes and critical binary
hashes. Logs are normalized only to remove ANSI escapes, embedded NULs and
trailing whitespace. Historical phase hashes remain unchanged: the commit hook
formatted 28 source files with only whitespace changes, and this audit rebuilt
the current source rather than claiming the historical binaries came from those
formatted bytes. The audit's final source snapshot records the tested bytes
before this checkpoint's formatting hook; any subsequent formatting is verified
separately rather than rewriting historical validation hashes.

## Reproduction

Run from `/home/harry/projects/DNDSR-array-geom`. The configured profiles use
GCC 13.3, OpenMPI 5.0.9, the project Python 3.12 venv, and CUDA 13.2/SM80 for
the device build. Native MPI tests cover np=1,2,4,8. Install and test each Python
profile sequentially because both install into the same `python/` directory.

```sh
export PATH=/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin:$PATH
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1
build_dir=build-refactor # Repeat with build-refactor-cuda.
cmake --build "$build_dir" -t dnds_unit_tests geom_test_mesh_reorder \
    dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 runtime_array_bridge_test -j8
ctest --test-dir "$build_dir" -R '^(dnds_|geom_mesh_reorder)' --output-on-failure
cmake --install "$build_dir" --component py
export PYTHONPATH="$PWD/python:$PWD/$build_dir/test/cpp"
python_exe=/home/harry/projects/DNDSR/venv/bin/python
for ranks in 1 2 4; do
    mpirun -np "$ranks" "$python_exe" -m pytest --with-mpi -q \
        test/DNDS/test_runtime_array.py test/DNDS/test_runtime_array_bridge.py \
        test/DNDS/test_runtime_row_builder.py test/DNDS/test_runtime_engine.py \
        test/DNDS/test_runtime_transient.py
done
"$python_exe" -m pytest test/DNDS test/Geom -q
```

The device-only reduction slice still executes CUDA kernels when OpenMPI's
accelerator is disabled. This allows memcheck without the previously isolated
OpenMPI finalization probe error:

```sh
/usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
    build-refactor-cuda/test/cpp/dnds_test_runtime_cuda_engine --no-colors
mpirun --mca accelerator null -np 1 \
    /usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
    build-refactor-cuda/test/cpp/dnds_test_runtime_row_transfer \
    --test-case='Native CUDA reduction dispatch*' --no-colors
mpirun --mca accelerator null -np 2 \
    /usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
    build-refactor-cuda/test/cpp/dnds_test_runtime_transient --no-colors
```
