# Runtime Array follow-up phase

The requested starting checkpoint is commit `eb71d55191cd00d58e7c977d52ee1c125b9253cf`
on `codex/refactor-dnds-array-geom`, pushed to harryzhou2000/DNDSR before this
phase began. This report describes the follow-up checkpoint authorized for
commit and push on 2026-10-22; its validation snapshots retain their original
source hashes and dates.

## Implemented scope

| Requirement | Interface and behavior |
|---|---|
| Shared allocation and metadata | ArrayStorage retains value Buffer and shared ArrayLayout; offsets/sizes are MetadataBuffers. External owners and custom AllocationResources retain foreign allocations. |
| Dynamic row construction | Table, padded and CSR layouts; private parallel row construction, exclusive unique edits, explicit Commit, detached editing when old views must survive. |
| Unified eager execution | StreamLauncher contract, inline/worker CPU and CUDA providers, void Submit, explicit fixed-prefix Record/Wait, dynamic loops and failure/drain behavior. |
| Async access and lifetime | Submission reservations, actual-completion publication, bare kernel views, chosen-stream final release, alias ownership and conservative ordinary destruction. |
| CUDA allocation/copies | cudaMallocAsync and zero initialization, explicit readiness, cudaFreeAsync on a selected downstream stream, DMA retention, prepared shared variable metadata. |
| MPI policy | Persistent or transient requests independently of direct/HostPacked transport; shared context and matched tags; known route input and shared SoA schedules. |
| Dynamic MPI and eager bridge | Source snapshot pin, private CSR destination, collective Commit after success; main-thread progress imported as explicit eager completion. |
| Python | One runtime element facade sharing native storage, numeric/POD access and retained views; row editing, eager orchestration, dynamic transfers and progress. |
| I/O and compatibility | Existing parallel Serializer with matched-rank/direct-partition reads; legacy DNDS Array/ArrayTransformer and Geom reorder regression coverage. |

The detailed contracts include class/object ownership diagrams:
[allocation lifetime](../architecture/ArrayAllocationLifetime.md),
[row editing](../architecture/ArrayRowEditing.md),
[eager execution](../architecture/EagerExecution.md), and
[MPI transfer policies](../architecture/ArrayTransientMPI.md).
No PETSc, hypre, Torch, Legion or new mesh dependency was introduced.

## Validation

| Profile | Verified result |
|---|---|
| CPU native | 58/58 CTest entries, including legacy DNDS and Runtime tests, MPI np=1,2,4,8 and Geom reorder |
| CPU Python Runtime/bridge/editor/engine/transfer | 48 passed, 21 CUDA-only skips per process at np=1,2,4 |
| CPU Python DNDS/Geom regression | 66 passed, 28 skips; MPI-specific and CUDA cases are separate matrix entries |
| CUDA native | 64/64 CTest entries, including legacy CUDA ArrayDOF and Runtime async engine/transfer tests |
| CUDA Python Runtime/bridge/editor/engine/transfer | 69 passed, zero skips per process at np=1,2,4 |
| CUDA Python DNDS/Geom regression | 84 passed, 10 MPI-specific skips |
| CUDA staged MPI memcheck | Five cases per rank at np=2; zero errors with OpenMPI host transport selected |
| Prior native CUDA async memory slice | Seven engine cases, 3890 assertions, zero Compute Sanitizer errors; source hashes recorded separately |

The [combined evidence directory](array_phase_validation_2026-10-21/) contains
logs and exact source hashes. The earlier
[async memory snapshot](array_async_memory_validation_2026-10-21/) covers the
native allocation/free slice. Historical snapshots are retained and are not
silently relabeled as validation of later source.

The default OpenMPI sanitizer run passed every assertion but reported
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED in accelerator_cuda_finalize at
MPI_Finalize. The installed OpenMPI 5.0.9 source calls
cuMemHostUnregister(&checkmem) as a context probe there (lines 199--217).
A [dependency-only control](array_phase_validation_2026-10-21/mpi_finalize_control.cpp),
with no DNDS code, reproduces the same report. Both the
[original failed sanitizer run](array_phase_validation_2026-10-21/cuda-mpi-memcheck-default.txt)
and [control output](array_phase_validation_2026-10-21/mpi-finalize-control.txt)
are retained. Selecting OpenMPI's null accelerator for the host-staged test
keeps native CUDA allocation/copies enabled and produces
[zero sanitizer errors on both ranks](array_phase_validation_2026-10-21/cuda-mpi-memcheck-host.txt).
No OpenMPI installation or global configuration was changed.

Representative cases use 257 six-equation rows with stride eight, 193 permuted
variable rows, independent fields sharing routes, 67--73 changing CSR rows,
empty rows/ranks, MPI POD extents with holes, and owned-only stencils during
pending ghost writes. They verify values/order, generations, retained views,
cross-stream lifetime, schema/tag/collision rejection, failed reducers and
collective publication, rather than merely successful API calls.

## Reproduction

Use `/home/harry/projects/DNDSR-array-geom`. Both build profiles use the project's
Python 3.12 venv, GCC 13.3 and OpenMPI 5.0.9. The CUDA profile uses CUDA 13.2,
architecture 80 and physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`). Test execution
uses `OMP_NUM_THREADS=2`.

```sh
export PATH=/mnt/ssd-SATARAID5/harry/tools/openmpi-5.0.9/install/bin:$PATH
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1
build_dir=build-refactor  # Repeat with build-refactor-cuda.
cmake --build "$build_dir" -t dnds_unit_tests geom_test_mesh_reorder -j8
ctest --test-dir "$build_dir" -R '^(dnds_|geom_mesh_reorder)' --output-on-failure

cmake --build "$build_dir" -t dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 runtime_array_bridge_test -j8
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

Run each profile's build/install/Python tests sequentially: installation is
shared. Keep headers unchanged throughout compilation, including the separate
native Python bridge translation units. Source changes require rebuilding
and reinstalling before Python tests.

The additional sanitizer check uses the already-built CUDA transient test:

```sh
mpirun --mca accelerator null -np 2 \
    /usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
    build-refactor-cuda/test/cpp/dnds_test_runtime_transient --no-colors
```

Omit the MCA option to reproduce the separately recorded dependency report.
The tested transfer uses HostPacked for device buffers, so selecting host MPI
does not bypass the Array's CUDA allocation, copy or lifetime code.

## Limits

Geom migration is optional and existing fields remain compatible. Eager mode
requires caller-supplied dependency and release ordering; automatic schedule
proofs, graph capture and replay remain future layers. New variable GPU metadata
is prepared during setup; async allocate_like shares it. Dynamic MPI preparation
is synchronous/collective and uses O(rank-count) discovery tables; payloads are
nonblocking. Portable MPI progress remains on the host thread, with host
observation before dependent CUDA submission. Tests establish functional
correctness on one machine, not multi-GPU/multi-node transport, CUDA-aware MPI
deployment capability, overlap speedup or solver numerical accuracy.
