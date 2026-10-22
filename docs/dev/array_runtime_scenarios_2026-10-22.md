# Euler workload and overlap audit, 2026-10-22

The previous audit was committed and pushed as
`244b390c4992d7336fd34e8de0c67f5a1651a136` on
`codex/refactor-dnds-array-geom`. The new goal was started after that push.
This snapshot was subsequently included with the
[caller-contract follow-up](array_runtime_contract_audit_2026-10-22.md).
Its evidence directory retains the original tested source/binary hashes.
No changes were made to legacy Euler,
CFV, Geom, or DNDS::Array implementation during this follow-up.

## Findings

1. **Fixed: equal storage width was mistaken for compatible element types.**
   A PayloadPlan made with `ElementSchema::Arithmetic<double>()` accepted
   `Array<int64_t>` and even a double reducer. C++ typed Bind previously checked
   layout and byte width only. The two-rank reproduction failed both expected
   rejection assertions on each rank; it does not execute the invalid reducer.
   See [before](array_scenario_validation_2026-10-22/schema-before.txt) and
   [after](array_scenario_validation_2026-10-22/schema-after.txt).
   Factory-generated schemas now retain a binding type tag. Typed Bind and
   custom reducer factories check it. Python schemas also retain a dtype tag,
   so same-width numeric and structured dtype substitutions are rejected.
   A final binding-path review reproduced the same bypass in Python dynamic
   CSR migration; it now validates collectively, including a one-rank wrong
   dtype. The [migration reproducer](array_scenario_validation_2026-10-22/python-dynamic-schema-before.txt)
   records the two failing cases per rank before that correction.
   Explicit MPI/POD descriptors without a tag still allow application-defined
   types and typemaps. Raw Buffer bindings deliberately remain untyped.
   These comparisons happen during binding/setup, with no kernel overhead.
   Anchors: `src/DNDS/Runtime/RowTransfer.hpp` ElementSchema, Reducer::Custom,
   PayloadPlan::Bind; `Runtime_bind.cpp` Schema and PayloadPlan.bind.

2. **Demonstrated: missing eager joins can silently produce stale results.**
   New CPU and CUDA tests gate a ghost writer, consume the old ghost values
   without joining, finish that read, then open the writer and perform a
   correctly ordered consumption. Every expected stale value is checked.
   The stale output remains valid according to access tracking; no exception
   is expected from the caller-ordered scheduled path. Immediate access to
   the same pending writer does throw. The gated sequence is deliberately
   race-free: it demonstrates an incorrect algorithmic order without relying
   on C++ undefined behavior or a device memory race. Array validity means
   that its operations did not fail, not that dependencies were correct.

3. **Complemented: actual operator overlap, rather than only final values.**
   CPU mock communication and core callbacks rendezvous before either can
   finish. Inline/one-worker compute uses a separate transport executor;
   two-worker compute runs both lanes in the same engine. Another test posts
   real MPI transfers on the main thread and completes them while a core
   worker callback is active; its bounded wait holds the callback until MPI
   completion. Ring work joins both prefixes. CUDA uses two single-block
   kernels with atomic rendezvous and a bounded clock timeout: serial kernel
   execution fails the test. This demonstrates simultaneous kernel residency.
   These are correctness tests, not measurements of useful-work overlap or
   a claim of network bandwidth/speedup.

4. **Complemented: collective failure handling and dynamic solver flow.**
   One-rank shape/stride/edge-width errors and immediate lease conflicts fail
   collectively. Different rank schemas/reduction policies fail before data
   transfer. Rejected starts preserve destination validity and values, release
   reservations, and allow the same binding to be retried successfully.
   Manufactured forward/backward triangular sweeps exercise ordered in-place
   increments, temporary ownership, and data-dependent eager stopping.

5. **Caught during validation: an installed CPU module can masquerade as a CUDA run.**
   The final CPU/CUDA bindings linked within the same second. A subsequent
   CUDA installation reported the already-installed CPU binding as up to date.
   That run exited successfully but skipped all 21 CUDA cases; checking the
   expected counts caught it. ELF build IDs confirmed the wrong module.
   The misleading logs are preserved with `-wrong-profile` suffixes. Generated
   installed artifacts were preserved in build-local backup directories before
   reinstalling each profile. Final tests require matching ELF build IDs for
   all four extensions and libdnds_shared, matching import paths, and the
   expected `Runtime.cuda_enabled` value. The read-only
   [profile verifier](array_scenario_validation_2026-10-22/verify_python_profile.py)
   makes that check reproducible; this audit does not change CMake installation
   policy. A successful pytest exit alone is insufficient evidence of CUDA coverage.

## Coverage grounded in the current solver

This maps the **data and execution contracts** of Euler on the audited branch.
It does not claim that Runtime::Array has replaced Euler's array classes or
that the mock operators validate Euler fluxes, reconstruction, convergence,
or a complete migrated implicit solver. Legacy numerical/linear-algebra
convenience operations still belong to the legacy classes; new runtime
operators consume views and use the existing MPI reduction layer.

| Production use | Source anchor | Runtime coverage |
|---|---|---|
| Conservative state, RANS/species widths, limiter scalars | `src/Euler/EulerSolver.hpp:114`, `:119`; `src/CFV/FiniteVolume.hpp:165` | State-like rows of width 1, 4, 5, 6, 7, 13; physical stride exceeds logical width; every logical component checked |
| Matrix-shaped reconstruction/gradients, borrowed halo routing | `src/CFV/VariationalReconstruction.hpp:820`, `:845` | Width 114 represents 19 reconstruction coefficients times 6 variables; existing padded-to-CSR transfer, shared routes for SoA fields; row permutation and native POD typemaps |
| Variable connectivity and per-cell reconstruction matrices built incrementally | `src/CFV/VariationalReconstruction.cpp:536`, `:828` | Parallel RowBuilder growth, publication, shared metadata lifetime and explicit detach, exclusive editing; variable/empty rows, retained old shape; existing 257-row builder tests |
| Mean/reconstruction/limiter halo pulls | `src/Euler/EulerSolver.hxx:202`, `:362`, `:695` | Repeated epochs, persistent/transient requests, direct/HostPacked transport; 67 owned and 19 permuted ghost rows; separate owned/ghost access controllers |
| Core and ring adjacency accesses | Euler halo/RHS sequence at `EulerSolver.hxx:695` | Existing owned-only CSR stencil tests, plus active CPU/MPI and CUDA overlap, ring joins both communication and core prefixes |
| In-place directional SGS and repeated increment halos | `src/Euler/EulerSolver.hxx:1183` | Manufactured triangular sweeps on inline/1-/2-worker engines; forward/backward order, exact-solution oracle, early stop; repeated MPI field exchange |
| GMRES/ODE temporary fields and algebra | `src/Euler/EulerSolver.hxx:1205`; `EulerSolver_Init.hxx:885` | Fork/join execution, same-stream aliases, temporary ordered release, retained views, allocation after queued work; mock stencil/scale/add operators |
| Component-wise global norms and validity checks | `src/Euler/Euler.hpp:269`, `:287` | Per-component distributed SUM after the mock RHS, checked against a closed-form rank/row oracle; builtin reduction families and error propagation retained from prior audit |
| Custom element/mesh metadata | `src/Geom/Mesh/Mesh_DeviceView.hpp:16`; `src/CFV/VRDefines.hpp:53` | Native POD struct with MPI holes/extent and custom reduction; Python native POD bridge and retained field views |
| Reordering and redistribution | `src/Geom/Mesh` reorder integration | Existing 193-row permuted SoA transfer, dynamic CSR migration over changing epochs, version-stale plan rejection, legacy Geom reorder regressions |
| Parallel restart and direct partition reads | `src/Euler/EulerSolver_PrintData.hxx:983`, `:1102`, `:1171` | Existing Serializer adapter: matching ranks, reversed uneven partitions, empty ranks, changed communicator size, JSON/POD schema checks |
| Python debugging/native sharing | `src/DNDS/Runtime/Array_bind.hpp` | Rebuilt C++/Python shared-storage bridge; numeric/POD access, views and lifetime, CPU/CUDA debugging copies, full runtime suites at np=1/2/4 |

The new CPU tests live in `test/cpp/DNDS/test_EagerEngine.cpp`, the MPI
workload/failure matrix in `test_RuntimeTransient.cpp`, and the device tests
in `test_CUDAExecution.cu`. The MPI workload repeats three epochs for
2 worker counts x 7 widths x 2 request modes x 2 transports. Manufactured
values encode rank, row, component and epoch so stale data, wrong permutations
and wrong strides cannot pass by comparing two identically initialized arrays.
The MPI global sum has a closed-form reference independent of the mock kernel.

## Errors versus unchecked contracts

| Case | Required behavior / evidence |
|---|---|
| Wrong native arithmetic or Python dtype with equal byte width | New binding/custom-reducer rejection tests |
| One-rank stride/width mismatch, policy disagreement, immediate receive read lease | New collective failure and retry tests; no destination modification |
| Receive collisions without reducer, duplicate tags, active binding reuse, stale version | Existing native regressions, rerun in both profiles |
| Reducer failure, repeated failed Wait/Test/progress import | Previous checkpoint regressions rerun; destination invalid, dependent work suppressed |
| Failed/skipped CPU/CUDA writer, rejected submission, read-only import | Existing engine/storage regressions, rerun |
| Missing caller join | New CPU/CUDA deterministic stale-value oracle; scheduled path intentionally accepts it |
| Wrong semantic row order in a self-consistent Known route | Caller contract; rank/row/component oracle in permutation tests detects regressions in the implementation |
| Wrong explicit POD schema, false imported dtype, untracked raw pointer, unchecked stencil index | Caller contract; generic storage cannot infer field meaning or prove pointer use. Bounds-checked access exists, but bare hot-loop views deliberately omit checks |
| Native final release without joining all independent uses | Caller contract; positive multistream lifetime tests cover proper joins. No intentional use-after-free is executed |

No schedule validator, automatic dependency inference, portable stream-native
MPI, graph capture, or new regional lease mechanism was introduced. CPU worker
tests initialize MPI with FUNNELED support and progress MPI on the main thread.
Portable CUDA/MPI still observes producers and completion on the host. The
simultaneous CUDA communication kernel is a mock device transport, not an
assertion that OpenMPI is captureable or stream ordered.

## Validation and reproduction

Build environment: GCC 13.3, OpenMPI 5.0.9, Python 3.12 venv, CUDA 13.2,
physical GPU 1 (one A100 exposed as logical device 0), OMP_NUM_THREADS=2.
Source and binary hashes and full logs are in
[the evidence directory](array_scenario_validation_2026-10-22/).

| Validation | Result |
|---|---|
| Full DNDS native + legacy Geom reorder, CPU | 58/58 CTest entries |
| Full DNDS native + legacy Geom reorder, CUDA | 64/64 CTest entries |
| Final expanded CPU engine/MPI scenarios, both profiles | 5/5 CTest entries in each profile, MPI np=1/2/4/8 |
| CPU Python Runtime suites, np=1/2/4 | 53 passed, 21 CUDA-only skips per process |
| CUDA Python Runtime suites, np=1/2/4 | 74 passed per process |
| Python DNDS/Geom regression | CPU 71 passed/28 skips; CUDA 89 passed/10 skips |
| CUDA engine memcheck | 10 cases, 42981 assertions, zero errors |
| CUDA staged MPI transient memcheck, np=2 | 8 cases per rank, zero errors on both ranks |

The full native matrices passed before the final test-only expansion (width
4, same-engine CPU transport, and global norm checks). The affected engine
and MPI targets were rebuilt and rerun in both profiles afterward; the final
MPI sanitizer run also covers that expansion. Native runtime code was
unchanged between those runs. The subsequent Python dynamic-transfer check
was rebuilt and the Python matrices repeated in both profiles. All four
production Python modules and the bridge were rebuilt and installed before
each profile's Python run.
The commit hook changed five checkpoint files in whitespace only; the new
builds use those actual bytes. Historical pre-hook hashes were not rewritten.

From `/home/harry/projects/DNDSR-array-geom`:

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
"$python_exe" docs/dev/array_scenario_validation_2026-10-22/verify_python_profile.py "$build_dir"
for ranks in 1 2 4; do
    mpirun -np "$ranks" "$python_exe" -m pytest --with-mpi -q \
        test/DNDS/test_runtime_array.py test/DNDS/test_runtime_array_bridge.py \
        test/DNDS/test_runtime_row_builder.py test/DNDS/test_runtime_engine.py \
        test/DNDS/test_runtime_transient.py
done
"$python_exe" -m pytest test/DNDS test/Geom -q
/usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
    build-refactor-cuda/test/cpp/dnds_test_runtime_cuda_engine --no-colors
mpirun --mca accelerator null -np 2 \
    /usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
    build-refactor-cuda/test/cpp/dnds_test_runtime_transient --no-colors
```

The MPI sanitizer invocation disables OpenMPI's accelerator component to avoid
the dependency-only finalization probe documented in the previous audit. CUDA
allocation/copy/kernels and staged MPI still run. Ordinary CUDA CTest uses the
normal accelerator configuration. These runs establish no multinode/multi-GPU
transport or performance result, and are not a full CFD accuracy study.

If the profile verifier rejects an installed artifact, preserve the generated
installed extension/shared-library files outside their install destinations
and rerun the installation before testing. Do not count a CUDA run that takes
the CPU-only skips. The preflight logs record the previous and intended build
IDs and the backup directories used for this audit.
