# Runtime Array completion and interoperability audit — 2026-10-23

Branch: `codex/refactor-dnds-array-geom`. Baseline:
`04e051e037e6ffb3ea7dfe21ca5f130ec92f4f21`. This phase completes the agreed
Array foundation and mock integration scope. Legacy Array/Euler/CFV/Geom APIs
remain available; production Euler and full Geom migration are deferred.

## Changes and review findings

1. **Registered POD schema mismatch:** a factory schema for a registered Python
   POD class accepted raw opaque records with the same size. Static binding and
   dynamic transfer now check the native element type tag. NumPy representation
   checks for custom reducers are separate, preserving POD reducers and numeric
   native/Python interoperability. Tests reject mismatches collectively, including
   rank-mixed inputs, without poisoning source/destination storage.
2. **DLPack ownership/access:** added a small optional Python adapter with a pinned,
   licensed DLPack header. Numeric tables export zero-copy, including row padding;
   contiguous numeric 1D/2D producers import with shared ownership. Export retains
   ArrayStorage, metadata and an exclusive write lease through the last consumer
   alias. CUDA release fences the consumer stream. Unsupported representations
   fail explicitly. No LibTorch native dependency or autograd claim is introduced.
3. **Foreign-owner teardown deadlock:** delayed worker release of an imported
   owner could need the GIL while engine destruction held it and joined the worker.
   The reproduction timed out after eight seconds. Python engine destruction now
   releases the GIL during drain; the regression and sanitizer run pass.
4. **Production-shaped integration:** `test/demos` uses native CPU/CUDA stencil
   kernels, separate owned/ghost storage, core computation during active halo
   transfer, global norms, permutation/migration and parallel restart. An independent
   NumPy global field is the oracle. Other tests cover ragged/padded hindexed
   transfer, persistent reuse, transient requests and collision reduction.
5. **Coverage/publication gaps:** CI now builds/imports the native bridge, enables
   MPI-marked tests and runs two-rank Runtime tests. A rank wrapper assigns distinct
   GPU UUIDs before CUDA/MPI initialization. Two older final build logs referenced
   by the previous audit are included despite the generic `build*` ignore rule.

The direct MPI datatype path was already the default. This phase preserves that
default and validates it on distinct GPUs; HostPacked remains explicit. Collision
reduction needs contribution workspace. Direct means no DNDS packing, not a claim
about MPI's internal staging or GPUDirect.

## Validation performed

| Check | Result |
|---|---|
| CPU native DNDS plus Geom reorder, MPI np=1/2/4/8 | 58/58 CTest entries passed |
| CUDA-build native DNDS, MPI np=1/2, distinct GPUs | 34/34 configured entries passed |
| Final CUDA Runtime Python and mock, np=2 | 132 passed per rank, no skips |
| CUDA Runtime Python and mock, np=1 before the last two queued-owner cases | 130 passed, no skips; final 32-case DLPack sanitizer suite includes those two added cases |
| Final CPU Runtime Python and mock, np=4 | 91 passed and 41 CUDA-only skips per rank |
| CPU Runtime Python and mock, np=2 before the last two queued-owner cases | 90 passed and 40 CUDA-only skips per rank |
| CPU broader DNDS/Geom/mock Python suite | 114 passed, 41 expected skips, before those last two cases |
| CUDA native engine memcheck | 11 cases, 61,593 assertions; zero errors |
| Final DLPack/Torch memcheck | 32 passed, zero errors |
| Installed CPU/CUDA binding profiles | All four extension ELF build IDs and libdnds_shared match the selected builds; Runtime.cuda_enabled checked |

Tests include pointer equality, all ten supported numeric dtypes, mutations,
escaped tensor slices, parent reset/destruction, custom C++ POD/native sharing,
capsule single consumption, unconsumed capsule cleanup, offset/empty imports,
rejected readonly/unsupported layouts, non-default CUDA streams with explicit
joins, queued CPU/CUDA owner retention and destructor drain. Existing gated
overlap and deterministic missing-edge tests remain in the native/Python suites.

The first singleton Python sanitizer attempt failed to attach and was stopped;
it is retained as a failed diagnostic, not counted as validation. A Torch-only
sanitizer probe passed. Running the full interop suite through `mpirun -np 1`
then passed. That run uses `OMPI_MCA_accelerator=null` to avoid the previously
isolated MPI accelerator/finalization issue and sends **no device buffers through
MPI**. GPU communication coverage comes from the normal two-GPU Direct suites;
the interop memcheck result is not presented as sanitized GPU MPI coverage.

## Environment and evidence

The new environment is `/home/harry/projects/DNDSR-array-geom/venv-interop`:
Python 3.12.3, NumPy 2.4.6, mpi4py 4.1.2, Torch 2.14.1+cu130 (CUDA 13.0 wheels).
Native CUDA builds use toolkit 13.2, SM80, GCC 13.3 and Open MPI 5.0.9.
The two GPUs are A100 80GB PCIe, driver 595.71.05, connected across SYS/NUMA;
no NVLink transport or communication speedup is claimed. The venv occupies 5.6 GB;
about 37 GB remained free on `/home`. No persistent wheel cache was retained.
The original DNDSR venv was not changed. Existing builds use its Python 3.12 ABI;
all interoperability tests execute the new environment's Python.

[Evidence and reproduction commands](array_completion_validation_2026-10-23/README.md)
include normalized logs, exact package pins, source/binary hashes and profile
verification. Remote CI execution is not claimed by local validation.

## Remaining integration boundary

The supported Array phase has no unresolved failure from this validation matrix.
Passing tests do not prove universal bug freedom. [Caller contracts](../architecture/RuntimeCallerContract.md)
and [DLPack ownership](../architecture/ArrayDLPack.md) remain part of the API:
eager dependencies, semantic row identity, raw-pointer uses and foreign aliases
cannot all be inferred or checked by the runtime.

Later phases include full Geom mutation/migration, production Euler integration,
atomic mesh-field bundles, optional schedule verification/capture, other devices,
cross-device copies in one process, ragged tensor component sharing and adjoints.
No PETSc, hypre, NVSHMEM or Torch dependency was added to the native solver.
Torch's test wheel has its own CUDA dependencies; this does not select a new DNDS
communication backend. The solver-wide plan records these boundaries separately.
