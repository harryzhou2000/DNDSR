# Native geometry, periodic topology and output checkpoint

This is a further GN3–GN6 implementation checkpoint. **GN7 remains open**:
production orchestration, native local cell ordering, periodic import/refinement,
some compatibility outputs and retirement of legacy mesh authority still need
work. The consumer inventory records the individual paths. No solver trajectory
or numerical-method accuracy claim follows from these unit tests.

## Changes exercised

- Boundary extraction constructs Runtime arrays directly, including line-cell
  surfaces from 2D volumes, with a new namespace and durable source-ID ancestry.
  Destination guards reject hidden legacy allocations before publication.
- Native coordinate transforms, owner callbacks and boundary rectification
  preserve publication/lease contracts and refresh owner/ghost values. Native
  wall distance uses the existing geometry mathematics and publishes native
  vectors. A legacy ghost-copy ordering defect found by parity tests was fixed.
- Native coordinate hashing preserves the historical weak, partition-dependent
  hash. Exact facial neighbor graphs and all six direct ordering options work
  without a legacy geometry workspace, including empty owners and local parts.
- Periodic primary/topology construction accepts canonical node identities and
  ordered image labels. Independent fixtures cover repeated IDs, physical self
  faces versus inverse sets, quadratic face agreement, rigid frames, migration
  and archives. CGNS periodic identification is still a separate missing step.
- Native VTKHDF output uses output-only buffers and a dedicated congruent
  communicator. The common CGNS writer reads scoped facade queries. A round-trip
  deadlock exposed CGNS's process-global MPI communicator: parallel output now
  restores the self communicator, and local serial parsing selects it explicitly.
- Python has explicit native read/prepare/extract/archive helpers with retained
  ownership and rank-consistent BC names. Legacy defaults remain available.

## Acceptance evidence

The CPU profile is `build-refactor`; the CUDA/Python profile is
`build-refactor-cuda`. MPI is Open MPI 5.0.9; `OMP_NUM_THREADS=2`.

The broad CPU Geom run executed 139 CTests: 131 passed and eight failed because
two older UTs still expected newly supported native operations to throw. Those
checks were migrated to active-reader rejection and legacy/native graph parity,
preserving their isolation and rollback purposes. All eight corrected tests pass
at MPI 1/2/4/8. Production sources were unchanged by that UT correction. The first
run and focused rerun are retained separately, not presented as one all-pass run.

All three CUDA executables pass: the full facade suite on GPU 1, and direct-MPI
overlap plus entity-field tests on two GPUs, one MPI rank per GPU. Each MPI rank
sees only its assigned GPU. The native-only stencil runs before and after
redistribution/archive restart. Device core work completes during an active
receive-lease interval; ring reads follow completion and stream joins. This
demonstrates correctness and simultaneous in-flight work, not network speedup.

The rebuilt CFV and Euler categories pass all 9 and 17 CTests respectively,
including reconstruction and evaluator pipelines. Euler's modified
`RunImplicitEuler` NS instantiation also passes a syntax-only compile; its
existing deprecated SGS-call warnings are retained. These are unit-test
pipelines, not an executed production Euler trajectory.

Python build/install and imported-artifact verification are recorded with the
final Python run manifests. All eight shared-library/extension ELF IDs match
the installed CUDA build and `Runtime.cuda_enabled=true`. Native helper,
extraction, construction, CGNS and refinement tests pass: 24 plus one expected
single-rank skip at MPI 1, and 25 per rank at MPI 2/4/8. Existing facade/edit/basic
tests pass 19 per rank at MPI 1/2. Two old Python tests needed the same supported
operation migration as the C++ tests; held-reader rejection, immutable snapshots,
entity-order validity and exact transformed coordinates remain checked. The
intermediate failure log is retained; the first two-failure Python log was
overwritten before collection, so only its status is recorded in the manifest.
The four binding targets must be rebuilt and the
`py` component installed before reproducing those runs; a CUDA-enabled package
does not by itself prove that each Python test uses a GPU.

## Reproduction and provenance

```sh
cmake --build build-refactor --target geom_unit_tests -j10
ctest --test-dir build-refactor -R '^geom_' --output-on-failure --timeout 120 -j8
cmake --build build-refactor --target geom_test_mesh_edit geom_test_native_mesh_publication -j8
ctest --test-dir build-refactor -R '^geom_(mesh_edit|native_mesh_publication)_np(1|2|4|8)$' --output-on-failure
cmake --build build-refactor --target cfv_unit_tests euler_unit_tests -j8
ctest --test-dir build-refactor -R '^(cfv_|euler_)' --output-on-failure --timeout 120 -j8
cmake --build build-refactor-cuda --target geom_test_mesh_facade_cuda geom_test_mesh_facade_overlap_cuda geom_test_mesh_entity_field_cuda dnds_pybind11 geom_pybind11 cfv_pybind11 eulerP_pybind11 -j8
cmake --install build-refactor-cuda --component py
```

CUDA/Python run manifests retain exact commands, environments and exit statuses.
The pre-commit manifest records source and executable hashes before hook
formatting; formatting is not silently described as a newly executed binary.
Stored text logs remove ANSI control sequences, NUL terminators in MPI library
version strings, and trailing whitespace;
the normalization manifest retains original and stored hashes. Earlier failures
are diagnostic evidence, not additional tests in the acceptance count.

The periodic memory audit distinguishes forbidden globally indexed tables from
genuinely large requested node stars and repeated boundary-candidate payloads.
The wall search still replicates surface triangles; it is not the sparse ghost
construction algorithm. Native topology mutation remains a host operation.
