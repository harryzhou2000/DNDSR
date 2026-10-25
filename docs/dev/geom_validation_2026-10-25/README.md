# Native construction checkpoint evidence — 2026-10-25

This is an intermediate GN0–GN4 checkpoint, not completion of the native Geom
work order. See [the migration ledger](../geom_native_migration_2026-10-25.md)
and [the ownership/constructor contract](../../architecture/NativeMesh.md).

## Executed checks

| Record | Result and scope |
|---|---|
| `focused-final-cpu-tests.txt` | 20/20 CTest registrations: native construction, primary builder, named registry, native transition and existing transition, each at MPI 1/2/4/8. Includes exact-boundary and conflicting-same-key publication rejection. |
| `overlap-final-cpu-tests.txt` | 4/4, MPI 1/2/4/8: existing CPU inline/worker communication-computation overlap and restored-field stencil after native redistribution. The fixture now checks distinct row order while preserving native logical domain. |
| `dsl-cpu-tests.txt` | 8/8: native connectivity and relation access at MPI 1/2/4/8. |
| `runtime-cpu-tests.txt` | 2/2: Runtime Array and legacy Array bridge suites, including sized row access. |
| `python-native-np{1,2,4}.txt` | 12 passed per rank: shared buffers/POD values, uniform CSR/padded metadata, escaped readonly rows, empty ranks and collective input rejection/retry. |
| `python-existing-np{1,2}.txt` | 18 passed and one explicit CUDA-only skip per rank: existing mesh facade, edit and basic Geom tests. |
| `cuda-gpu1-final-builder.txt` | Native-array construction → device placement → direct CUDA stencil, including source destruction while a completion is pending: 1 case, 22 assertions. |
| `cuda-gpu1-runtime-tests.txt` | Runtime CUDA Array suite: 7 cases, 749 assertions; also an earlier native-builder run. |

The historical broad run (`broad-cpu-before-overlap-fix.txt`) passed 103/107
registrations. Its four overlap failures were the old test assumption that a
redistributed mesh must have a different logical domain. The corrected fixture
checks a different row order, unchanged native domain and valid retained source;
the four final reruns passed. The broad run and focused follow-ups are separate
invocations, not one final all-suite run. Some publication targets were refreshed
during the broad run; only the final focused records establish the latest
publication checks.

All four Python modules were built before testing, and the Python component was
installed again after the last publication fix. `python-final-imports.json`
records actual loaded extension/shared-library paths and matching ELF build IDs,
including `Runtime.cuda_enabled=false`. CMake rewrites RPATH on installation,
so raw built/installed file hashes differ without implying stale code.
`python-final-runs.json` records commands and environment.

## GPU resource limit

The current native builder failed its first CUDA launch on GPU 0 with
`cudaErrorMemoryAllocation`. An unrelated process held about 80.5 GiB there;
only about 442 MiB remained. That process was left untouched. The identical
binary passed on GPU 1. Earlier development code ran this case on two distinct
A100 GPUs, but that is not counted as final-source two-GPU acceptance. The
updated CUDA overlap/EntityField tests compile; their current two-GPU run is
pending sufficient free device memory.

## Reproduction and provenance

Checkout: `/home/harry/projects/DNDSR-array-geom`; CPU build `build-refactor`,
CUDA build `build-refactor-cuda`; Open MPI 5.0.9; CUDA 13.2, architecture 80;
Python 3.12 from `/home/harry/projects/DNDSR/venv/bin/python`.
Text logs omit terminal color/NUL bytes and trailing whitespace, and convert
carriage-return progress updates to lines. `log-normalization.json` records
original and normalized hashes; test outcomes are unchanged.

```sh
cmake --build build-refactor --target geom_unit_tests -j12
ctest --test-dir build-refactor --output-on-failure -R '^geom_' -j4
ctest --test-dir build-refactor --output-on-failure -R '^geom_(native_mesh_construction|native_mesh_builder|native_relation_registry|native_mesh_transition|mesh_transition)_np' -j4
ctest --test-dir build-refactor --output-on-failure -R '^geom_mesh_facade_overlap_np' -j4
cmake --build build-refactor-cuda --target geom_test_mesh_facade_cuda dnds_test_runtime_cuda geom_test_mesh_facade_overlap_cuda geom_test_mesh_entity_field_cuda -j6
```

CUDA log headers contain the actual GPU selection and launch commands. The
source/binary hashes in `precommit-artifacts.json` identify the tested source
checkpoint before the repository's formatting hook. That hook may change
formatting at commit. These records do not claim exact final-source GN7 coverage;
the phase still requires native restart/readers, face/edge/periodic construction,
remaining Geom workflows, consumer migration and the final validation matrix.

Known next DSL work: give mathematical inverse sets a distinct native descriptor
from ordered physical facade slots (especially two-sided FaceCells). Legacy
inverse adapters currently preserve historical support keys. Full named variants
are retained by state but `RegisterMesh` still registers fixed facade slots only.
