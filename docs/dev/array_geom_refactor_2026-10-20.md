# Array/Geom refactor implementation record

Branch: `codex/refactor-dnds-array-geom`.
Worktree: `/home/harry/projects/DNDSR-array-geom`.
Base: latest fetched `origin/dev/harry`, `03891f8b5869017de355468d93294e3eb25f0feb`.
Design: private `harryzhou2000/dndsr-next-plan`, commit `1c2b1fe`.

See [Array runtime architecture](../architecture/ArrayRuntime.md) for the implemented API, ownership rules and deliberately unsupported paths. This record describes the initial foundation; it is not completion of the next-generation solver roadmap.

## Build isolation

The implementation worktree has independent CPU (`build-refactor`) and CUDA (`build-refactor-cuda`) builds. Existing external headers/libraries are referenced read-only from the main checkout. `/tmp/dnds-refactor-deps.cmake` supplies the resolved DNDS_EXTERNAL_INCLUDE_* cache paths from the main checkout, with FORCE to replace an initial NOTFOUND result. Input mesh fixture entries in the worktree's data/mesh are symlinks to the main checkout's fixture files; they are ignored test inputs, not source changes. GeneratedMultiBlock2x2_OneSided.cgns and GeneratedMultiBlock3x3_OneSided.cgns are exceptions: their links were removed and the CGNS test regenerated local regular files, so the generator cannot overwrite the original checkout's fixtures.

The CGNS multizone test also hardcodes external/cfd_externals/install/lib/libcgns.so relative to its source checkout. An ignored external/cfd_externals/install symlink points to the main checkout's dependency installation for this read-only library lookup. A fresh worktree needs this lookup available before running that test; configuring the CMake prefix alone does not alter the test's hardcoded generator path.

The initial bare configure failed because a new worktree does not contain ignored external header trees. Supplying the install prefix alone does not provide Eigen/Boost/CGAL/header-source paths. Configuration then succeeded with the dependency cache. No external package was rebuilt or installed into the main checkout.

Regenerate that local dependency cache from the existing configured main checkout:

```sh
python3 - <<'PY'
from pathlib import Path
cache = Path('/home/harry/projects/DNDSR/build/CMakeCache.txt')
entries = []
for line in cache.read_text().splitlines():
    if line.startswith('DNDS_EXTERNAL_INCLUDE_') and ':PATH=' in line:
        key, value = line.split(':PATH=', 1)
        if not value.endswith('-NOTFOUND'):
            entries.append(f'set({key} "{value}" CACHE PATH "Shared read-only dependency" FORCE)')
Path('/tmp/dnds-refactor-deps.cmake').write_text('\n'.join(entries) + '\n')
PY
```

```sh
cmake -S . -B build-refactor -C /tmp/dnds-refactor-deps.cmake \
  -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
  -DDNDS_CFD_EXTERNALS_INSTALL=/home/harry/projects/DNDSR/external/cfd_externals/install \
  -DPython_EXECUTABLE=/home/harry/projects/DNDSR/venv/bin/python
cmake -S . -B build-refactor-cuda -C /tmp/dnds-refactor-deps.cmake \
  -DDNDS_BUILD_TESTS=ON -DDNDS_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DDNDS_CFD_EXTERNALS_INSTALL=/home/harry/projects/DNDSR/external/cfd_externals/install \
  -DPython_EXECUTABLE=/home/harry/projects/DNDSR/venv/bin/python
```

CPU compiler: GCC 13.3.0. MPI: Open MPI 5.0.9. CUDA compiler: toolkit 13.2. Device: NVIDIA A100 80 GB PCIe. Native CUDA computation uses device 0, with an additional wrong-stream/device rejection check using device 1. This does not establish cross-device transfer or multi-GPU solver support.

## Implemented scope

- Native single-allocation host/CUDA Buffer, allocator policy, shared alias controller, const/mutable spans, move-only access leases and CUDA completion events.
- Explicit copy operations and allocation retention after handles are replaced; no compulsory host mirror in the native path.
- Existing Array host allocation backed by Buffer, preserving row APIs and deep-copy behavior. Native host access guards coexist with an explicitly documented legacy raw-access boundary.
- Dense Field layout/distribution metadata separate from storage; immutable ExchangePlan routing with request-owned leases, forward copy and transpose addition using host MPI.
- Selective mesh dependency generations, scoped mutations, shared control for shallow mesh copies and fail-closed abandoned transitions. Existing reorder methods are guarded collectively.
- ReorderEntitiesWithFields applies one mesh correspondence to explicitly registered legacy/native fields. The native host adapter transfers stable IDs and histories, increments distribution epochs and retains old field aliases. It does not perform interpolation or automatically discover solver state.

No PETSc or hypre dependency was added. hypre remains the chosen optional AMG integration target for a subsequent solver-layer change.

## Validation record

All four CPU aggregate targets built successfully after the final source changes. The selected 100 CPU CTest entries have passing results across the broad run and focused reruns; this is not a claim that the first broad invocation was completely green.

| Check | Result | Evidence |
|---|---|---|
| Broad DNDS/Geom/CFV/Euler CPU run | 98 passed; CGNS generator dependency lookup failed; reactive split-indicator executable was not built by the existing aggregate target | [Original broad log](array_geom_validation_2026-10-20/cpu-broad.txt) |
| Final CPU refresh and focused rerun | 11/11 passed, including both corrected checks and all new runtime/exchange/reorder tests; MPI np=1,2,4,8 | [Final focused log](array_geom_validation_2026-10-20/cpu-final-focused.txt) |
| Final CUDA refresh | 3/3 passed: native runtime and legacy ArrayDOF np=1,2 on A100 | [CUDA log](array_geom_validation_2026-10-20/cuda.txt) |
| Diff integrity | git diff --check passed | Run from implementation worktree |
| Plan integrity | 23 original topics, 23 traceability rows, 17 Markdown files, 57 local links, 2 JSON files, 29 matching source records; original brief hash unchanged | Plan validator at commit 1c2b1fe |

The CGNS issue was test setup, resolved as described above. The missing euler_test_reactive_split_indicator aggregate dependency was added to test/cpp/CMakeLists.txt so subsequent euler_unit_tests builds include the registered test. No numerical test failure remained. The main checkout's generated fixture SHA-256 values remained unchanged: 2x2 `33888c99c33b96c295d60d5c6bd4ecf6ee6c74faf4525942d2878fb27e06ad3c`, 3x3 `1046445d8bbff6131a63539a07291d76098821896cdfec9388793fb5a61bf94a`.

```sh
cmake --build build-refactor --target dnds_unit_tests geom_unit_tests cfv_unit_tests euler_unit_tests -j8
ctest --test-dir build-refactor --output-on-failure -R '^(dnds_|geom_|cfv_|euler_)' --timeout 180

# After correcting fixture lookup, aggregate dependencies and refreshing final headers:
cmake --build build-refactor --target dnds_unit_tests geom_unit_tests cfv_unit_tests euler_unit_tests -j8
ctest --test-dir build-refactor --output-on-failure \
  -R '^(dnds_runtime$|dnds_runtime_exchange_np[1248]$|geom_mesh_reorder_np[1248]$|geom_mesh_cgns_multizone$|euler_reactive_split_indicator$)' --timeout 180

cmake --build build-refactor-cuda --target dnds_test_runtime_cuda dnds_test_array_dof_cuda -j8
ctest --test-dir build-refactor-cuda --output-on-failure \
  -R '^(dnds_runtime_cuda$|dnds_array_dof_cuda_np[12]$)' --timeout 120
```

The runtime tests exercise conflicting aliases, allocation replacement, deep versus shallow copies, zero-sized storage, allocator lifetime, selective cache invalidation, abandoned mutations, shallow mesh controller sharing, explicit CUDA copies/kernels/completion and invalid stream/device access. Exchange/reorder tests cover empty ranks, duplicate leaves, transpose sums, mismatched collective modes/widths, rank-local access conflicts, stale routing stamps, local permutations, registered native IDs/histories and collective mesh read exclusion. Existing CFV reconstruction/limiter and Euler gas/RANS/reactive/evaluator tests provide CPU compatibility coverage.

No Python tests were run: pybind modules were not rebuilt or installed in this worktree. Existing NumPy/raw C++ aliases retain legacy access semantics. No solver-scale performance measurement, full CUDA Euler port, GPU MPI exchange, conservative overset remap or hypre integration is claimed. The remaining migration items are in the architecture guide.

No solver implementation commit or push has been requested. The design commit is local; no additional plan push was performed.
