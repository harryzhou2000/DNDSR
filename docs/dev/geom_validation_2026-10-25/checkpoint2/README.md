# Native import, topology, refinement and archive checkpoint

This is an intermediate GN2–GN6 checkpoint, **not GN7 completion**. The work
ledger describes the production paths still requiring migration. Tests ran in
`/home/harry/projects/DNDSR-array-geom`, using `build-refactor` and
`build-refactor-cuda`, Open MPI 5.0.9 and `OMP_NUM_THREADS=2`.

## Recorded checks

- `geom-native-checkpoint2-cpu-tests.txt`: 32/32 CTests at MPI 1/2/4/8 for native
  construction, connectivity, relation registry, transition, archive, topology,
  refinement and MeshEdit. Tests preserve exact oriented rows and persistent IDs;
  they include empty ranks, remote node owners, malformed archive/provenance
  rejection, full node-star repair and conflicting high-order face-node IDs.
- `geom-native-checkpoint2-legacy-tests.txt`: 16/16 CTests for independently
  rebuilt existing mesh pipeline, distributed reader, overlap and new native
  CGNS. The large pipeline fixture includes Ball2. The standalone CGNS multi-zone
  result is in `geom-native-checkpoint2-multizone-tests.txt`.
- `geom-native-elevation*-tests.txt`: earlier focused refinement/topology passes.
  These are retained evidence, not added to the latest test count.
- `geom-native-checkpoint2-dsl-tests.txt`: 16/16 CTests for legacy connectivity,
  interpolation and ghost algorithms plus relation access at MPI 1/2/4/8. This
  specifically checks callers affected by mathematical inverse descriptors.
- `geom-native-checkpoint2-cuda-gpu1-tests.txt`: full CUDA facade suite passed
  with `CUDA_VISIBLE_DEVICES=1`, including the independent native builder stencil
  held behind a stream gate with source handles destroyed before completion.
- `geom-native-checkpoint2-cuda-tests.txt`: failed GPU-0 probe (`cudaErrorMemoryAllocation`)
  and interrupted two-rank attempt. GPU 0 briefly had about 9 GB free, but another
  process then occupied about 80 GB. The test process on GPU 0 held only about
  80 MB while its peer waited; our launcher and its test children were terminated.
  The unrelated process was untouched. Two-GPU overlap/field acceptance remains
  pending. No network speedup claim follows from these tests.
- Python `native-np{1,2,4}`: **18 passed per rank** (builder, CGNS/topology/archive,
  refinement/provenance). Existing facade/edit/basic `np{1,2}`: **18 passed and
  one CPU-profile CUDA skip per rank**. Four binding targets were rebuilt and
  the `py` component installed before testing. Imported paths, ELF build IDs,
  source hashes and exact invocations are in the Python JSON/SHA256 manifests.
  `Runtime.cuda_enabled=false`; these are CPU Python results.
- Python `initial-native-np1.txt` retains a test-oracle failure: ancestry spans
  preserve element edge orientation, whereas that initial test sorted them.
  The corrected test enumerates explicit oriented element spans. Production
  behavior was unchanged. Earlier closure/refinement cases similarly retain
  exact per-element slot oracles rather than relying only on counts.

## Reproduction

```sh
cmake --build build-refactor --target geom_test_native_mesh_refinement geom_test_native_mesh_topology geom_test_native_mesh_archive geom_test_mesh_edit geom_test_native_mesh_transition geom_test_native_mesh_construction geom_test_native_relation_registry geom_test_native_connectivity -j8
OMP_NUM_THREADS=2 ctest --test-dir build-refactor --output-on-failure -j4 -R '^geom_(native_mesh_(refinement|topology|archive|transition|construction)|native_relation_registry|native_connectivity|mesh_edit)_np(1|2|4|8)$'
cmake --build build-refactor --target geom_test_mesh_pipeline geom_test_mesh_cgns_multizone geom_test_mesh_distributed_read geom_test_native_mesh_cgns geom_test_mesh_facade_overlap -j8
OMP_NUM_THREADS=2 ctest --test-dir build-refactor --output-on-failure -j4 -R '^geom_(mesh_pipeline|mesh_cgns_multizone|mesh_distributed_read|native_mesh_cgns|mesh_facade_overlap)_np(1|2|4|8)$'
OMP_NUM_THREADS=2 ctest --test-dir build-refactor --output-on-failure -R '^geom_mesh_cgns_multizone$'
cmake --build build-refactor --target geom_test_relation_access geom_test_mesh_connectivity geom_test_mesh_connectivity_ghost geom_test_mesh_connectivity_interpolate -j8
OMP_NUM_THREADS=2 ctest --test-dir build-refactor --output-on-failure -j4 -R '^geom_(relation_access|mesh_connectivity|mesh_connectivity_ghost|mesh_connectivity_interpolate)_np(1|2|4|8)$'
cmake --build build-refactor-cuda --target geom_test_mesh_facade_cuda geom_test_mesh_facade_overlap_cuda geom_test_mesh_entity_field_cuda -j8
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 ctest --test-dir build-refactor-cuda --output-on-failure -R '^geom_mesh_facade_cuda$'
```

Python invocations use `/home/harry/projects/DNDSR/venv/bin/python` and this
checkout's `python/` package, as recorded by `python-runs.json`. Do not copy that
package's CPU result to a CUDA-profile claim. CUDA and CPU installs share a
destination and must be validated serially.

`precommit-artifacts.json` records dirty source and tested executable hashes
before the commit hook's formatting. Python has its own manifest. The hook's
formatting is not silently represented as a new executed binary. Logs copied
here have ANSI color escapes and trailing whitespace removed; test text and failure evidence are retained.
No solver trajectory or numerical-method accuracy claim is made by this suite.
