# Geom facade and reorder validation evidence — 2026-10-24

Evidence directory for the GE1 facade slice and reorder follow-up. Logs are
kept compact as plain text; no logs have been removed. Commands below describe
the recorded invocations in `build-refactor` / `build-refactor-cuda`; do not
interpret this as a claim that every listed gate is complete.

## Baseline before facade changes

`baseline-cpu.txt` records the pre-change affected CTest baseline:

```sh
ctest --test-dir build-refactor --output-on-failure -R '^(geom_|cfv_|euler_)'
```

Result: 49/49 registered tests passed in 161.64 s. This is the saved source
baseline at commit `232df1a069ae7aecc442b7ed71bb98fe8cf78768`; it is not the
post-change validation result. `configure-cpu.txt` and `configure-cuda.txt`
record configuration only.

## Targeted failure reproductions and follow-ups

The `*-before` and intermediate `*-after` logs preserve failures used to drive
the reorder fixes. Keep them as historical evidence; the failure is intentional
in the sense that the targeted case was run to reproduce a known gap, not that a
passing result was expected from that source state.

| Log | Recorded scope / outcome |
|---|---|
| `reorder-face-before.txt` | `mpirun -np 2 build-refactor/test/cpp/geom_test_mesh_reorder --test-case="ReorderEntities: Face-explicit identity and arriving cell rows"`; reproduced invalid target face lookup / rank failure. |
| `reorder-face-after-np2.txt` | Same focused case after an intermediate change; still failed while resolving `face2cell` against relocated cell rows. |
| `reorder-follow-before.txt` | Focused `FollowSpec Bnd follows Node via bnd2node` MPI-2 case; reproduced invalid `bnd2cell` state. |
| `reorder-follow-after-np2.txt` | Follow-up intermediate state still failed the full face/cell consistency assertion. |
| `reorder-coverage-before.txt` | Targeted target-face coverage check failed before coverage/fix work. |
| `reorder-periodic-initial-check.txt` | Initial periodic-preservation check passed; a later assertion in the same log exposes invalid face-to-cell indices in another targeted path. Read the complete log; do not summarize it as an all-pass run. |
| `reorder-build*.txt` | Iterative targeted rebuild records; some files end at target link progress, and are not test evidence. |

Final targeted records:

- `reorder-coverage-after-np2.txt`: MPI-2 target-face coverage regression, 1
  case passed, 36 other cases skipped by test-case filter.
- `reorder-face-final-np2.txt`: MPI-2 face-explicit identity and arriving-cell
  reorder checks passed (2 cases, 22,766 assertions); 35 unrelated cases
  skipped by test-case filter.
- `reorder-follow-final-np2.txt`: MPI-2 follow case passed (1 case, 2
  assertions); 36 unrelated cases skipped by test-case filter. This test checks
  the follow behavior but does not establish full ghost/sons reconstruction.

These targeted successes close only their focused assertions. In particular,
they do not establish every source-documented distributed face movement or
post-reorder ghost-rebuild scenario.

## Final full reorder suite

The `reorder-full-final-np{1,2,4}.txt` logs are the final all-case reorder
executable runs at each listed rank count. Invocation form:

```sh
mpirun -np 1 build-refactor/test/cpp/geom_test_mesh_reorder
mpirun -np 2 build-refactor/test/cpp/geom_test_mesh_reorder
mpirun -np 4 build-refactor/test/cpp/geom_test_mesh_reorder
```

Results: 37/37 cases passed and 0 skipped for np=1, 2, and 4. Assertion counts
are respectively 2,639,781; 23,866 (one rank's report; companion report
25,088); and 12,424 (companion reports 12,560 and 12,583). MPI ranks can each
print a doctest summary, so repeated summaries are rank-local output, not
separate invocations. `reorder-full-iteration5-np2.txt` is an intermediate
failed run and is retained for history.

The full executable currently has rank/fixture guards in individual test
bodies. A zero `skipped` count means doctest selected all test cases; it does
not prove every guarded branch ran on every rank. The new targeted MPI-2 face
test specifically covers one previously guarded path. The older source comment
about a broader face pull-set gap and the follow test's limited ghost-state
coverage remain relevant when claiming broader closure.

## Historical initial facade build

`facade-build-cpu.txt` and `facade-build-cuda.txt` are live append-only build
logs for the initial facade targets. The CPU build exited with both
`dnds_test_legacy_array_bridge` and `geom_test_mesh_facade` built successfully;
its log has no compiler or linker errors. The CUDA command exited nonzero after
successfully building the `geom` and `dnds` libraries. It failed compiling
`test_MeshFacade_CUDA.cu`: a compile-time static assertion at
`src/Geom/Mesh/MeshView.hpp:112` says device coordinate output requires bounded
inline storage (`MeshQueryCoords`). The instantiation path reaches `LoadCoords`
and `GetCoordsOnCell` at `MeshView.hpp:217,211,209` with
`TResult=Geom::tSmallCoords`. Therefore `geom_test_mesh_facade_cuda` did not
build; this is a CUDA compile failure, not a device-test result. Neither CPU
target's runtime tests are represented by these build logs.

## Integrated checkpoint, before native authority and detached transitions

The initial CUDA coordinate-output error above was fixed by using bounded
`MeshQueryCoords`. Later results supersede that initial build limitation:

| Record | Result and exact boundary |
|---|---|
| `integration4-geom-tests.txt` | 10 passes: snapshot, relation, mutation and real-mesh restart/pipeline at MPI 1/2, CUDA query and two-GPU direct-MPI overlap. |
| `integration5-geom-regression.txt` | 12 passes: reorder, facade and inline/worker overlap at MPI 1/2/4/8. |
| `integration5-geom-high-ranks.txt` | 8 passes: snapshot, relation, mutation and pipeline at MPI 4/8. |
| `integration5-native-facade-memcheck.txt` | Native query/lifetime CUDA cases: 9 cases, 454 assertions, zero memory errors; excludes the separate legacy parity case. |
| `integration5-facade-memcheck.txt` | All 10 case assertions pass, but the legacy parity path reports a CUDA/CUB duplicate-kernel registration diagnostic. This is not a clean sanitizer result. |
| `integration5-consumers-cuda-build.txt` | Real CUDA EulerP shared consumer, CFV reconstruction and legacy device-transfer test build successfully. |
| `integration6-geom-tests.txt` | Fresh CPU Geom and bridge matrix: 60/60 passes, including native transient inverse/ghost routes at MPI 1/2/4/8. |
| `integration6-profile-cpu.txt`, `integration6-python-cpu.txt` | Installed ELF identities verified; CPU Geom/CFV/EulerP Python: 29 passed, one explicit CUDA-only skip. |
| `integration6-python-consumers-np2-fixed.txt` | MPI-2 facade/CFV/EulerP Python: 27 passed, one CUDA-only skip on each rank. The barycenter test now checks the global union rather than expecting each partition to contain every coordinate. |

The full MPI-2 Python basic Geom run exposed a separate multilayer local-reorder
lookup failure (`integration6-python-multilayer-repro.txt`). The retained ghost
list includes deeper rows absent from the owned-row incidence pull set. Its fix
and closing regression are in progress at this checkpoint. The earlier broad
CUDA CTest attempt (`integration6-cfv-cuda-tests.txt`) started before requested
executables existed; its launch failures are orchestration failures, not kernel
test results. The subsequent built reconstruction run passed MPI 1/2/4, then
exposed a singleton-partition test setup at MPI 8; the tiny nine-cell fixture
must not request a two-way local Metis split. Closing runs are recorded separately.

These are scoped checkpoints, not full GE0–GE6 completion or performance results.

## Native publication, edits, output and transitions: integrations 18–21

These records supersede the early feature-specific failures for the listed
scope. The typed entity-field binding and native ghost/interpolation setup work
started afterward; these results do not validate that later source.

| Record | Result and boundary |
|---|---|
| `integration18-new-tests.txt` | 16/20 focused registrations passed; four managed-edit failures exposed globally numbered legacy inverse setup rows. |
| `integration19-edit-tests.txt` | Corrected inverse selection: managed edits passed at MPI 1/2/4/8. |
| `integration18-euler-tests.txt` | 17/17 Euler registrations, including native-authority residual/timestep/update and volume-output comparisons. |
| `integration19-python-cpu.txt`, `integration19-python-cpu-np2.txt` | 54 passed, one CUDA-only skip per rank. Includes installed Geom read/edit, headless overset, CFV and EulerP. Numba was installed in the checkout-local interop venv; overset cases executed. |
| `integration19-profile-cpu.txt` | Imported CPU extension paths and ELF build IDs matched the selected build. |
| `integration19-python-cpu-np4.txt` | Not an accepted source/binary pair: guard tests changed during execution against earlier installed binaries. Preserve this failed invocation; rerun after final build/install. |
| `integration20-checkpoint-tests.txt` | MPI 1/2/4/8 coherent mesh-plus-four-history H5 checkpoint after detached redistribution; exact original-ID field/coordinate oracle and rebuilt ghost/faces. |
| `integration21-geom-tests.txt` | 83/83 Geom registrations, including native mutation rejection and malformed self-periodic side rejection. The subsequent two-mesh destination guard and typed/DSL extensions require their own builds/tests. |
| `integration21-dnds-tests.txt` | 55/55 DNDS registrations, including the Array bridge and direct transient/persistent transfers on SELF and split communicators. |
| `integration21-cfv-tests.txt`, `integration21-euler-tests.txt` | 9/9 CPU CFV and 17/17 CPU Euler registrations. |
| `integration21-gpu-tests.txt` | Native facade/legacy parity and two-rank direct-MPI CUDA mock passed. One rank is assigned to each physical GPU. |
| `integration21-native-memcheck.txt` | Native cases pass their assertions, but MPI CUDA-accelerator shutdown reports `cuMemHostUnregister` error 713. This is not a clean sanitizer run. |
| `integration21-native-memcheck-no-mpi-accelerator.txt` | Query-only native CUDA run with `OMPI_MCA_accelerator=^cuda`: 9 cases, 768 assertions, zero memory errors. The one legacy parity case is excluded because of the separately recorded registration diagnostic. This setting is not used to claim GPU-aware MPI coverage. |

The broad invocations are `ctest --test-dir build-refactor -R '^geom_'`
(and corresponding `^dnds_`, `^cfv_`, `^euler_`), with
`--output-on-failure`. The GPU invocation selects
`^geom_mesh_facade(_cuda|_overlap_cuda_np2)$` in `build-refactor-cuda`.
For the query-only sanitizer invocation, use the checkout-local
`test/demos/one_rank_per_gpu.py` launcher with `mpirun -np 1`,
`compute-sanitizer --tool memcheck --error-exitcode 99`, and
`--test-case-exclude='Native mesh CUDA queries match existing const legacy pair views'`.

Legacy `Mesh_DeviceView.hpp`, `ArrayPair.hpp` and `ArrayAdjacency.hpp` remain
unchanged against the Array phase baseline `232df1a0`. New views have their own
device compile/execute coverage; legacy read-only CUDA support is preserved.
Native authoritative publication, detached mesh/field transitions and final
post-change regression remain active work. Keep rank/fixture/hardware guards and
explicit skips visible when reporting coverage. Earlier failed logs are retained
and must not be combined into a fictitious all-pass run.
