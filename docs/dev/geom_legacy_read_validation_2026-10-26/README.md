# Legacy read facade validation

The [implementation report](../geom_legacy_read_churn_2026-10-26.md) describes the
native-only storage contract and restored numerical read spellings. This evidence
pins the validated working tree over `1841a2e2`, compared against base `03891f8b`.
The original manifests retain that pre-publication snapshot. The publication
manifests/table separately record required formatting; this is not a merge into
`dev/harry`.

| Gate | Result | Evidence |
|---|---|---|
| CPU facade tests, MPI 1/2/4/8 | 4/4 registrations pass | `geom-legacy-read-cpu-facade-tests.txt` |
| CPU CFV/Euler/facade/pipeline/overlap | 38/38 registrations pass | `geom-legacy-read-cpu-consumer-tests.txt` |
| CUDA mesh/overlap/entity fields | 3/3 registrations pass | `geom-legacy-read-cuda-tests.txt` |
| CUDA CFV | 1/1 registration passes | `geom-legacy-read-cuda-cfv-tests.txt` |
| CPU Python MPI 1/2/4/8 | 46 passed + 3 skipped per rank | `geom-legacy-read-python-cpu-matrix.json` and np logs |
| CUDA Python MPI 1/2 | 48 passed + 1 skipped per rank | `geom-legacy-read-python-cuda-matrix.json` and np logs |
| Installed binary identity | Eight IDs match each profile; final CUDA restored | profile JSON files |

The focused host invocation selects two of the four new facade cases and passes
308 assertions; it supplements, not replaces, the full facade registration. CUDA
parity uses independent legacy const-array reference data and native pointer
identity/lifetime gates. The Python suite adds a full CPU-versus-CUDA EulerP RHS
comparison, covering its normal argument-owner/default-stream completion path.

CPU skips are the CUDA-only mesh case, CUDA RHS case, and either the single-rank
mixed-authority case or the serial-only WaveTester fixture. CUDA skips only that
mixed-authority/WaveTester choice. No discovery/compile-only result counts as a
runtime pass. GPU Python's launcher records unique physical UUIDs before imports.

`source-manifest.json` pins all 21 changed/new source and test files. The
[consumer table](consumer-diff.md) and [JSON](consumer-diff.json) retain exact
per-file before/after counts for the 28-file source inventory. The new support
header/owner integration and tests are outside those consumer totals.
`contract-source-audit.json` records unchanged acquisition/dependency/bridge/stage
marker counts as an audit aid, not a standalone correctness proof.

[Publication counts](publication-consumer-diff.md) include 146 restored
whitespace-only EulerP declaration/comment changes: 2,288 total changed lines,
2,092 ignoring whitespace. `publication-format-audit.json` compares Clang raw
token spellings (including literals) across the five modified source paths.
`publication-source-manifest.json` pins final formatted source. The original
`source-manifest.json` remains the runtime-validated pre-format snapshot.

`artifact-manifest.json` records original and stored SHA256 values. Text logs
remove ANSI escapes, escape NUL bytes and strip trailing whitespace. Build outputs
and the two unrelated OpenFOAM drafts are excluded. The first CUDA compile failed
on an extended-lambda restriction; `build-cuda2` and the CUDA binding build pass
after the empty-functor correction. Historical external sanitizer diagnostics
have not been suppressed or relabeled as fixed by this phase.

Python profiles share one installed package. Each has all four bindings rebuilt
and CMake-installed, followed by explicit copies of the eight selected binaries
and ELF-ID verification to avoid timestamp-based profile mixing. The final
installed profile is CUDA. The retained scripts and matrix command records show
the exact local paths, MPI sizes, profile choices and physical GPU assignment.
