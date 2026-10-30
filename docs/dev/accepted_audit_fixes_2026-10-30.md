# Accepted retrospective fixes and next work orders

2026-10-30. Follow-up to [the retrospective](refactor_retrospective_2026-10-30.md).
The earlier defect table is historical; its field-policy failure is corrected
here. This phase also commits the preceding audit's five local fixes.

## Implemented

1. **Storage-oriented EntityField.** Only element type belongs to the handle.
   Layout projection is selected at read/write access. Migration/halo storage
   uses an explicit Runtime TransferLayout; compact CSR is default, while table
   stride and padded capacity can be retained or explicitly selected. Incoming
   shape failures are collectively rejected before payload/publication. Source
   data and retained old scopes survive. See [the public contract](../architecture/EntityField.md).
2. **Immutable compiled ghost programs.** Private shared data contains diagnostic
   roots and executable levels. Only compilation constructs it; const inspection
   replaces public mutable vectors. Agreement uses a complete execution signature
   including collection flags and semantic descriptors. The assembler uses an
   optional result until collective compilation succeeds, not an invalid default
   program. Existing branch/legacy fixture tests retain their original purposes.
3. **Narrow MPI helpers.** DNDS/MPI/CollectiveStage.hpp contains local-preparation
   error agreement and candidate agreement. Connectivity no longer includes field
   migration just to obtain these helpers. Existing field helper names forward
   to them for compatibility. No top-level library was introduced: Runtime,
   Algebra and MPI helpers remain in DNDS.
4. **Write-check placement verified.** DofField's width check runs before
   SubmitArrays in the host operation prologue. CPU and CUDA numerical loops
   are unchanged. This establishes no per-element overhead, not a benchmark
   claim of identical timing for tiny submissions.

The generic adjacency owner already used Runtime Arrays; this correction removes
the remaining access-policy/storage-policy coupling in associated field migration.
Dense storage remains appropriate for fixed-width hot fields. Cartesian numerical
specialization and spatial-interface managers are planned, not implemented here.

## Tests and evidence

Logs: ignored `data/accepted-audit-fixes-20261030/`.

- CPU: 28/28 registrations at MPI sizes 1/2/4/8, with no skips: 24 for entity
  fields, local/distributed reorder, compiled ghost traversal, native
  connectivity, mesh transition and DOF algebra; four for the CFV reconstruction
  consumer, including the wall-dependent cache regression.
- CUDA: 2/2 registrations. The entity-field test verifies distinct devices with
  UUIDs and uses one rank per GPU, direct CUDA-aware MPI halo, typed device views
  and detached field migration (1,294 assertions per rank). DOF algebra passes
  77,940 assertions. No skips or new sanitizer-clean claim.
- New tests check table/padded/CSR payloads by original entity ID across migration
  and halo, retained old values, static stride/capacity views, rank-local capacity
  rejection/recovery, immutable program inspection and mismatched endpoint flags.
- Initial builds caught an assembler default-construction site and a test CAPTURE
  macro arity error. Both were corrected; failed logs remain alongside retries.

Python validation rebuilt and installed all four production modules and the
native bridge before testing eight Array/Geom/CFV files: np1 and each np2 rank
passed 87 tests, with 22 skips requiring the CUDA Python profile. The real CUDA
C++ gates above are separate evidence. The prior audit's focused I/O/CSR/cache
regression results remain in its own record.
No new performance benchmark was run; the prior parallel timing target is not
redefined by these correctness tests.

## Next phase

The standalone plan repository now separates the executable work orders:

- `docs/28-fv-facility-work-order.md`: compositional CFV frontend; extraction and
  reorganization of local setup mathematics; native plans/workspaces/operators;
  simplified Cartesian Taylor bases and shared reference quadrature tables.
- `docs/29-geom-spatial-interface-work-order.md`: generic boundary geometry,
  stencils, spatial matching and transfer routes; single-rank point interpolation
  first, with distributed matching/updating contracts preserved.

Geom can provide these generic services without owning solver domain scheduling
or depending on CFV. No conservative interface flux is required for the initial
point-interpolation milestone. Full solver, CUDA numerical CFV and tree mesh
implementation remain separate work packages.
