# Bounded CFV/Euler query migration, 2026-10-24

This is a bounded consumer migration supporting legacy host storage and explicit
native publication. Legacy Python properties remain compatibility access, and
the reference device views remain unchanged. The
[checkpoint report](geom_refactor_completion_2026-10-24.md) records the phase's
validation and retained adapters. Source presence and syntax compilation alone
are not runtime or numerical-equivalence evidence.

## Migrated operators

One retained read scope surrounds each migrated numerical operator. Its plain
view supplies rows, counts, element/zone information, physical coordinates, and
face-side queries to the existing loops; there is no per-element lease creation.
Numerical formulas and iteration order are preserved.

- CFV finite-volume geometry and reconstruction coefficient setup, reconstruction
  sweeps/gradients/SOR, smoothness indicators, WBAP limiters, and model RHS.
- Euler timestep evaluation, LU/SGS operations, initialization, reconstruction
  limiter/positivity routines, reactive split-indicator stencil, residual/local
  step smoothing, wall-distance collection/AABB/Poisson, and BC anchors/profiles.
- CFV and Euler face/cell helpers use semantic connectivity, element/zone,
  coordinate and periodic queries. Their cached native host view has no
  per-call lease or version lock; callers retain the outer operator scope.
- Local LU neighbor and partition queries use `LocalCellNeighbors()`,
  `NLocalParts()`, and `LocalPartStart/End()`. Native publication retains these
  setup-derived vectors independently of the editable legacy workspace.
- Checkpoint writers/readers retain a mesh read and use `OriginalEntityId()`
  for cell correspondence. The writer materializes a temporary original-ID
  array to preserve the existing checkpoint schema without exporting raw
  `cell2cellOrig` from the detached workspace.
- `EulerSolver::PrintData` prepares volume/point/boundary quantities through a
  retained volume query view. Deferred output closures retain the shared scope
  through execution. The worker moves its retained captures into local variables
  so a completed `std::future` cannot indefinitely retain the mesh lease.
  Native authority supports distributed volume output with synchronous detached
  export preparation. Serial-reader output and reduced boundary output still
  require legacy authority; the retained query scope alone does not certify
  those separate derived output objects.
- The Python mesh-drawing demo reads physical element-coordinate snapshots;
  the basic Geom wall-distance check uses a managed read before device transfers.
- OversetCart geometry packing, boundary extraction, distance/interpolation
  geometry and face-midpoint queries use managed reads. Entity descriptors
  supply owner, owner-local-row and local/global conversions for requests;
  geometry packing inside loops borrows one caller-retained scope. Cell-field
  indexing uses `Geom.borrow_legacy_field_indexing` through a named adapter,
  validating and cloning canonical entity maps in either authority mode.

Operators that dereference both cells of every captured face request
`allFaceCellsLocal`. Generic owned-cell stencils retain the narrower `CellFlux`
closure contract. Periodic unmatched boundary faces use `UnInitIndex` through the
facade; they are checked before face dereference. Self-periodic side selection
continues to use the actual cell-face slot.

CFV face geometry, periodic variable transforms, and anisotropic functional
helpers use semantic mesh queries under the enclosing operator's retained read
scope. Published native queries use their cached immutable host view without
per-element leases. Full metric construction requires publication with
`allFaceCellsLocal`; wall-distance bases additionally require `wallDistance`.
Legacy wall-distance cache validation retains its buffer identity/version
checks, while immutable native distances depend on the mesh publication epoch.
`WriteSerializeRecMatrix` remains an explicit legacy workspace serializer and
rejects native authority before changing auxiliary mesh state.

## Cache publication and geometry edits

FV metrics depend on topology, geometry, distribution and layout, with a separate
publication generation for a rebuild in the same mesh epoch. Granular setup
must finish with `FinalizeMetrics()` (`finalize_metrics()` in Python). Each
stage requires its current prerequisites; rebuilding an input clears completion
bits for all dependent stages. Matching array sizes and an earlier completed
bundle cannot satisfy an abandoned rebuild. `SetCellAtrBasic()` and the full
`ConstructMetrics()` entry start a fresh construction sequence.

Reconstruction bases additionally record the metric publication generation;
coefficients require valid bases. Full metric, base and coefficient rebuilds
clear validity before fallible work and publish only after success. Repeated
base construction clears boundary-point caches. Rebuild in the order
`ConstructMetrics()`, `ConstructBaseAndWeight()`, `ConstructRecCoeff()` after a
geometry/order change or settings change. Mutable legacy attribute/coefficient
references remain compatibility access: their caller must serialize edits and
perform the affected rebuilds.

Euler's `dWall`/`dWallFace` cache is distinct from mesh `nodeWallDist`. It records
the mesh identity, topology/geometry/distribution/layout stamp, FV identity and
metric publication generation. RANS and extended-model RANS operators validate
it once at their outer boundary. `GetWallDist()` clears its stamp before work
and publishes after both cell and face distances succeed. Rebuilding FV metrics
therefore requires `GetWallDist()` again, even if row counts are unchanged.

Solver startup uses `ApplyGeometryTransform()` for proper rotation and positive
uniform scaling. This transforms coordinate and nodal wall-distance vectors
and updates the periodic frame consistently. `RectifyBoundaryCoordinates()`
selects components by boundary zone, reduces selections to node owners and
refreshes coordinate ghosts. The optional generic `TransformCoords()` snapping
step and boundary rectification preserve pre-snap wall-distance metadata;
applications requiring geometric distances afterward must rebuild them.
These managed edits declare mesh generations and reject retained managed reads.
They do not automatically rebuild FV, reconstruction or Euler wall caches.

## Explicit compatibility allowlist

These are bounded migration exceptions, not a claim that arbitrary solver mesh
member access is safe:

| Boundary | Existing access and caller obligation | Remaining migration |
|---|---|---|
| Standalone numerical helpers | CFV/Euler helpers dispatch through semantic queries, and output-picker boundary callbacks use `BoundaryCells()`/`BoundaryFace()`. They borrow the caller's scope; callback registration does not pin a read for the evaluator's lifetime. | Independent callers must retain an equivalent read scope or use a serialized setup phase. |
| Local symbolic solver caches | Mesh neighbor and partition vectors have semantic native backing. Factorization structures such as `symLU` and its row-position correspondence remain evaluator/setup caches. | Rebuild the evaluator's symbolic/factorization data after topology, distribution or ordering transitions; native mesh publication does not rebuild solver objects. |
| Reduced boundary mesh export | Boundary meshes currently publish the reduced `ForBnd` topology, which does not meet the full primary query contract. `PrintData` retains their version read lease through deferred export; `cell2parentCell` and boundary-zone helpers remain in the named export adapter. | Add a reduced boundary query/export view with explicit parent identity. |
| Visualization export | Native solver `PrintData` supports distributed volume output synchronously. Geom writers retain current query authority, prepare an explicit output workspace and map point fields by entity identity. Source boundary reverse caches may be explicitly dropped from the output copy only. | Native serial-reader and reduced boundary output require separately refreshed adapters and currently reject. Native setup runs before/background-free on the source communicator; prepared legacy VTKHDF writers use their isolated output communicator. |
| Serialization and restart | `OriginalEntityId()` supplies current checkpoint correspondence under a retained read; the existing field/file schema and legacy field serializers remain. No query view exposes a transformer. | Field identity, redistribution policy and persistent checkpoint semantics remain explicit caller/I/O contracts. |
| Startup and geometry setup | Mesh reading, partitioning, elevation/bisection, boundary construction and serial-reader routes remain setup orchestration. Rotation/scaling and boundary rectification now use managed mesh methods. Generic callback transforms do not maintain every dependent geometric field. | Serialize setup; use declared transition/edit APIs and rebuild affected mesh/FV/evaluator caches before resuming operators. |
| Legacy device references | `Mesh_DeviceView.hpp` and legacy Array device views remain unchanged references. EulerP now uses `UnstructuredMeshView` through a retained native snapshot adapter; CFV's device view accepts either mesh-view type. | Keep FV/physics device placement and metric rebuilds under their existing phase barrier; the adapter currently retains default-stream work, not arbitrary external stream usage. See `EulerPNativeMeshSnapshot.md`. |
| Legacy FV device benchmarks | `BenchmarkFiniteVolume.hpp` still obtains legacy mesh rows through `FiniteVolume::deviceView`. That path rejects native authority before returning a legacy mesh view. | These benchmark entrypoints are guarded compatibility consumers; production EulerP uses the retained semantic view above. |
| Python compatibility and typed field setup | Legacy array properties and the existing post-device-transfer basic assertion remain outside managed host queries. Overset query/identity helpers and scalar field indexing are migrated. The field bridge supports scalar/dynamic-width real transformers only; sizing, MPI requests and rebuilding bindings after numbering transitions remain caller obligations. The native host bridge deliberately rejects an attached legacy device mirror. | Extend typed bridges only for concrete consumer needs, and finish the supported device-placement workflow before replacing remaining compatibility paths. |

The current supported Python spelling is `mesh.acquire_read(...)`; no
`mesh.read()` alias is required by the implementation. Retained readonly NumPy
rows/slices extend the scope; physical coordinate snapshots do not. Release all
shared rows before tracked collective mesh mutation.

## Validation boundary

Representative manual syntax checks use existing configured include/definition
flags and write `/tmp/*-consumer2-syntax.log`, `/tmp/cfv-stage-*-syntax.log`,
`/tmp/euler-wall-*-syntax.log`, and `/tmp/euler-native-oracle-syntax.log`; they do
not rebuild shared libraries. Rebuild/install every affected pybind module
before Python execution.
Existing independent CFV constant/polynomial reconstruction, limiter, Euler
pipeline, periodic, and restart tests remain the numerical/runtime gates. The
new facade tests independently verify ordered CGNS geometry, source mapping
identity, strict closure, invalid element shapes, retained host/CUDA lifetimes,
and legacy/native halo values.

Added cache tests cover rejected stale prerequisites after a geometry change,
transitive invalidation after same-epoch granular input rebuilds, failed full
metric publication, and RANS wall-cache rebuilding after geometry/metric
publication changes. The Euler pipeline oracle restores complete initial
solution/reconstruction halos, then compares every owned timestep, RHS and
increment value between legacy authority and native authority after detached
workspace corruption. Its fixtures cover periodic 2-D NS, viscous SA, and
periodic 3-D NS; two-equation and variable-species instantiations have separate
syntax checks. These descriptions specify coverage in source, not test results.
Runtime outcomes must be read from the integration logs for the built revision;
this note does not assert that these tests were executed by its author.
The headless overset regression compares travelling cell/boundary geometry and
global node IDs before/after native publication and deliberate legacy geometry
corruption. It also checks immutable boundary coordinate snapshots and ordinary
and periodic face-coordinate equivalence, without rendering or an overset solve.
Typed field tests perform actual scalar/dynamic-width halo pulls after detached
workspace map corruption, compare legacy/native values and reject stale entity
descriptors, row-count mismatches and incompatible communicators. The multilayer
Geom test now reads its neighbor closure through a managed query scope.

The CFV native-authority regression reconstructs metric and coefficient caches
after deliberately changing detached raw coordinates, face connectivity/zones,
periodicity, and wall distances. It compares legacy/native metric values,
coefficient matrices, periodic face transforms, and a nonzero reconstruction
iteration for periodic centroid-difference and wall-distance functionals.
