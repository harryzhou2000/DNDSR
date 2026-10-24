# Bounded CFV/Euler query migration, 2026-10-24

This is a partial GE2 migration over legacy host storage. It does not close the
approved facade phase, make legacy Python properties safe, or migrate the
reference device views. Source presence and syntax compilation are not runtime
or numerical-equivalence evidence; current build/test logs establish execution.

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
- `EulerSolver::PrintData` prepares volume/point/boundary quantities through a
  retained volume query view. Deferred output closures retain the shared scope
  through execution. The worker moves its retained captures into local variables
  so a completed `std::future` cannot indefinitely retain the mesh lease.
- The Python mesh-drawing demo reads physical element-coordinate snapshots;
  the basic Geom wall-distance check uses a managed read before device transfers.

Operators that dereference both cells of every captured face request
`allFaceCellsLocal`. Generic owned-cell stencils retain the narrower `CellFlux`
closure contract. Periodic unmatched boundary faces use `UnInitIndex` through the
facade; they are checked before face dereference. Self-periodic side selection
continues to use the actual cell-face slot.

## Explicit compatibility allowlist

These are bounded migration exceptions, not a claim that arbitrary solver mesh
member access is safe:

| Boundary | Existing access and caller obligation | Remaining migration |
|---|---|---|
| Per-face/per-cell numerical helpers | `fluxFace`, `EvaluateCellSource`, CFV `FFaceFunctional`, boundary RHS helpers, and periodic geometry/variable transformation helpers still use legacy mesh reads. Their migrated parent operators retain a read scope throughout the call; acquiring a separate scope for every quadrature point would be inappropriate. | Pass the existing plain view explicitly through helper overloads, preserving legacy entry points. Direct standalone helper callers still obey the legacy serialized-mutation contract. |
| Local symbolic solver caches | `cell2cellFaceVLocalParts` and local partition bounds are accessed while the outer LU/SGS or reconstruction scope is retained. They are solver/setup-derived vectors, outside ordinary relation storage. | Add explicit cache dependency stamps and dedicated read descriptors with the transition work. |
| Output picker callbacks | `InitializeOutputPickerBnd` stores callbacks that map boundary IDs through legacy `bnd2cell`/`bnd2faceV`; ordinary output pickers retain semantic mesh helpers. They execute under the output operation's retained mesh scope. Callback registration itself must not pin a read lease for the lifetime of the evaluator. | Bind output extractors to an operation-scoped export snapshot; independent callers must provide equivalent scope/lifetime protection. |
| Reduced boundary mesh export | Boundary meshes currently publish the reduced `ForBnd` topology, which does not meet the full primary query contract. `PrintData` retains their version read lease through deferred export; `cell2parentCell` and boundary-zone helpers remain in the named export adapter. | Add a reduced boundary query/export view with explicit parent identity. |
| Serialization and restart | Reader-owned serial-output routes and `cell2cellOrig` serialization/identity correspondence remain in `PrintRestart`/`ReadRestart` and serializer adapters. No generic query view exports their transformer. | Coherent mesh/field checkpoint snapshots and persistent identity handling belong to the transition/I/O work. |
| Geometry and periodic setup | Solver initialization and legacy Python coordinate transforms still use explicit setup/mutation APIs. Generic coordinate transforms do not refresh periodic frames, wall distance, FV metrics, or reconstruction caches automatically. | Managed geometry transactions with declared dependent refresh/publication. |
| Legacy device references | `Mesh_DeviceView.hpp` and legacy Array device views remain unchanged references. EulerP now uses `UnstructuredMeshView` through a retained native snapshot adapter; CFV's device view accepts either mesh-view type. | Keep FV/physics device placement and metric rebuilds under their existing phase barrier; the adapter currently retains default-stream work, not arbitrary external stream usage. See `EulerPNativeMeshSnapshot.md`. |
| Python compatibility and overset setup | Legacy array properties, overset distributed owner/ID lookup code, and the existing post-device-transfer basic assertion remain outside managed host queries. The native bridge deliberately rejects an attached legacy device mirror. | Expose entity descriptors/export identities in Python and a supported device-placement workflow before replacing these paths. |

The current supported Python spelling is `mesh.acquire_read(...)`; no
`mesh.read()` alias is required by the implementation. Retained readonly NumPy
rows/slices extend the scope; physical coordinate snapshots do not. Release all
shared rows before tracked collective mesh mutation.

## Validation boundary

Representative manual syntax checks use existing configured include/definition
flags and write only `/tmp/*-consumer2-syntax.log`; they do not rebuild shared
libraries. Rebuild/install every affected pybind module before Python execution.
Existing independent CFV constant/polynomial reconstruction, limiter, Euler
pipeline, periodic, and restart tests remain the numerical/runtime gates. The
new facade tests independently verify ordered CGNS geometry, source mapping
identity, strict closure, invalid element shapes, retained host/CUDA lifetimes,
and legacy/native halo values.
