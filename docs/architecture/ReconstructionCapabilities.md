# Reconstruction preparation capabilities

`VariationalReconstructionPlan::ConstructRecCoeff(metrics, settings, capabilities)`
freezes an explicit CPU operation contract. Omitting the third argument retains
the full standalone API. `ReconstructionCapabilities::ForSweeps(withWBAP)` is
the Cauchy profile: reconstruction sweeps, optional WBAP/change-of-base tables,
and the existing inverse diagnostics. Cauchy omits sweep tables for constant or
gradient reconstruction and requests change-of-base support for WBAP and active
LVR; its flow Jacobian/preconditioner is a different facility and does
not require the reconstruction linear-system tables.

| Operation | Retained numerical tables |
| --- | --- |
| Jacobi/SOR sweep | `matrixAInvB`, `vectorAInvB`, `boundaryInjection` |
| WBAP | `secondary` |
| Original operator/RHS/residual | `localMatrix`, `rawMatrixB`, `rawVectorB`, `rawBoundaryInjection` |
| Local preconditioner | `localInverse` |
| Inverse diagnostics | `inverseConditions`, global inverse summary |
| Sampling, always available | cell moments and geometry/topology descriptors |

`settings.cacheDiffBase` and its derivative order independently select cached
face-basis values. They do not imply any matrix capability. Omitted numerical
tables are not allocated, including on empty ranks. Face-system preparation
scratch also omits unused blocks: a sweep without WBAP does not construct the
two secondary change-of-base matrices. SVD is skipped when no sweep,
preconditioner or inverse diagnostics requests it. Inverse-only setup still
forms the coupling needed to correct a self-periodic cell's diagonal.

Raw matrices are assembled directly, never recovered by multiplying through
the filtered inverse. This matters when SVD truncation discards nonzero modes:
the raw operator must remain unchanged, while the preconditioner may annihilate
those modes. The public capability flags are independent and may be combined;
preparation derives their shared scratch dependencies centrally.

Each operator calls `Require` at its entry before acquiring output access or
performing communication. Missing capabilities throw; no lazy rebuild, implicit
fallback or first-use allocation exists. Capability masks must agree across
the preparation communicator; this mandatory protocol agreement remains when
optional collective safety preflight is disabled. Workspace and boundary
ownership remain bound to the exact plan publication, not just matching flags.
`Require` checks operation support only. A caller constructing a manual async
wrapper calls it before queueing work; the numerical operator still pins the
mesh and validates its generation before consuming prepared geometry. Support
checking does not replace those leases or generation checks.
Device publication currently requires both sweep and WBAP support because its
public API provides both; there is no implicit download or CPU fallback.

`StorageReport()` is rank-local and reports each retained numerical table's
logical payload bytes and underlying allocation extents, including array metadata
buffers. The total allocation bytes deduplicate shared allocation identities.
`preparationRuntimeScratchBytes`
records the face-system allocation; `preparationRuntimePeakBytes` is its sum
with the retained Runtime payload allocations that coexist during setup.
These are **not process RSS or allocator high-water marks**: shared metrics,
mesh storage, workspaces, native Eigen/std scratch and
allocator bookkeeping/pools are excluded. A zero-payload table on an empty rank
can still be a prepared table; `allocated` distinguishes that case from an
omitted table. No alignment/padding or RSS savings are inferred from logical
payload counts. Measure RSS separately in a bounded assessment.

Focused coverage lives in `test_ReconstructionCapabilities.cpp`: 2D/3D degrees
1–3, Jacobi and logically ordered SOR, active direct coefficient exchange during
core work, WBAP, rotated and self-periodic cases, empty ranks, sampling without
matrices, and independent raw/preconditioner profiles under complete truncation.
The older symmetry/Jacobi oracle remains in `test_FVCompletion.cpp`.

The scoped reconstruction basis view exposes `FaceMatrix(point, side)` as a
device-callable column-major derivatives-by-modes view. Both `(derivative, mode)`
and `[derivative][mode]` retain extent/stride information; keep the enclosing
basis read scope alive through use. This replaces the unshaped pointer accessor
without adding ownership work to numerical loops or changing cache policy.
Element shape data is separate: host setup uses
`Geom::Elem::GetCachedShapeAtQuadrature(type, order, point)` for immutable N and
parametric first derivatives. CFV/Cauchy no longer reach into that cache's
private storage. Device operators continue to use explicitly published views.
