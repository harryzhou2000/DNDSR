# DNDS, Geom and CFV: status and next implementation scope

This records the 2026-10-31 design discussion. It does **not** report new
implementation or validation. Current numerical/performance evidence is in
[the CPU facility ledger](fv_interface_2026-10-30.md); current APIs and limitations
are in [FVFacility](../architecture/FVFacility.md).

**Historical discussion:** implementation has since advanced. Read
[the completion ledger](fv_completion_2026-10-31.md) for the executed scope and
remaining limits. Statements below such as “LU” and “single-rank only” describe
the pre-execution snapshot, not the current implementation.

Later accepted decisions in the standalone plan repository, documents 31/32,
supersede proposals here: use node-star cell core classification consistently,
use full SVD without a LU fast path, correct the legacy filter first, and allow
generic entity-anchored locality ordering. Absolute cutoff defaults to zero.

## Current status

| Layer | Implemented foundation | Remaining facility work |
| --- | --- | --- |
| DNDS | Runtime arrays, checked access, shared storage, DOF algebra, eager ordering/lifetime machinery, transient/persistent transfers, serialization and Python sharing | Reuse these contracts; new CFV operators are not yet asynchronous launchers |
| Geom | Native mesh construction, adjacency registry/DSL, explicit ghost assembly, entity ghost mappings, mesh views; local boundary projection/search and stencil plans | Adjacency-derived work partitions, distributed boundary search/routing and updates |
| CFV | CPU metric/reconstruction plans, degree 1–3 isotropic VR, Jacobi/SOR, order-wise WBAP, explicit boundary values and characteristic matrices, point evaluation | Broader weights/BC parity, independent FV gradients, robust deficient-system inversion, complete periodic/axisymmetric support, caches, activation and linear operator interfaces |

Current local boundary searching rejects multiple ranks; its tests do not prove
distributed matching. Current VR supports translations, not general rotated
periodicity. Current local LU rejects deficient systems; it is not a
pseudoinverse. Current weights and default boundary penalties are a subset of
legacy behavior. Existing timing comparisons apply to the tested shared subset,
not to all additions below. No new fluid solver is included in this scope.

## Geom work partitions and exact adjacency semantics

The existing names have distinct meanings:

* `Adj::Cell2Cell` / `MeshRelation::CellCells`: cell–node–cell node-star
  adjacency, excluding the source cell.
* `Adj::Cell2CellFace` / `MeshRelation::CellFaceCells`: face-neighbor adjacency,
  stored already, with one neighbor per cell-face slot and an exterior sentinel
  at physical boundaries. `MeshView::CellFaceCells(iCell)` exposes it.

The second is the usual choice for face-stencil CFV core cells. Do not rename or
silently change the first to mean face adjacency. The construction is in
`Connectivity/NativeMeshTopology.cpp`; both descriptors are registered in
`Connectivity/AdjacencyRegistry.cpp`.

Add a Geom work-partition facility taking owned output entities and an explicit
input adjacency descriptor or composed path. A core output has only owned input
dependencies; a halo-dependent output has at least one ghost input dependency.
For example, cell–face–cell inputs are cell values; cell–node inputs are node
values. Multiple input dependencies require intersection of their core sets.
Work lists use signed indices, retain deterministic order and require no mesh
reordering. Empty sets are valid.

Classification is entirely MPI-local over already published connectivity.
Required intermediate ghost rows and terminal mappings must be complete;
missing closure is an error, never a silently skipped dependency or implicit
communication. Distinguish legitimate exterior sentinels from unresolved global
indices. Ownership alone cannot prove that an adjacency row is complete: use
the publication/closure contract as well as index validation. Cache partitions
against topology, adjacency and entity-mapping generations.

Geom proves this declared dependency partition, not the caller's actual kernel
read/write behavior, halo freshness, or boundary-data readiness. Callers must
declare every input dependency and arrange communication and execution. A
physical boundary can remain core if its supplied BC data are ready locally;
remote interface BC data introduce an additional dependency.

## Reconstruction overlap and logical SOR ordering

Expose all/core/halo-dependent work selection on heavy reconstruction and WBAP
operators. Construct owned-only input views for core execution; acquire ghost
read access only for the halo-dependent invocation. Lease/version checks belong
in submission/operator prologues, never per-cell hot loops. Lists and plans are
prepared outside iteration. Shared scratch must not permit simultaneous writers.

Jacobi and each WBAP order use immutable input and separate output. The intended
sequence is begin input halo, compute core, complete halo, compute dependent
cells, then publish output. Preserve the required coefficient exchange between
successive descending WBAP orders.

SOR gains an explicit configuration mode for **core before halo-dependent**
logical ordering, preserving relative order within each set. This changes the
local SOR permutation and must be compared with an unsplit reference using that
same permutation, not claimed bit-identical to the original cell ordering.
Keep the existing order as a separately selectable mode.

Direct MPI sends hold read access to their source owned storage. In-place SOR
must not modify that storage while sending. Use a frozen exchange snapshot and
a separate owned working iterate, or finish the sends before modifying their
rows. In the snapshot scheme, both core and dependent phases see preceding
owned SOR updates in the working iterate and the exchanged frozen remote
iterate. This is a numerical scheduling/storage decision, not MPI packing.

## Central basis, isotropic functional and caches

Centralize mode ordering, degree, evaluation of derivatives, normalization,
cell moments and frame transforms in one basis definition. Plans and kernels
consume that definition; do not duplicate basis switches across setup,
reconstruction, limiters and samplers. Start with global axes, isotropic scalar
length and zero-mean Taylor polynomials. Keep basis choice separate from the
functional and its weights, so Cartesian-specialized implementations can later
share the mathematical contract without sharing unstructured setup code.

The isotropic functional uses Frobenius products of derivative tensors:
`sum_m w_m^2 * length^(2*m) * <D^m p, D^m q>_F`.
For symmetric derivative storage, multi-index multiplicities are `m!/alpha!`.
This tensor contraction is rotationally invariant; arbitrary changes to those
multiplicities are not. Legacy compatibility options must not be described as
invariant when they change that metric. Anisotropic tensor metrics remain a
clean functional-policy extension, deferred for now.

Cache basis values **and derivatives** at cell/face quadrature points, equivalent
in purpose to legacy `cacheDiffBase`. Cache identity includes basis, degree,
geometry/frame, scaling and quadrature rule. Arbitrary interface query points
can have a separate prepared basis-evaluation cache, invalidated by motion or
matching changes.

“Gradient/Hessian sampling” simply means evaluating the first/second spatial
derivative of the reconstructed polynomial at requested points. It is not a
separate reconstruction algorithm or an extra flow solve. Keep derivative order
and coordinate frame explicit in the sampling API.

Separate nonhomogeneous-boundary quadrature controls can wait. Such a choice is
made during setup, but determines cached integrals, boundary buffer lengths and
later operator behavior; it is not merely a setup-performance switch.

## Local inversion: SVD first, certified LU only as an optional optimization

Use an explicit SVD pseudoinverse policy to remove unresolved singular modes;
FullPivLU's pivot-rank test alone does not guarantee the SVD rank decision.
Specify relative cutoff, equality convention and coefficient metric. Avoid
silently changing the minimum-norm solution through column rescaling.

A sufficient full-rank LU acceptance certificate can be based on a candidate
inverse X. If validated bounds give `rho >= ||I-A X||_2`, `rho < 1`,
`x >= ||X||_2`, and `a >= ||A||_2`, then
`sigma_min(A) >= (1-rho)/x`. Therefore `(1-rho)/(x*a) > tau` certifies that every
singular value exceeds the relative cutoff `tau*sigma_max(A)`. Otherwise use
SVD. Frobenius upper bounds are sufficient, though conservative. Roundoff in
forming the residual must be enclosed too: a floating-point residual estimate
alone is not a mathematical certificate. Until such bounds and tests exist,
retain the SVD path rather than label a heuristic LU threshold guaranteed.

Compatibility caveat requiring explicit resolution before implementation:
legacy VR uses ordinary SVD solve when `svdTolerance==0`, but a nonzero setting
calls `EigenLeastSquareInverse_Filtered(..., mode=1)`. That mode retains singular
values below a cutoff based on the **minimum**, unlike ordinary truncation of
small singular values relative to the maximum. Preserve this distinction in
tests/documentation; do not silently copy or change the unusual filter while
claiming standard pseudoinverse semantics. Standard truncated SVD and an exact
legacy-filter compatibility policy must be distinguishable.

## FV gradients, periodicity and axisymmetric geometry

Implement Green–Gauss, weighted least squares and GGMP in the FV layer, returning
gradients. VR wraps those gradients into its normalized basis coefficients.
Supply boundary sample values/positions through buffers, not callbacks. Audit
legacy ghost-value versus physical-face-value conventions explicitly; copying
a factor of one-half without that convention would change the boundary method.

For least squares, a proposed initial established choice is inverse-distance
squared weighting of squared residuals (row scaling by inverse distance).
Document length normalization and boundary weights; solve with QR/SVD rather
than relying on normal equations for difficult stencils. Test linear exactness,
skewed and stretched cells and rank deficiency. GGMP's corrected system and
iteration convention require separate parity tests.

Complete periodic support includes mapped geometry, polynomial derivatives,
vector-component block rotations and face/cell frame conventions, including
MPI-crossing periodic neighbors. Scalar fields need no component rotation.
Keep transforms explicit and reusable with a compact public interface. The
interface search also needs explicit periodic image identity when enabled.

Include axisymmetric metric integration, moments and reconstruction/gradient
consistency. Use one documented radius-weighted measure convention across
volume and face integrals; weighted divergence identities need their geometric
correction. Handle the axis by integral limits, not unguarded division by radius.
Equation source terms remain outside CFV. No new flow solver is required.

## Weights, boundaries, activation and linear evaluation

Expose independently configurable legacy weight families: directional
Factorial/HQM_OPT/TEST_OPT/manual weights and their combination controls;
geometric none/HQM_SD/SD_Power with bias/power parameters; applicable scale
choices and multipliers. Do not retain only the current bundled HQM boolean.
Document which old options require deferred anisotropic functionals. LVR
penalties are deferred explicitly, not silently approximated.

Match legacy boundary penalty orders and normalization in the new scheme.
The current new default is value-only whereas legacy defaults weight all
derivative orders; current comparisons deliberately align those settings.
Represent required BC coefficients and targets as data with clear physical
meaning. Homogeneous derivative penalties must not be advertised as supplied
nonhomogeneous derivative targets.

Accept a general scalar Runtime array and threshold for limiter activation:
`indicator >= threshold` activates the output cell. Specify nonfinite values as
errors and leave inactive outputs equal to their inputs. An inactive neighbor
is still an input dependency. Retain characteristic transforms over a selected
component block based on cell averages. CWBAP and other limiter variants are
deferred behind an explicit limiter-policy extension.

Expose reconstruction linear application, RHS construction, residual and local
preconditioner application separately from sweeps. Freeze geometry, weights,
boundary linearization and activation during a linear solve; separate affine
BC data from the homogeneous perturbation operator. Provide the field algebra,
halo contract and nullspace/projection semantics needed by GMRES and flexible
PCG. Flexible PCG additionally requires the appropriate symmetric positive
operator/subspace and admissible positive preconditioner; arbitrary left-scaled
VR operators or one-way SOR do not automatically qualify. GMRES/FGMRES is the
general alternative. Krylov drivers belong in the existing solver infrastructure,
not duplicated inside CFV, and no fluid solver is part of this work.

## Distributed boundary routing and interpolation

Use direct evaluation of the donor reconstructed polynomial as the initial
interpolation scheme. This is point interpolation, not conservative flux or FSI
force transfer. Geom owns geometry matching and entity routing; CFV owns basis
evaluation; the future solver coordinates domains and updates.

Provide two geometric-search modes:

1. All-gather a small configurable collection of coarse boxes per rank partition
   (roughly 1–10), route queries to candidates and perform exact projection and
   containment on the geometry owner. Boxes must conservatively cover the patch
   after motion. Use persistent entity/image IDs to resolve duplicate matches.
2. All-gather the selected boundary geometry as a simple correctness baseline.
   Explicitly document replicated-boundary memory cost; never replicate volume
   mesh data or introduce global-sized ghost-construction scratch.

Both modes produce equivalent donor decisions. Boundary-geometry ownership may
differ from donor-cell ownership: resolve the latter explicitly. Evaluate on
the donor-cell owner using ready means/coefficients, then transfer sampled
values to receivers through canonical DNDS transfer facilities. The baseline
need not replicate solution fields. Cache stable routes persistently; use
lightweight plans for short-lived matches. Invalidate geometry, stencil, basis
and transfer caches at their respective identity/version boundaries on sliding,
repartition or topology changes. Unmatched/ambiguous points must be reported.

## Implementation sequence and acceptance work (not executed)

1. Add Geom work partitions and core/halo CFV entry points; test face versus
   node-star adjacency, mixed entity kinds, incomplete closure, physical
   sentinels, empty/interleaved sets and mutation invalidation.
2. Centralize basis/functional definitions, add SVD policy and quadrature caches;
   independently check tensor rotation invariance and deficient-system modes.
3. Add FV gradients, full periodic transforms and axisymmetric support, with
   manufactured fields and legacy comparisons under explicitly matched options.
4. Expand weights and boundary penalties; add indicator arrays and linear
   operator interfaces, checking symmetry/nullspaces where PCG is offered.
5. Validate actual MPI/core overlap with immutable send inputs; compare split
   Jacobi/WBAP and core-first SOR against their matching unsplit references.
6. Add distributed interface routing and compare coarse boxes against replicated
   boundary matching, including moving patches, multiple candidates, empty
   ranks, donor/face ownership differences and update invalidation.
7. Recheck setup and iteration performance against matched legacy VR cases;
   preserve the distinction between performance evidence and numerical scope.

CUDA kernels, Cartesian tree implementation, anisotropic functionals, LVR
penalties, CWBAP variants and separate nonhomogeneous quadrature controls are
deferred. Preserve their extension points without implementing speculative
backends. The existing DNDS runtime remains the execution/storage foundation;
numerical algebra stays within DNDS, geometry matching within Geom, and domain
orchestration within the future solver.
