# Cauchy native implicit operator contracts

The coupled DITR method, its independent intermediate-stage workspace and
accepted/trial history contracts are documented in [DITR.md](DITR.md).

## Steady BDF1 limit and physical endpoints

`time.steady=true` uses the analytic infinite-timestep limit of BDF1. The
existing stage solver receives physical mass zero, history source zero and
RHS scale one. Pseudo-time continuation, PP, SGS/ILU/GMRES and optional p-FAS
remain the same operators. Physical time stays frozen; no infinity enters
floating-point arithmetic or BDF history. This matches Euler's `steadyQuit`
behavior: its BDF2 driver takes only its first, BDF1 startup step at dt1e200.

Configuration resolution forces `method=bdf1`, at most one outer solve, no
physical final-time/CFL cap and no timestep-halving retries, recording each
override's provenance. `dt` remains a required positive finite placeholder
and is ignored by the steady equation. The nonlinear stage must actually
converge before publication; reaching its correction cap rejects the candidate
and leaves accepted means, time and history unchanged. A finite-dt BDF1
transient with an outer residual stop is not this steady mode.

`implicit.consoleInterval` enables rank-zero progress at selected nonlinear
iterations; zero disables it. These norms are the uncompressed equation
residual, volume-weighted L2 per component, tested against the initial stage
norm and absolute floor. Euler's current stopping rule instead uses density's
volume-weighted L1 ratio to the running maximum. A baseline comparison must
record this difference; other components above Euler's density threshold do
not imply Euler failed its own stopping rule.

`implicit.requireConvergedReconstruction=true` additionally requires the
finest VR update to meet `numerics.reconstructionTolerance` before accepting
the equation, including p-FAS. Variational reconstruction then requires a
positive tolerance. The default false retains Euler-style fixed-sweep mean
acceptance; a soft VR early-exit threshold alone does not guarantee a settled
polynomial at the mean stop. Inspect the reported VR update and use the strict
guard for a settled-VR qualification. It uses the norm already reduced by CFV,
without another collective or a check inside a numerical kernel. Boundary
quadrature values are refreshed from the current polynomial on each VR sweep.

The physical frontend uses `FieldTimeStepper::ReachedTimeTarget` to avoid a
roundoff-sized extra step at a requested endpoint. The tolerance scales with
the actual times, preserving tiny nonzero physical horizons. Genuine positive
steps which cannot advance a floating-point clock are rejected explicitly.
The relative endpoint guard also covers addition error from the tested800-step
vortex and2500-step cylinder sequences, rather than only one rounded addition.

Restart compatibility allows reconstruction sweep-budget/early-exit changes.
They do not change stored means or a compatible polynomial coefficient schema;
physics, flux and active PP policies retain their existing compatibility checks.
Exact replay still requires compatible numerical controls and stored guesses.

## Polynomial multigrid

`Cauchy/Multigrid/PolynomialMultigrid` connects the generic
[FAS cycle](FAS.md) to same-cell spatial operators. `implicit.multigrid.levels=0`
disables it. Level1 is direct-gradient second order, level2 additionally uses
first-order convection; physical viscosity/gradient calculation, equation and
boundary conditions remain enabled. The finest operator requires variational
degree2 or3. This follows the supported Euler pMG fidelity hierarchy, without
copying its recursive orchestration lambda. No spatial coarsening or new mesh
partition is performed.

```text
Domain physical trial / time equation
  PolynomialMultigrid -> finest uncompressed convergence test
    Solver::FASCycle
      VR high-order   -> NonlinearSolver bounded PTC/SGS or GMRES smoothing
      direct O2      -> independent state, FV workspace and matrix/linear solver
      first-order    -> independent state, FV workspace and matrix/linear solver
      identity state/residual/correction maps on the same cell identity/order
```

Every operator uses `mass*U-rhsScale*R(U)/V`. Consequently all three same-cell
maps are identity in this normalized convention; an integrated-residual or
different-mesh hierarchy must supply its own transfer. Cell identity/order is
checked during setup. Means, fine/coarse matrices and mutable auxiliary/gradient
buffers are distinct. The current Cauchy profile smooths the whole coupled
flow5 block; generic FAS accepts independent fields of arbitrary component
width, permitting future selected-block adapters without erasing other fields.

`preSmooth`, `postSmooth` and `coarseSmooth` bound nonlinear corrections, and
`cycles` caps V-cycles. Each level carries its pseudo-CFL between smooth calls;
the next physical stage resets it. Coarse correction `damping` and physical PP
reuse the same candidate-before-publication policy as the fine smoother.
Convergence checks the original finest equation with its original source,
per-component norms and configured tolerances. A coarse forcing tolerance or
a V-cycle count never implies physical stage acceptance. Reports contain
cycle counts, corrections per level and total corrections/linear work across
levels. No acceleration claim follows from the correctness tests alone.

Physical discard/retry still restores the fine reconstruction guess. Coarse
levels have no persistent VR polynomial guesses and restriction overwrites
their state each cycle. A failed cycle drops its scratch hierarchy; the next
attempt constructs fresh scratch before reuse. The physical caller still owns
restoring any changed fine trial. Failed/capped stages never publish accepted
state or time history. Same-cell pMG tests cover manufactured nonconstant P3
equations, mixed DOF layouts, active distributed halos, constant fixed points,
physical discard/retry and commit rejection by a retained debug reader.


`Cauchy/Implicit/LinearProblemContext` owns the numerical matrix for one fixed
native DOF space. Its immutable sparse columns use the same owned-plus-ghost
local numbering as the input field. It stores rank-owned rows only, separate
diagonal-entry or full diagonal blocks, and full off-diagonal blocks. Blocks use
`DNDS::Algebra::BlockMatrixStorage`; diagonal values and inverses are native
Arrays. No legacy ArrayDOF or EulerJacobian adapter is involved.

The constructor checks the core set against actual matrix columns. Its logical
scan visits `[core][noncore]`, without permuting physical arrays or communication
mappings. `Apply(Core)` acquires only owned input; All and HaloDependent require
completed ghost input. Output buffers must be distinct from inputs and matrix
storage. Array allocation or shape replacement invalidates the fixed stencil.
Bindings retain the exact `ArrayStorage`, including its payload range and shape
metadata. Allocation identity alone is insufficient: two disjoint views can
share an allocation and its mutation epoch. Replacing a bound handle with such
a view also requires rebuilding the workspace. These O(1) checks run at wrapper
entry, outside numerical loops. Ordinary shallow handle copies retain the same
storage and remain valid. Auxiliary provenance uses the same storage pins plus
state/output versions and time; cell auxiliary halo writes may refresh values,
but may not replace the bound ghost storage.
Fill finite coefficients before use; the low-level action does not rescan every
coefficient or input value on each matrix-vector multiplication.

`InvertDiagonal` explicitly prepares scalar reciprocals or robust block inverses
using the shared active-Euler LU/SVD policy. Changed or actively written diagonal
values reject SGS until inversion is repeated. `ApplySGS` exposes an unsplit
convenience over the staged symmetric-GS kernel: RHS, old iterate, forward scratch and result are
separate buffers. Remote old values stay frozen through both scans. No MPI or
middle exchange is hidden in the numerical operator. ILU composition and
Krylov/nonlinear drivers remain separate phase items.

`FiniteVolume::BuildLinearProblem` allocates the cell-face-cell stencil on the
prototype's existing row space. `PrepareJacobian` fills its explicit numerical
arrays from frozen means, prepared face spectral radii and an owned scalar
inverse-pseudo-time array. For integrated residual `R=d(VU)/dt`, its convention
is `A = V/dTau + physicalMass*V - residualScale*J_R_approx`. No division by cell
volume is hidden in the action. Diagonal inversion/factorization is a later
explicit operation.

Scalar dissipation uses the Euler spectral diagonal and off-diagonal
`0.5*residualScale*weight*(A_normal(U_other)-lambda*I)`. Its boundary contribution
is the same scalar spectral diagonal, without a BC derivative. Full-block Roe
mode forms equation-owned characteristic dissipation from mean-state numerical
wave magnitudes and adds the physical exterior-state Jacobian on boundaries.
Viscosity enters the approximate spectral dissipation; this does not differentiate
the complete gradient reconstruction or viscous flux. Source Jacobians are not
part of the initial source-free profile. Off-rank columns remain in the operator.

Periodic neighbor states and increment columns are transformed into the current
cell's frame. Multiple incidences accumulate into one sparse block. A periodic
self-neighbor contributes to the diagonal block; diagonal-entry storage retains
only that block's diagonal entries. Core classification stays conservative with
the mesh's node-star halo. The producer performs no MPI, input/reconstruction
mutation or hidden inverse update. It rejects stale auxiliary provenance,
missing required signals, incompatible spaces and input/output aliases before
filling. Its caller coordinates rank-local errors before later MPI operations.

Distributed tests compare assembled action against independent centered
differences of a frozen-spectral first-order flux residual on a periodic torus
and a cube with physical slip boundaries. They include integrated time/pseudo-time
scaling, both diagonal representations where applicable, and stale-input errors.
These validate the approximate operator; convergence of the eventual nonlinear
solver and equivalence to a full high-order residual derivative are separate
questions. Linear/nonlinear drivers and native physical methods are connected;
the complete implicit-phase qualification remains in progress.

## Local ILU composition

`PrepareILUStructure(fill, coreFirst, localParts)` constructs an independent
rank-local template, then `FactorizeILU` copies current native coefficients and
factors them. Fill levels0..5 and complete LU(-1) match the generic Solver
facility. Local factors exclude off-rank columns, which remain in the full
matrix action. Explicit local parts additionally exclude cross-part columns
from the preconditioner, giving independent part solves. The Cauchy adapter
builds these induced graphs before symbolic fill: the generic symbolic routine
preserves every supplied edge and does not perform that partitioning itself.

Optional core-first ordering precedes fill and stays within each local part.
Incomplete templates publish dependency-safe forward and backward ready rows;
the latter account for the actually available forward rows. An upper ready core
can be empty. Complete LU has no bounded-core promise. No physical field/halo
mapping is reordered by this setup.

`ApplyILU(rhs, forward, result)` uses separate native fields and the guarded
local triangular solve. It acquires no RHS ghost lease and posts no MPI between
scans. Matrix coefficient generations are checked while source reads are pinned;
changed/active diagonal or off-diagonal values require numeric refactorization.
Symbolic preparation and numerical factorization are independent of SGS inverse
preparation. Dense-local reference tests check optional ordering, diagonal modes,
fill, independent local parts, held ghost writers, empty owners and stale factors.
The generic legacy factor algorithm and its first-build arithmetic stay intact.

## Staged symmetric GS and publication

`BeginSGS(rhs, old, forward, result, middleExchange=false)` creates a move-only
`SGSIteration`. Phase calls execute CPU numerical work synchronously. They do
not post MPI, choose streams or perform rank agreement. The caller serializes
phase calls and coordinates local failures before later distributed operations.

```cpp
auto halo = oldBinding.Begin();
auto iteration = matrix.BeginSGS(rhs, old, forward, result, refreshMiddle);
iteration.Lower(LinearRegion::Core);       // no ghost lease or ghost access
halo.Wait();
iteration.Lower(LinearRegion::HaloDependent);
if (refreshMiddle)
    forwardBinding.Begin().Wait();        // explicit, exposed inter-scan join
iteration.Upper(LinearRegion::HaloDependent);
iteration.Upper(LinearRegion::Core);
iteration.Finish();                      // publishes complete result
```

Either scan also accepts All. Lower Core precedes Lower HaloDependent; the
reverse scan requires Upper HaloDependent before Upper Core. Full and staged
calls use the same logical ordering, coefficients and arithmetic. The native
layout and transfer mapping are unchanged. Wrong, repeated or incomplete phase
sequences throw at the phase boundary, outside cell/block loops.

The iteration holds input/matrix reads and exclusive output access. Lower core
work cannot see the input ghost allocation. Dependent lower work acquires it
only after the caller observes the input exchange. The default upper scan
retains that same frozen remote input. The optional middle mode instead reads
the completed forward halo; it rejects both an active write and an unchanged
generation from an earlier iteration. Generation checks establish publication,
not semantic routing identity: the caller binds the intended shared mapping.

Forward scratch becomes readable when the complete lower scan finishes, so its
halo can be sent then. The result stays exclusively held even after a partial
upper scan and becomes readable only at Finish. Abandoning a started iteration
invalidates unfinished scratch/results. An unused iteration cancels its unused
writes without changing their values. Reinitialize failed buffers with a full
Array write or replace them before reuse; a failed partial result is not an old
valid iterate. A failed Finish leaves the scope available to complete its missing
phases. Calling it again after successful publication is an error.

```mermaid
flowchart LR
    Context[LinearProblemContext] --> Space[DOF space / communication ownership]
    Context --> Pattern[Shared immutable stencil and order]
    Pattern --> Identity[Partition identity only]
    Context --> Coefficients[Native coefficient and inverse arrays]
    Iteration[SGSIteration] --> Pattern
    Iteration --> Reads[Matrix / RHS / input read scopes]
    Iteration --> Forward[Forward write then read scope]
    Iteration --> Output[Result write scope until Finish]
```

Numeric scopes retain allocations, row metadata and partition identity after
original handles/context expire. They do not retain communicator ownership,
which avoids moving MPI resource destruction into a worker's numerical scope.
Read/write views and dense adapters are constructed once per phase; the hot
loop uses bare views, with a stack flow5 vector and a general-component fallback.
No row-mask, shared-pointer copy or lease lookup occurs in block arithmetic.

Focused distributed tests run active direct-MPI input exchanges with core lower
work, compare staged/full default results exactly, and compare the refreshed
middle option with an independent global forward-scratch oracle. They exercise
phase/readiness errors, retained lifetimes, output publication, unused scopes and
abandonment for AoS/SoA/AoSoA. The eventual engine wrapper must declare scheduled
accesses/events for queued execution. These synchronous CPU entry points do not
claim CUDA submission or graph support.

## CPU linear orchestration

`Implicit/LinearSolver` owns reusable native vectors and persistent input halo
bindings, while `LinearProblemContext` owns coefficients and numerical kernels.
It supports standalone SGS and restarted left-GMRES with a fixed SGS or local
ILU preconditioner. Every SGS preconditioner application starts from zero and
uses the configured fixed number of sweeps. ILU never inserts a middle exchange;
the optional SGS middle exchange is explicitly exposed. Matrix read scopes pin
coefficients throughout a solve. Both overlapping core/dependent and unsplit
wait-then-All execution use the same numerical order.

All domain ranks construct/call the solver together with agreed controls.
Callers serialize access, release conflicting debug/Python views, and bind each
input to the intended entity ghost mapping. Checked execution coordinates local
kernel errors before more MPI; disabling those checks is caller-ordered eager
execution and transfers distributed failure handling to the caller. Control and
solve-entry checks remain collective. A preflight rejection is reusable; a
failure after iteration starts invalidates that workspace for reuse. Reports
distinguish convergence, finite iteration limits and unusable breakdowns. The
borrowed solution changes at the next solve; it cannot serve as that solve's RHS.

Distributed tests solve an independently assembled global block matrix, using
real persistent ring transfers, both overlap policies, both SGS middle modes,
SGS/GMRES+SGS/GMRES+ILU and AoS/SoA/AoSoA. They check warm/repeated solves,
unchanged RHS generations, rank-inconsistent controls and rank-local conflicting
accesses. These gates validate linear orchestration, not nonlinear CFD acceptance.

## Implicit PP and physical time building blocks

The unlimited VR warm guess and the filtered flux polynomial are distinct
fields when WBAP or reconstruction PP is enabled, as in Euler's default
`preserveLimited=false` path. `ReconstructionGuess()` is the unlimited field
retained for the next residual, rollback and optional restart.
`FluxReconstruction()` is the prepared polynomial consumed by RHS and surface
sampling. Its independent halo binding follows its own allocation; it is
invalidated by reset/restore and the next residual entry. Do not use its ghost
values until the corresponding coefficient exchange completes.

```text
unlimited guess -> VR sweeps -> next unlimited warm guess / optional restart
                                  |
                                  +-> owned copy -> WBAP -> PP -> halo -> RHS
```

Only coefficient payloads and their halo bindings are separate. Immutable
plans and temporary/scratch buffers are shared between the serial VR/filter
phases. Each simultaneous residual still requires its own Workspace. No extra
coefficient allocation or copy occurs when filtering is disabled. The public
standalone `LimitReconstructionPP()` starts from a fresh unlimited guess on
every call; inside Evaluate, PP instead consumes the preceding WBAP result.
Means remain unchanged. Reconstruction metadata identifies the coefficient
role: older ambiguous saved guesses cold-start, while compatible means remain
readable. No byte-identical cross-version continuation is claimed for them.

`Physics/ImplicitPositivity.hpp` ports Euler's active scheme-one internal-energy
compression, relaxed increment fraction and flow5 increment repair as pure
functions. Density is handled before energy; the legacy safety, compression
powers and repair factors are retained. Temperature and pressure constraints are
an intersection of two affine energy bounds for the initial perfect-gas model.
The same concave rule applies to both. A future EOS must supply valid energy
reference/bound information, rather than substituting gamma in numerical code.
Callers supply finite admissible old states, valid relaxation/power controls,
check the resulting state, and perform the field/global reductions themselves.
An exhausted energy repair returns NaN; it must be rejected, never applied.
These functions do not yet replace the existing explicit flux-blend path.

`Physics/ReconstructionPositivity.hpp` also ports the quadrature-common
density fraction, active Euler scheme-one energy fraction and beta power/safety
as a pure view-based rule. It accepts an admissible lower-order reference and
finite high-order point values, with the same two affine thermal constraints.
Its matrix-style test oracle calls active Euler Gas compression and covers
unequal reference points, shifted energy, beta cutoff, good-polynomial identity
and an exact thermal-bound mean. `FiniteVolume::LimitReconstructionPP` connects
it to owned-cell face/volume samples. Default `positivityReconstruction="Euler"`
uses a common beta and the cell mean reference. Setting
`positivityCompressToMean=false` progressively truncates the central Taylor
basis to find an admissible reference, discarding failed higher orders as in
Euler. `positivityBetaPower=1` gives beta squared; the safety/cutoff apply only
when limiting is necessary. A good polynomial remains bitwise unchanged.
`positivityVolumePoints` defaults true. The resulting stored coefficients are
sampled again and checked against the strict density/thermal constraints.

These synchronous CPU projections allocate their scratch once in the FV
workspace. They use explicit input/output coefficients and owned means only,
so a mean halo transfer can be active. Callers must complete coefficient
transfers and conflicting accesses before projection; use the exact state
storage bound by `BuildWorkspace`. No time advancement, reconstruction solve,
halo communication or mean mutation is hidden in this method. Diagnostics
report this call's count and minimum *applied* beta (zero after cutoff).

`positivityReconstruction="Segment"` explicitly reproduces the earlier Cauchy
pointwise rule, with the original default order/power controls. A pre-control
checkpoint is interpreted as Segment when PP was enabled, so reading it with
Euler PP rejects the numerical-policy mismatch. Inactive PP controls do not
invalidate an older non-PP restart. This reconstruction migration is separate
from the remaining global alpha variants and physical-dt PP field policy; do
not claim the entire Euler PP suite is already migrated.

`Solver/FieldTimeStepper.hpp` owns native-field time scratch and accepted BDF
history. BDF1/2, Trapz and ESDIRK2/3/4 retain Euler's coefficients; DITR uses
its original coupled U2R2 equations and an independent intermediate-stage factory.
A callback solves `mass*U-rhsScale*F(U,t)=source`; nonlinear/PP,
physical accumulation and MPI orchestration remain its caller's responsibility.
This initial profile uses identity accumulation in the semidiscrete time
equations. Different physical mass maps need an explicit adapter. Preparation
changes only trial state/scratch. Commit history only after accepted-state
publication succeeds; discard/retry never changes accepted history. Variable
BDF2 uses the last accepted timestep, with BDF1 startup or an explicit reset.
Independent nonautonomous exact-stage tests verify orders and retry semantics.
Domain now connects these time equations to the nonlinear stage solver below.
Full physical qualification and the remaining Euler PP measures are outstanding.

## Nonlinear stage and physical transaction

`Implicit/NonlinearSolver` solves the explicit stage equation by pseudo-time
continuation. Each iteration evaluates the uncompressed physical defect,
checks per-component volume-weighted norms, prepares local pseudo-time and
the approximate Jacobian, solves a distinct correction, and publishes a checked
candidate. Root `implicit` settings separate nonlinear, linear, SGS, ILU and PP
choices. CPU backend selection is explicit. Exact constructor control agreement
prevents rank-dependent iteration/PP branches from mismatching MPI calls.

For `numerics.positivity`, the implicit path uses Euler's RHS compression or
dTau compression, relaxed delta compression and final flow5 increment repair.
Convergence always uses the uncompressed defect. Reconstruction quadrature PP
now uses the Euler rule by default (explicit Segment compatibility remains).
Physical-dt PP remains phase work. This is not yet the full PP suite.

`implicit.positivity.rhsReduction` selects Euler's `cell` (default),
`minimumLimited` (`MIN_IF_NOT_ONE`) or `minimumAll` (`MIN_ALL`) policy. Select
the global fraction before applying either RHS compression or dTau compression.
The limited-cell counter counts the original alpha<1 triggers, even when every
row is subsequently scaled. Default cell mode retains its fused local apply;
only global modes allocate one reusable owned scalar alpha array and perform
a second application pass. The two existing count/min statistics reductions
also validate statistics and scope agreement, without an extra collective.
No RHS statistics reduction is needed when PP/RHS compression is disabled.
This scope applies to RHS/dTau only, not an invented global increment limiter.
Physical-dt PP remains separate phase work.

Corrections are computed into separate candidate storage, then admissibility
is agreed before any trial-state write. This required safety agreement remains
enabled when optional per-kernel collective checks are disabled. A bad correction
therefore rejects without poisoning the bound state allocation. Optional-check
disablement still transfers handling of unexpected local kernel/access failures
to the caller; it is not an MPI exception-recovery mechanism.

`Domain` preserves the reconstruction warm guess before an implicit attempt.
Discarded/rejected attempts restore it before retry. Accepted state and clock
are published before committing BDF history. An external debug reader can reject
publication collectively without advancing history; releasing it allows the
same prepared candidate to commit. A stage correction cap rejects the physical
candidate, never labels the capped inner solve converged. CSV output records
stage count, nonlinear/linear iterations and residual-to-target ratio.

Distributed tests recover a manufactured constant solution from nonconstant
means, check unusable/bad correction handling, and exercise the connected
BDF1/2, Trapz and ESDIRK2/3/4 paths. The P3 transport test checks periodic
conservation and exact discard/retry versus fresh-run agreement. They are
focused implementation gates, not the final IV/cylinder qualification matrix.

Cauchy resolves ordered SOR to the same logical core-first scan for both
execution policies, including direct C++ FV construction. The resolved root
records the effective `cfv.coreFirst` and its ordering provenance. Merely
toggling overlap therefore preserves finite-sweep mathematics. The standalone
CFV plan still exposes natural-order unsplit SOR. The regression leaves the
requested coreFirst flag at its default false and was proven red before this
frontend resolution fix.

The Python Domain constructor honors the configured physical time method.
Stage reports expose the final equation evaluation's reconstruction sweep count
and relative update. CSV `implicit_final_vr_sweeps`/`implicit_final_vr_update`
are their maxima across the step's stage reports, distinct from `vr_sweeps`/
`vr_relative_update` produced by the independent observation workspace. They
are not cumulative VR work counts or maxima over all inner RHS evaluations.
Polynomial FAS reports the finest residual evaluation's VR, rather than a
coarse smoother's last call. Failure-only correction-cap diagnostics include
the actual initial/final component residuals, work counts, pseudo-CFL and this
stage VR snapshot; accepted state/time/history remain unchanged on rejection.
Observation reconstruction is seeded afresh from the accepted numerical guess,
or its preserved backup during a pending/rejected trial. It neither carries its
own previous output's polynomial forward nor resets a converged boundary
polynomial to zero. Repeated diagnostics therefore remain deterministic for
the same accepted numerical state and cannot change the stepping trajectory.
Restart without saved coefficients intentionally cold-starts that numerical
guess, as documented; exact diagnostic/trajectory continuation then need not
match a warm restart.
`implicit_stages` returns copied read-only report objects; `flow` remains a
shared native Array handle with normal lease/lifetime rules. Reports include
convergence, initial/final component residuals, nonlinear/linear iteration
counts and PP activations. Release array access scopes before conflicting
updates or publication. Focused MPI Python tests verify BDF2/ESDIRK4 dispatch,
shared state identity through stepping/restart and retained allocation lifetime.
They require rebuilding/installing bindings and pytest's `--with-mpi` flag;
a skipped collection is not validation.

## Restart correspondence and history

Cauchy checkpoint format4 adds `metadata_json` with the flow schema, energy
convention, EOS parameters, polynomial interpretation and time-history method.
Native Array archives still own their own element/row/physical-layout schemas;
the Cauchy envelope does not serialize C++ static row types. Formats2/3 remain
readable. Reads shard file rows across the current communicator, match exact
canonical `cell_ids`, and use one reusable ordinary transfer plan to redistribute
flow, mesh records, optional coefficients and history. No global ID-sized array
or VTK row correspondence is used. Numerical payloads follow native direct I/O
and canonical direct MPI transfers; the ID directory is separate cold metadata.

The default omits reconstruction. Set `output.checkpointReconstruction=true`,
pass `includeReconstruction=true` in C++, or `include_reconstruction=True` in
Python to preserve a compatible warm guess. Missing coefficients or changed
polynomial metadata cold-start at zero. Mean restoration permits changed MPI
ownership/local order, execution layout and P-degree. Actual per-cell topology,
coordinates, periodic frames, equation schema and other retained physical/
numerical configuration must still match. Time/output/execution choices and
polynomial setup are excluded from the compatible-config comparison; ordinary
Riemann/BC/physics changes are not silently accepted. Earlier archives missing
the Riemann stencil flag retain its explicit manual interpretation.

BDF2 records previous accepted means and dt. Compatible history is remapped;
absent/incompatible history uses BDF1 startup. A replacement time workspace and
its restored history are prepared before accepted publication, then swapped
only after publication succeeds. A failed load leaves accepted clock/history
unchanged. Caller serializes checkpoint access and releases conflicting views
before publication. Only ready CPU arrays are supported by the current direct
I/O backend; device serialization does not silently stage through host memory.

Focused tests change writer/reader rank counts, perform a real mesh RCM
permutation, change P2 to P3 and AoS to SoA, and inspect the reserialized actual
coefficients to prove cold-start behavior. They also verify optional nonzero
warm-guess storage, exact variable-step BDF2 continuation and reader-conflict
rejection/retry. The legacy matching-partition exact-continuation tests now
explicitly opt into coefficient storage, preserving their original purpose.

Mesh envelope2 stores a periodic face's canonical axis tag and the intrinsic
frame of its nodes in the owning cell's reference coordinates. Runtime
`faceInfo.zone` main/donor choice and `cell2facePbi` instead refer to the selected
stored face image, which can change when a fresh partition chooses the other
ordered parent. They are not persistent physical mesh identifiers. Older
envelopes compare exact cell/node IDs, cell-node bits and coordinates, neighbor
IDs, periodic axis and global transforms; their valid face-image gauge bits
are not compared to a different partition's gauge. Physical boundary zones
remain exact. This changes checkpoint interpretation only, not Geom's runtime
face representation or numerical kernels.

The original strict envelope passed generated-mesh tests but rejected a real
CGNS IV20 archive when read at np4 after writing at np8. Diagnostics identified
complementary periodic tags (-1/-4, -2/-5), while coordinates matched exactly.
A permanent CGNS repartition test was proven failing before this correction.
It now passes at np1/2/4/8. The actual old format3 archive also loads and takes
its next ESDIRK4 step after repartition, preserving loaded means and diagnostics.
