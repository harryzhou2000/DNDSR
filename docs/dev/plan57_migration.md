# Plan 57 integration ledger

Status: **source integrated; acceptance in progress**. Merge
`0e1111747f7d51c3e767816d5b2786bc57bdb475`; BSSCA source candidate
`7493806bb90f436316e9cc17855abd59497cef90` includes the tracked halo audit
follow-up. Source parents: audit
`c036961f1977a11880975f5c19212a3b777db5ef`, refactor
`b22e69e43657f2fe7d1e4f1aa054782d2c8f7cc7`; common base `aec8d0e4`.
Implementation lives in the isolated `codex/cauchy-audit-integration` worktree.
The original checkouts and historical experiment outputs remain untouched.
See the [second source audit](plan57_source_audit.md) for the direct comparison
against the optimized parent and the prior audit plans.

## Semantic integration decisions

1. Named `FiniteVolumeMetricOptions` separates geometry from validation, with
   typed Python options and keyword-only legacy-adapter additions.
2. Runtime bindings resolve Inherit/Collective/AbortOnFailure once. Cauchy's
   master off disables optional launch checks even with explicit collective policy.
   Group members must share their effective launch policy.
3. PP's unchanged-flow fast exit separately proves RANS needs no repair;
   reduced-order SA and SST keep independent scalar repair semantics.
4. Shared evaluated Jacobian geometry uses centered incidences, scalar face
   area and cached column slots, preserving all repeated-face contributions.
5. pMG retains the finest solver facade by value while retaining upstream
   rate reuse, independent smoother controls and diagnostics.
6. The production Cauchy controller aborts on caught exceptions around
   collectively ordered work, including matched point-to-point exchanges.
   Local checks can throw and coordinated UTs can catch them. Normal numerical
   rejection is a structured outcome; legacy Euler code is unchanged.
7. Upstream zero-start SGS, ILU, source sampling/linearization and equation-view
   optimizations remain; E5 worker and remaining plan56 features are not part
   of this integration.

## Verification ledger

- Fresh CPU configuration: completed; paired build uses canonical externals.
- Fresh solver and focused test binaries compiled; affected targets were
  rebuilt after the production and test follow-ups.
- Focused np2 CPU gate and affected reruns passed the equation FV (23 cases),
  linear (11), nonlinear (13), CFV facility (9) and reduced SA (1) suites.
  The C++ frontend and unchecked solver-local abort CTests passed at np2.
  The pMG relink against the current static library passed 11 cases and
  1367/1344 assertions at np2. Logs and hashes: `plan57-tests/resumed3/`
  and `plan57-tests/resumed4/` in the paired build.
- Numerical rejection returns `StepResult { accepted, reason }` through
  Domain/Simulation/C++ frontend/Python. Actual failures never drive dt retry.
  Runtime and production fatal subprocess results are recorded below.
- The compact CPU complement passed Runtime Array (31 cases), Cauchy
  boundary inputs (two per rank), CFV capabilities (four per rank), Euler SST
  and RANS (four), and Geom policy np2/4 (four per rank). The first Geom
  invocation lacked the canonical UniformSquare fixture; the unchanged binary
  passed after it was linked into the isolated tree. Receipts:
  `plan57-tests/acceptance-compact/`.
- The tracked-halo change passes the full finite-volume np2 suite (39 cases
  per rank) at source `7493806b`; the PMPI case confirms equal ordinary and
  tracked setup counts with checks off and two numerical reductions during
  P3 RHS evaluation at np2 and np4. The StepResult gate passed eight
  distributed suites on its first run; checkpoint passed all 13 cases at
  np2 after correcting its stale field assumptions. The tooling tests pass
  22/22. Receipts: `plan57-tests/{step-result-gate,dofhalo-gate,final-audit}/`
  and `plan57-tests/source-audit-tooling.log`.
- Python modules rebuilt and installed. The focused np2 rerun passed per rank:
  CFV 13 (four NCCL skips), Cauchy Domain 23, Cauchy operators two.
  `plan57-tests/final-audit/` retains the final source-matched logs; the first
  repaired run is retained in `plan57-tests/python-callers-fixed/`. The old default-policy
  and split-field assertions failed before their test callers were updated;
  the rerun compares logical blocks across the new default coupled layout and
  explicit split/permuted fields. The install emitted stub-generation warnings;
  runtime imports and the focused APIs worked.
- CUDA descriptor build and device 0 run passed: 12 cases and 3595 assertions,
  no skips, OMP1. Binary SHA256
  `e7271aeb8aa523646120112486c21e72fd4a940b9c2905c8a86c5d81c5a7c995`.
  `build/plan57-cuda-explicit-mpi/` retains the source-matched receipt. The
  first configuration with MPI compiler wrappers hid include flags from nvcc;
  a separate configure with native C/C++ compilers and explicit FindMPI paths
  fixed it. The gate tests descriptors, not NCCL or distributed GPU execution.
- Maintained MG factor matrices: all 70 cylinder and 126 NACA applicable
  32-correction members passed the receipt audit; these are capped trials.
  All 42 full convergence runs passed on cylinder and NACA0012. For all
  14 refactor/integrated Cauchy pairs, logged component histories and linear
  work match exactly; persistent-ID fields and aligned wall pressures are
  bit-identical. Shared Euler terminal observables pass the historical
  envelope, and cell areas agree to Euler's printed precision. Euler
  fields were not compared bitwise. The final release gate is
  `build/plan57-cluster/final-audit/verified.json`. Median nonlinear time
  fell 4.0% on cylinder and 11.0% on NACA versus pre-merge Cauchy.
- BSSCA auxiliary, reconstruction/RHS, matrix, linear and whole-window
  clocks passed the stock numerical gate for 21 cylinder and 21 NACA
  instrumented members. The 256-correction NACA LU-SGS repeat retained
  exact work and traces and took 10–12% less nonlinear time than the
  refactor. The instrumented builds were kept separate from the production
  timings and from the application and library sources in this branch.
- Source `7493806b` is the production benchmark pin. Subsequent changes
  are limited to tests and audit records. The
  [plan 57 experiment index](https://github.com/harryzhou2000/dndsr-next-plan/tree/main/experiments/cauchy-cpu-next/plan57-migration)
  records the new BSSCA runs. Historical measurements keep their own pins.

## Audit commits (all 47)

| Commit | Obligation | Disposition / proof |
| --- | --- | --- |
| `7f60eda6` | fix(cauchy): separate RANS reconstruction PP and correct SST blend | Adapted in Reconstruction/EquationRANS; flow exit proves independent scalar repair. RANS FV np2 and reduced SA passed; SST model oracle retained. |
| `01803350` | fix(cauchy): validate worker capabilities and retain finest solver owners | Retained MPI startup status and ValidateMPIThreadSupport before engine setup; pMG owns facade, not engine. Config, moved-facade and frontend fixtures retained; pMG and frontend np2 passed. |
| `de8d66e5` | refactor(audit): share private reductions and encapsulate buffer address queries | Retained AuxiliaryReduction, MeshAdjacencyBindings and Buffer metadata helpers; edits to callers preserve row order and borrow lifetimes. |
| `88f21255` | test(audit): restore aggregate coverage and separate routine integration costs | Retained CMake aggregate dependency checker, committed split-SOR test, 3D routine/golden split and lazy Geom fixtures; paired configure passed. |
| `16b9a447` | feat(runtime): make prepared MPI safety agreement an explicit context policy | Adapted setup context to binding Inherit/override; checked and unchecked starts and Cauchy no-extra-Allreduce gate passed. |
| `b673177c` | feat(cfv): select prepared VR storage by supported operations | Retained ReconstructionCapabilities allocation/entry checks; named metric API independent. Python and four focused C++ capability cases passed. |
| `602ac520` | feat(geom): inherit preflight policy through owned mesh preparation | Retained NativeOwnedMeshData context through topology, registry and assembler; paired ghost/payload source reviewed, Geom policy cases passed at np2 and np4. |
| `4197fee8` | feat(geom): carry preflight policy through import and mesh transitions | Retained context injection in Cauchy Simulation and native CGNS/OpenFOAM owned preparation/transition; Cauchy frontend np2 passed. |
| `246c169a` | perf(cfv): qualify capability-selected storage with isolated RSS probes | Retained opt-in CFV storage probe and original isolated RSS receipts; no new RSS timing inferred. |
| `66994afd` | docs(audit): retain compressed E2 evidence and measured source snapshots | Historical evidence retained; not relabeled as integration validation |
| `df5a3a75` | perf(audit): isolate Geom acquisition allocations and operator entry costs | Historical evidence retained; not relabeled as integration validation |
| `647b6dd6` | feat(geom): retain preflight policy through output and local setup | Retained output, preparation and wall-distance context propagation; output restart Python and frontend np2 passed. |
| `f984cb9a` | refactor(algebra): share local products and qualify Runtime ownership | Adapted private products to inverse scales without merging owners; linear np2 and Python split/permuted operators passed. |
| `18a2cd48` | feat(geom): honor preflight policy during native coordinate edits | Retained checked/unchecked coordinate stages in MeshEdit and MeshGeometryEdit; added scoped fatal catch without changing coordinate arithmetic. |
| `4a174863` | feat(geom): carry preflight policy through native smoothing | All six code/test paths equal audit parent; smoothing/refinement options still receive preparation context. |
| `d4d6bae5` | perf(audit): isolate reconstruction coefficient publication costs | Retained opt-in coefficient probe and original results; production keeps logical-row publication. |
| `b5c0de03` | fix(cauchy): honor flow5 collective policy in local FV stages | Retained flow5 FV/aux context stages; exact PMPI test observes only two numerical reductions with checks off. |
| `ba456643` | fix(cauchy): share the agreed preflight policy in equation FV | Retained equation FV/mean-gradient context stages; equation FV 23 cases per rank passed. |
| `ebde0383` | fix(geom): honor preflight policy in distributed surface setup | All ten code/test paths equal audit parent; sparse surface/interface checks inherit context. |
| `78d0e2af` | test(euler): qualify shared physics against the original solver | Retained independent frozen Euler transport and physics probes plus harness, unchanged; original results remain at their source pins. |
| `895dfbdc` | fix(cauchy): fail fast for unchecked implicit local phases | Adapted LinearSolver/NonlinearSolver/EquationNonlinearSolver/pMG local stages to scoped solver abort; coordinated UTs still throw. Focused solver and abort tests passed. |
| `cddf355a` | refactor(geom): avoid duplicate sibling mapping agreement | All three code/test paths equal audit parent; EntitySpace same sibling/rank-order protocol retained. |
| `43e099fe` | perf(audit): prepare matched coefficient publication attribution | Historical evidence retained; not relabeled as integration validation |
| `d770484f` | docs(audit): record completed E7 acquisition assessment | Historical evidence retained; not relabeled as integration validation |
| `416d914b` | refactor(audit): clarify checkpoint protocol and sampling preflight | Retained ArrayIO protocol validation; split codec checks from collective HDF5 using CodecStage. Checkpoint/frontend np2 passed. |
| `8a1c308f` | refactor(cauchy): share evaluated Jacobian geometry traversal | Adapted JacobianDetail::BuildStencilRows to centered samples/areas/cached slots; flow5/equation snapshot types remain separate. Jacobian and equation FV np2 passed. |
| `e60b85ba` | perf(audit): record matched copies and qualify native-copy candidate | Historical evidence retained; not relabeled as integration validation |
| `3a606f75` | style(cauchy): scope readable formatting to Python bindings | Retained local binding formatter and updated StepResult/collective docstrings; focused Python APIs passed. |
| `a7cd4bf5` | docs(audit): record matched coefficient copy candidate outcome | Historical evidence retained; not relabeled as integration validation |
| `b6887a40` | test(audit): curate Cauchy ranks and add frozen Jacobian probe | Retained audit tier rules and frozen E6 probe; upstream registrations coexist under same aggregate checker. |
| `85f4bf73` | perf(cauchy): specialize exact identity Jacobian transforms | Adapted exact identity-frame shortcut to centered flow5 geometry; equation six-component specialization retained separately. Rotated action tests passed. |
| `8a230e82` | fix(audit): retain frozen executable sidecar names | 1 code/test paths match audit tip; retained source, no new experiment inferred |
| `d745756c` | test(audit): observe publication cost in composed Cauchy RHS | Retained opt-in composed coefficient probe and publication entry; cached polynomial updates still publish into stable storage. |
| `01c06db9` | test(audit): record bounded composed-RHS publication campaign | 2 code/test paths match audit tip; retained source, no new experiment inferred |
| `513d03e4` | perf(cauchy): keep identity dispatch out of quadrature payloads | Retained one flow5 plan/geometry identity flag, with no extra per-point flag from the rejected audit trial. |
| `05ee5459` | refactor(geom): keep incidence assertions in compact row vectors | 1 code/test paths match audit tip; retained source, no new experiment inferred |
| `a6fa6e51` | docs(audit): record actual flow5 RHS publication fractions | Historical evidence retained; not relabeled as integration validation |
| `4356d964` | feat(runtime): add opt-in prepared CPU transfer grouping | Adapted opt-in group to require same effective launch policy, with one preflight. Separate messages and checked reservation rollback passed np2/4. |
| `fc9ed384` | test(audit): assess grouped and worker transfer scheduling | Retained opt-in scheduling probe unchanged; Cauchy still uses separate controller starts. |
| `f773d320` | test(audit): measure publication within split mixture RHS | 3 code/test paths match audit tip; retained source, no new experiment inferred |
| `4decc781` | fix(geom): preserve companion transfer preflight policy | Retained companion source validation and isolated BeginRows inheriting source mapping policy; PMPI fixture retained. |
| `cd013bd0` | docs(audit): record reacting publication and worker scheduling results | Historical evidence retained; not relabeled as integration validation |
| `c7a40e76` | docs(audit): retain compact raw campaign records | Historical evidence retained; not relabeled as integration validation |
| `8964181b` | test(audit): derive compute-worker communication tail from completion times | 1 code/test paths match audit tip; retained source, no new experiment inferred |
| `af951ad7` | docs(audit): retain final routine acceptance and fixture failures | Historical evidence retained; not relabeled as integration validation |
| `9ccb3dbd` | test(audit): measure composed RHS scheduling and nonlinear update costs | Retained benchmark-only RHS and nonlinear assessment sources and registrations; no production worker or different default adopted. |
| `c036961f` | docs(audit): close composed RHS and nonlinear update assessment | Historical evidence retained; not relabeled as integration validation |

## Refactor commits (all 50)

| Commit | Accepted change or evidence | Integration proof |
| --- | --- | --- |
| `7dff3897` | fix(cauchy): retain coarse gradients and fail fast on distributed errors | Adapted terminal scopes and explicit StepResult; direct coarse source gradients retained. Runtime failure, implicit and pMG np2 tests passed. |
| `cde3087c` | docs: record bounded pMG outcomes and local preflight contract | 1 code/test paths match refactor tip; retained source, no new experiment inferred |
| `a300d2ee` | feat(cauchy): configure independent polynomial multigrid smoothers | Retained independent coarse smoothers and defaults in PolynomialMultigrid/SimulationConfig; np2 pMG 11 cases passed. |
| `826a7353` | fix(cauchy): match Euler reconstruction wall targets | Retained reflected reconstruction wall target in flow/equation paths; BoundaryInputs fixture retained. |
| `878ecf22` | fix(cauchy): align wall and farfield protocols with Euler | Retained thermal wall and farfield protocols in BoundarySystem/BoundaryOperators, separate trace and approximate-Jacobian inputs; equation FV np2 passed. |
| `828b714a` | perf(cauchy): remove repeated FV and implicit preparation work | Retained unique-face flux/gather, cached auxiliary/rate path, midpoint and P0 gradients, small-block kernels and fixed-count controls. Scope-only changes in `FiniteVolume`. |
| `310fdccf` | fix(cauchy): honor optional pMG residual preflight checks | pMG optional residual check adapted through RunLocalStage; numerical residual reductions remain. Focused pMG np2 passed. |
| `480a6c64` | perf(cauchy): preserve static blocks and direct row access in CPU kernels | Retained EquationViews, bounded row dispatch, static block inverse, zero-start SGS and direct evaluation with generic fallback. Jacobian/linear/Equation FV np2 passed. |
| `f4d6ddb9` | test(cauchy): distinguish candidate validation from collective waiting | 4 code/test paths match refactor tip; retained source, no new experiment inferred |
| `f21610f1` | fix(cauchy): honor auxiliary entry validation policy | Equation auxiliary entry follows metric context, with rank-local abort in production; RANS/auxiliary gates passed. |
| `f123e4dc` | test(cauchy): attribute assembly through four external diagnostic batches | Retained diagnostic source/extractors and opt-in registration; timing follows its own phase boundaries. |
| `defa1936` | test(cauchy): preserve exact extraction through formatter whitespace changes | 1 code/test paths match refactor tip; retained source, no new experiment inferred |
| `2cba525e` | test(cauchy): verify selected six-component Jacobian principal blocks | Retained selected six-component oracle in RANS FV; np2 suite passed. |
| `4fbfd581` | test(cauchy): check selected direct Jacobian actions | Retained selected direct-action oracle in RANS FV; np2 suite passed. |
| `7e40134b` | fix(cfv): maintain face-centered metrics for midpoint FV | Retained faceMean setup, separate RHS quadrature, host/device views and midpoint test with named options. CFV fixture np2 passed. |
| `f91de7e9` | fix(cauchy): use face-centered approximate Jacobians | Retained centered incidence and area Jacobian geometry in both producers; assembled/evaluated and rotated oracles passed. |
| `6869b266` | feat(cauchy): separate direct FV gradients and residual phases | Retained split direct gradient, flux overwrite and source-add paths; Python operators and equation FV np2 passed. |
| `b25b01de` | test(cauchy): verify direct-gradient SA source Jacobian | Retained direct-gradient SA source oracle; RANS FV np2 passed. |
| `26c3f799` | test(cauchy): audit correction source refresh work in phase diagnostics | 4 code/test paths match refactor tip; retained source, no new experiment inferred |
| `c29b6677` | perf(cauchy): apply the first zero-start ILU correction directly | Retained first direct zero-start fixed ILU correction and original arithmetic in LinearSolver; linear np2 passed. |
| `eb196cb1` | perf(cauchy): borrow direct gradients for source Jacobian preparation | Retained source Jacobian direct-gradient leases in `FiniteVolume`; diff changes preparation policy only. |
| `27d48bce` | test(cauchy): check high-order SST auxiliary face bounds | Retained high-order SST auxiliary face oracle; RANS FV np2 passed. |
| `98807307` | test(cauchy): keep auxiliary oracle ranks in collective order | Retained collective-order auxiliary oracle and empty-owner handling; RANS FV np2 passed. |
| `c2dbe166` | perf(cauchy): borrow direct gradients for midpoint volume sources | Retained midpoint RHS borrowed physical-gradient views in `FiniteVolume`, including P0 source gradient. |
| `f17dbbb2` | perf(cauchy): skip exact identity-frame flux block transforms | Retained exact identity-frame equation Jacobian products and rotated fallback; both state types retain specialized paths. |
| `69cf6264` | docs(cauchy): clarify borrowed source gradients and identity frames | Retained upstream documentation and gradient/frame contracts. |
| `455284fc` | perf(cauchy): assemble block diagonals through borrowed output views | Retained output block-diagonal borrowed views and no row ownership change; Jacobian/Equation FV np2 passed. |
| `b80d26e0` | test(cauchy): expose untimed rank work and cover rotated Jacobians | Retained diagnostic rank-work fields and rotated equation test; focused equation FV np2 passed. |
| `2afe198b` | docs(cauchy): explain untimed rank work diagnostic metadata | 1 code/test paths match refactor tip; retained source, no new experiment inferred |
| `1c19cc58` | perf(cfv): isolate cached value sampling from derivative evaluation | Retained cached EvaluateValues dispatch for PP and unchanged value kernel tokens apart from formatting. Generic sampling remains. |
| `a37c2593` | test(cauchy): preserve uncompressed SA reconstruction policy | Reduced-order SA oracle retained in separate macro target; np2 passed. |
| `164810d9` | perf(cauchy): resolve source row spans before component loops | Retained source row spans outside component loops; scope-only changes to `FiniteVolume`. |
| `e17a2084` | perf(cauchy): specialize six-component assembled face blocks | Retained bounded six-component assembly plus generic path. Repeated/rotated/selected tests retained; Jacobian and RANS np2 passed. |
| `68b4cb10` | test(cauchy): cover repeated neighbor faces and dirty assembly buffers | Retained repeated neighbor faces and dirty assembly buffers in RANS FV; np2 passed. |
| `7337156e` | perf(cfv): sample source values and gradients in one pass | Retained bundled value/gradient kernel and fallback; only terminal catch calls added to sampling source. |
| `11df9b7f` | perf(cauchy): inline bounded six-component evaluated products | Retained six-component evaluated product and selected direct fallback; Jacobian and linear np2 passed. |
| `314a50e8` | fix(cauchy): restore independent source Jacobian diagnostic | Retained standalone source-Jacobian diagnostic independent of cached RHS gradient; production reuse unchanged. |
| `f1d6f7ee` | perf(runtime): compact fully static row descriptors | ArrayView header identical to refactor parent. CUDA device 0, CPU Runtime Array and distributed row-transfer gates passed. |
| `e6776121` | perf(cauchy): reuse current direct source gradients | Retained direct-gradient readiness/publication check and source use; no source resampling on current gradients. |
| `ed3c9fca` | perf(cauchy): share mean thermodynamics in source linearization | Retained mean EOS/source preparation and published ownership, with stale state guards; equation FV np2 passed. |
| `fffcfbe1` | test(cauchy): trace auxiliary readiness and rank arrival chronology | 4 code/test paths match refactor tip; retained source, no new experiment inferred |
| `aeb52231` | perf(cauchy): use prepared incidences in common Jacobian assembly | Retained prepared incidence lookup in common assembled six-component leaf; no replacement with old quadrature. |
| `08845fef` | feat(cauchy): close coupled AoS and quadrature source path | Retained coupled AoS default, RHS quadrature source option and generic split layouts; focused Python verifies coupled/split parity. |
| `a46bbfaf` | test(cauchy): validate both source-Jacobian modes | 1 code/test paths match refactor tip; retained source, no new experiment inferred |
| `4b693ab2` | perf(cauchy): cache Jacobian column slots | Retained cached column slots in common assembly, now also used by shared evaluated geometry helper; repeated faces remain ordered. |
| `ebf3d60e` | test(cauchy): record auxiliary ghost donors | 1 code/test paths match refactor tip; retained source, no new experiment inferred |
| `bdb532be` | docs(cauchy): record auxiliary refresh and Jacobian slot contracts | Retained auxiliary freshness and cached incidence documentation and implementation. |
| `f6ab39f7` | feat(cauchy): expose flow5 finite-volume profile sampling | Retained flow5 profile hook and distinct gradient scope. All seven policies passed cylinder stock and supplemental clock checks. |
| `2d5a0dcd` | fix(cauchy): align residual logging and iteration labels | Retained integrated L1/L2 residuals and one-based labels; plan runner uses new convention and counts the Euler cap sentinel separately. |
| `b22e69e4` | test(cauchy): retain split-domain IV experiment controls | Retained IV experiment registration/controls; caller adapted to StepResult. No IV work or outputs modified. |

## November 9 failure scope correction

An intermediate implementation made local validation and already coordinated
preflight rejection fatal. That was wider than requested. The original local
and coordinated rejection tests have been restored; their assertions on
unchanged buffers, reservation cancellation, and successful subsequent valid
operations remain meaningful. The terminal subprocess fixtures now exercise
an unsafe domain factory failure, unwinding with an active MPI transfer, a
throwing reducer after MPI posting, and unchecked launch rejection. The local
negative cases removed from the abort fixture have returned to routine tests.
Normal numerical rejection still uses the explicit `StepResult`.

Earlier focused runs and gate logs remain historical. The clarified policy
was verified again as listed above. The CGNS policy case passed after an
ignored symlink to canonical IV10_2.cgns was supplied. The final Python
gate includes a rebuild and installation after the production correction.
The original plan-repo matrix wrapper at `ef6021f` predates the completed
matched runs and their updated reducers.


## Production solver boundary

The executable selects `MPI::SolverFailureScope` on its controller thread.
Lease operations retain their throwing API. In production, preflight and
collective operator catches report locally and abort before cleanup or more
communication, regardless of the optional preflight agreement. A worker
executor created in that scope uses process abort on a worker failure without
calling MPI there. Safe UT checks still use the same throwing API without the
production scope. Checked and unchecked production launch fixtures exercise an
actual rank-local lease conflict; a worker fixture exercises the MPI-free abort
path. Numerical wind-back remains an agreed `StepResult`, not a caught exception.

The frozen local CPU build passed the complete Runtime row transfer and
transient suites at np2 and np4 (27 and 14 cases per rank). All six distributed
failure fixtures passed at both rank counts, including the two production
preflight modes and the worker termination. Both standalone np2 unchecked
launch fixtures also passed. The first legacy Python harness run expected an
obsolete diagnostic; it passed after requiring the explicit DNDS fatal
handler and rank-local reason. Evidence: paired build
`plan57-tests/resumed2/` and `plan57-tests/resumed3/`. The broader Cauchy/CFV
and Python follow-ups are listed above, separately from the cluster gates.
