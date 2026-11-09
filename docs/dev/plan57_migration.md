# Plan 57 integration ledger

Status: **integration in progress, not accepted**. Source parents: audit
`c036961f1977a11880975f5c19212a3b777db5ef`, refactor
`b22e69e43657f2fe7d1e4f1aa054782d2c8f7cc7`; common base `aec8d0e4`.
Implementation lives in the isolated `codex/cauchy-audit-integration` worktree.
The original checkouts and historical experiment outputs remain untouched.

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
- First compilation: solver and 17 focused binaries compiled; incremental builds continue.
- Focused np2 CPU gate and affected reruns passed the equation FV (23 cases),
  linear (11), nonlinear (13), CFV facility (9) and reduced SA (1) suites.
  The C++ frontend and unchecked solver-local abort CTests passed at np2.
  The pMG suite passed before the last local-stage routing change; relink and
  rerun before final acceptance. Logs and hashes: `plan57-tests/resumed3/`
  and `plan57-tests/resumed4/` in the paired build.
- Numerical rejection returns `StepResult { accepted, reason }` through
  Domain/Simulation/C++ frontend/Python. Actual failures never drive dt retry.
  Runtime and production fatal subprocess results are recorded below.
- Python rebuilt/installed and API tests: pending.
- Runtime CUDA descriptor checks: first configure passed, but nvcc did not
  receive the wrapper's implicit MPI include path. Retry with native C/C++
  compilers and explicit FindMPI wrappers. No CUDA device evidence yet.
- Maintained MG factor/full convergence matrices: unrun.
- Before acceptance, every code obligation below needs a final location and
  test/source-proof entry; inherited history alone is not verification.

## Audit commits (all 47)

| Commit | Obligation | Disposition / proof |
| --- | --- | --- |
| `7f60eda6` | fix(cauchy): separate RANS reconstruction PP and correct SST blend | Review/validation pending |
| `01803350` | fix(cauchy): validate worker capabilities and retain finest solver owners | Review/validation pending |
| `de8d66e5` | refactor(audit): share private reductions and encapsulate buffer address queries | Review/validation pending |
| `88f21255` | test(audit): restore aggregate coverage and separate routine integration costs | Review/validation pending |
| `16b9a447` | feat(runtime): make prepared MPI safety agreement an explicit context policy | Review/validation pending |
| `b673177c` | feat(cfv): select prepared VR storage by supported operations | Review/validation pending |
| `602ac520` | feat(geom): inherit preflight policy through owned mesh preparation | Review/validation pending |
| `4197fee8` | feat(geom): carry preflight policy through import and mesh transitions | Review/validation pending |
| `246c169a` | perf(cfv): qualify capability-selected storage with isolated RSS probes | Review/validation pending |
| `66994afd` | docs(audit): retain compressed E2 evidence and measured source snapshots | Historical evidence retained; not relabeled as integration validation |
| `df5a3a75` | perf(audit): isolate Geom acquisition allocations and operator entry costs | Historical evidence retained; not relabeled as integration validation |
| `647b6dd6` | feat(geom): retain preflight policy through output and local setup | Review/validation pending |
| `f984cb9a` | refactor(algebra): share local products and qualify Runtime ownership | Review/validation pending |
| `18a2cd48` | feat(geom): honor preflight policy during native coordinate edits | Review/validation pending |
| `4a174863` | feat(geom): carry preflight policy through native smoothing | 6 code/test paths match audit tip; integration gate pending |
| `d4d6bae5` | perf(audit): isolate reconstruction coefficient publication costs | Historical evidence retained; not relabeled as integration validation |
| `b5c0de03` | fix(cauchy): honor flow5 collective policy in local FV stages | Review/validation pending |
| `ba456643` | fix(cauchy): share the agreed preflight policy in equation FV | Review/validation pending |
| `ebde0383` | fix(geom): honor preflight policy in distributed surface setup | 10 code/test paths match audit tip; integration gate pending |
| `78d0e2af` | test(euler): qualify shared physics against the original solver | Review/validation pending |
| `895dfbdc` | fix(cauchy): fail fast for unchecked implicit local phases | Review/validation pending |
| `cddf355a` | refactor(geom): avoid duplicate sibling mapping agreement | 3 code/test paths match audit tip; integration gate pending |
| `43e099fe` | perf(audit): prepare matched coefficient publication attribution | Historical evidence retained; not relabeled as integration validation |
| `d770484f` | docs(audit): record completed E7 acquisition assessment | Historical evidence retained; not relabeled as integration validation |
| `416d914b` | refactor(audit): clarify checkpoint protocol and sampling preflight | Review/validation pending |
| `8a1c308f` | refactor(cauchy): share evaluated Jacobian geometry traversal | Review/validation pending |
| `e60b85ba` | perf(audit): record matched copies and qualify native-copy candidate | Historical evidence retained; not relabeled as integration validation |
| `3a606f75` | style(cauchy): scope readable formatting to Python bindings | Review/validation pending |
| `a7cd4bf5` | docs(audit): record matched coefficient copy candidate outcome | Historical evidence retained; not relabeled as integration validation |
| `b6887a40` | test(audit): curate Cauchy ranks and add frozen Jacobian probe | Review/validation pending |
| `85f4bf73` | perf(cauchy): specialize exact identity Jacobian transforms | Review/validation pending |
| `8a230e82` | fix(audit): retain frozen executable sidecar names | 1 code/test paths match audit tip; integration gate pending |
| `d745756c` | test(audit): observe publication cost in composed Cauchy RHS | Review/validation pending |
| `01c06db9` | test(audit): record bounded composed-RHS publication campaign | 2 code/test paths match audit tip; integration gate pending |
| `513d03e4` | perf(cauchy): keep identity dispatch out of quadrature payloads | Review/validation pending |
| `05ee5459` | refactor(geom): keep incidence assertions in compact row vectors | 1 code/test paths match audit tip; integration gate pending |
| `a6fa6e51` | docs(audit): record actual flow5 RHS publication fractions | Historical evidence retained; not relabeled as integration validation |
| `4356d964` | feat(runtime): add opt-in prepared CPU transfer grouping | Review/validation pending |
| `fc9ed384` | test(audit): assess grouped and worker transfer scheduling | Review/validation pending |
| `f773d320` | test(audit): measure publication within split mixture RHS | 3 code/test paths match audit tip; integration gate pending |
| `4decc781` | fix(geom): preserve companion transfer preflight policy | Review/validation pending |
| `cd013bd0` | docs(audit): record reacting publication and worker scheduling results | Historical evidence retained; not relabeled as integration validation |
| `c7a40e76` | docs(audit): retain compact raw campaign records | Historical evidence retained; not relabeled as integration validation |
| `8964181b` | test(audit): derive compute-worker communication tail from completion times | 1 code/test paths match audit tip; integration gate pending |
| `af951ad7` | docs(audit): retain final routine acceptance and fixture failures | Historical evidence retained; not relabeled as integration validation |
| `9ccb3dbd` | test(audit): measure composed RHS scheduling and nonlinear update costs | Review/validation pending |
| `c036961f` | docs(audit): close composed RHS and nonlinear update assessment | Historical evidence retained; not relabeled as integration validation |

## Refactor commits (all 50)

| Commit | Accepted change or evidence | Integration proof |
| --- | --- | --- |
| `7dff3897` | fix(cauchy): retain coarse gradients and fail fast on distributed errors | Retained in merge; validation pending |
| `cde3087c` | docs: record bounded pMG outcomes and local preflight contract | 1 code/test paths match refactor tip; integration gate pending |
| `a300d2ee` | feat(cauchy): configure independent polynomial multigrid smoothers | Retained in merge; validation pending |
| `826a7353` | fix(cauchy): match Euler reconstruction wall targets | Retained in merge; validation pending |
| `878ecf22` | fix(cauchy): align wall and farfield protocols with Euler | Retained in merge; validation pending |
| `828b714a` | perf(cauchy): remove repeated FV and implicit preparation work | Retained in merge; validation pending |
| `310fdccf` | fix(cauchy): honor optional pMG residual preflight checks | Retained in merge; validation pending |
| `480a6c64` | perf(cauchy): preserve static blocks and direct row access in CPU kernels | Retained in merge; validation pending |
| `f4d6ddb9` | test(cauchy): distinguish candidate validation from collective waiting | 4 code/test paths match refactor tip; integration gate pending |
| `f21610f1` | fix(cauchy): honor auxiliary entry validation policy | Retained in merge; validation pending |
| `f123e4dc` | test(cauchy): attribute assembly through four external diagnostic batches | Retained in merge; validation pending |
| `defa1936` | test(cauchy): preserve exact extraction through formatter whitespace changes | 1 code/test paths match refactor tip; integration gate pending |
| `2cba525e` | test(cauchy): verify selected six-component Jacobian principal blocks | Retained in merge; validation pending |
| `4fbfd581` | test(cauchy): check selected direct Jacobian actions | Retained in merge; validation pending |
| `7e40134b` | fix(cfv): maintain face-centered metrics for midpoint FV | Retained in merge; validation pending |
| `f91de7e9` | fix(cauchy): use face-centered approximate Jacobians | Retained in merge; validation pending |
| `6869b266` | feat(cauchy): separate direct FV gradients and residual phases | Retained in merge; validation pending |
| `b25b01de` | test(cauchy): verify direct-gradient SA source Jacobian | Retained in merge; validation pending |
| `26c3f799` | test(cauchy): audit correction source refresh work in phase diagnostics | 4 code/test paths match refactor tip; integration gate pending |
| `c29b6677` | perf(cauchy): apply the first zero-start ILU correction directly | Retained in merge; validation pending |
| `eb196cb1` | perf(cauchy): borrow direct gradients for source Jacobian preparation | Retained in merge; validation pending |
| `27d48bce` | test(cauchy): check high-order SST auxiliary face bounds | Retained in merge; validation pending |
| `98807307` | test(cauchy): keep auxiliary oracle ranks in collective order | Retained in merge; validation pending |
| `c2dbe166` | perf(cauchy): borrow direct gradients for midpoint volume sources | Retained in merge; validation pending |
| `f17dbbb2` | perf(cauchy): skip exact identity-frame flux block transforms | Retained in merge; validation pending |
| `69cf6264` | docs(cauchy): clarify borrowed source gradients and identity frames | Retained in merge; validation pending |
| `455284fc` | perf(cauchy): assemble block diagonals through borrowed output views | Retained in merge; validation pending |
| `b80d26e0` | test(cauchy): expose untimed rank work and cover rotated Jacobians | Retained in merge; validation pending |
| `2afe198b` | docs(cauchy): explain untimed rank work diagnostic metadata | 1 code/test paths match refactor tip; integration gate pending |
| `1c19cc58` | perf(cfv): isolate cached value sampling from derivative evaluation | Retained in merge; validation pending |
| `a37c2593` | test(cauchy): preserve uncompressed SA reconstruction policy | Retained in merge; validation pending |
| `164810d9` | perf(cauchy): resolve source row spans before component loops | Retained in merge; validation pending |
| `e17a2084` | perf(cauchy): specialize six-component assembled face blocks | Retained in merge; validation pending |
| `68b4cb10` | test(cauchy): cover repeated neighbor faces and dirty assembly buffers | Retained in merge; validation pending |
| `7337156e` | perf(cfv): sample source values and gradients in one pass | Retained in merge; validation pending |
| `11df9b7f` | perf(cauchy): inline bounded six-component evaluated products | Retained in merge; validation pending |
| `314a50e8` | fix(cauchy): restore independent source Jacobian diagnostic | Retained in merge; validation pending |
| `f1d6f7ee` | perf(runtime): compact fully static row descriptors | Retained in merge; validation pending |
| `e6776121` | perf(cauchy): reuse current direct source gradients | Retained in merge; validation pending |
| `ed3c9fca` | perf(cauchy): share mean thermodynamics in source linearization | Retained in merge; validation pending |
| `fffcfbe1` | test(cauchy): trace auxiliary readiness and rank arrival chronology | 4 code/test paths match refactor tip; integration gate pending |
| `aeb52231` | perf(cauchy): use prepared incidences in common Jacobian assembly | Retained in merge; validation pending |
| `08845fef` | feat(cauchy): close coupled AoS and quadrature source path | Retained in merge; validation pending |
| `a46bbfaf` | test(cauchy): validate both source-Jacobian modes | 1 code/test paths match refactor tip; integration gate pending |
| `4b693ab2` | perf(cauchy): cache Jacobian column slots | Retained in merge; validation pending |
| `ebf3d60e` | test(cauchy): record auxiliary ghost donors | 1 code/test paths match refactor tip; integration gate pending |
| `bdb532be` | docs(cauchy): record auxiliary refresh and Jacobian slot contracts | Retained in merge; validation pending |
| `f6ab39f7` | feat(cauchy): expose flow5 finite-volume profile sampling | Retained in merge; validation pending |
| `2d5a0dcd` | fix(cauchy): align residual logging and iteration labels | Retained in merge; validation pending |
| `b22e69e4` | test(cauchy): retain split-domain IV experiment controls | Retained in merge; validation pending |

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

Earlier focused runs and gate logs are historical; the clarified policy needs
fresh verification. The CGNS policy case passed after an ignored symlink to
canonical IV10_2.cgns was supplied. Python bindings built before this correction
but were not installed or exercised. The plan-repo matrix wrapper was pushed
at `ef6021f`; matched cluster runs remain unrun.


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
`plan57-tests/resumed2/` and `plan57-tests/resumed3/`. Broader Cauchy/CFV,
Python bindings, and the cluster MG matrices remain separate pending gates.
