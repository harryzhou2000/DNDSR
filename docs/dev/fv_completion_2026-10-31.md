# CFV and Geom facility completion ledger

Implements the accepted work orders in standalone plan documents 31/32. The
public architecture/caller contract is [FVFacility](../architecture/FVFacility.md).
Implementation checkout: `/home/harry/projects/DNDSR-array-geom`.

## Sequence and scope

1. Legacy fix on `dev/harry`: `2fb93c44` corrects VR's nonzero-tolerance SVD
   filter and OpenMP reductions, with spectral/legacy setup tests. `4495df8d`
   fixes the aggregate test dependency. Nonzero-tolerance production cases can
   change numerically; no CRM/Orion acceptance is implied.
2. Refactor branch rebased onto that tested base, preserving the prior
   uncommitted facility work with recovery stash `34430194`. Checkpoint
   `cfc80270` restores it; `af3cef2c` adds native SVD, explicit work partitions,
   sparse batches, distributed geometric routing and generic locality ordering.
3. `154f9098` adds core/dependent Jacobi/WBAP, core-first SOR with a frozen
   send snapshot, scalar activation and standalone GG/weighted-LS/GGMP gradients.
4. Completion adds independent functional/zone BC weights, rigid periodic
   derivative/vector transforms, radius-weighted integration and gradients,
   face/query basis caches, derivative sampling and unscaled linear/RHS/residual/
   local-preconditioner operators. The default boundary penalty now weights all
   orders like legacy. Manufactured polynomials explicitly use value-only BCs.
   Geom adds 1..10 conservative boxes/rank, tolerance-envelope bounds and
   traversal-independent nearest/seam arbitration. Protocol allocations agree
   errors collectively before entering later communication.

The completion tests include rank-separated rotational-sector cells, independent
axisymmetric volume/linear-gradient oracles, analytic derivative/cache snapshots,
bilinear symmetry, raw operators under a rank-zero inverse and actual existing
GMRES driven by native fields with an independent final residual. Distributed
sliding tests use generated regular meshes, empty ranks, distinct boundary/cell
owners, changing poses and direct return transfers; baseline boundary replication
checks routing while analytic point/ID oracles check geometry independently.

## Validation record

Earlier milestones passed nine focused tests and then 21 expanded registrations.
The frozen completion C++ gate passes all 29 registrations at np1/2/4/8,
including the existing GMRES/native-field mock, and repeats after commit.
Fresh built/installed Python tests pass all four cases per rank at np1/2/4.
All five existing legacy reconstruction/limiter registrations pass. Benchmark
smokes at np2 degree2/3 also pass reconstruction and full WBAP output agreement.
One intermediate run mixed object files across a changed diagnostics-structure
ABI during ongoing edits; its four failures are retained as invalid build
evidence. Rebuild all dependent targets after the source is frozen before using
the test result. Never weaken an assertion to accommodate a stale binary.

## Matched performance checkpoint

BSSCA job `12412961`, candidate `154f9098daa65d0c37084ea0c6af698dbb8b004f`, completed
on exclusive `n2202`, OMP=1, one core per MPI rank, exit 0:0, elapsed 5:18.
The 32-task request allocated the entire 64-core node. Five trials/configuration
alternate old/new execution order; both consume the same mesh, means, BC
penalty and halo. Maximum coefficient discrepancy across 25 trials: `3.33e-16`.

| Case | Metric legacy/new s | Preparation legacy/new s | Sweep legacy/new ms | WBAP legacy/new ms |
| --- | ---: | ---: | ---: | ---: |
| 262144 hex, np32, degree1 | .4652/.1736 | .6240/.2813 | 15.890/4.728 | 31.204/13.050 |
| 262144 hex, np32, degree2 | .6727/.2282 | 2.6647/.9292 | 36.285/18.049 | 99.499/55.803 |
| 6400 quad, np1, degree2 | .0659/.0154 | .2901/.0777 | 9.631/2.737 | 22.888/11.977 |
| 6400 quad, np2, degree2 | .0347/.0080 | .1500/.0404 | 4.910/1.404 | 11.443/5.921 |
| 6400 quad, np4, degree2 | .0184/.0044 | .0776/.0217 | 2.543/.754 | 5.904/3.103 |

These are max-rank medians, not universal speedups. Raw logs/source manifest/
accounting: `data/cfv-completion-20261031/cluster-run-12412961/`. Remote source,
build and logs remain under `/public4/home/sc56867/zhy/dndsr-audit-perf-20261029/`
`cfv-completion-20261031/`. Compact records belong in plan
`experiments/fv-completion-20261031/`. Final completion-source timing is recorded
separately below; do not attribute this checkpoint's timing to later code.

## Final completion-source performance

Candidate `52d90e18ddbadf2d8953a33eae7609d1f08e6e0e` contains implementation
`e76c9b65` plus a stronger benchmark check of every limited coefficient after
WBAP. BSSCA job `12413077` completed exit0:0 in5:28 on one exclusive EPYC7452
node `n1515`, OMP1, 32 tasks requested/64 CPUs allocated. GCC12.2/OpenMPI4.1.2
launcher/libraries/components were verified against one installation. Eight
configurations have five trials each; all40 pass. Maximum reconstruction/WBAP
differences are `3.37e-15`/`1.71e-14`. The benchmark rejects differences >=1e-8.

| Case | Metric legacy/new s | Preparation legacy/new s | Sweep legacy/new ms | WBAP legacy/new ms |
| --- | ---: | ---: | ---: | ---: |
| 262144 hex, np32, degree1 | .4825/.2082 | .6237/.3137 | 15.223/7.510 | 30.381/13.856 |
| 262144 hex, np32, degree2 | .6883/.2594 | 2.6970/1.3129 | 35.458/19.977 | 94.994/56.384 |
| 6400 quad, np1, degree2 | .0663/.0163 | .3063/.0771 | 9.274/4.446 | 22.686/11.977 |
| 6400 quad, np1, degree3 | .0676/.0159 | .5831/.1929 | 15.878/7.676 | 44.016/23.474 |
| 6400 quad, np2, degree2 | .0348/.0088 | .1564/.0411 | 4.839/2.291 | 11.158/6.086 |
| 6400 quad, np2, degree3 | .0352/.0089 | .2949/.1038 | 8.037/3.920 | 22.067/11.987 |
| 6400 quad, np4, degree2 | .0185/.0047 | .0803/.0227 | 2.507/1.192 | 5.818/3.138 |
| 6400 quad, np4, degree3 | .0191/.0047 | .1512/.0579 | 4.268/2.079 | 11.593/6.272 |

These are max-rank medians on matched source/options/inputs within each trial.
Setup and reconstruction order alternate; the single WBAP timing always runs
native first. That ordering is a limitation of the WBAP performance comparison,
not of the independently checked outputs. The final code retains more caches
and raw linear blocks than checkpoint154f; compare each against its matched
legacy control rather than treating different nodes/footprints as identical.
Large-case combined setup is1.572 versus3.385 s and sweeps19.977 versus35.458ms,
so the comparable-or-better setup/iteration gate is satisfied for the tested
envelope. Rotated/axisymmetric cases have manufactured-field UT evidence,
not separate timing claims. No full flow-solver accuracy is implied.

Raw evidence: `data/cfv-completion-final-20261031/cluster-run-12413077/`.
Compact records: plan `experiments/fv-completion-20261031/final-12413077/`.
The canonical remote checkout and dependencies were unchanged; source/build were
paired in an isolated directory. Local validation hashes are retained in
`data/cfv-completion-final-20261031/local-validation/` and the plan's compact
`local-validation.json`. The plan validator still verifies all23 original topics
and their unchanged brief hash; it is not a numerical validator.

## Explicit limits

No new flow solver, CUDA VR kernels, tree mesh, anisotropic functional, LVR
penalty, CWBAP variant, variable degree, nonhomogeneous derivative targets or
separate BC quadrature control is implemented here. The isotropic `CellMax`
length is symmetric/aligned, deliberately not the old asymmetric inertia-box
formula. Interface matching supports linear/bilinear faces and point sampling,
not conservative flux/FSI/ALE. Domain pose is external to mesh geometry.
Collective callers coordinate settings, epochs and producers; eager scheduling
does not infer PDE dependencies. Local SVD preconditioners do not guarantee SPD
under arbitrary cutoffs; PCG eligibility/nullspace handling remain caller duties.
