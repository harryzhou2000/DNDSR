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
including the existing GMRES/native-field mock. Fresh Python build/install and
the final matched benchmark remain pending at this checkpoint.
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
`experiments/fv-completion-20261031/`. Final completion-source timing remains a
separate required check; do not attribute this checkpoint's timing to later code.

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
