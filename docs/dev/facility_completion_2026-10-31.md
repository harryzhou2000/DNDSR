# CFV/Geom completion execution checkpoints

The legacy prerequisite is committed on `dev/harry` as `2fb93c44` (SVD filter,
rank/condition reporting and OMP reductions), with aggregate-test registration
in `4495df8d`. The refactor was rebased from `c171c318` to `6bd7af2c`, including
the upstream AUSM change; restored CPU facility work is committed as `cfc80270`.
Recovery stash `34430194d7a3b21e205120682a3ef6bb6a225330` remains retained.

## Implemented in the current follow-up

* Every native VR local system uses explicit SVD truncation. Relative cutoff is
  `1e-13` by default; absolute cutoff is zero. Equality is discarded explicitly,
  independent of Eigen's rank threshold equality convention. Deficient systems
  get a truncated minimum-norm solution. Global owned-cell counts and per-cell
  raw/retained conditions/rank use the existing factorization.
* `EntityWorkPartition` performs MPI-local declared-path classification, with
  conservative node-star cell policy, immutable lists and generation validation.
* `SparseBatchExchange` reuses direct transient row transfers for fixed-record
  tables, count discovery, candidate fan-out and variable reply counts.
* `DistributedInterfacePointMap` supports one coarse rank box and a replicated
  selected-boundary baseline. Exact proposals return to query origins, which
  resolve seams or reject ambiguous interiors. It distinguishes geometry owner
  and donor-cell owner. Poses leave mesh reference coordinates unchanged.
* `DistributedInterfaceSamplingPlan` routes donor points once, computes on actual
  donor-cell owners and returns values through a reversed row route. Callers can
  use an explicit reusable workspace and persistent result binding. Geometry,
  sampling and field halos remain separate responsibilities.

## Local correctness evidence

The post-rebase baseline passed ten CTest registrations: spectral + complete
legacy reconstruction at np=1/2/4; initial FV and boundary interfaces np=1/2/4.
The new SVD tests pass three spectral cases (29 assertions) and native FV at
np=1/2/4. The equality test caught Eigen's inclusive threshold behavior; the
implementation now applies the strict predicate itself. A zero-functional
native test verifies zero rank, counters and the minimum-norm result.

Work-partition tests pass np=1/2/4 against a generated Cartesian ownership
oracle, including nodal inputs, multi-hop traversal, empty peers, stale layout,
geometry-only preservation and incomplete deep closure. Built-in adjacencies
use the established fixed-slot read facade; named/custom entries use descriptor
lookup. This distinction was found by the tests and corrected.

Sparse-batch tests pass np=1/2/4 for a registered POD record, multi-peer rows,
variable reply counts, unsent rows, zero-count peers and collective invalid
width/tag rejection.

Distributed matching and sampling pass eight CTest registrations at np=1/2/4/8.
Regular Cartesian boundary fixtures have independent donor IDs, geometry/cell
ownership and reference-point answers. Tests include rotating/translating poses,
seams, empty geometry/query ranks, unmatched points, duplicate accepted interiors,
collectively inconsistent settings and matching-mode agreement. Sampling uses
two manufactured linear polynomials with distinct owner offsets, exact output
order and changed donor values through a persistent result binding. The
all-gather baseline validates routing independently of the boxes but shares the
projection kernel; physical point oracles independently check that kernel.

Fresh Python extensions built and installed successfully; all four facility
tests pass in explicit MPI mode at np=1/2/4. No new
matched cluster performance result is recorded yet. The prior LU timings do not
establish SVD setup performance. Core/halo numerical splitting, locality reorder,
expanded FV gradients/functional/weights, full periodic/axisymmetric support,
and Krylov operator interfaces remain outstanding.

## Second numerical checkpoint

Selected-row reconstruction and WBAP entry points now keep Jacobi input
immutable, omit ghost leases for core work and require an explicit completed
output commit. Core-first SOR uses a distinct frozen exchange snapshot and an
owned working iterate. Scalar activation uses `indicator >= threshold`, with
inactive passthrough and finite-value validation.

Standalone Green–Gauss, inverse-distance-squared/face-measure weighted LS and
GGMP gradient plans use physical BC sample buffers. Explicit gradient-to-VR
conversion scales linear modes and zeros higher modes. Regular/sheared 2D and
3D manufactured linear fields test their intended exactness envelope. Periodic
and axisymmetric gradient support is still pending.

The overlap tests use real direct MPI operations and both inline and worker
eager CPU launchers, checking split/unsplit results and illegal early commit.
All eight overlap/gradient registrations pass np=1/2/4/8. A ninth math test
independently checks Taylor chain-rule pullbacks, tensor Frobenius invariance
through degree three and legacy directional/geometric weight formulas. These
central math helpers are tested building blocks; public native VR weight-option
integration and rotated-periodic kernels remain work in progress.

The local matched benchmark target builds and its 9-cell degree-one smoke
passes, with maximum coefficient difference `8.33e-17`. This smoke validates
the executable/gate, not setup performance. Matched one-node cluster timing is
planned separately in the standalone plan repository.
