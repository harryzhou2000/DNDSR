# Euler mesh and numerical migration parity

The reference is `03891f8b5869017de355468d93294e3eb25f0feb`, immediately before
the Array refactor. The migration checkpoint is `0231ddba`. Both checkpoints
are distinct from later unrelated changes on `dev/harry`. The isolated legacy
snapshot's 398 tracked source files were verified against their Git blob hashes;
only a build target for the common probe was overlaid outside `src/`.

The same [probe](../../test/cpp/Euler/MigrationParityProbe.cpp) is compiled
against both implementations. Builds use the same GCC, Release options,
dependencies and OpenMPI 5.0.9. Local runs use one or two MPI ranks, two OpenMP
threads per rank (`OMP_NUM_THREADS` **and** `DNDS_DIST_OMP_NUM_THREADS`), and one
OpenBLAS thread. The experiment plan and checked orchestration/comparison scripts
live in the independent `workspace/dndsr-next-plan/experiments/euler-migration-parity`
repository directory. Raw data stays under the refactor checkout's
`data/euler-migration-parity-20261026/`.

## Findings and corrections

1. Source KWAY input was distributed before partitioning, selecting ParMETIS
   instead of legacy reader-owned METIS. The serial METIS options also differed.
   Source helpers now retain reader-owned ingress and restore the historical
   unweighted options. An independent vertex-intersection graph/direct-METIS
   oracle checks KWAY/RB, seeds, cut counts and a nonzero reader rank.
2. Cell reordering consumed face-slot graph order, whereas legacy setup used
   sorted, duplicate-free neighbors before face interpolation. Local ordering
   now restores that graph order. Face topology is refreshed after permutation,
   retaining persistent IDs, provenance and local execution partitions. A
   parent-major element-face oracle checks oriented nodes, parent slots and
   periodic image bits for primary-only and already-facial input.
3. Periodic boundary placement followed the minimum owner of both coupled
   cells. It now follows its own physical side, as legacy source setup did.
   The coupled support remains available; explicit whole-support followers
   retain their original minimum-rank policy. A missing named side relation
   is derived from the canonical physical slot, without truncating support.
4. Pure scaling passed through an identity matrix multiplication, changing
   arithmetic order and potentially signed zero. Pure scaling/rotation and
   identity now preserve the legacy operations. Bit tests include subnormals,
   signed zeros, fractional coordinates and large finite magnitudes.
5. Setup profiling found redundant primary-topology completion and repeated
   generation-mutex checks inside already-leased incidence traversal. Partition
   now completes only missing faces. Private pinned indexing is used only by
   `RelationReadAccess`, which already owns source/target epoch leases; public
   descriptor queries keep their validation and all row/index bounds remain.

## What “same mesh” means

Raw representation, physical geometry/topology, and numerical values are
separate comparisons. Legacy periodic import compacted Node IDs after merging
periodic vertices; native import retains source IDs. The supplementary comparator
records an explicit bijection using **unique exact IEEE coordinate bits**, then
complete element/image/zone/side signatures. It rejects ambiguity, unmatched
entities, and coordinate differences; it never rounds coordinates or silently
intersects ID sets. The strict original comparison is retained separately.

Physical cells, face incidence/orientation, boundary own-side geometry, periodic
frames, cell/face metrics and numerical fields can therefore match exactly while
raw node labels or ghost-boundary storage differ. Legacy periodic `bnd2cell`
duplicated its physical-side cell in the reserved second slot. Native physical
incidence exposes the actual periodic partner there. That additional support is
reported as a representation/contract extension, not silently erased by comparison.

## Numerical evidence

The first completed matrix has 20 rows: one and two ranks for periodic quad and
triangle vortices, reordered/non-reordered KWAY and RB setup, rotation plus
non-power-of-two scaling, O2 elevation, bisection, small SA wall and NACA0012 cases,
and a periodic 3D box. All owned initialization, timestep, RHS, implicit increment,
cell-volume and barycenter components match IEEE bits. Face geometry and node
coordinates match through the explicit identity correspondence.

Four additional rows execute the actual P1/P2 implicit solver for two physical
steps and compare final solution and reconstruction bits. They match. These are
bounded execution fingerprints with an internal cap, **not convergence or
accuracy evidence**. The fixed-state evaluator probe deliberately uses zero
reconstruction; only the advance rows exercise P2 reconstruction evolution.

The comparator has synthetic bad-input tests for duplicate global owners,
missing fields, unmatched IDs, periodic-image changes, orientation reversal,
signed zero, one-bit corruption and ambiguous coordinate matching.

Three additional comparisons read an archive written by the actual legacy
solver: mode 1 on its two writing ranks, and mode 2 repartitioned onto one and
two ranks. All owned state and physical geometry bits match after the explicit
identity correspondence. This tests real legacy mesh input, in addition to
the dedicated solution-restart fallback tests from `7424be0f`.

Curved NACA0012 elevation was tested with both RBF variants on two ranks.
Local RBF (`meshElevationInternalSmoother=0`) matches all measured bits.
Global RBF (`=1`) preserves topology but is **not byte identical**: 1,190 of
186,732 coordinate components differ, by at most `1.1102230246251565e-16`.
An additional correspondence based on stable Cell IDs and identical element
node roles/images proves entity identity here without requiring equal geometry.
It still compares every coordinate/metric bit and rejects incomplete or
inconsistent mappings. Face normals differ by up to `6.55e-12`, and the resulting
RHS differs by up to `3.95605e-5` absolute; those differences are not hidden by
tolerances. Initialization remains identical. Constraint order, accumulation
order, and multiply/divide expressions differ from legacy global RBF; this is
a demonstrated limit of exact reproduction, not an accuracy or convergence
claim about the perturbed result.

## Limits

- No unconditional legacy-bit guarantee is claimed for every configuration.
  Overflow-safe integer edge weights intentionally differ from unsafe legacy
  weighted partition scaling. Exact-face topology also avoids the legacy
  vertex-intersection heuristic on exceptional/nonconforming input.
- Global RBF smoothing changed constraint/block ordering from unordered
  containers to deterministic ordering, and some floating-point expressions
  differ. The explicit curved-wall comparison above fails bit parity.
- Wall-distance vectors now rotate with geometry and refresh after snapping,
  as requested earlier. Those corrections can intentionally change legacy
  output for affected wall cases.
- EulerP's earlier HY correction uses total Roe speed instead of signed normal
  velocity. That physics correction is a distinct reason not to promise all
  EulerP configurations reproduce the old bytes.
- Different compilers/options, MPI sizes, thread schedules, devices or
  reduction order are separate experiments. Output container schemas,
  timestamps and diagnostic text are not numerical payload bytes.

## Performance measurement

Local timing is retained only as a diagnostic because this machine is shared.
It identified significant one-time setup overhead and near-legacy evaluator
timing after the redundant topology/generation checks were removed. The
authoritative comparison uses sequential paired runs on one exclusive BSSCA
node, with identical pinned cores, compiler, MPI and dependencies. Each of five
pairs contains 31 evaluator sequences; the first sequence is warmup. The report
uses the median of complete dt/RHS/matrix/solve sequences, then the median of
paired native/legacy ratios. It excludes reconstruction/limiting and full
time integration, so these numbers are not whole-solver speedups.

The private cluster script pins the MPI libraries and headers as well as the
wrappers: an inherited `CMAKE_PREFIX_PATH` otherwise mixed two MPI installs.
It also uses an available Make generator and complete tracked source snapshots.
Source-record hashes are authoritative; an archived source nested under the
canonical checkout can inherit that parent's unrelated Git version banner.
All failed startup attempts and their corrections remain in the evidence.

BSSCA job **12384513** completed on the exclusive node **s1609** (AMD EPYC
7452), with GCC 12.2, OpenMPI 4.1.0-para/UCX and two OpenMP threads per rank.
Five sequential pairs per case/rank count completed: **60/60 timed launches,
30/30 comparisons**. Every owned-state field matches bits in all 30. Periodic
IV/box raw Node IDs still differ as described above; those cluster rows do not
independently establish mapped geometry parity. NACA's strict geometry and
logical comparisons pass in all ten rows.

Times below are medians; ratios are medians of paired native/legacy ratios.
Setup includes mesh preparation and FV/evaluator initialization. Evaluator is
the complete warmed dt/RHS/matrix/SGS sequence, in milliseconds.

| Case (cells) | Ranks | Setup legacy/native (s) | Setup ratio | Evaluator legacy/native (ms) | Evaluator ratio |
|---|---:|---:|---:|---:|---:|
| Periodic IV (100) | 1 | 0.034 / 0.083 | 2.42× | 0.995 / 1.002 | 1.007× |
| Periodic IV (100) | 2 | 0.034 / 0.088 | 2.56× | 0.546 / 0.581 | 1.064× |
| SA NACA (20,816) | 1 | 0.840 / 5.678 | 6.76× | 410.196 / 394.945 | 0.960× |
| SA NACA (20,816) | 2 | 0.506 / 3.929 | 7.77× | 201.236 / 198.332 | 0.980× |
| Periodic box (32,768) | 1 | 3.618 / 29.520 | 8.14× | 820.363 / 790.216 | 0.962× |
| Periodic box (32,768) | 2 | 2.152 / 20.974 | 9.75× | 419.663 / 401.861 | 0.961× |

The [complete timing report](https://github.com/harryzhou2000/dndsr-next-plan/blob/main/evidence/euler-migration-parity-2026-10-26/bssca/timing-report.json)
contains every paired observation, stage medians and ranges. Evaluator ratio
ranges are 1.000–1.021/1.058–1.073 for IV, 0.950–0.964/0.980–0.994 for NACA,
and 0.957–0.976/0.957–0.964 for the box at one/two ranks. These are observed
ranges, not statistical confidence intervals. The tiny two-rank IV regression
is about 35 microseconds per measured sequence.

**Setup remains a material regression.** Removing duplicate completion and
per-incidence generation locks does not make it performance-equivalent to
legacy setup. Native topology publications, exact identity recovery, repeated
sparse exchanges, remapping and full validation remain setup work. The earlier
profile establishes these as optimization targets, not a complete post-fix
breakdown. Further setup optimization needs a separate measured pass; disabling
leases or weakening identity checks is not a justified shortcut. The measured
hot path is close to legacy, but reconstruction-heavy/time-step workloads need
their own timing before making whole-solver claims.

## Validation and reproduction

The [compact validation record](https://github.com/harryzhou2000/dndsr-next-plan/tree/main/evidence/euler-migration-parity-2026-10-26)
contains campaign manifests, source/input/binary hashes, per-field comparison
counts and retained test logs. The full raw probes remain in `data/`; large rank
payloads and the profiler recording are not copied into documentation.

The final focused CPU run passes **28/28** registrations: preparation,
geometry edits, leased relation access/conversion, native connectivity/registry,
and the real Euler evaluator pipeline, each at MPI 1/2/4/8. Before the last
formatting-only pass, another 24-registration geometry selection had 23 passes
and a 120-second timeout in the 958,994-cell Ball2 mesh pipeline at eight ranks.
The unchanged test passed when rerun alone with a 600-second bound (287.58 s).
That initial timeout is retained; the fixture and its purpose were not reduced.
The comparator/correspondence/timing reducer tests pass **20/20**, including
malformed and incomplete evidence. These tests do not substitute for the actual
paired solver runs.

The rebuilt CUDA mesh-view and two-rank compute/communication-overlap smoke
tests pass **2/2**. They exercise device access to the retained native facade;
they do not establish CUDA solver equivalence or performance. Python bindings
were not rerun in this C++ follow-up and must be rebuilt/installed before any
Python validation against these sources.

The probe is optional:

```sh
cmake --build build-refactor --target euler_migration_probe -j4
```

Compile the same probe against `03891f8b` in a separate source/build tree (the
experiment's `bundle.py` materializes this). Both trees reuse verified existing
dependencies. The checked campaign driver takes an explicit mesh root, MPI
launcher, case, rank counts, new output tag and wall bound. For example, from
the independent plan repository:

```sh
python3 experiments/euler-migration-parity/run.py \
  --root /home/harry/projects/DNDSR-array-geom \
  --case iv_p1 --np 2 --np 1 --tag new-iv-parity --repeats 5 --timeout 120
python3 -m unittest discover -s experiments/euler-migration-parity -p 'test_*.py'
```

`run.py` emits generated config copies, keeps maintained cases untouched,
serializes process runs, bounds each process group and refuses existing output
directories. `mapped.py` adds explicit identity correspondence without changing
the strict comparison. `report.py` requires the declared pair count and rejects
failed/duplicate/incomplete/nonfinite results before reducing timings.
