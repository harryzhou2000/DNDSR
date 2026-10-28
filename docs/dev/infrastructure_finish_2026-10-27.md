# DNDS/Geom infrastructure finish, 2026-10-27

> **Correction, 2026-10-28: full infrastructure acceptance is withdrawn.**
> NativeMeshAssembler reconstructs its own GhostSpec and omits the legacy
> Bnd→Node→Bnd boundary ring. An independent check of the recorded two-rank box
> payloads finds 270 required remote boundaries missing on rank 1. Initial
> construction includes the ring, but later assembly drops it. Mapped owned
> geometry/numerical parity does not establish local ghost coverage. The timings
> below remain measurements of the tested implementation, not performance
> acceptance for a closure-equivalent replacement. Builder/face construction
> should produce owned topology with temporary construction closure; the
> assembler must receive the final GhostSpec rather than invent it. That repair,
> independent closure regressions and corrected performance measurements remain
> pending. Historical results below are retained for provenance.

The work starts at `aa190a8ba4dce16e7811f19be1b1dd94c3be3bfe` on
`codex/refactor-dnds-array-geom`. The independent planning repository committed
the work order and new-solver milestones in `68ab0c5`. Implementation remains
in the existing refactor checkout. The three work-order acceptance gates are
complete with frozen v11 source and the evidence below. Changes remain
uncommitted; the phase does not implement the new fluid solver.

## Native algebra

[DNDS::Algebra](../architecture/NativeDofAlgebra.md) composes Runtime Arrays in
an immutable shared `DofSpace` and `DofField<T, Components>`. It provides float/
double owned-entry pointwise operations, local and global reductions, immediate
CPU convenience and explicit eager CPU/CUDA submission. Logical component width
is separate from physical stride. Ghosts and padding are excluded. Equal row
counts do not substitute for row association; callers create a new space after
reordering. Handle copies share storage and numerical copying is explicit.

Validation covers widths 5/6/7/114, static width 6, zero/uneven ranks, padding and
ghost sentinels, exact aliases and rejected partial overlap, retained temporary
fields, worker/inline scheduling, explicit cross-stream joins, collective
preflight failures/retry and an independently checked diagonal Krylov solve.
CUDA pointwise operations and reductions are functional baselines. The initial
CUDA reduction is sequential per component and has no throughput claim.

All four Python modules were rebuilt and installed in both CPU and CUDA
profiles. Installed extension and implementation-library paths/build IDs and
`Runtime.cuda_enabled` were checked before tests. CPU Python at MPI 1/2/4 passed
24 cases per rank with three CUDA-only skips. CUDA Python at MPI 1/2 passed 27
cases per rank, selecting distinct physical GPUs (one rank per GPU). CUDA algebra
passed 77,940 assertions and Compute Sanitizer reported zero errors. The detailed
caller/lifetime rules and public API are in the algebra documentation.

## Source boundaries

[Source organization](../architecture/source_organization.md) records 196
relocations and the separate symbol map. Old Array/ParArray/Pair/Transformer,
derived arrays, DOF numerics, bindings and generators live under `DNDS/Legacy`.
Old headers forward and original `DNDS::*` names remain the same classes;
`DNDS::Legacy::*` explicitly names them. `DNDS::Runtime::Array` remains the new
API. Root namespace promotion is reserved for an explicit later API cut.

Geom sources are grouped by Unstructured, Access, Connectivity, Transition,
Geometry, Refinement, IO and Compat responsibilities. The primary state and
snapshot names are `UnstructuredMeshState` and `UnstructuredMeshSnapshot`;
historical names are source aliases. This rename requires rebuilding compiled
C++ clients. Canonical native access/state headers do not pull in legacy Array;
the public compatibility facade and old serializer still have explicit legacy
dependencies. No second mesh storage authority was introduced.

Path/symbol, native-only include and old/new include coexistence checks passed.
All relocation targets exist and CMake explicitly enumerates relocated sources.
Current guide references were updated; historical source/line/hash evidence was
preserved. Existing draft files were excluded from snapshots and left untouched.

## Setup work and current evidence

Known-shape row construction now fills one final allocation, Array row queries
avoid layout copies, retained relation scopes validate contents once, and exact
aliases avoid repeated conversion/ghost transfer. Periodic role-key pruning was
checked against all eight XOR masks on 120,000 independently generated cases.
An unweighted reader-owned partition graph can be built from primary incidence
without first publishing faces. Weighted/distributed/already-facial inputs keep
the general path. Final topology, IDs and evaluator data remain checked.

The first pinned v4 comparison used one exclusive BSSCA node, four active cores,
20,816 NACA cells (5,204/core), and intended 32,768 box cells (8,192/core).
NACA MPI 1 medians were legacy 0.714930 s, previous native 5.566310 s, optimized
3.389756 s. One completed MPI 2 sample was 0.532470/3.919557/2.109520 s.
These results miss the planned performance target. Job `12386922` stopped on an
over-strict rank-count evidence check despite exact logical/numerical parity;
the incomplete campaign is retained and is not a completed performance result.
The preceding `12386889` failed at configuration because the private source
package omitted `examples/`; the next snapshot included the actual files.

A later allocation/alias pass retains exact local NACA and box representation,
geometry and numerical bits. A cache audit added the case where every rank has
an alias hit on a different earlier descriptor: reusing those different
collective results can silently select a donor's wrong row. Both ghost publication
and transition caches now require agreement on the same prior descriptor before
reuse. Nonuniform-payload MPI tests cover that distinction.

Step-level legacy/native profiling completed as a separate build-only overlay
(job 12387235). It places the excess in topology/publication, general transitions
and sparse traversal. FV/evaluator initialization is comparable to legacy.
The intermediate v5 path builds topology twice, assembles four times and
publishes ghost relations eight times (including the separate boundary mesh).
The reader graph itself is separate from the face completion after partition.
The plan repo's `SETUP_ALGORITHM_COMPARISON.md` records the source correspondence.

Uninstrumented v6 job 12387281 completed all 36 launches and paired correctness
checks. Setup medians (legacy/previous native/v6) were 0.688/5.389/2.186 s and
0.489/3.797/1.487 s for NACA MPI 1/2; box medians were 2.716/27.872/8.949 s and
2.098/20.347/6.549 s. These still miss the at-most-two-times-legacy target.
Later local work removes transferred copies of local DSL rows, repeated support
normalization, identity remapping of unchanged families, and intermediate row
allocations. It is not covered by those v6 timing claims.

Explicit `MeshAuditOptions` now separates optional repeated diagnostics from
construction. Persistent-ID uniqueness/ghost-ID and cell-face reciprocity checks
default to full. Trusted setup can opt out; `state.Audit()` rechecks a checkpoint
without rebuilding or changing policy. Required local closure construction,
layout/index/sentinel validity, periodic matching, leases and collective schedule
checks remain enforced. Policy propagation and deliberate corruption/retry tests
pass MPI 1/2/4/8. Future timing reports must label full and trusted modes separately.

Broader archive regressions exposed a byte-I/O compatibility bug: JSON's native
byte segment API rejected the established in-memory DOM mode. It now enforces
file mode when a file is open and retains DOM read/write behavior otherwise.
Direct in-memory Array/byte tests and 32 MPI registrations, including refinement,
transition and mesh-edit archive cases, pass with that fix.

A broader corruption-path run found an inverse-connectivity collective-order
bug. Deliberately corrupting a raw relation could throw during packet packing on
one rank while peers entered `Alltoall`. Packing now participates in the existing
collective error agreement before communication. The bad-input/retry regression
passes at MPI 1/2/4/8. The corrected broad Geom run passed 191 registrations;
DNDS passed 66. These are incremental checks, not evidence for later source edits.

Later matching work replaces per-candidate hash-node allocation with compact
sorted occurrence counting and uses fixed scratch arrays for small subentities.
Searching the smallest vertex's reverse list preserves the first accepted
entity while avoiding repeated rejected candidates on other vertices. Focused
composition and matching suites each passed 12 registrations. Local large-mesh
probes at MPI 2 retained exact native representation and numerical bits; their
timings are not used for performance acceptance.

The frozen v10 matching path passed 24 focused MPI registrations and another
191-test Geom run. Python checks passed 26 cases per rank at MPI 4 in the CPU
profile (two CUDA skips), 28 per rank at MPI 2 in the CUDA profile, and 67 per
rank for the full Python Geom directory (two skips). CUDA facade/overlap/algebra
checks passed three registrations and native algebra retained zero Compute
Sanitizer errors across 77,940 assertions. Extension paths, build IDs and CUDA
availability were checked after rebuilding and installing each profile.

The frozen v8 exclusive-node campaign (12387849) completed all 36 sequential
launches, with exact full/trusted comparisons. Full-audit setup remained
2.27–2.50 times legacy; trusted setup remained 2.23–2.48 times legacy. Optional
audits explain only a few percent. The subsequent v11 source removes full scans
used merely to obtain descriptors and validates pinned raw rows without repeated
public-access scope checks. Local indices are converted to global IDs only when
sorting or remove-self semantics need them; unavailable encodings always retain
range and mapping checks. Sixteen focused MPI registrations and both large-mesh
parity probes passed.

## Final acceptance

BSSCA job `12387989` completed with exit `0:0` in 23:02 on one exclusive node.
All 36 sequential launches used four pinned active cores: MPI 1/OMP 4 and
MPI 2/OMP 2. Cases contain 20,816 NACA and 32,768 periodic-box cells, or
5,204/8,192 cells per active core. Each entry below is the median of three
maximum-rank setup timers in seconds, including mesh and FV/evaluator setup.

| Case/ranks | Legacy | Full audit | Trusted | Full / legacy |
| --- | ---: | ---: | ---: | ---: |
| NACA/1 | 0.695267 | 1.308212 | 1.278531 | 1.882 |
| NACA/2 | 0.496530 | 0.932478 | 0.911317 | 1.878 |
| box/1 | 2.765091 | 4.961358 | 4.906347 | 1.794 |
| box/2 | 2.123894 | 3.795714 | 3.743936 | 1.787 |

Full audits meet the work-order limit of twice legacy in every case. Relative
to the previously recorded native medians, setup is 4.07–5.62 times faster;
that comparison crosses campaign nodes, whereas full/legacy pairs above share
the same node and active cores. Residual overhead is explicit: native setup is
still about 1.8–1.9 times legacy, not equal-cost. Audit opt-out is not required
to meet the target.

Strict v6-native/v11, v8-native/v11 and full/trusted comparisons preserve raw
representation, logical topology and finite numerical bits. Periodic legacy
labeling retains the established difference; independent cell-incidence mapping
verifies physical cell/face/boundary geometry and finite numerical parity.
Elevated and bisected MPI-2 probes also preserve earlier native representation
and all emitted numerical bits. Source/archive hashes, binary identities,
maintained input equivalence, actual bindings and audit flags are recorded in
the independent plan repository's v11 acceptance evidence.

Final local CPU checks passed 66 DNDS, 191 unique Geom and 11 CFV/Euler
registrations. The broad Geom process disappeared after 150 passes during the
goal continuation; the remaining 41 distinct registrations passed separately.
Python CPU MPI 4 passed 26 cases per rank (two CUDA skips). CUDA MPI 2 passed
80 combined algebra/Geom cases per rank (two skips), assigning one physical GPU
per rank. Three CUDA C++ registrations passed. The unchanged algebra backend
previously passed 77,940 assertions with zero Compute Sanitizer errors.

Switching build profiles exposed a CMake timestamp collision that left CPU Geom
installed alongside CUDA Runtime. This failed explicitly in the mesh placement
test. Reinstallation and ELF build-ID comparison of all four extensions and all
four implementation libraries resolved it; the corrected combined run passed.
The plan repository contains a reusable installation verifier and retained
failure/provenance records. The final installed local profile is CUDA.

Raw payloads, profiles and logs stay under
`data/euler-migration-parity-20261026/setup-finish-20261027/` and the corresponding
private BSSCA campaign directories. Compact source/build/run records and final
results belong in the independent plan repository. This report makes no
whole-solver speed, physical convergence, new-fluid-solver or GPU-solver claim.
