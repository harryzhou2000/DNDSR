# Runtime and mesh audit, 2026-10-29

Audit baseline: `1fdc3f33`. Work is in progress; performance acceptance has not
been established. The plan/evidence repository contains
`docs/26-runtime-mesh-audit-parallel-performance.md` and reproducible profiling
tools. The 5% target applies to parallel mesh setup, excluding serial reader
construction and FV initialization.

## Findings

1. Typed projections from uniform CSR/padded arrays inspected metadata through
   the kernel view on the host. CUDA pointers are invalid for that inspection.
   Projection now validates retained host `RowLayout`, then emits borrowed
   device pointers. Direct leases and eager submissions use the same helper.
   Tests cover valid uniform projections and rejected irregular/padded shapes.
2. Signed public row indices were backed by unchecked unsigned shape extents.
   Empty arrays could accept a count beyond `INT64_MAX`, then expose a negative
   row count without allocating a large buffer. Layout creation now validates
   signed representability, including cumulative offsets. Known MPI routes now
   reject negative extents collectively before publication; previously an empty
   route reached a later rank-local exception.
3. One-shot setup commonly exchanged identical row widths twice: first to
   allocate the destination, then to compile its payload. PayloadPlan now offers
   an inferred-destination overload, reusing the first exchanged lengths. The
   row route remains independent of payload shape, and the explicit-destination
   API retains its complete validation. Sparse holes, zero widths, different
   rank-local shapes, colliding incompatible widths and transfer-policy mismatch
   have explicit tests.
4. Blocking Geom ghost transfers duplicated a communicator for each payload.
   Their binding dies after the wait, so they can share the route context and
   reuse the retired tag. Independent asynchronous starts still isolate their
   namespace. TransferBinding now agrees mode/tag/reduction/transport together
   before context selection, avoiding a second redundant agreement round.
5. The documentation index directed new Array users to the historical Array
   hierarchy, while the Runtime guide still called Geom migration optional.
   Links and scope now describe the actual ownership boundary.
6. `MeshAdjacency::Read` sampled its cached value generation before acquiring
   the data lease. An alias writer could finish in that interval. Acquire the
   lease first, then compare generations and rescan while writes are excluded.
   Added coverage alternates retained reads, rejected concurrent writers,
   invalid alias writes and repaired writes on copied adjacency handles.
7. Plain payload copies compiled reduction-only CSR offsets, contribution
   groups and receive datatypes. These are now derived only when a reduction
   binding needs them; ordinary payload plans retain only direct send/receive
   layouts. Sparse idle peers share one committed empty datatype per plan.
8. Three payload construction error reductions guarded consecutive local work.
   The work now accumulates failure and agrees once before publication. Binding
   context/tag validation similarly agrees on the route communicator before
   using the selected context, and Begin combines active/stale and lease checks
   before posting requests. Reordered-context and rank-local tag failure tests
   prove rejection and subsequent tag reuse; this does not remove validation.
9. Redistribution rebuilt identical CSR offsets and converted each global row
   before remapping it. It now shares the immutable layout and fuses the copy
   with remapping. Persistent ID companions reuse the validated row route when
   source extents agree. Cone completion uses the pinned owned interval instead
   of locking the version tracker and finding a rank for every incidence.
10. A topological descriptor and a sorted descriptor could send the same
    adjacency twice. Redistribution now reuses one unchanged-slot wire payload
    and sorts an independent receiver copy only where its semantic descriptor
    requires it. Every descriptor is validated before reuse, including a new
    test with an invalid sorted alias on one rank. Already sorted receiver rows
    retain the original storage. Cross-rank alias agreement remains exact.
11. The adjacency remap now also applies the sender's partition-bucket row
    permutation. The adjusted known route preserves peer correspondence and
    destination rows, while direct MPI sees contiguous source row intervals.
    The original mesh and its semantic transfer maps remain unchanged. This
    requires no payload pack/unpack workspace and works at every rank count.
12. Enumeration caches each created entity's sorted vertex key instead of
    reconstructing it for every candidate. A custom query with a different
    vertex count under the same type tag uses the original prefix comparison.
    Scratch scales with locally enumerated entities, not global mesh size.
13. Binding controls now use one exact integer min/max agreement instead of
    serializing fixed integer fields through string broadcasts. Custom reducer
    names still use exact string agreement. A plain direct copy has no workspace
    to prepare, so it no longer executes an empty workspace-result reduction.
14. Boundary extraction fetched ordered face cones/periodic bits from their
    owners although the retained volume mesh already published those rows.
    It now reads its owned/ghost face view directly, retaining node-set and
    physical-frame validation. The high-order fixture explicitly checks that
    boundary owners read remotely owned faces. Parent-ID cache construction
    also uses retained array views instead of re-entering the ID API per row.
15. An old `mpi.size == 1` payload bypass remained in generic redistribution.
    It is removed: only an explicitly requested local permutation uses local
    movement. Single-rank migration exercises the same route as other sizes.
16. Terminal ghost-tree expansion sorted all reached local incidences before
    discarding owned IDs. It now filters owned targets at terminal nodes only.
    Intermediate nodes keep complete frontiers even when collected. A new
    independent oracle follows owned nodes into remote cell/node endpoints.
17. Distinct periodic node IDs determine one candidate uniform frame mask;
    testing eight masks and sorting each was redundant. Repeated-ID patterns
    retain the full multiset search. The independent full-search oracle now
    explicitly exercises both unique and repeated IDs for widths 1 through 10.
18. Adjacency descriptor comparisons serialized names, paths and provenance
    on every ordered-registry lookup. Comparison now uses the semantic fields
    directly; the wire signature stays unchanged. Registry order remains
    deterministic, but is not a persisted descriptor ordinal. The oracle checks
    equality against wire signatures, including ignored legacy `via`, native
    paths, punctuation, different name lengths and multi-digit versions.
19. Distributed inversion looked up an incidence's owner twice and reacquired
    version checks through public descriptor queries. Its existing adjacency
    scope already pins both maps. It now searches once for owner/local row and
    verifies received source IDs against that peer's immutable rank interval.
    Collective input rejection and full incidence validation remain in place.
20. Typed payload binding evaluated an Array's shape before entering collective
    validation. An unpublished/failed shape or moved-from handle on one rank
    could throw locally while peers waited in a collective. Binding now pins
    published storage and captures its buffers inside the preflight failure
    boundary. Shape editing is excluded while pinned. Both copy and reduction
    take this path; tests cover active and abandoned edits, consumed handles,
    nontrivial table strides, collective rejection and successful tag reuse.
21. Local reordering queried owned row zero to find its rank offset, which
    fails on ranks made empty by redistribution. It now uses the retained rank
    interval. An independent gather-to-one-rank/reorder/assembly test checks
    persistent IDs and every cell cone. Remap scratch collects only remote
    references before sorting, and uses retained views outside the loops. The
    remaining single-rank `PartitionOwned` bypass is removed as well.
22. Redistribution waited for every payload before remapping the next one.
    It now retains direct transfer operations with distinct tags, allowing
    communication to progress during subsequent remapping. Explicit joins
    precede receiver sorting and output publication. The operations retain
    buffers and leases; no pack buffer or progress thread is introduced.
23. Uniform table payloads still exchanged an identical length for every
    edge. Their existing schema agreement now also proves a common table
    width across ranks. Only that proven case omits the width exchange;
    mixed layouts and unequal widths retain it. Tests cover sparse destination
    holes, physical strides, rank-dependent widths and mixed CSR/table input.
24. Sender ordering covered adjacency values but left coordinates, metadata,
    provenance and IDs scattered. A subsequent candidate applies the same
    explicit row permutation to these companion arrays before direct MPI.
    This allocates an independent ordered array and copies its logical rows;
    that cost is included in setup timing. It does not change Runtime's direct
    transport default or introduce MPI pack/unpack. ID allocations retaining a
    ghost suffix keep their checked original route. A gather/reverse-scatter
    oracle covers nontrivial order, strided values, coordinates and POD zones.
    Focused mesh tests and the np4 SA exact comparison passed. Matched job
    12403115 passed both orders: high-lift parallel setup was 0.129150/0.155137 s
    legacy/native (1.2012x), cube 0.747558/0.713718 s (0.9547x). High-lift still
    fails the target; this is not an isolated attribution of every improvement.
25. Geom rebuilt a Known route after reordering already-validated send rows,
    repeating peer-count exchange and collision sorting. DNDS now derives that
    schedule through `RowTransferPlan::ReorderSourceByPeer`. Unique in-range
    source rows with complete coverage prove a bijection; the returned route
    retains receive slots, collision flags, context and epoch dependencies.
    Geom applies the separately returned permutation to arrays. CPU MPI tests
    pass at np1/2/4/8, and the CUDA transfer build passes 26 cases per GPU rank.
    Rebuilt/installed CUDA Python modules pass all 13 transient tests at np1
    and on each np2 rank. Full CPU/Geom validation passes 204 registrations;
    high-lift np4 agrees exactly with the preceding native candidate. Final
    direction-agreement checks also pass CPU np1/2/4/8 and two-GPU 26/26.
26. Construction closure manually expanded complete CellCells or node-star
    rows outside the ghost tree. It now registers those adjacencies and
    compiles/evaluates the actual recipe. Registry publication can select only
    required payloads while retaining all discovered entity maps. Unknown,
    duplicate and rank-inconsistent selection is rejected collectively. The
    closure no longer has privileged access to adjacency internals. Independent
    two-triangle expectations exercise complete-graph, retained-node-star and
    newly-inverted support, including empty ranks. Final focused tests pass
    8/8 and the full CPU/MPI selection 204/204 in 232.08 s. The np4 high-lift SA
    no-advance probe agrees exactly with the preceding native candidate in
    topology, representation and tested finite numerical fields. Source hashes
    remain unchanged after testing. The rebuilt/installed CUDA Python profile
    passes all 44 tests in five Geom modules at np1 and on each np2 rank, with
    no skips. Matched performance is pending; no time-evolved solver-history
    claim is made.
27. Follow placement, remote-number lookup and payload remapping separately
    acquired the same input descriptor's read scope during one transition.
    The transition now retains one normal MeshAdjacencyRead per descriptor
    through completion. Its leases keep values and numbering immutable; all
    descriptor-specific validation remains in the adjacency class, and byte
    aliases under different descriptors still acquire distinct scopes. No
    validation rules or unchecked view are introduced in transition code.
    This change is included in the validated closure-tree CPU candidate above.
28. EntityGhostMapping's TransferRows and
    BeginRows query the supplied Array layout before entering their failure
    Allreduce. An unpublished shape edit on one rank can throw locally while
    peers enter that collective. The typed PayloadPlan binding was already
    hardened. The new candidate captures published shared storage inside a
    catch/agreement boundary and retains it through binding. Both entry points
    have rank-local active/abandoned shape-edit and consumed-handle rejection
    tests, followed by successful strided-row transfers to prove recovery.
    This is not included in the frozen closure-tree candidate; validation of
    the new candidate passes the full CPU/MPI selection 204/204 in 222.79 s and
    focused native connectivity np1/2/4/8. The np4 SA smoke matches the
    closure-tree checkpoint exactly in topology, representation and tested
    finite numerical fields. Python rebuild/validation is in progress.
29. Tree evaluation transferred remote adjacency rows, then publication could
    transfer the exact same rows again. The registry now retains the last
    fetched frontier per descriptor. Reuse requires the exact ordered canonical
    ghost IDs plus an unchanged value version under a new read lease, agreed
    across ranks. Input scopes already pin the immutable source/target order.
    Successful reuse consumes scratch into a single publication; later
    publications get independent ghost buffers. Writes to earlier output,
    different equal-sized frontiers and larger endpoint unions are covered by
    independent row expectations. No input-mesh ghost
    buffers or inferred route-tree identities are reused. This bounded sparse
    scratch optimization passes the CPU checks above; no speedup is claimed yet.

## Responsibility and abstraction review

Performance acceptance also requires preserving these boundaries. Do not move
mesh semantics into DNDS, bypass lease/schema/epoch validation, expose storage
details to consumers, or discard published adjacencies to improve the timer.
Any reuse of adjacency-validation results belongs in the adjacency access
layer; transition code must not maintain a second copy of those rules.
Specialized paths within these facilities are allowed. The DSL, ghost tree,
adjacency objects and EntityGhostMapping remain the authority and the route
through setup; a specialization must implement their contracts rather than
introduce an independent replacement mesh pipeline.

The basic separation is sound: Buffer owns allocation and exclusion, ArrayStorage
owns shape publication, ArrayAccess retains leases, and ArrayView exposes raw
pointer indexing. Static/table views eliminate layout dispatch in the kernel;
generic views intentionally retain it. Python shares storage and metadata through
the same ownership record. Serializers and MPI both use ElementSchema; custom
POD binding validates its NumPy definition against that registration.

RowTransferPlan owns row correspondence, PayloadPlan owns byte layouts and MPI
datatypes, TransferBinding owns pointers/tags/transport and TransferOperation
owns active requests and reservations. Shape inference belongs in PayloadPlan,
not in Geom's topology or a mutable route cache. Retaining separate owned and
ghost controllers permits core-only reads while receives write ghost storage.

Geom owns semantic adjacency descriptors, entity numbering and GhostSpec;
DNDS does not know cells, nodes or periodic images. Owned primary construction,
face construction with temporary sparse closure, and final assembly remain
separate operations. The assembler publishes the exact tree endpoints plus
required cone closure; composition is a different operation that produces rows.
Global/local conversion uses the retained target space, and persistent IDs are
not used as adjacency offsets.

Compatibility aliases do not own extra storage. Remaining Native-prefixed
service names and forwarding headers are deliberate staged migration. The large
registry implementation combines traversal and interpolation algorithms; that
is a maintainability issue rather than a second mesh authority. Further file
splitting is optional and should not obscure the measured fixes.

## Validation so far

- View fix: CPU 18 cases / 501 assertions; CUDA 10 cases / 1039 assertions.
- Signed extent fixes: runtime Array and transient MPI np=1/2/4/8, 5/5 CTest.
- CUDA memcheck ran the tests successfully but reports one pre-existing duplicate
  CUB `EmptyKernel` registration API error; this is not a clean sanitizer result.
- Payload/context changes: 204/204 selected CPU/MPI and Geom CTest registrations
  passed, including np=1/2/4/8 mesh fixtures. CUDA runtime/engine and the single
  rank transfer registrations passed 3/3; two-rank transfer tests passed with
  one rank per local GPU. A focused success-report rerun confirmed the direct
  device MPI branch executed on both ranks.
- Matched timing remains pending. The first cluster build succeeded, but its
  driver rejected an np=32-only matrix during serial defaults generation;
  that harness error produced no performance evidence.

The first corrected timing-window run (BSSCA 12401824, np=32, OMP=1) measures
0.045942 s legacy versus 0.086877 s native for NACA, and 0.713819 s versus
0.881869 s for the periodic cube. Neither meets the 5% **parallel** target.
These are one-sample diagnostics, with matching geometry and finite numerical
outputs. The earlier unbarriered comparison included waiting for serial METIS
and cannot support parallel acceptance. Serial ingress and FV remain separate.

The fused-remap candidate passed 204/204 CPU/MPI registrations; deferred
reduction setup passed 8/8 transfer/transient registrations and all 21 CUDA
transfer cases on each of two GPU ranks. Combined collective checks passed the
same transfer matrix. The new context/tag failure case passed np=1/2/4/8.
Generation-cache changes passed 204/204 registrations; duplicate-detection
changes passed 8/8 transfer registrations. Destination sorting passed the full
204 selection, with its final shared-storage optimization passing the four
transition registrations. Vertex-key caching passed 20 connectivity/topology
registrations. Sender row permutation passed 12 transition/preparation/scenario
registrations. Binding control changes passed 8/8 CPU MPI registrations and
22 CUDA transfer cases per rank on the two local GPUs, with no skips.

BSSCA `12401916` completed both profile orders on one exclusive node. Median
parallel windows were 0.050361 s legacy / 0.081495 s native for NACA (1.6182x),
and 0.782275 s / 0.896614 s for the cube (1.1462x). All geometry and numerical
gates passed. This candidate predates the later combined checks, destination
sorting, vertex caching and sender permutation; it does not meet the target.

## Plan 25 completeness qualifications

| Plan items | Audit disposition |
| --- | --- |
| 1–3, 15, 26: OMP=1 timing and contained reader copy | Copy oracle and isolated A/B retained. New parallel-only timing window is separate; acceptance pending. |
| 4–6: names and signed logical indices | Public `RowCount()` implemented with compatibility aliases. Helper names and setup-loop signed-index cleanup are partial. |
| 7–11, 21–22: typed handle/view access | One `ArrayHandle<T>` behind compatibility aliases; static/dynamic/table/padded/CSR views. Fixed width and fixed stride are separate policy parameters. CUDA metadata projection defect fixed in this audit. |
| 12–14, 23: bounded scratch | Reviewed closure, traversal and row-route allocations; local/request-sized append/sort storage, no global-ID bitmap. No scratch-size UT added. |
| 16–18, 24: exact payload alias reuse | Descriptor semantics remain separate; mapping identity and physical view identity are O(1) keys. Existing cross-rank disagreement fixtures retained; invalid sorted aliases additionally tested. |
| 19–20, 25: communication modes | Direct h-indexed payload default, explicit packing/reduction workspace, shared routes. Known routes still validate peer counts; persistent Begin still performs collective admission checks. |
| 27: branch/endpoint semantics | Shared prefixes and separate frontiers; explicit endpoint markers, including non-leaves. Additional oracle prevents premature owned-target pruning. |
| 28: integer agreement | Exact bitmask/small-domain and min/max/large-domain paths, tested around crossover and with absent/conflicting values. |

The handle/view split, generic row access, endpoint-marked branch evaluation,
O(1) mapping identity in payload reuse, bounded closure scratch and optional
integer agreement are implemented. Public row access uses signed indices, but
some setup loops and metadata storage still use `size_t`; the original request
to use signed logical indices throughout is not fully completed. Fixed width
and fixed stride remain distinct: `FixedRows<N>` permits a runtime stride,
while `FixedRows<N,S>` carries both constants. Compatibility aliases and
forwarding headers still exist and own no alternative payload storage.

Source review confirms closure candidates are appended and sorted once per
stage; row-route collision checks also append request-sized scratch and sort
once, since no dynamic membership query is needed during route construction.
Neither allocates a bitmap by global entity count. MPI metadata all-to-all
exchanges remain separate from canonical direct h-indexed payload traffic.
The known-route constructor verifies peer counts with an Alltoall, even though
it skips row-ID discovery; “no discovery” must not be read as “no collectives.”

Profiler provenance is also audited. Job `12402028` is invalid for its intended
destination-sort candidate: an older staged source timestamp left the previous
object/binary in use. Its output is retained, not credited to that change.
The prior fused and compact build logs do show recompilation of the changed
production sources and distinct binary hashes. Future staging refreshes affected
dependency timestamps and verifies the compile log. Timing events now buffer
plain records and format them only after the outer interval; native's more
detailed diagnostic scopes must not pay more formatting work inside setup.

Caller limitations remain explicit: borrowed views require a live lease,
untracked external writers cannot be detected, eager callers must provide
cross-stream dependencies, and MPI collective setup must occur in matching order.
These limitations are not replaced with per-element checks or a scheduler that
claims to infer arbitrary kernel dependencies.

## Latest measured checkpoint

BSSCA `12402055` (sender permutation and exact binding controls, old timer
formatting) completed one order at np=32, OMP=1. The parallel windows were
0.0460108 s legacy / 0.0661448 s native for NACA (1.4376x), and
0.707017 s / 0.710523 s for the cube (1.0050x). Both geometry and exact finite
numerical gates passed. The cube is a passing diagnostic sample, not repeated
acceptance. NACA still fails the target.

Source-level attribution remains bounded: the sender candidate contains several
changes, so its improvement is not an isolated measurement of one optimization.
Rank-zero PMPI counts in NACA's native assembly are 347 Allreduces and 70
Bcasts, versus four Allreduces in the legacy PrepareMesh scope. These scopes
contain different subsets of work and must not be used as a like-for-like
timing ratio; they expose the cost of layered collective validation. The native
array path keeps collective failure handling enabled even with mesh audits off.

The later frontier candidate passed 204/204 local CPU/MPI CTest registrations
and a two-rank periodic migration-probe smoke. BSSCA `12402117` rebuilds both
profiles with the buffered timer and measures both profile orders. Its
results remain pending. Python bindings are being rebuilt and installed before
the corresponding language-interface checks.

Both orders of `12402117` have now completed. Median parallel setup is
0.718554 s legacy / 0.713740 s native for the 262,144-cell cube (0.9933x),
and 0.0475832 s / 0.0657236 s for the 33,834-cell NACA (1.3812x). Geometry,
boundary and exact finite numerical gates pass. The user prioritizes larger
meshes at np=32; NACA H3M remains the small per-rank overhead diagnostic. A
larger unstructured fixture and final-candidate validation are being checked.

The comparator/inversion/preflight candidate `12402781` also completed both
orders on an exclusive node. Cube median parallel setup was 0.755492 s legacy
/ 0.728623 s native (0.9644x); small NACA was 0.0455541 s / 0.0649363 s
(1.4255x). Every geometry and exact finite numerical gate passed. These tests
evaluate the recorded initial FV/reconstruction/RHS pipeline; they do not
establish bitwise equality of an arbitrary time-evolved solver history.

The final local CPU/MPI matrix passed **204/204** registrations, serially, in
221.76 s after the typed-binding shape fix. The selection is
`^(dnds_runtime_(array|row_transfer|transient)|geom_)` in `build-refactor`;
it includes np=1/2/4/8. Rebuilt and installed Python modules passed 147 tests
with 56 optional/backend skips at np=1 and on each np=2 rank. The direct
two-GPU transfer run passed all 24 cases on each rank (one visible GPU/rank,
no skips). Logs and build/source hashes are under
`data/runtime-mesh-audit-20261029/`, particularly `ctest-regression-final.log`,
`hash-regression-final.log`, `pytest-python-final-np{1,2}.log` and
`test-cuda-shape-two-gpu.log`. CUDA engine and required-Torch checks are recorded below.

BSSCA job `12402888` tests the final source archive with a 125,828-cell
unstructured high-lift fixture and the cube. It runs np=32 first, then
np=1/2/4/16, both profile orders, all with OMP=1 and a pinned core/rank.
The corresponding plan records the archive/mesh hashes and bounded script.

The final CUDA runtime/engine registrations passed 2/2. Rebuilt and installed
CUDA Python modules passed the required Torch/DLPack suite on both local GPUs:
32 tests per rank, no skips. The optional pre-existing interop environment was
reused; no new dependency installation was needed.

Job `12402888` failed in the driver script before any probes; its successful
builds are not timing evidence. Replacement `12402927` completed both np=32
orders: the cube measured 0.764256 s legacy / 0.737213 s native (0.9646x),
while the 125,828-cell high-lift mesh measured 0.131314 s / 0.167538 s
(1.2759x). All geometry/boundary/exact finite numerical gates passed. The
high-lift case keeps this goal open; cube parity alone is insufficient.

Findings 21–23 postdate those timings. Their focused CPU MPI transfer tests
passed 4/4 registrations and mesh transition/preparation tests passed 12/12,
both including np=1/2/4/8. CUDA runtime/engine passed 2/2; the full transfer
suite passed 25 cases on each GPU rank with no skips. A local high-lift np=4
SA probe matched the preceding native candidate exactly in topology, raw
representation and finite numerical fields. Their cluster timing is pending.

The frozen overlap-width candidate then passed the complete 204-registration
CPU/MPI selection in 221.92 s (`ctest-overlap-width-full.log`). Python bindings
were not rebuilt for this later candidate; the earlier Python/Torch results
above apply to their explicitly recorded source checkpoint.

The companion-order normal sample gives a more comparable stage diagnosis.
Rank-zero intervals (milliseconds, legacy/native) are: parallel-window start
through PrepareMesh start 83.170/86.047, PrepareMesh 43.602/59.095, boundary
extraction 1.317/9.161, and the complete parallel window 128.803/154.887.
The initial interval includes legacy primary ghosts; comparing only legacy
distribution with native transition would omit that work. The remaining
difference is concentrated in preparation and extraction. Native preparation
includes 24.056 ms owned topology and 28.842 ms final assembly; these are
nested diagnostics, not extra terms to add to PrepareMesh. Within assembly,
tree evaluation is 7.593 ms, ghost publication 6.778 ms and localization
4.045 ms. These values are one rank/sample, not the repeated max-rank
acceptance statistic. No isolated gain is attributed to a new optimization.
