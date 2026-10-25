# Native Geom migration ledger

Date: 2026-10-25. Status: **native import/topology/refinement/archive checkpoint; migration not complete**.

This ledger supports [the native Geom construction work order](/home/harry/projects/DNDSR/workspace/dndsr-next-plan/docs/19-native-geom-construction.md).
It records old Geom behavior that must survive migration to Runtime Arrays and
new acceptance gaps. Existing tests are not evidence for native construction
unless they construct mesh state without the legacy reader/workspace. The
inventory below records the starting state; subsequent implementation is tracked
in the checkpoint section immediately below.

## Native construction checkpoint

Implemented directly over Runtime arrays:

- Independent primary Cell/Node/Boundary construction, canonical entity spaces,
  shared coordinates and POD metadata, retained publication leases, and a
  Python constructor sharing the same allocations. Dense, padded and uniform
  CSR coordinate/metadata inputs preserve their backing values and access control.
- Sparse inverse, named composition variants and closure discovery with complete
  owned-node stars, independently distributed node/cell ownership and empty ranks.
  Registry keys retain path, predicate, ordering, periodic policy and provenance;
  inverse incidence and node-composed relations coexist without endpoint aliasing.
- Durable mesh namespace and per-kind persistent-ID high-water metadata;
  distributed duplicate checks use routed IDs, never a global-sized lookup.
- Native-to-native reorder/redistribution with sparse target renumbering, ordered
  companions, staged field publication and rollback. Preserved face/edge topology
  can move through this route, including periodic node-bit companions.
- Thin Runtime row access usable from CUDA, an explicit legacy setup adapter,
  and native-only header boundaries. Legacy DNDS Array remains supported.

The next implementation checkpoint adds:

- An owned-global assembly contract shared by native transitions and archives.
  JSON matching-partition and H5 matching/even direct reads preserve all fixed
  and named relations, entity families/IDs, ordered periodic companions,
  coordinates and metadata. Archive tests include different reader rank counts,
  independent family ownership, empty ranks, malformed metadata and exact IDs.
- Native face/edge completion with sparse closure derived from owned cell cones.
  Physical oriented FaceCells and mathematical inverse sets remain distinct.
  Shared high-order faces with inconsistent midside IDs are rejected. Exact
  boundary matching and nonmanifold rejection remain explicit contracts.
- A single CGNS parser backed by Runtime build arrays, with direct native
  distribution and a compatibility copy for the existing local serial API.
  Multi-zone, mixed elements, PointList and reader-rank failure checks remain.
- Native O1 elevation and O2 bisection using existing element mathematics. New
  meshes have new namespaces/IDs and durable owned ancestry. Exact shared span
  nodes, signed quad/tet measures, boundary metadata, sparse original IDs, source
  destruction, redistribution and archive round trips are checked. CSR ancestry
  permits empty/multiple/repeated sources and different storage layouts by rank.
- Native staged MeshEdit for coordinates, cell metadata and neighbor ordering,
  sharing unchanged buffers, retaining old readers and invalidating wall vectors
  after coordinate changes. Order detection now reads native metadata rather than
  silently succeeding on empty legacy storage.

The primary/topology/refinement builders still reject periodic input. Existing
periodic state can be preserved by transition/archive; that is not evidence for
independent periodic construction. Boundary extraction, remaining geometry/wall
and output workflows, production reader orchestration and final removal of legacy
mesh authority remain required. No GN7 completion is claimed.

Focused validation and exact artifact provenance are retained under
`geom_validation_2026-10-25/`. The original tests below remain a purpose ledger,
not a declaration that every fixture has migrated. New constructor, relation,
transition, Python and CUDA tests supplement it; transition checks that inspect
published native destinations now use scoped views while retaining their exact
ID, slot, periodic-bit and field-value oracles.

The second checkpoint's evidence is under `geom_validation_2026-10-25/checkpoint2/`.
Its latest focused static run passes 32 CTests (construction, native connectivity,
registry, transition, topology, refinement, archive and MeshEdit at MPI 1/2/4/8).
The CGNS and existing pipeline/multi-zone regressions are separately recorded.
The full CUDA facade suite passes on GPU 1. A two-device attempt is retained as
incomplete: another process increased GPU 0 usage to about 80 GB during the run,
the probe reported CUDA allocation failure, and the waiting test processes were
terminated without touching that other process. This does not close two-GPU
acceptance. The validation README records final Python artifacts independently.

## Inventory snapshot

- The Geom mesh tree has 63 source files, with construction and transformations
  spread across `Mesh.hpp`, `Mesh.cpp`, `Mesh_Serial_*`, `Mesh_ReadSerializeDistributed.cpp`,
  `MeshConnectivity*`, `Mesh_InterpolateHelpers.hpp`, `Mesh_Reorder.cpp`,
  `Mesh_Elevation.cpp`, `Mesh_WallDist.cpp`, `Mesh_Plts.cpp`, `MeshEdit*`,
  and the native state/access/transition files.
- A broad source scan finds about 630 `.trans` spellings in Geom. These include
  Eigen's matrix transpose as well as ArrayPair transfer members; inventory each
  occurrence by receiver type before changing it. `Mesh.cpp` and the serial
  reader/partition path are the key legacy construction seam; connectivity,
  interpolation, ghosts, reorder, geometry edits and I/O consume or mutate the
  resulting arrays.
- The C++ Geom directory contains 36 files, including unit tests, CUDA tests,
  fixtures and helpers. Four Python files match `test_*.py`; the broad
  `test/Geom` tree additionally contains mesh tools and OversetCart examples.
- Raw consumer scan finds 12 direct `mesh->...` connectivity/coordinate matches
  in CFV/Euler/EulerP and app sources; false positives are possible. The
  authoritative follow-up must inspect output/restart, entity identity/hash,
  wall-distance cache, concrete `ArrayPair` signatures, device views, and
  solver-owned transfer arrays, not just these member spellings. Preserve
  solver-owned storage and communication; bridge mesh reads through semantic
  views/facade and route actual mesh mutation through Geom.
- CodeGraph could not be used in `/home/harry/projects/DNDSR-array-geom` because
  that checkout has no `.codegraph/` index. No index was initialized.

## Existing test purposes to preserve

“Old purpose” is the requirement/oracle guarded by the original test. Keep that
independent check when migrating its fixture and APIs; do not replace it with a
count-only smoke test. Status for every item below: **not yet migrated/accepted
as native-only by this ledger**.

| Original test files | Original purpose / oracle | Native migration requirement |
|---|---|---|
| `test_MeshConnectivity.cpp` | Exact inverse rows and inverse round-trip; compose/filtered neighbors; face-share threshold; regressions against mesh recovery; typed cone/support access; fixed-width conversion; periodic-image filter regression | Native DSL tests need independent exact rows/order, including Cell-Node-Cell vs Cell-Face-Cell, direct inverse Cell-Face vs Face-Node-Cell, named any/all/exact selection, empty rows, self and periodic incidences. Preserve legacy oracle only as baseline. |
| `test_MeshConnectivity_Interpolate.cpp` | Face/edge interpolation on quad, triangle, tet, hex, mixed element, shared incidences, periodic and distributed meshes; ordered slot/companion checks and legacy parity | Native construction and interpolation directly from Runtime arrays; preserve exact ordered node/parent rows, element metadata, periodic bits and empty-rank collective behavior. |
| `test_MeshConnectivity_Ghost.cpp` | Ghost-chain validity, prefix merging, unsupported chain detection, hash/equality, available closure, diagnostic dump; 1-hop/2-hop/multi-ring union traversal; no ghosts at np=1; periodic tiled traversal | Use named relation variants and independently enumerated exact closure; test node-star closure, missing closure failure, empty ranks, sparse IDs and that no production allocation scales with global ID range. |
| `test_MeshPipeline.cpp` | End-to-end reader/partition/reorder/recovery/ghost/interpolation pipeline; state/index round-trips; face/node/boundary/edge validity; periodic companions; boundary extraction; VTK connectivity; elevation/bisection/smoothing; multilayer ghost counts and local closure | Add an independent Runtime-array-native fixture. Migrate pipeline in GN2–GN5; preserve geometric/count facts and replace legacy-vs-DSL comparisons with independent combinatorial/geometric expected rows where legacy is removed. Include requested node-centered views and a numerical mock. |
| `test_MeshIndexConversion.cpp` | Global/local/ghost conversion round-trips, invalid and `UnInitIndex` behavior, negative local encoding, all primary/facial/N2CB relation bounds and round-trips | Native entity-space mapping tests must check wrong kind/order, absent vs unavailable, valid owned/ghost and invalid input behavior, independent of legacy index maps. |
| `test_MeshReorder.cpp` | Permutation and distributed transfer values; cell/node/face/edge/boundary movement; source/target relocation/remap; companions and periodic slot alignment; follow-owner; invalid states; ghost identities; field staging/rejection and exact source preservation | Migrate to native-to-native transitions. Test all entity kinds alone/together, persistent IDs as payload (never remapped), exact topology rows and field values, ghosts/routes after redistribution and delayed failure preserving source. |
| `test_MeshDistributedRead.cpp`, `test_MeshCGNSMultiZone.cpp` | Same-rank/cross-rank count and topology sanity, persistent original IDs, node resolution; multizone deduplication and one-entry boundary PointList parsing | Direct import/restart into native buffers; include matching-rank and direct-partition modes, exact identity/topology oracle, serial CGNS input as explicit reader mode, no legacy import workspace. |
| `test_MeshFacade.cpp`, `test_MeshFacade_CUDA.cu`, `test_MeshFacadePipeline.cpp`, `test_MeshFacadeOverlap.cpp`, `test_MeshFacadeOverlap_CUDA.cu` | Scoped CPU/CUDA views, retained allocation/metadata lifetime, checked invalid access, row compatibility, stale distributions, real overlap with live halo, native-backed query parity, checkpoint and real CGNS pipeline | Retain these as boundary/interop tests, but add direct-native-array creation and ensure all queries run without a legacy-built mesh. Validate actual stream completion and two-GPU direct MPI when available. |
| `test_RelationAccess.cpp`, `test_NativeConnectivity.cpp` | Native storage-neutral inverse/compose, relation access failures, native halo route, ordered interpolation, periodic self/owner-frame behavior, empty ranks; current-authority-to-native setup | Expand to full descriptor identity and coexistence for same-endpoint variants, semantic selection, node-star and cell-star closure, sparse memory behavior, and independent expected order/set. The current native setup still crosses legacy authority and is not GN2 acceptance. |
| `test_NativeMeshSnapshot.cpp`, `test_NativeMeshPublication.cpp`, `test_MeshMutation.cpp`, `test_MeshEdit.cpp`, `test_MeshGeometryEdit.cpp`, `test_MeshEntityField.cpp`, `test_MeshEntityField_CUDA.cu` | Snapshot retention/publication; managed mutation, alias isolation and stale generations; edit validation/abort; coherent geometry transforms; entity-bound fields, collective rejection, payload movement and CUDA halo | Preserve lifetime/failure/field contract over one native storage authority; add independent constructor, metadata conflict, retained-alias write protection and source-preserving failed publication. |
| `test_MeshTransition.cpp`, `test_MeshCurrentOutput.cpp`, `test_MeshFieldCheckpoint.cpp` | Detached native transition, exact field/companion state, preparation and delayed publication failures; current-numbered VTKHDF; matching-partition restart | Extend to direct native source and destination through output/restart; exact all-kind IDs and topology after round-trip; no whole legacy export/import fallback. |
| `test_MeshAudit.cpp`, `test_LocalPerformance.cpp`, `test_MeshMutation.cpp` | Audited wall distance/smoothing contracts, cache/lookup performance properties, reader/writer exclusion, mutation generations and invalidation | Preserve independent numerical/geometric answers and cache dependency rules under native storage; demonstrate bounded sparse traversal allocations. |
| `test_Elements.cpp`, `test_Quadrature.cpp`, `test_QuadratureCachePeer.cpp` | Element traits/topology, shape functions and derivative checks, normals/orientation, edge uniqueness, quadrature exactness/integration and cache identity | These are geometry/math unit tests, not legacy storage tests. Keep their present numerical oracles while changing any row-view adapters they use. |
| `test/Geom/test_basic_geom.py` | Python mesh read, node/face/wall distance values, initialized mesh state, multilayer ghost counts and local neighbor closure | Bind native builder/read facade; retain exact query and locality assertions; do not rely on implicit mutable row writes. |
| `test/Geom/test_mesh_facade.py`, `test_mesh_edit.py` | Python checked read scope, immutable rows, escaped slice/metadata lifetime, snapshots, CUDA copy semantics, periodic access, explicit publication/edit validation and callback failures | Keep view ownership/edit contracts with Python arrays sharing native dynamic storage. Add construction from Runtime inputs and escaped rows surviving replacement. |
| `test/Geom/test_overset_facade.py` | Overset traveling geometry agrees with current facade numbering, coordinates and boundary metadata; field mapping remains independent of mesh indexing | Preserve exact traveling item IDs/rows and route associations using semantic native read views; do not expose mutable canonical maps. |
| OversetCart helpers/scripts, `draw_meshes_2D.py`, `cart_overset_test_2D.py` | Reusable mesh-consuming demos/algorithms; several helpers assume current Geom read and field APIs | Inventory separately from tests. Update raw reads to facade/views and writes to explicit edit/session operations; keep illustrative behavior and executable examples. |

## Production migration scope and external seam

| Area / representative files | Existing responsibility | Required native-only destination |
|---|---|---|
| Primary construction: `Mesh.hpp`, `Mesh.cpp`, `Mesh_Serial_ReadFromCGNS.cpp`, `Mesh_Serial_Partition.cpp`, `Mesh_Serial_BuildCell2Cell.cpp`, `Mesh_ReadSerializeDistributed.cpp` | Serial arrays, import, partition/reorder, runtime pair setup, CGNS/HDF5 reads and distributed placement | Direct native Runtime array constructors/readers; optional backend scratch is allowed, but no required legacy mesh workspace or whole-mesh conversion. Keep explicit serial import mode separate from sparse ghost logic. |
| Topology: `MeshConnectivity.hpp`, `MeshConnectivity_StateChecked.hpp`, `MeshConnectivity_Ghost.cpp`, `MeshConnectivity_Interpolate.hxx`, `Mesh_InterpolateHelpers.hpp`, `NativeConnectivity.cpp`, `RelationAccess.hpp` | Inverse/compose, generated face/edge incidence, periodic filtering, ghost schedules | One typed native DSL/registry; complete descriptor keys preserve path/predicate/order/periodic semantics. Implement all node and cell variants and sparse owner-directed traversal. |
| State/space/access: `NativeMeshState.*`, `NativeMeshSnapshot.*`, `Mesh_Access.cpp`, `Mesh_EntitySpace.cpp`, `MeshView.hpp`, `Mesh_DeviceView.hpp`, `MeshAccess_bind.hpp` | Native snapshots, entity maps and scoped reads over existing mixed authority | Native space creation independent of legacy import, read requirements/closure explicit for cell and node profiles, native Array pair and retained row metadata. Keep compatibility facade at module boundary. |
| Reorder/edit/fields: `Mesh_Reorder.cpp`, `ReorderPlan.hpp`, `MeshTransition.*`, `MeshEdit.*`, `Mesh_GeometryEdit.cpp`, `FieldTransfer.hpp` | Legacy reorder plus staged native fields/transitions and managed edits | Native-to-native transitions with IDs/companions, stale-plan rejection and atomic managed publication. Mesh edits use named operations; external field arrays remain their owners' arrays. |
| Derived geometry/output: `Mesh_Elevation.cpp`, `Mesh_Elevation_Smooth*`, `Mesh_WallDist.cpp`, `Mesh_Plts.cpp`, serialization and cache/hash helpers | Elevation, bisection, extraction, smoothing/wall distance, VTK/CGNS/restart | Native new-mesh construction, optional durable provenance, dependency-stamped caches, output and legacy checkpoint decode directly to/from native arrays without a complete legacy export. |
| External consumers: `src/CFV`, `src/Euler`, `src/EulerP`, `app/` | Read coordinates/connectivity/topology, setup outputs/restarts/hashes/cache, hold some concrete pair/transfer assumptions | Keep solver fields and their communication intact. Replace direct mesh-array dependencies with read facade/semantic view. Capture each output/restart/hash/cache/device path. Mesh changes are named Geom edits. No `.trans` compatibility on mesh objects is required. |

Concrete consumer items visible in source include Euler's coordinate hash and
restart/output transfer setup, CFV wall-distance/cache and reconstruction data,
EulerP's concrete pair transfer assumption, and CPU/CUDA mesh views. A complete
line-numbered caller ledger remains required before GN6 exit; the 12-match raw
scan above is only an entry-point count, not proof of complete coverage.

## Gates GN0–GN7

Every phase gate below remains **open** until its complete acceptance row is
satisfied. The checkpoint above advances several gates without claiming that
the broader reader, topology, restart or consumer obligations are finished.

| Gate | Required work and evidence | Dependencies / hard stop |
|---|---|---|
| GN0 — baseline and semantic DSL | Correct native `Compose`/`Inverse` variant propagation; inventory entry points and preserve legacy test purposes. Add independent expected ordered rows for C-N-C vs C-F-C, inverse C-F vs F-N-C, any/all/exact predicates, direct incidence, empty/self/periodic cases, registry coexistence and collective ghost traversal. | Root/DSL owners. Stop if endpoint-only keys collide, an ambiguous lookup succeeds, or expected results are only legacy parity/counts. |
| GN1 — thin Runtime access | Reuse `ArrayPair`, leases, builder/draft rows and common sized row access. Split core headers from legacy array/transformer dependencies. Cover dense/fixed/padded/CSR/POD, CPU/CUDA access and retained metadata. | Depends GN0 descriptor contracts. No duplicated pair allocator or per-element ownership work. |
| GN2 — native workspace and spaces | Create/publish primary mesh and entity spaces directly from Runtime arrays; no `ReadFromCGNSSerial`/legacy workspace required for direct native fixture. Add persistent ID arrays for Node/Cell/Boundary/Face/Edge. Verify alias protection, invalid/conflicting construction and matching-rank read. | Root owns builders. Native fixture must be able to construct usable mesh without calling legacy import/capture. |
| GN3 — topology and closure | Migrate inverse/interpolation/ghost routines. Exact 2D/3D, periodic, multi-ring cell/node closure across MPI 1/2/4/8 and empty ranks; node-star includes complete incident cell nodes/coords plus requested incidences. Audit C++ scratch allocations as well as Runtime arrays for global-size/range dependence. | Depends GN1/GN2. Fail if node closure silently omits unavailable rows or global-sized visited/old-new arrays appear. |
| GN4 — reorder/redistribution | Native A→B reorder/redistribution and all five entity kinds; preserve persistent IDs as payload, ordered topology/periodic slots, explicit field companions, rebuilt ghosts/routes. Failed preparation leaves source intact; run an actual stencil before and after transition. | Depends GN2/GN3. No native→legacy candidate→native staging. |
| GN5 — remaining Geom workflows | Elevation, bisection, extraction, geometry changes, wall distance, output/restart native. Verify new namespace/ID policy, optional provenance after source destruction, matching and direct-partition restart, VTK/CGNS output. | Depends GN2/GN4. No whole legacy export to serialize current native state. |
| GN6 — consumers and Python | Complete callsite ledger for CFV/Euler/EulerP and app including output/restart/hash/cache/device-view. Migrate to read facade and named Geom edit calls. Python properties are retained native views; demos using writes move to explicit edit builders. Verify CPU and CUDA artifacts actually loaded. | May progress incrementally with GN1–GN5. Keep solver-owned array storage/communication and numerical methods. |
| GN7 — native-only closure | Remove production dual authority switch and mesh fallback; docs list explicit exclusions. Run native-array + file/restart end-to-end path, all-kind IDs/incidences, node/cell mock, bounded-memory audit, overlap/CUDA gates, consumer compatibility and Python tests. Record exact source/build/binary/MPI/device provenance and failures. | Depends all prior gates. Legacy DNDS Array APIs/tests remain; no claim based solely on legacy-built fixture or successful submission/startup. |

## Starting gaps (retain as audit history)

1. Native connectivity is still fed from current/legacy authority in existing
   paths; a standalone Runtime-array constructor and independently valid
   EntitySpace are not demonstrated by this inventory.
2. Existing connectivity descriptors cannot be treated as endpoint-complete
   identity. `via`/selection/order/periodic semantics must survive registry,
   schedule, serialization and reorder lookup without collision.
3. Existing broad Geom `.trans` count is not specific to MPI transfer; type-aware
   inventory is needed before editing. The same applies to raw `.trans` and mesh
   member matches in external consumers.
4. Current historical tests largely use CGNS + legacy reader/workspace fixtures.
   They establish algorithms/oracles, but not the direct native build, native
   restart, sparse memory limit, or complete node-star contract.
5. Node closure must be explicit and complete when node ownership differs from
   incident cell ownership; no node-centered acceptance mock is identified in
   the preexisting test inventory.
6. Output/restart and hash/cache/device access must be accounted for alongside
   evaluator loops. The raw member-scan count is intentionally not a completion
   metric.
