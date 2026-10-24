# GE0 Geom refactor inventory — 2026-10-24

Status: source inventory only; no implementation edits, builds, or tests run.

## Scope and method

Inspected checkout `/home/harry/projects/DNDSR-array-geom`, branch
`codex/refactor-dnds-array-geom`, baseline `232df1a069ae7aecc442b7ed71bb98fe8cf78768`.
The checkout was clean when inspected. `.codegraph/` is absent, so the CodeGraph
index is unavailable here; findings below come from source search and line-level
inspection. This is a categorized migration ledger, not an exhaustive list of
every textual occurrence. The approved sequence and acceptance intent are in
`/home/harry/projects/DNDSR/workspace/dndsr-next-plan/docs/17-geom-facade-migration.md`.

## Consumer inventory

| Area | Existing access and representative anchors | Category | GE2 action / compatibility boundary |
|---|---|---|---|
| CFV algorithms | `VariationalReconstruction_Reconstruction.hxx:204-239,327-367,560-597` iterates raw `cell2face`, reads `GetFaceZone`, and mixes this with CFV geometry queries; `VariationalReconstruction.cpp:190-280,837-846` reads `face2cell` and borrows `cell2cell.trans`; `FiniteVolume.cpp:599-641` borrows cell smoothing scale mapping | Query + field mapping setup | Replace topology reads with one retained read scope and row/face-side semantics. Keep FV/reconstruction field storage solver-owned. |
| CFV DOF setup | `DOFFactory.hpp:60-79,129-148` selects `cell2node.trans`, `coords.trans`, or `face2node.trans` for field `BorrowGGIndexing`; `VariationalReconstruction.hpp:832-897` repeats cell/node/coordinate bindings | Field mapping setup | Bind fields to Cell/Node/Face entity distribution descriptors through a named legacy-field adapter; document scalar/vector payload layout separately from entity identity. |
| Euler kernels and loops | `EulerEvaluator.hxx:61-65,206-220,316-366,1050-1125,2723-2821`; `EulerEvaluator_EvaluateDt.hxx:128,466-579,886-1033,1171`; `EulerEvaluator_EvaluateRHS.hxx:190-195,678-687` | Query | Migrate raw rows and row positions to scoped relation views; preserve self-periodic side semantics (`EulerEvaluator.hxx:216-221`) and periodic transforms (`EulerEvaluator.hpp:136-141,1075-1081`). Existing semantic methods such as `CellFaceOther`, `GetFaceElement`, `GetCoordsOnFace/Cell`, zones and volumes should remain available on the view. |
| Euler mesh setup and output | `EulerSolver_Init.hxx:98-100,315-381` edits periodic translations/coordinates and pulls coordinate ghosts; lines `557-586` bind output arrays to coordinate and reader serial-output maps. `EulerSolver_PrintData.hxx` contains restart/output paths that consume mesh identity/indexing (surveyed via mesh references) | Geometry mutation + I/O + field mapping | Move geometry/periodic changes to declared edit/refresh transaction. Output/restart should consume coherent export snapshots and explicit mappings; reader-owned serial-output transfer objects remain an I/O compatibility boundary. |
| EulerP host evaluator and buffers | `EulerP_Evaluator.cpp:91,132-150` reads mesh MPI/counts and invokes mesh VTK output; `EulerP_Evaluator.hpp:155-191,283-320,333-365,417-431` sizes entity fields, obtains `fv->mesh->device()`, combines FV device view, and checks entity sizes | Query + field mapping + device | Use entity-space counts and retained device-compatible read view. Preserve trivially-copyable kernel view shape and validation of Cell/Face/Node field association. |
| EulerP device kernels | `EulerP_Evaluator_impl_common.hxx:73,156,276,422,759-860,926-944,1095-1103,1248-1253` reads zone/connectivity rows, face-cell slots, and FV geometry through a device view; launch glue is in `EulerP_Evaluator_impl.cu:189-216` | Device view | Adapt the mesh portion of the existing view to the facade's POD device view; kernel bodies must not retain Array or transformer ownership. Keep CUDA transfer/launch lifetime gates explicit. |
| CFV device path | `FiniteVolume_DeviceView.hpp:177-198`, `BenchmarkFiniteVolume.hpp:30-32,134-136`, and `BenchmarkFiniteVolume.cu:19-62` consume mesh device views/CSR rows in host/device code | Device view + query | Preserve a trivially-copyable view and row ordering; audit helper aliases, not just `mesh->` patterns. |

The approved plan's text survey reports direct `mesh->` references in 9 CFV files
(298 occurrences), 8 Euler files (495), and 2 EulerP files (25); those counts
include comments and suitable semantic queries, and exclude aliases/device/Python.
They should be treated as rough scope counts, not edit counts.

### Legacy mapping borrowing

Current borrowing is not limited to adjacency reads. The explicit sites located
are:

- DOF fields from `cell2node`, `coords`, and `face2node`: `CFV/DOFFactory.hpp:60-79,129-148`.
- Reconstruction fields from `cell2node` and `coords`: `CFV/VariationalReconstruction.hpp:832-897`.
- Matrix/smoothing fields from `cell2cell`: `CFV/VariationalReconstruction.cpp:837-838`, `CFV/FiniteVolume.cpp:599-601`.
- Node output from coordinate distribution and reader serial maps:
  `Euler/EulerSolver_Init.hxx:557-586`.
- Geom algorithms also directly borrow maps for mesh-derived data, e.g.
  `Mesh_Elevation_SmoothV2.cpp:120-122` and `Mesh_WallDist.cpp:220-222`;
  retain these as explicit managed Geom adapters during GE1/GE3, not as solver
  compatibility exceptions.

## Python surface and helpers

`Mesh_bind.hpp:50-79` exports `coords`, primary adjacency/element arrays,
face/edge arrays, periodic-index arrays and `nodeWallDist` with `def_readonly`
and `reference_internal`. This prevents replacing the property but does not make
the returned Array object or its contents immutable. Methods expose mesh build,
recovery, conversion, elevation, bisection, reorder, serialization and export
operations (`Mesh_bind.hpp:81-145`). `UnstructuredMeshDeviceView` is not a Python
read lease. Preserve current helper-level behavior while migrating the raw
properties to managed read views and explicit edits.

The Python workflow itself is semantic and orchestration-heavy: `Geom/utils.py`
uses recovery/ghost/conversion steps at `118-128`, CGNS partition/elevation/
bisection at `130-188`, H5 distribution at `191-210`, and public read/prepare/
boundary/FV/serialization helpers later in the file. Keep these helpers as the
stable high-level API; replace underlying raw-array access where present rather
than requiring users to reproduce setup sequences. Python tests and examples
include `test/Geom/test_basic_geom.py:149-150`, `test/CFV/test_basic_fv.py`,
`test/CFV/test_fv_correctness.py:19-25`, `test/EulerP/test_basic_eulerP.py:116-118`,
and `test/Geom/OversetCart/*` plus mesh drawing scripts. Overset utilities are
an indirect Python mesh consumer and should be inventoried by entity operations
during GE2; they are not represented by the C++ `mesh->` survey.

## Mutation and invalidation ledger

`UnstructuredMesh` owns public mesh arrays, periodic metadata, cached auxiliary
vectors/maps, and `Runtime::VersionTracker versions` (`Mesh.hpp:38-87,102-162,
168-184`). Its public topology/geometry includes primary cells/nodes/boundary,
faces, edges, periodic companions, inverse relations and on-demand cell-face
relations. There is no automatic interception of writes through these public
legacy arrays.

| Mutation family | Representative sites | Current coverage / required disposition |
|---|---|---|
| Geometry values | `Mesh.hpp:1023-1029` `TransformCoords` writes every coordinate directly; `EulerSolver_Init.hxx:98-100,315-381` writes periodic transforms and coordinates / refreshes coordinate ghosts | No visible version mutation guard in `TransformCoords`; direct element/periodic writes bypass tracking. Require geometry edit declaration, periodic frame update, and a named ghost-refresh boundary. |
| Topology construction and conversion | `Mesh_bind.hpp:84-105,128-135` exposes ghost build, global/local conversion, face/edge interpolation; `Geom/utils.py:118-180` orchestrates recovery and topology stages | These methods mutate mappings, relations and derived arrays. Ensure each routes through managed mutation protocol or a confined builder phase. |
| Reorder / redistribution | `Mesh_Reorder.cpp:643-646` starts mutation with Distribution/Layout/Allocation and optional Topology; `:648-680` destroys face relations and clears `bnd2faceV`/`face2bndM`; `:978` starts another layout/allocation/distribution mutation | Existing VersionTracker coverage is concentrated in reorder. Keep transactional validation and explicit topology vs layout/distribution change sets; do not assume all public edits participate. |
| Derived/cache state | `Mesh.hpp:121-125` has `bnd2faceV` and `face2bndM`; `Mesh_Reorder.cpp:677-680` clears them on face destruction. CFV owns geometry-dependent caches such as `VariationalReconstruction.hpp:166-178` cell/face base caches and boundary point cache | Add dependencies/generation stamps for topology, geometry, distribution/order and element metadata. Current cache clear shown is face-destruction-specific; audit all invalidation paths before claiming complete coverage. |
| Derived exports | `Mesh.hpp:173-184` VTK connectivity and recreated periodic arrays; `BuildVTKConnectivity`/`RecreatePeriodicNodes` at `:1031-1034`; wall distance API at `:1089` | Treat as derived data with topology/geometry stamps. Wall distance is also a coordinate-derived field; `Mesh_WallDist.cpp:220-222` maps its ghost distribution to coordinates. |
| I/O | `Mesh.hpp:1033-1061`, serializer methods on reader/writer, and `EulerSolver_PrintData.hxx` | Snapshot mesh/identity and fields coherently; keep serial reader/writer mappings behind named export adapter. |

## Existing gates, skips, and known gaps

The current registered suites are listed in `test/cpp/CMakeLists.txt`: Geom
tests around lines 240-319; CFV tests around 321-370; CUDA CFV device-transfer
tests are conditional and only `np=1` (`:351-370`). Existing Python suites are
under `test/Geom`, `test/CFV`, `test/Euler`, and `test/EulerP`; AGENTS.md requires
rebuilding and installing all affected pybind modules before running them.
No tests were run for this inventory.

Source-visible test conditions require careful recording, not assumed coverage:

- `test_MeshReorder.cpp:2034-2047` explicitly limits face-explicit identity
  reorder to `np=1`; at `np>1` the target-face pull-set can miss old face globals
  reachable after cell rows relocate. This is an acknowledged gap, not a pass.
- Other reorder tests contain early rank/fixture returns; examples at lines
  575, 770, 825, 1097, 1233, 1268, 1316, 1377, 1437, 1493, 1523, and 1576.
  Inspect each condition when selecting gates, and fail required cases instead of
  silently skipping if a capability is part of the gate.
- `test_MeshReorder.cpp:2024-2027` says the preceding follow case does not
  preserve full ghost+sons state for `Recover*` / `BuildGhost*` / local conversion.
- `test_MeshPipeline.cpp:881-937` compares DSL/legacy ghost mappings, useful as
  equivalence coverage but not an independent semantic oracle.
- GE0/GE4 should add a reproducible distributed face-preserving reorder with
  relocated cells, independently check target face identity/mapping, and then
  perform complete ghost reconstruction and local/global conversion.

## Compatibility allowlist recommendation

Allow temporary raw member access only in narrowly named boundaries:

1. `Geom/Mesh` legacy storage implementation and a `LegacyMeshStorageAdapter`
   that implements the new facade over current arrays during GE1/GE3.
2. A `LegacyFieldBridge` for solver-owned `ArrayPair` fields, allowed to borrow
   entity distributions but forbidden to expose mesh relation arrays to CFD
   algorithms. Remove mesh `ArrayTransformer` borrowing from migrated callers.
3. A `MeshExport/LegacySerializerAdapter` for CGNS/HDF5/restart/output and a
   reader-owned serial partition adapter. These consume coherent snapshots and
   explicit identity mappings; they do not grant mutable access to published mesh
   state.
4. A POD `MeshDeviceView` adapter during EulerP/CFV device migration. It may
   contain kernel pointers/strides/spans and stable metadata, never Array owners,
   transformers, or host-side mutable aliases.
5. Python legacy array properties may remain as explicitly deprecated
   compatibility exports during migration, but are outside managed-view and
   mutation guarantees. New supported Python paths should use scoped views/edit
   contexts; avoid representing these properties as safely read-only.

Do not allow solver algorithm code, generic demos, or ordinary Python helpers to
read `UnstructuredMesh` Array members/transforms after GE2. Keep solver-owned
solution/reconstruction arrays and FV operator internals outside this mesh-member
allowlist.
