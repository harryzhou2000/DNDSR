# Native geometry consumer migration inventory, 2026-10-26

## Current integration after checkpoint 4

Native OpenFOAM Hex8 import, weighted KWAY/RB partitioning, reader-owned ingress,
all three native elevated-node smoothers and guarded archive ingress are now
integrated. C++ Euler initialization selects native construction. CFV benchmark
wrappers retain semantic CPU/CUDA read scopes. Euler serial and boundary output
uses native geometry plus maps-only solver-field adapters. The final checkpoint
5 evidence records the exact acceptance matrix and remaining limitations.

Python default orchestration and retained pair-property compatibility are still
pending. The remaining legacy construction/edit authority and fallback must be
retired before GN7 closes. The sections below retain the checkpoint 4 audit
history rather than claiming those historical gaps describe current code.

## Checkpoint 4 history

The entries below the checkpoint section retain the earlier audit history.
The native implementation now includes local cell permutation, distributed
unweighted facial-graph partitioning, explicit preparation helpers, serial
output snapshots, direct primary checkpoint export and old primary archive
decoding. Periodic import, elevation, bisection and boundary extraction are
implemented. Native allocation accounting includes shared value and shape
buffers on host and device, counting allocation capacity once per identity.

Serial output gathers only output payloads to the selected writer. It does not
construct a legacy mesh. Its maps-only field bridge accepts existing solver
output arrays; it neither owns those fields nor proves their association from
matching row counts. Geometry/order changes require rebuilding the descriptor.
The checkpoint writer stages one wire-compatible payload at a time; the native
archive decoder constructs Runtime arrays directly. Both preserve the existing
Serializer dependency. Old archives lack a mesh namespace, so decoding creates
a new namespace while preserving the stored original IDs.

`WriteSerializeRecMatrix` exports exact ordered face-neighbor slots without
changing mesh state. It still uses a temporary legacy diagnostic field for the
established file schema; this is not legacy geometry storage. Periodic boundary
extraction resolves durable source boundary IDs into current source rows for
the existing Euler boundary output mapping.

Production C++ initialization still selects the legacy input orchestration.
The Python EulerP native route is explicit (`native=True`). Its numerical parity
gate exposed an orientation-dependent entropy-fix input and a zero-gradient
write during an active halo exchange. Both corrections and their before/after
evidence are recorded in checkpoint 4. No native-default acceptance is claimed.
The remaining work is:

1. Complete native OpenFOAM import, configured weighted/RB partition policies
   and all three elevated-node smoothers before migrating default setup.
2. Route Euler initialization/output field transfer setup through native helpers
   and replace CFV benchmark/device wrappers with retained semantic read scopes.
3. Migrate Python defaults/demos and preserve
   old test purposes with native fixtures.
4. Retire production dual mesh authority and the old construction fallback;
   finish the native-only closure audit and its exact validation record.

OpenFOAM/smoothing/weighted partition work is being prepared separately and is
not part of checkpoint 4 acceptance until promoted, rebuilt and tested. Global
boundary datasets in the established wall/smoothing algorithms remain explicit
scalability limits; the prohibition on global-sized ghost/remap storage remains
unchanged.

This is a source audit after checkpoint `1ee2c041`, with subsequent work in
progress. It separates legacy **mesh storage** dependencies from legacy solver
and metric field arrays. No solver run or scientific validation is claimed.

## Current consumer work

Coordinate hashing and native direct-factor graph setup are implemented in the
working tree and passed their focused MPI tests at 1, 2, 4 and 8 ranks:

- `UnstructuredMesh::CoordinateHash` dispatches to the historical legacy hash or
  reads native owned and ghost coordinate rows. It preserves the weak XOR hash,
  per-array row counts, and rank-ordered MPI gather. It is partition dependent,
  includes ghosts, and is not a canonical geometry identity or equality proof.
- `EulerSolver::RunImplicitEuler` now calls this facade method instead of
  `mesh->coords.hash()` (`src/Euler/EulerSolver.hxx`).
- Native `GetCell2CellFaceVLocal` requires prepared facial topology and traverses
  exact cell-face slots. It excludes diagonals, ghost cells and, when requested,
  neighbors outside the current local part. No primary-topology approximation
  or legacy workspace fallback is used.
- `NativeMeshState::WithLocalCellSetup` retains the current part boundaries and
  publishes owned local neighbor metadata without changing entity order,
  coordinate buffers, mapping stamps or mesh epochs. Direct ordering consumes
  that metadata rather than detached legacy facade members.
- Empty cell owners participate in the bandwidth reductions used by RCM
  ordering codes 3 and 4, for both storage modes.

The focused target is `geom_test_native_mesh_caches`. Its three cases check hash
parity with legacy owned/ghost coordinate storage before and after redistribution,
all six direct ordering codes and symbolic factors, exact local-part filtering,
exclusion of ghost neighbors, retained read epochs and empty-rank participation.
Test source: `test/cpp/Geom/test_NativeMeshCaches.cpp`. Evidence:
`/tmp/geom-native-caches-tests.log`. An initial test used an unbound
`CaptureLegacy` result as a published state and correctly failed; its log is
retained as `/tmp/geom-native-caches-tests-initial.log`. The fixture now uses
`PublishNative` before passing `CurrentNativeState` to the facade.

## Remaining production paths, in priority order

Native periodic primary construction now accepts canonical node IDs and ordered
image companions directly in C++, without a legacy mesh. Native topology builds
faces/3D edges, retaining physical self incidences and exact frame companions.
Focused periodic/connectivity tests pass at MPI 1/2/4/8, including an independently
specified one-cell doubly periodic torus, a partitioned two-cell strip, quadratic
midside agreement/rejection, rigid rotation, and archive/redistribution. Evidence:
`/tmp/geom-native-periodic-second-tests.log`. The earlier negative matcher oracle
was itself transform-equivalent; its failure is retained in
`/tmp/geom-native-periodic-first-tests.log` and the corrected nonmatching pattern
passes. Periodic identification from CGNS/coordinate pairing, Python periodic
builder arguments, periodic refinement and periodic boundary extraction remain
unsupported. These tests do not constitute solver or scientific validation.

| Priority | Current path | Required native integration |
| --- | --- | --- |
| 1 | `src/Euler/EulerSolver_Init.hxx`, `ReadMeshAndInitialize`: constructs volume/boundary legacy meshes and `UnstructuredMeshSerialRW`, then invokes `ReadMeshFromCGNS`, `ReadMeshFromH5Parallel`, `ReadMeshFromH5` and `PrepareMesh`. | Select native CGNS/archive input, native transition/repartition and topology completion before constructing the facade. Preserve BC name mapping, ghost requirements and configuration semantics. |
| 1 | `src/Geom/Mesh/Mesh_Helpers.hpp`, `PrepareMesh`: legacy local reorder, face interpolation, N2CB global/local transitions, raw `node2cell.father` checks and optional serial setup. | Replace orchestration with native construction/transition and explicit closure validation; do not call legacy mutators on a native facade. |
| 2 | `src/Euler/EulerSolver_Init.hxx`, boundary construction and unconditional `RecreatePeriodicNodes`/`BuildVTKConnectivity`; `src/Geom/Mesh/Mesh_Serial_BuildCell2Cell.cpp` reads legacy adjacency and transformer mappings. | Native extraction and its facade entry are now implemented separately; replace old orchestration and remove unconditional legacy output-cache calls. |
| 2 | `src/Euler/EulerSolver_PrintData.hxx`, `meshBnd->cell2parentCell` maps boundary output cells back to the volume boundary. | Resolve native extraction ancestry/source persistent IDs into current volume boundary rows. A persistent ID must not be treated as a local row number. |
| 2 | `src/Geom/Mesh/Mesh_Plts.cpp`, native PLT/VTU compatibility output calls `CloneCurrentMeshForSetup`; `src/Geom/Mesh/MeshTransition.cpp` explicitly selects `TransitionAuthority::LegacyWorkspace`. | Native VTKHDF direct output and shared facade CGNS output pass MPI 1/2/4/8 tests. PLT/VTU compatibility exports and native serial gathering remain separate work. Serial reader output still rejects native flag 0. |
| 3 | `src/Geom/Mesh/Mesh_WallDist.cpp`, `BuildNodeWallDist`, previously required legacy backing. | The native branch computes from retained reads and publishes owned/ghost vectors while retaining coordinates. Wall and audit tests pass at MPI 1/2/4/8. Integrate the facade path during native solver setup. |
| 3 | `src/Geom/Mesh/Mesh_GeometryEdit.cpp`, rotation/scale and boundary snapping; `Mesh.hpp`, arbitrary `TransformCoords`. Called by solver initialization. | Native geometry replacement/edit operations pass focused and broad tests at MPI 1/2/4/8. Verify orchestration and wall-vector transformation/invalidation when switching defaults. |
| 3 | `EulerSolver_Init.hxx` elevation smoothing calls; `Mesh_Elevation_SmoothSolver.cpp` and `Mesh_Elevation_SmoothV2.cpp` use managed legacy mutation. | Native elevation/bisection already replace topology construction; curved boundary/internal smoothing requires its own native integration. |
| 4 | `src/CFV/VariationalReconstruction.hpp`, `WriteSerializeRecMatrix`: explicit legacy guard, `BuildCell2CellFace`, legacy pair serialization. Optional call in solver initialization. | Serialize the exact native cell-face-neighbor relation and reconstruction payload without modifying mesh index state. |
| 4 | `src/CFV/FiniteVolume.hpp`, generic `deviceView`; `FiniteVolume_DeviceView.hpp` defaults to `UnstructuredMeshDeviceView`; `BenchmarkFiniteVolume.hpp` accesses its `cell2face`. | Use a retained native read view for these benchmark/general wrapper paths as EulerP already does. |
| 4 | `src/Geom/Mesh/Mesh.hpp`, `getArrayBytes`, loops only legacy device arrays; Python `build_fv` prints that value. | Count native allocations/placements without double-counting shared backing. A pure native facade currently reports zero through this compatibility accounting path. |
| 1, Python | `python/DNDSR/Geom/utils.py` now has `read_mesh(native=True)`, `prepare_native_mesh`, native extraction and archive dispatch; legacy defaults remain. | Runtime acceptance is recorded in checkpoint3. Native local reorder and switching production defaults remain; reject mixed authority before collective branches. |
| 1, Python | `python/DNDSR/EulerP/EulerP_Solver.py`, `ReadMesh` calls the old helpers; `BuildEval` expects `name2Id.n2id_map` and passes it into `BCHandler`. | Native import now returns a rank-consistent BC name map through the new companion binding. Migrate EulerP orchestration to consume the native result and explicit preparation. |

Subsequent focused guard validation passed geometry and native extraction at MPI
1/2/4/8 (`/tmp/geom-native-guards-tests.log`, eight CTests). Geometry operations
reject mixed authority before branching and preserve coordinates/versions on
failure. Boundary publication rejects legacy backing in any destination group,
including an optional array on just one rank, while preserving both meshes and
allowing fresh/native destination reuse. Native wall-distance and output test
results are maintained separately by their owners; final checkpoint evidence
must identify their final source/binary snapshot.

## Paths that already consume native geometry

- `src/CFV/FiniteVolume.cpp` metric construction uses `AcquireRead`/host views.
- `src/CFV/VariationalReconstruction.hpp`, `WallDistanceIdentity`, has an
  explicit native branch; the raw legacy wall array checks are confined to its
  legacy branch.
- `src/Euler/EulerEvaluator_EvaluateDt.hxx`, wall-distance collection/evaluation,
  uses native mesh read views. This is distinct from mesh node-wall-vector
  construction listed above.
- `src/EulerP/EulerP_Evaluator.hpp` retains `NativeMeshSnapshot` and
  `MeshReadAccess`; its kernels receive `UnstructuredMeshView`.
- `DOFFactory.hpp`, reconstruction field builders, cell smoothing fields and
  reconstruction matrix fields use `LegacyFieldBridge::BorrowIndexing` with
  native `EntitySpace` metadata. Their `.father`, `.son` and `.trans` operations
  belong to **field** storage, not a duplicate legacy geometry workspace.
- Local-part iteration and `LocalCellNeighbors` facade getters already dispatch
  to native state. The setup producer, rather than those hot loops, needed work.

## Compatibility surfaces and explicit gaps

C++ facade legacy arrays remain public mutable members. Python `def_readonly`
pair properties prevent attribute rebinding, not edits through returned arrays.
Native facades leave these members empty; they are not native buffer views.
Production reads must use retained mesh read scopes or explicit native IDs.

`TransformCoords` preserves its historical rank-local legacy callback contract.
For native authority it is collective: callbacks run on owned nodes, failures
agree across ranks, and owner values refresh ghosts before publication. Callers
must use the same authority on all ranks when selecting this entry point.
`ApplyGeometryTransform` and `RectifyBoundaryCoordinates` are collective in both
modes and reject mixed authority before entering mode-specific communication.

The old setup applications `app/CFV/vrBasic_Test.cpp`, `vrStatic_Test.cpp`,
`app/Euler/jacobiLUTest.cpp` and `app/Geom/partitionMeshSerial.cpp` deliberately
exercise legacy pipelines. The last two also mutate periodic metadata directly.
They are compatibility applications, not evidence of a native default driver.

Periodic identification, OpenFOAM import, curved elevation smoothing, native
serial output and a native partitioner policy remain separate capabilities.
An initial nonperiodic inviscid solver path can omit optional wall/smoothing and
serial output, but must remove unconditional boundary/output-cache setup first.
