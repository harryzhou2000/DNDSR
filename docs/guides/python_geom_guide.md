# Python Geom: native mesh workflow {#python_geom_guide}

`DNDSR.Geom` reads, constructs, prepares and queries native Runtime-backed meshes.
`UnstructuredMesh` has no legacy geometry workspace. Readonly compatibility pair
names remain available; solver-field Array/ArrayPair and Torch interfaces retain
their independent contracts.

## Read and prepare

```python
from DNDSR import DNDS, Geom
from DNDSR.Geom.utils import read_mesh, prepare_mesh, build_bnd_mesh, build_fv

mpi = DNDS.MPIInfo()
mpi.setWorld()
result = read_mesh("data/mesh/UniformSquare_10.cgns", mpi=mpi, dim=2)
prepare_mesh(result.mesh, result.reader)
mesh = result.mesh                         # same facade, prepared in place
boundary = build_bnd_mesh(mesh)
fv = build_fv(mpi, mesh)                    # includes finalize_metrics()
```

DNDS initializes MPI when imported. To control initialization/thread level,
initialize `mpi4py` before importing DNDS. Keep the supplied communicator alive
through mesh, field and pending communication use. Do not substitute
`COMM_WORLD` for a mesh's actual communicator.

`MeshReadResult` contains `mesh`, `reader` and `name_to_id`. The reader is an
output-only `UnstructuredMeshSerialRW` adapter over that native facade. CGNS
returns its boundary-name map on every rank; archive input returns
`name_to_id=None` because the separate name dictionary is not archived.
Numeric boundary zone IDs are preserved. Retain/provide a name map separately
when restarting applications that select boundary conditions by name.

All setup helpers are collective. Every rank, including empty ranks, must
provide the same logical options and follow the same call order on the supplied
communicator. Python option/configuration control flow is also caller
responsibility: do not have only one rank call a helper or take a failing local
branch while peers enter MPI. Finish asynchronous input producers before host
setup. Callbacks must not enter MPI collectives or mutate the source mesh.

### Input options and partitioning

| `read_mesh` argument | Default / meaning |
|---|---|
| `mesh_file`, `mpi`, `dim` | Required input path, MPI context and dimension |
| `read_mode` | Auto from extension, or `"cgns"` / `"h5"` |
| `partition_options` | CGNS: KWAY defaults below; archive: matching saved ownership when omitted |
| `cell_layers` | 1; positive cell closure depth |
| `complete_node_star` | False; opt in to complete cells/boundaries incident to owned nodes |
| `elevation` | `""` or `"O2"` |
| `bisect` | 0..4; do not combine a nonzero value with `elevation="O2"` |
| `periodic_geometry`, `periodic_tolerance` | CGNS import frames / absolute matching tolerance (1e-9) |
| `serializer_factory` | Default H5 factory for archive input |
| `native` | True; `False` explicitly rejects before input I/O |

CGNS defaults are `metisType="KWAY"`, `metisUfactor=20`, `metisSeed=0`,
`metisNcuts=3`; `edgeWeightMethod=0` is unweighted. Explicit method 1 weights the
facial graph by physical face diameter. Distributed ownership uses ParMETIS
KWAY. RB requires reader-owned serial ingress and runs METIS before distribution;
the high-level reader selects it when requested. Unsupported options reject;
there is no silent legacy fallback.

```python
result = read_mesh(
    "data/mesh/UniformSquare_10.cgns", mpi, 2,
    partition_options={"metisType": "KWAY", "metisSeed": 42},
    cell_layers=2,
)
```

Matching archive input preserves saved partitions and original IDs. Explicit
`partition_options` requests even-split H5 loading, including a different MPI
size, followed by graph partitioning. Per-rank JSON requires matching saved MPI
partitions; it does not support changed-size loading. The C++ file helper
coordinates archive open/decode/close failures and propagates `cell_layers`
through assembly and repartitioning. Low-level serializer users still own
consistent serializer mode, open/close ordering and file lifetime.

```python
matching = read_mesh("mesh.dnds.h5", mpi, 3)
repartitioned = read_mesh("mesh.dnds.h5", mpi, 3,
                         partition_options={"metisType": "KWAY"}, cell_layers=2)
```

### Preparation preserves the in-place contract

`prepare_mesh(mesh, reader, ...)` returns None and updates the same native
facade. It prepares missing topology, reorders local cells, applies requested
geometry operations, optionally builds wall vectors, and refreshes serial output
when requested. It does not call a legacy interpolation/ghost/conversion chain.
A supplied reader must refer to that same mesh on every rank. `reader=None`
creates a temporary output adapter when needed.

| Preparation option | Default |
|---|---|
| `ghost_layers` | 1 |
| `reorder_parts`, `reorder_inner_parts` | 1, 1 |
| `build_serial_out` | True |
| `coord_scale`, `coord_rot_z_deg` | 1.0, 0.0 |
| `rectify_near_plane`, `rectify_threshold` | 0, 1e-12; bit mask x=1, y=2, z=4 |
| `wall_dist_predicate`, `wall_dist_options` | None; options require a predicate |

Allocate solver fields after preparation. Reorder/replacement does not
implicitly transfer arbitrary external fields or rebuild finite-volume metrics.
Use the C++ native transition field maps when fields already exist.

`prepare_native_mesh(mesh, ...)` instead returns a detached facade; retain the
return value. It partitions/reorders only when explicitly requested and does
not build serial output caches. `Geom.prepare_native_topology`,
`partition_native_mesh` and `reorder_native_mesh` are the corresponding detached
setup functions. `mesh.PrepareNativeTopology(...)` and `mesh.ReorderLocalCells(...)`
are tracked in-place setup methods.

The deprecated `create_mesh_from_CGNS` and `create_bnd_mesh` preserve their tuple
contracts by delegating native helpers. Their historical names do not select
another storage backend. SerialRW's former read/deduplicate/partition methods
have been removed.

## Periodic import and derivation

High-level CGNS `read_mesh` keeps the established default unit translations when
`periodic_geometry=None`. An explicit dictionary overrides any of these nine
keys: `translation1..3`, `rotationCenter1..3`, `eulerAngles1..3` (each a 3-vector).
For example:

```python
periodic = read_mesh("data/mesh/IV10_10.cgns", mpi, 2,
                     periodic_geometry={"translation1": [10, 0, 0],
                                        "translation2": [0, 10, 0],
                                        "translation3": [0, 0, 10]})
```

Low-level `Geom.read_native_cgns` / `read_native_cgns_with_names` use None to
**disable** periodic identification; `{}` enables default frames. A nondefault
tolerance requires an explicit geometry dictionary. Archive input retains its
saved frames and rejects periodic overrides. There is no mutable
`SetPeriodicGeometry` binding.

Periodic identity is canonical node identity plus ordered incidence-image bits.
Do not discard or reorder bits independently of cell/face/node slots. The reader
requires matching main/donor labels and exact image reconstruction within the
absolute tolerance; it does not guess geometry-only pairs.

Elevation inserts shared O2 span nodes; bisection splits O2 elements using the
established element subdivision rules. Native derivation creates a new namespace
and fresh IDs, preserves physical periodic images, and records source-ID
ancestry. It does not interpolate solution fields or project new points onto a
curved boundary automatically.

```python
quadratic = Geom.elevate_native_mesh(mesh)
refined = Geom.bisect_native_mesh(quadratic)
# Or use the destination-style compatibility spelling:
quadratic2 = Geom.UnstructuredMesh(mpi, mesh.getDim())
quadratic2.BuildO2FromO1Elevation(mesh)
assert quadratic2.IsO2()
```

For a 10x10 Quad4 grid, elevation keeps 100 cells and changes 121 nodes to 441;
bisection produces 400 Quad4 cells while retaining those O2 nodes. Elevation
covers Tri3/Quad4/Tet4/Hex8/Prism6/Pyramid5 and their supported O2 counterparts.
`native_mesh_provenance(mesh)` exposes source namespace, source kinds and retained
Runtime source-ID arrays. Zero/multiple CSR parents are valid. These values are
persistent IDs, never current source indices, and survive native transitions and
full archives. Retaining ancestry does not require the source mesh to remain alive.

## Scoped reads and readonly aliases

```python
with mesh.acquire_read(faces=True, inverse=True) as read:
    owned, ghosts, combined = read.counts["cell"]
    if owned:
        info = read.cell_info(0)             # detached ElemInfo value
    for cell in range(owned):
        cone = read.cell_nodes(cell)        # shared readonly NumPy row
        points = read.cell_coords(cell)     # independent immutable copy
        for slot, face in enumerate(read.cell_faces(cell)):
            other = read.cell_face_other(cell, int(face), slot)
            # Check absence / unavailable encoding before using other as an index.
```

Counts include `node`, `cell`, `face`, `boundary`, `edge` tuples
`(owned, ghost, combined)`. `NumCell()` also counts owned rows;
`NumCellProc()` includes ghosts. Requirements `faces`, `inverse`, `edges`,
`cell_face_neighbors`, `wall_distance` expose only the requested prepared groups.
Unrequested groups are hidden even if present in the publication. Acquisition
never constructs a missing group. `read.relation_state(Geom.MeshRelation.CellNodes)`
reports local compute state; there is no `mesh.adjPrimaryState` or mutable
adjacency `idx` API.

Rows, individual coordinates, periodic bits and wall vectors share readonly
storage. A NumPy buffer owner retains the scope, values and row metadata;
slices and `np.asarray` retain that lifetime. Assignment and making the buffer
writable are rejected. `array.copy()` is independent and writable. Bulk
`coordinates`, `cell_coords`, `face_coords`, `boundary_coords` return independent
immutable snapshots and do not retain a facade read lease.

```python
with mesh.acquire_read() as read:
    escaped = read.cell_nodes(0)[::2] if read.counts["cell"][0] else None
# New queries on read reject after close; escaped still retains the native lease.
del escaped                               # release before tracked mutation
```

Scopes/readonly rows can outlive the Python mesh. Keep communicators alive.
Every rank must follow the same mutation workflow: a retained facade read on
any rank blocks tracked mutation. A plain C++ view owns nothing; its owner must
outlive host workers and joined GPU completion. An independent detached snapshot
retains old values without blocking later source publication.

Compatibility properties such as `mesh.coords`, `mesh.cell2node`,
`mesh.cellElemInfo`, and their `father` (owned) / `son` (ghost) parts are native
read handles, **not** legacy mutable Arrays. Numeric indexing returns readonly
rows; `(row, column)` or call indexing returns scalars. `data()` exports only a
uniform contiguous arithmetic part. Ragged rows and split owned/ghost buffers
must be read by row or explicit part; no fabricated contiguous view is created.
POD metadata is a readonly handle; `.snapshot()` returns a detached mutable value.
These aliases also retain leases, so release aliases as well as scopes before
mutating the facade. Prefer scoped semantic queries in new code.

```python
if mesh.NumCell():
    info = mesh.cellElemInfo[0, 0]          # readonly NativeElemInfoRead
    kind, zone = info.getElemType(), info.zone
    copy = info.snapshot()                 # detached mutable ElemInfo
# Editing copy does not edit mesh metadata.
```

`Geom.Elem.ElemType` includes Line2/Line3, Tri3/Tri6, Quad4/Quad9, Tet4/Tet10,
Hex8/Hex27, Prism6/Prism18 and Pyramid5/Pyramid14. Mesh `IsO1()` / `IsO2()` are
exposed. The full C++ Element shape-function class is not a Python mesh export.

## Solver fields, Torch and entity spaces

`mesh.entities(Geom.EntityKind.Cell)` retains immutable canonical distribution
metadata. Current globals are rank-offset addresses; `original_entity_id(kind,row)`
returns persistent IDs. Descriptors do not lock mutation. After a numbering
change, refresh descriptors, field indexing, communication requests and caches;
equal row counts do not prove matching order.

The compatibility bridge supports real scalar/dynamic-width field transformers:

```python
cells = mesh.entities(Geom.EntityKind.Cell)
field = DNDS.ArrayEigenVectorPair(1)
field.father = DNDS.ArrayEigenVector(1, init_args=(mpi,))
field.son = DNDS.ArrayEigenVector(1, init_args=(mpi,))
field.father.Resize(cells.owned)
field.TransAttach()
Geom.borrow_legacy_field_indexing(field.trans, cells)
field.trans.createMPITypes()
# Fill owned rows, then perform the collective halo exchange.
field.trans.pullOnce()
```

The bridge clones maps into the field transformer and checks freshness, counts
and communicator ordering. Callers own sizing, attachment, requests, phase
ordering and rebuilds. It does not return a mutable mesh transformer or attach a
permanent mesh dependency to every later field operation.

Existing solver-field `DNDS.Array`, ArrayPair/transformer and Torch interfaces
remain available. Their access, placement, tensor lifetime and MPI active-buffer
contracts are unchanged. A native mesh readonly alias is not such a writable
field. Managed mesh reads provide no writable Tensor/DLPack export: copy data
before creating a writable tensor, and never bypass readonly storage with a
foreign pointer or tensor alias. Async consumers require explicit dependency
record/wait and completion; validity alone does not order streams.

## Edits and geometry

Release retained scopes/aliases first. `mesh.edit()` stages supported owned
coordinate, cell-info and neighbor-order changes, and publishes only on explicit
collective commit. Returned values are copies; context exit discards uncommitted
work. Neighbor rows may only be permuted with the original exact multiset;
general connectivity or element-type changes are unsupported.

```python
with mesh.edit() as edit:
    if mesh.NumNode():
        point = edit.coord(0)
        point[0] += 0.1
        edit.set_coord(0, point)
    edit.commit()
```

`transform_coordinates(callback)` accepts nonperiodic meshes only and calls the
callback on independent `(3,)` owned-point copies, refreshes ghosts and invalidates
wall vectors. Return a
finite `(3,)` value or modify the copy and return None. Failure before publication
preserves the source. Use the validated rigid/affine APIs for periodic geometry;
general point callbacks reject it.

`apply_geometry_transform(rotation, scale)` applies a proper rotation and
positive scale coherently. `apply_affine_geometry_transform(linear, offset)`
requires a finite orientation-preserving invertible 3x3 map, transforms
coordinates/frames, and invalidates wall vectors. Conjugated periodic rotations
must remain rigid. `rectify_boundary_coordinates(component_for_zone, ...)` uses
owner/ghost boundary masks; selected coordinate changes invalidate wall vectors.
The old writable bulk `TransformCoords` export is removed.

After geometry/order changes, rebuild metrics, reconstruction bases and
coefficients in dependency order. Wall-anisotropic bases also depend on wall
allocation/version. Low-level granular FV construction must finish with
`finalize_metrics()`; `build_fv` does this automatically. Auxiliary fill-order
cache setup is pure and can run under retained reads without changing geometry
or generations; use the acquired scope/captured state for references retained
across cache or placement-handle replacement.

## Boundary extraction and wall vectors

`build_bnd_mesh(mesh)` returns `BndMeshResult(mesh_bnd, reader_bnd)` with an output
adapter; `build_serial_out=False` omits output assembly. Extraction selects
external boundaries, omits periodic seams and preserves tangential images. It
creates a new namespace and Node-to-Node / Cell-to-Boundary ancestry. A 2D volume
produces 1D line cells; a line-cell face topology is not synthesized. The
`ConstructBndMesh` compatibility method additionally provides current-pair parent
caches, which are not durable ancestry.

```python
name_map = result.name_to_id.n2id_map
wall_ids = {value for name, value in name_map.items() if "WALL" in name.upper()}
options = Geom.UnstructuredMesh.WallDistOptions()
options.method = 1
options.subdivide_quad = 5
options.minWallDist = 1e-10
mesh.BuildNodeWallDist(lambda zone: zone in wall_ids, options)
with mesh.acquire_read(wall_distance=True) as read:
    if read.counts["node"][0]:
        wall_vector = read.wall_distance(0)
```

Wall search retains methods 0/20/1, execution modes and minimum-distance policy.
Publishing wall vectors shares coordinates without advancing geometry generation.
The search still replicates selected wall-triangle coordinates: this surface
scratch is a scaling limit separate from sparse mesh ghost construction.

## Placement and detached snapshots

```python
mesh.place_native(-1)                       # current CUDA ordinal
mesh.to_device("CUDA")                     # same native-placement route
mesh.to_host()                             # resident immutable host state
snapshot = mesh.snapshot(faces=True)
with snapshot.acquire_read() as read:
    values = read.cell_coords(0) if read.counts["cell"][0] else None
```

Placement prepares native buffers before GPU operators; it does not make mesh
values device-mutable. Python managed indexing remains host-only.
`snapshot.to_device(ordinal)` returns an independent CUDA snapshot;
`device_snapshot.to_host()` returns a synchronized host debug snapshot. A stale
`source_current` flag is provenance, not lost snapshot storage. Never export a
CUDA pointer as a host NumPy row.

## Restart and output

```python
from DNDSR.Geom.utils import serialize_mesh
serialize_mesh(mesh, "prepared", mpi)
restored = read_mesh("prepared.dnds.h5", mpi, mesh.getDim())
```

The full native archive preserves every published family, full named relation,
periodic metadata, typed companions, wall vectors, persistent IDs, namespace,
high-water marks and ancestry. Only owned rows are serialized; sparse closure
is rebuilt at the requested depth. Solver fields and boundary-name dictionaries
are separate checkpoint data. `mesh.write_native_archive(serializer, name)` /
`Geom.UnstructuredMesh.read_native_archive(...)` are lower-level alternatives.

Old primary-wire input is decoded directly into Runtime arrays. It preserves
valid original IDs and generates a fresh namespace because that schema had none.
It does not reconstruct a legacy mesh. Historical derived periodic archives
with duplicate original IDs reject explicitly; no silent renumbering occurs.
`WriteCurrentPrimaryMeshSnapshot` (and `WriteCurrentMeshSnapshot`) exports only
primary topology with a rebuild marker. It does not preserve optional-family
field numbering; wall vectors require explicit `DropAuxiliary` policy. Use a
full native archive for authoritative restart.

`UnstructuredMeshSerialRW(mesh, 0).BuildSerialOut()` builds native serial output
buffers and field routes for existing serial output consumers, including
physical periodic images. Rebuild the adapter after changing its source mesh.
VTK/CGNS output uses native buffers; it does not toggle adjacency globals or
require `BuildVTKConnectivity`. Globally empty or periodic parallel CGNS output
is explicitly unsupported. Output buffer preparation is not an automatic solver
field transfer or a numerical validation result.

## EulerP and removed setup APIs

`Solver.ReadMesh` uses native ingestion, in-place preparation and compact
boundary extraction before field allocation. Native archive restart requires
`boundary_names` when the evaluator needs names. `Solver.to_device("CUDA")` and
`Solver.to_device(DNDS.DeviceBackend.CUDA)` place mesh topology on the current
CUDA device before FV/evaluator placement. `native=False` raises before I/O.

The removed APIs include `publish_native`, `use_legacy_workspace`, mutable raw
mesh topology, in-place `AdjGlobal2Local*` / `AdjLocal2Global*`, the SerialRW
input/partition chain, `SetPeriodicGeometry`, and writable bulk `TransformCoords`.
Use native construction, tracked geometry/edit operations, or C++ detached
`NativeRelation::Convert`; live compute topology stays local.

EulerP's intentional Roe entropy-threshold correction uses total Roe speed,
matching the existing Euler convention, rather than signed normal velocity.
Its `zero_grad=True` path clears gradients before starting the halo pull, so it
does not write active MPI buffers or allow late receives to restore old ghosts.
These are numerical/caller-contract changes, not consequences of a storage alias.
Current test logs, not this guide, establish their validation status.

## References

- [Native construction and identity](../architecture/NativeMesh.md)
- [Scoped query lifetime](../architecture/MeshFacade.md)
- [Managed edits](../architecture/ManagedMeshEdit.md)
- [Connectivity and ghost closure](../architecture/MeshConnectivity.md)
- `python/DNDSR/Geom/utils.py` and `src/Geom/Mesh/Mesh_bind.hpp`
- `test/Geom/test_native_mesh_helpers.py` and `test/EulerP/test_native_solver.py`
