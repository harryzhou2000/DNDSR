# Scoped mesh queries: current migration boundary

`UnstructuredMesh::AcquireRead` provides
a retained read scope and a storage-independent `UnstructuredMeshView`. Legacy
mesh arrays remain the default setup authority; explicit `PublishNative` installs
private immutable native authority. Solver field setup can obtain
[entity-space descriptors](EntitySpace.md) instead of borrowing mesh
transformers directly. CFV setup/reconstruction and selected Euler operators use
one scope per operator. Python has managed read scopes and tracked coordinate
transforms. This is a bounded migration with explicit legacy setup/output
adapters; the [checkpoint report](../dev/geom_refactor_completion_2026-10-24.md)
lists its compatibility boundary and validation evidence.

## Acquire once per operator

```cpp
auto access = mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
auto view = access.HostView();
for (DNDS::index cell = 0; cell < view.NumCell(); ++cell)
{
    auto faces = view.CellFaces(cell);
    for (DNDS::rowsize slot = 0; slot < faces.size(); ++slot)
    {
        auto face = faces[slot];
        // face is a local face required by this owned-cell stencil.
        auto other = view.CellFaceOther(cell, face, slot);
        // Test for a boundary sentinel before using other as a subscript.
    }
}
```

`Primary()` acquires cell/node/boundary primary fields. `CellFlux()` additionally
requests facial/C2F relations and face information. Explicit requirement flags
request inverse support, edges, face-neighbor cells, or wall distances. Absent
groups are not silently captured; use `HasRelation` when writing optional queries.

The scope retains the mesh version read lease, value allocations, row offsets and
row lengths. It validates relation state, source/target-map identity, companion
row shape, supported element types and incidence widths, periodic slot shape,
and required local closure at acquisition. Legacy edge construction independently
builds maps for `edge2cell`, edge information, and edge-node PBI. That explicit
compatibility boundary accepts distinct source maps only when their complete
ordered global offsets, ghost rows, and push/pull lists match the canonical edge
mapping. Other source families require shared map identity. The plain
view and its row spans own no allocation, tracker, lock, or shared pointer. Their
native element access performs no lease acquisition or identity lookup.

## Local indices and partial closure

Local numbering appends ghost entities after owned entities. `NumCell`, for
example, counts owned rows; `NumCellProc` includes owned plus ghost rows. Geometry
relations such as cell-to-node and face-to-node require locally available nodes.
Faces used by owned cells and owned faces require locally available adjacent
cells, apart from an external face's second-cell sentinel.

Operators that dereference cells on every ghost face request
`allFaceCellsLocal=true` together with facial relations. This requires both
face-cell entries to be local on every captured face, except the permitted
external second-cell sentinel. Generic `CellFlux()` keeps the narrower owned-cell
stencil contract. Timestep evaluation uses the stronger requirement; face-area
and mean-normal setup request it when their geometry checks dereference cells.

Outer ghost support can contain negative encoded unavailable globals. Inverse
node support may also lack the wider cell closure. These entries preserve legacy
meaning and must be tested before dereferencing. `UnInitIndex` denotes permitted
absence, for example the second cell on an external boundary face. It is distinct
from an encoded unavailable global. A relation being local does not make every
entry usable as an array subscript.

Periodic coordinates use incidence-slot PBI data and the captured periodic frame.
`CellIsFaceBack` and `CellFaceOther` require the actual cell-to-face slot for a
self-periodic face whose two adjacent cells have the same index. Checking only
the cell ID loses its side. Preserve incidence order when using element face/node
slots.

## Host and CUDA placement

In legacy mode, host acquisition shares existing host allocations through the
Runtime bridge and retains their access controllers. CUDA acquisition in that
mode validates and copies the requested snapshot to the chosen device. Native
mode instead retains the existing immutable placement; `PlaceNative` must prepare
each device explicitly before acquisition. `DeviceView()` returns the same plain query type
with device pointers. `HostView()` rejects a CUDA scope and vice versa.

For repeated native execution, explicitly capture a
[NativeMeshSnapshot](NativeMeshSnapshot.md). It validates and copies once, then
serves the same query contract over private Runtime Arrays on host or CUDA.
Repeated reads avoid the legacy bridge's metadata conversion and full topology
validation. Its independent snapshot lifetime does not establish compatibility
with fields reordered since capture.

Use `MeshQueryCoords` for coordinate matrices inside a CUDA kernel. It has inline
capacity for every supported element; resizing the logical width does not
allocate. The dynamic `tSmallCoords` compatibility output is host-only. Coordinates
and connectivity remain read-only in the query interface.

## Lifetime, completion, and mutation

Retain the access owner until every use of its view and row spans has completed.
Deleting the parent mesh does not invalidate allocations retained by the scope.
Copying a plain view or an escaped span does not retain the scope. A host worker
must retain its owner through execution; a raw view captured by itself is unsafe.

The scope can move; the moved-from scope rejects new view requests. `Complete`
seals a scope after its last submitted use and attaches a completion from the
acquisition context. That completion must cover every use, including work joined
from other streams. `Wait` seals if needed and waits for the declared prefix.
Both reject later view acquisition. Allocation and mesh read leases remain held
until the scope is destroyed, even after waiting. Do not reuse previously copied
raw views after sealing.

Validity and eager submission do not establish cross-stream order. Callers must
record and wait for dependencies and ensure the declared completion joins all
consumers. Keep streams and MPI communicators alive through their required use.
Supported tracked mutation rejects active mesh readers. Direct writes through
public legacy arrays, raw pointers, or periodic metadata remain compatibility
escape hatches and must not race an acquired scope. Coordinate/periodic edits
must declare the appropriate generations and refresh dependent data.

## Evidence and remaining work

`test_MeshFacade.cpp` checks synthetic ordered connectivity, ordinary and
self-periodic geometry, invalid association/shape cases, scope lifetime, stale
descriptors, and legacy/native halo setup. CUDA-specific tests launch direct
query kernels with retained snapshots and completion/lifetime checks.
`test_MeshFacadePipeline.cpp` prepares `UniformSquare_10.cgns` and periodic
`IV10_10.cgns` through the real partition/ghost/face pipeline. Its serial CGNS
coordinate oracle precedes deduplication and partitioning. It rechecks queries
and legacy/native halo values after local reordering and independent face/cell
redistribution with complete ghost reconstruction. These are implementation test
definitions; successful execution must be established by the current validation
logs, not inferred from their presence.

Explicit native authority and resident device placement are described in
`NativeMeshPublication.md`. Detached mesh-plus-field transitions and their
supported inventory are described in
`../dev/geom_detached_transition_2026-10-24.md`. Native inverse/composition and
field-bundle staging are documented in `geom_relation_access_2026-10-24.md` and
`geom_field_staging_2026-10-24.md` under `docs/dev`. The progress ledger and
current validation logs remain the authority for completed gates; these APIs
alone do not establish full numerical solver equivalence or every cross-mesh
mutation policy.

## Finite-volume and reconstruction dependencies

Metrics, reconstruction bases, and coefficients have separate mesh stamps for
topology, geometry, distribution, and layout. Migrated operators acquire their
mesh read scope and validate dependencies once before numerical loops; geometry
getters remain plain lookups. Auxiliary interface/value publication does not
invalidate geometry metrics. Wall-distance anisotropic bases additionally check
the wall-distance allocation identities/versions, so rebuilding that auxiliary
field requires rebuilding those bases and coefficients.

After geometry changes or cell reorder, rebuild `ConstructMetrics`, then
`ConstructBaseAndWeight`, then `ConstructRecCoeff`. Each invalidates its own and
downstream stamps before work and publishes only on success. Repeated base
construction clears boundary caches. Rebinding to a different mesh and parsing
reconstruction settings also require rebuilding.

Standalone `FiniteVolume` users finish the granular construction sequence with
`FinalizeMetrics()` (`finalize_metrics()` in Python). A completion mask plus one
construction stamp requires every cell/face stage to succeed in the same mesh
epoch; matching array extents alone cannot republish old values. Independent
granular constructors retain legacy getter use but do not certify a complete
operator-ready bundle. `SetCellAtrBasic` starts a fresh build; the high-level
`ConstructMetrics` finalizes automatically.
