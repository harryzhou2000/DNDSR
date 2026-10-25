# Native mesh construction and access

`NativeMeshBuilder` builds primary unstructured topology directly from
`Runtime::Array` storage. `UnstructuredMesh::FromNative` wraps the result in the
existing public mesh facade. Neither operation constructs a legacy mesh array.
The ongoing migration is tracked in
[the Geom work ledger](../dev/geom_native_migration_2026-10-25.md).

## Ownership and publication

```mermaid
flowchart TD
    C[C++ Runtime arrays] --> A[Shared values and row metadata]
    P[Python Runtime.Array handles] --> A
    B[NativeMeshInput] --> A
    B --> W[Native sparse topology construction]
    W --> S[NativeMeshState]
    S --> A
    S --> E[Canonical entity spaces]
    S --> R[Named relation registry]
    S --> L[Retained read leases]
    F[UnstructuredMesh facade] --> S
    Q[C++ or Python read scope] --> S
    Q --> V[Plain HostView or DeviceView]
    V -. borrows .-> A
    X[Escaped Python row or slice] --> Q
```

Publication adopts existing coordinate and metadata buffers and retains read
leases. A caller can keep its original Array handle, but cannot write through
that alias while the publication or an escaped read still owns the allocation.
Replacing an Array handle does not retarget a mesh or an escaped row. A plain
C++ view does not own storage: retain its `MeshReadAccess` through all CPU uses
or attach a completion that joins every GPU stream using it.

Entity spaces describe the current owned/appended row order. Their global
indices are rank-offset addresses, separate from persistent entity IDs.
Replacing a descriptor handle shares its controller; it does not invalidate the
old descriptor. Actual mesh changes use the mutation/publication contract.

Persistent identity is the tuple `(mesh namespace, entity kind, ID)`. Publication
checks owned-ID uniqueness using distributed hash buckets and verifies ghost IDs
against owners. Each kind has a next-ID high-water mark that survives transitions;
deleting an entity must not lower it. New construction generates one collective
UUID unless a saved namespace is supplied. Native archive persistence for this
metadata remains part of the restart migration.

## Native construction

`NativeMeshInput` contains sibling owned-only Cell, Node and Boundary spaces,
owned coordinates, element metadata, and cell/boundary node cones in global
numbering. Optional persistent-ID columns must match the owned row counts.
Coordinates have three components even for two-dimensional geometry.

```cpp
NativeMeshBuildOptions options;
options.cellLayers = 2;
options.completeNodeStar = true;
auto state = NativeMeshBuilder::Build(std::move(input), options);
auto mesh = UnstructuredMesh::FromNative(mpi, state);
auto access = mesh->AcquireRead(state->Requirements());
auto view = access.HostView();
// Borrow view inside loops; keep access alive until all uses finish.
```

Construction is collective. Every rank, including empty ranks, must use the same
communicator, options and schedule. Do not invoke setup concurrently on the same
MPI communication context. Finish asynchronous producers before passing input
arrays to the host builder. Recoverable local input errors are agreed before
subsequent communication. Invalid communicator participation itself is a caller
error and cannot be repaired by an MPI collective on a different communicator.

The builder obtains inverse support rows at their owners, constructs distinct
neighbor/containment relations, discovers requested ghosts, and exchanges typed
companions through shared entity routes. Storage is proportional to local rows,
requested/served rows and rank metadata. It never indexes a local vector by the
global entity count or the magnitude of a persistent ID.

`completeNodeStar` includes every cell and boundary incident to an owned node,
including when none is owned locally. Without that option, outer support rows
may contain explicit unavailable global references. Cell node cones always have
complete local coordinates. This is topology support; node-centered control
volumes are not implemented by this constructor.

The first builder currently accepts nonperiodic primary meshes. Every external
boundary must be contained in exactly one cell. Unsupported input is rejected;
it is not converted through a legacy mesh. Facial/edge construction, periodic
builder integration and legacy reader migration remain work in the phase ledger.

## Relation identity

A relation key records endpoints, traversal path, selection, self inclusion,
ordering, periodic policy, and a named/versioned definition. It also records its
construction provenance. Full keys are compared across MPI ranks; a descriptor
hash alone never establishes equality.

Direct Face-to-Cell incidence, Face-to-Node-to-Cell union, all-node containment,
and exact incidence matching can coexist. An exact/custom matcher needs an
explicit definition; a lambda address does not identify its semantics.
`SortedUnique` means strictly increasing decoded global targets, even when the
local owned/ghost numbering has a different order. Topological incidence rows
preserve their ordered slots and permitted boundary sentinels.

`NativeMeshState::Relation(key)` is a retained host setup query. Its named
registry is independent of fixed solver-facing slots; extra named variants are
not silently substituted into those slots. The fixed `UnstructuredMeshView`
continues to provide CUDA-callable direct row access. Native setup headers do
not include legacy Array/ArrayTransformer definitions; legacy constructors are
provided by the explicit `RelationAccessLegacy.hpp` adapter.

## Python

`Geom.build_native_mesh` accepts the same shared Runtime arrays: float64
coordinates, int64 connectivity and IDs, and `Geom.ElemInfo` metadata arrays.
The binding validates types collectively before creating entity spaces. It does
not instantiate a new Python Array class for every C++ fixed width.

```python
mesh = Geom.build_native_mesh(
    mpi, 2, coordinates, cell_nodes, cell_info, boundary_nodes, boundary_info,
    complete_node_star=True,
)
with mesh.acquire_read(inverse=True) as read:
    cell_nodes_view = read.cell_nodes(0)
# The escaped readonly NumPy row retains its scope, values and row metadata.
```

Python setters on POD elements update array values through the Runtime access
contract before publication. A retrieved POD value is a copy; write it back to
the Array to persist changes. Published mesh read views are readonly. The native
constructor tests exercise alias protection and escaped rows after both the
mesh and original input handles are released.

## Migration and validation boundaries

`NativeMeshTransition` remaps topology through sparse owner requests and moves
rows, IDs and companions through native transfer plans. Persistent IDs are
payload, never inputs to adjacency renumbering. Read scopes over the old mesh
remain attached to the old allocations. `PreparedMeshTransition` stages field
changes and validates source attachments again before publishing the bundle.

Current native transition limitations are explicit: topology destruction and
output-cache rebuilding need their native builders. Legacy readers, refinement,
output and some edit workflows are still being migrated. Native primary
construction passing tests is not evidence that every legacy Geom path has
already been replaced. CPU, worker, MPI, Python and CUDA evidence is recorded
separately in the work ledger.
