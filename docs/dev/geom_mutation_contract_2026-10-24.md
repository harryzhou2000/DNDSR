# Managed mesh mutation contract, 2026-10-24

Public managed mutators now acquire an exclusive mesh version writer before
touching topology, geometry, maps or exposed caches. Collective setup methods
first agree that every rank can mutate, so an active read on one rank rejects
the operation before peers enter its payload exchange. Local index conversion
and periodic metadata setters remain local operations. Constructor setup still
starts with a fresh controller and no published descriptors.

`MeshMutationToken` is an explicit owner/thread-bound capability. Public
entrypoints always acquire a new writer; they cannot infer permission from a
thread-local depth counter or an existing writer. Private `*Impl(token,...)`
calls alone join the outer edit. Nested implementation scopes union their
declared change masks, and the outer scope publishes one generation increment
per changed component. A nested exception sets a sticky failure flag even if
an intermediate implementation catches it. Uncommitted/failed edits invalidate
the mesh controller; recovery means an explicit replacement/restart rather
than clearing that failed state.

## Managed implementation coverage

1. Primary/facial/C2F/inverse/C2CFace/edge local-global conversions, primary
   and inverse relation recovery, primary/face/inverse/edge ghost builds,
   face/edge interpolation, boundary matching and cell-face-neighbor building.
   Composite face/edge setup uses private token calls and commits once.
2. Distributed and local mesh reorder, including the legacy local reorder
   path. Native fields still prepare before the destructive mesh writer;
   the consuming mesh implementation runs under the token, then successful
   wrapper bookkeeping publishes the prepared field handles.
3. Periodic geometry changes, managed coordinate transformation, wall-distance
   replacement, elevated boundary displacement preparation and the four
   coordinate smoothing implementations.
4. Elevation and bisection destination publication. A source mesh read lease
   is retained during construction; source/destination controller aliasing is
   rejected collectively. Bisection now deep-copies coordinate/node-ID fathers
   because its destination has an independent mutation controller: later
   managed coordinate changes cannot silently alter source geometry.
5. Direct deserialization and collective read/distribute substeps, including
   nested recovery and redistribution through explicit private capabilities.
   Serial partition publication and the periodic-deduplication live flag are
   guarded, as are legacy recovery/primary-ghost/face-interpolation paths.
6. Periodic output-coordinate and VTK connectivity caches. Boundary-mesh
   construction checks compatible dimensions/communicators, independent idle
   controllers, then holds both source-cache and destination writers because
   it updates `node2bndNode` in the source as well as destination arrays.

## Change masks

| Operation | Declared generations |
|---|---|
| Index representation conversion | Layout, Values, Interface |
| Recovery / face or edge construction | Topology, Distribution, Layout, Allocation, Interface |
| Ghost reconstruction | Distribution, Layout, Allocation, Interface |
| Face boundary metadata matching | Above topology group plus Values |
| Cell-face-neighbor auxiliary relation / wall distance | Interface, Values |
| Coordinate smoothing | Geometry, Values, Interface |
| Periodic geometry metadata | Geometry, Interface |
| Read/replacement and elevation/bisection | All |
| Boundary parent cache / output caches | Interface (plus Values for output caches) |

Masks are conservative where multiple legacy groups share one controller.
Auxiliary wall-distance/cell-face-neighbor publication does not stamp primary
topology/layout. CFV geometry-cache dependencies are coordinated separately.
Native replacement may later introduce per-group allocation/distribution
generations without weakening this outer publication boundary.

After native publication integration, public wrappers obtain their token from
`BeginManagedMutation`, which rejects native authority. The collective availability
gate agrees that rejection before any legacy payload. Only explicit authority
selection/import (`UseLegacyWorkspace`, `PublishNative`) and prepared native edit
publication bypass that default. Selecting the legacy workspace does not export
current native values. Authority changes stamp all components; private nested
steps retain the same capability.
Passive elevation/bisection source algorithms explicitly require the legacy
workspace until a corresponding current-state geometry source adapter exists.
`WriteSerialize` is exclusive setup because legacy serialization can compress
CSR shapes. It validates global host workspace completeness and idle buffers
before side effects, rejects retained readers, and stamps Allocation/Interface;
native authority uses the separate collective `WriteCurrentMeshSnapshot` exporter.

## Remaining escape paths and limits

Public legacy arrays, adjacency states, maps, parent vectors and metadata can
still be assigned or edited directly. Those writes are not intercepted. Raw
legacy device-mirror `to_device/to_host` operations and arbitrary direct
transformer calls retain their existing caller synchronization contract.
Serial-reader scratch construction, output settings/serializer state,
`ObtainLocalFactFillOrdering`'s local symbolic cache and direct borrowed
mapping mutation are not all wrapped. These are explicit compatibility
surface restrictions, not a complete mesh immutability claim.

The token provides serialized exclusion/invalidation and failed-state
semantics, not rollback or atomic detached mesh publication. Arbitrary user
callbacks that enter unmatched MPI, MPI process failure and universal
allocator fault recovery remain outside the guarantee. Ordinary collective
preflight prevents reader rejection divergence; legacy algorithm-internal
failure points retain their existing MPI contracts.

`test_MeshMutation.cpp` adds active-reader rejection before side effects,
nested face setup with one increment per component, sticky nested failure,
rejected public callback reentry, failed coordinate edit rejection, and
bisection/source geometry isolation under managed transformation.
`geom_validation_2026-10-24/integration6-geom-tests.txt` records fresh CPU passes
at MPI 1/2/4/8, including these tests. The earlier syntax and binary logs retain
their original narrower scope.
