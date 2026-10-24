# Entity-space descriptors and legacy field setup

`UnstructuredMesh::Entities(EntityKind)` describes the current row numbering of
one mesh entity family: cell, node, face, boundary, or edge. It retains the global
and ghost maps without exposing mesh arrays or their transformers. The entity
arrays must already have attached, prepared indexing; descriptor lookup performs
no MPI collective.

The descriptor provides `NumOwned`, `NumGhost`, `NumLocal`, and `NumGlobal`, plus
`LocalToGlobal`, `GlobalToLocal`, `Owner`, and `OwnerLocalRow`. Local numbering
appends ghost rows after owned rows. `GlobalToLocal` returns `std::nullopt` when a
global ID is unavailable locally. Negative boundary sentinels and encoded
unavailable indices are never converted into usable local rows.

`MeshIdentity()` is an opaque process-local identity. Do not serialize it or
compare its numeric value between MPI ranks. Legacy shallow mesh copies share
this identity and the version controller; independently constructed logical
meshes have different identities. Global indices are the current rank-offset
numbering, not persistent IDs that survive mesh redistribution.

The descriptor's version stamp depends on topology, distribution, layout, and
allocation. A tracked change to any of these makes its `Valid()` false and its
mapping operations reject use. Geometry-only changes leave entity numbering
valid. `SameDomain` compares mesh identity and entity kind; `SameOrder` also
requires both descriptors to be valid and to retain the same mapping objects.
Descriptors retain maps and the version controller, but do not hold a mesh read
lease or block mutation. Keep the communicator alive throughout their use.

## Borrowing indexing for an existing legacy field

Use `LegacyFieldBridge` where field setup previously borrowed a mesh array's
transformer. Keep the existing allocation and MPI datatype/request setup:

```cpp
auto cells = mesh->Entities(Geom::EntityKind::Cell);
// Allocate and size u.father/u.son using the existing field-specific layout.
u.TransAttach();
Geom::LegacyFieldBridge::BorrowIndexing(u.trans, cells);
u.trans.createMPITypes();
u.trans.initPersistentPull();
```

`BorrowIndexing` checks the descriptor stamp, owned row count, ghost row count,
and communicator rank ordering before assigning any maps. An initially empty
ghost allocation is permitted because legacy `createMPITypes` sizes it. An
already populated ghost allocation must have the descriptor's ghost count.
Congruent MPI communicators are supported; communicators with a different rank
ordering are rejected. The caller chooses the correct entity kind: an untyped
legacy field of coincidentally equal size does not identify its association.

The bridge performs no collective, allocation, resize, datatype creation, or
request creation. It does not expose a mesh transformer. Reader-owned serial
output routes remain separate legacy mappings.

For native payloads, `BuildRowTransferPlan()` explicitly constructs a collective
owned-to-ghost route. Every rank calls it in the same order, including ranks with
no owned or ghost rows. Each call constructs a route; retain and share the
returned immutable plan for reuse. The plan carries the descriptor's dependency
stamp so tracked numbering changes invalidate it.

## Transitional limits

Public legacy mesh arrays and maps still permit untracked edits. Such callers
must declare the appropriate mesh version changes; descriptor stamps cannot
detect arbitrary direct mutation of retained map objects. A borrowed legacy
transformer also retains its existing direct-call contract: after mesh changes,
reacquire a descriptor, borrow the new maps, and rebuild datatype/request bindings
before exchanging data. The bridge cannot automatically stamp later calls made
directly to that transformer.
