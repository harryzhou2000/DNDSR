# Native field bundle staging, 2026-10-24

`RegisterField` and `RegisterArray` retain their existing direct `fn` callback
and now also supply staged preparation callbacks. The mesh wrapper validates
the collective callback schedule, attachments, schemas and dispositions,
prepares every native destination using the existing mesh correspondence,
then begins the consuming mesh mutation. Native field handles publish only
after mesh bookkeeping succeeds. Publication consists of no-throw move
assignments; no field handle publishes merely because its own transfer ended.

Ordinary preparation failures discard all destinations. Every native source
handle, value, allocation identity and field epoch remains unchanged,
including fields prepared before a later field failed. Source aliases retain
their original allocations after successful publication as well.

Preflight checks owned row counts, idle/valid source storage, Field association,
host placement for Field, Array schema/type/context compatibility, and Field
domain/epoch agreement within each entity kind and across ranks. Field schema
signatures include component count and stride. Array signatures include
element schema/type and memory space; ragged widths remain per-row metadata
validated by Runtime routing/payload checks. `RetainUnchanged` and
`ProhibitReorder` reject an active reorder of the attached entity family.
An Array has no intrinsic entity association/domain: its declared association
remains the registering caller's contract.

The native adapters agree on local destination-allocation/description failure
before the next payload phase. Runtime `RowTransferPlan` and `PayloadPlan`
retain their collective route/schema/row-width checks before traffic. The
same `NativeRowTransfer` correspondence is used for native Array and Field;
Dense Field stride padding is transferred as part of each exact byte row.

This is field-bundle staging, not a detached atomic mesh transition. Legacy
`fn` callbacks still consume their arrays in place. A later mesh or legacy
callback failure leaves the mesh in its existing failed mutation state, while
unpublished native field sources remain unchanged. Mesh rollback, arbitrary
custom callback MPI divergence, MPI process failure and universal allocator
fault tolerance are not provided. Preparation/execution/publication remains
serialized; callers retain all referenced field objects until the call ends.

`ReorderPlan::apply` preserves its old local-only legacy path without adding
MPI. Applying a staged/policy registry is collective and requires matching
registration mode/order on every rank. `prepareCompanions` is always
collective and can be used explicitly; the mesh wrapper always uses it and
checks registration agreement before mutation.

Added regressions in `test_MeshReorder.cpp` cover a second preparation failing
on rank zero after the first field was prepared, with unchanged source values,
identities and epochs; and association/disposition/domain/epoch/preparation
failures before mesh mutation, with unchanged mesh stamps and both field
handles. Existing success tests retain exact histories, IDs, ragged Array rows,
local permutation, distributed receive order and old aliases.

`integration5-geom-regression.txt` and the subsequent fresh CPU
`integration6-geom-tests.txt` cover these staged field cases at MPI 1/2/4/8.
Earlier `reorder-full-*` logs predate field staging and establish only their
earlier source state. A detached prepare/publish API is separate active work.
