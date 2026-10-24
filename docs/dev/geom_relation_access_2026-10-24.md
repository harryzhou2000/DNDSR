# Storage-neutral relation setup increment, 2026-10-24

`RelationReadAccess` retains explicit source and target `EntitySpace`
descriptors, their version-controller read leases, and host Array value/layout
read leases. Native inputs are not copied into a legacy adjacency. Legacy
inputs share the existing host allocation/controller and require source global
mapping identity plus local-target mapping identity. Native relation metadata
is bound explicitly by `NativeRelation`; `Read(source,target)` rejects another
domain/order even when dimensions agree. Raw native constructor association is
the caller's setup contract; no algorithm can discover a false declaration
from coincidentally equal array lengths.

Reader state is explicit global or local. `UnInitIndex` means absent. A local
negative `-1-global` encoding means an existing global target unavailable in
the retained local target mapping; it is not an absent boundary incidence.
Global sparse input accepts off-rank targets without requiring target ghosts.
Global input rejects local negative encodings, and local encodings that claim
a target is unavailable despite its presence in the supplied mapping are
rejected. Row slots and repeated incidences are retained by the reader.

`MeshConnectivity` now overloads `Inverse`, `Compose` and `ComposeFiltered`
for these readers and returns owned native CSR `NativeRelation` output built
through `Runtime::RowBuilder`. Composition calls extracted shared row mathematics
in `ConnectivityRows`, counts incidences, applies the existing shared-count/match
predicate and sorts output. Native inverse separately deduplicates and sorts
support rows with the same incidence meaning. Empty rows remain explicit CSR rows.

Distributed native inverse routes `(target-owned row, source-global ID)`
contributions to each target's owning rank. One native CSR batch per peer keeps
setup routing bounded by communicating peers; `MPI_Alltoall` exchanges presence
only. `RowTransferPlan::Known` and `DestinationLayout` discover the transfer and
row widths; `PayloadPlan` sends native `index` values with transient direct MPI.
No full topology allgather or legacy reverse-push/scratch adjacency is used.
Ranks with no owned sources or targets still participate. Received contribution
IDs are checked collectively before sorted unique support rows are published.

`RelationGhostRoute(sourceSpace)` explicitly creates an entity-bound collective
route. Copies retain shared native route state. `relation.WithGhostRows(route)`
requires the same source domain/order and global relation entries, discovers
native CSR ghost widths, and performs a transient direct transfer. It returns a
retained owned/ghost relation without changing the original owned relation. Local
index payloads are rejected because their target indices would need translation
into the receiving rank's target map; explicitly publish global entries first.
Wrong entity kinds/domains and stale source orders are rejected collectively.

Native composition is noncollective and requires every referenced intermediate
row to be available in its retained reader. The intermediate source/target
entity order must match exactly. `removeSelf` additionally requires equal
initial and final entity orders, rather than comparing coincidentally equal
numbers from different entity families.

This increment bypasses `AdjVariant`'s concrete legacy-width alternatives for
inverse/compose. The ghost-tree/interpolation registries still use their
existing legacy variant and have not been universally migrated. CUDA setup
readers, topology publication, PBI/orientation output for new incidence
generation, and detached full-mesh transitions remain separate work.

New `test_RelationAccess.cpp` covers native/legacy inverse parity, deterministic
sorted support, repeated and empty input rows, distributed sparse targets,
explicit native intermediate ghost transfer before composition, exact self
filtering, absent versus unavailable entries, wrong entity identity/map order,
and mesh mutation exclusion while a reader is retained. Native inverse tests add
independent transpose expectations, owner-routed sparse contributions, duplicate
incidences, zero-owned source/target ranks, collectively rejected malformed input,
and reusable typed ghost-route domain/order/staleness checks. Initial syntax checks
passed for new source and test in
`geom_validation_2026-10-24/relation-access-syntax.txt`; build/MPI validation
is owned by parent orchestration and must be reported separately.
