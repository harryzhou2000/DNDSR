# Sparse fixed-record batch exchange

`DNDS/Runtime/SparseBatchExchange.hpp` adds setup-time request/reply batch
semantics over the existing Runtime row-transfer implementation. It does not
add a separate MPI payload backend. Call `Build(context, sourceRows, sendRows)`
with one ordered source-row list per destination rank, then `Exchange(source)`
on a host table. A row can be unsent or sent to several peers.

Count discovery uses `MPI_Alltoall` with one signed row count per peer. Receives
are allocated exactly and grouped by source rank; `ReceiveOffsets()` describes
those groups. `RowTransferPlan::Known`, `PayloadPlan` and direct transient
bindings perform payload movement. Application construction of a query/proposal
table is distinct from staging an existing payload. A POD record uses the same
registered `ElementSchema` as ordinary Runtime transfers.

Fixed-size means fixed element/row width within an exchange. Peer batches may
contain different numbers of records. More than one match means more proposal
records, not a ragged record field. Zero reply counts complete a candidate's
response; no-message probing must not be used as an epoch-termination test.

All ranks, including empty owners, participate in the same Build/Exchange order.
Record widths and tags must agree; inconsistent values are rejected collectively.
Finish CPU producers before Exchange. Source values, row metadata and storage
are protected by canonical transfer leases until the synchronous Exchange
returns. Callers retain MPI and the context, and finish uses before finalization.
For explicit nonblocking use, reuse `Route()` in an ordinary payload binding.
Concurrent bindings need distinct tags or contexts under the existing contract.

The adapter validates storage bounds and counts. Application records must carry
stable request keys and update epochs: equal counts cannot detect the wrong
query/proposal correspondence. Query origin, geometric candidate evaluation,
ambiguity resolution and donor priority belong to Geom, not this adapter.

Metadata is O(number of ranks plus actual routed row incidences); there is no
global-mesh-sized scratch. At very large rank counts, a sparse count-rendezvous
implementation can replace the initial all-to-all control phase while retaining
the same row-transfer payload contract. This first version is a collective
setup protocol, not a rank-independent mailbox or CUDA graph operation.
