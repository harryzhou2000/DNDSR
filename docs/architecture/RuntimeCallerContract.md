# Runtime Array caller contract

For external numeric tensors, also follow [the DLPack contract](ArrayDLPack.md):
export holds a write lease through the last foreign alias, CUDA release fences
the consumer stream, and the caller joins other streams. Import retains foreign
ownership but cannot discover repeated imports or intercept later external writes.

The Runtime checks access conflicts, shapes, typed payload compatibility and
declared versions at API boundaries. Fully eager execution does **not** prove
dependency ordering. A successful submission or `IsValid()` result is not proof
that an operator read the intended time level. These obligations apply equally
to inline CPU, worker CPU and CUDA execution.

## Execution, access and final release

1. Declare every accessed allocation and its intent in `SubmitArrays`. Use
   `Read`, `Write` or `ReadWrite` consistently with the kernel. `Write` does not
   zero storage or guarantee that the kernel initializes every element. A raw
   `Submit` callback must implement equivalent lifetime/access discipline itself.
2. Order conflicting uses on different streams explicitly: record the producer's
   prefix and enqueue `WaitFor` on each consumer. `Record()` covers only work
   already submitted. Scheduled reservations retain/protect storage against
   immediate access; they intentionally do not reject conflicting scheduled
   uses or infer a join. Missing joins can produce silently stale values or data
   races. A race is outside the API contract, not an expected recovery path.
3. Kernel `ArrayView` values are borrowed pointer/offset descriptors. Keep their
   access lease or submitted operation alive through the actual use, including
   device completion. Do not retain a raw pointer for later independent work.
   Bounds, stencil validity and matching device placement are caller obligations;
   there is no per-element lease or bounds check in the hot path.
4. Launch device work on the supplied `ExecutionContext` stream and check launch
   errors. Work submitted to another native stream must be joined before the
   enclosing operation publishes completion or relinquishes ownership. Keep
   borrowed native streams alive through all uses and ordered frees.
5. Join **all** independent uses before final release. `ReleaseOn(std::move(a), s)`
   consumes one shared handle; it is not a request to free an allocation with
   other live owners. The final decrement determines the actual release. Releasing
   aliases on different streams is safe only if whichever can be final is ordered
   after all uses. Reference counting supplies ownership, not this ordering proof.
6. Observe errors with `WaitHost`, operation `Wait`/`Test`, or engine `Synchronize`.
   Destructor cleanup is a safety fallback and cannot reliably report errors.
   Check failure before publishing results; coordinate rank-local recovery before
   subsequent MPI collectives. Do not concurrently mutate the same C++/Python
   Array handle; independently owned aliases do not make value races safe.

```cpp
// Both lanes read temp. Neither read is ordered after the other.
R::SubmitArrays(engine, compute, computeKernel, R::Read(temp), R::Write(result));
R::SubmitArrays(engine, comm, deviceCommKernel, R::Read(temp), R::Write(packet));
comm.WaitFor(compute.Record()); // comm already orders its own use
engine.ReleaseOn(std::move(temp), comm); // final owner may retire on comm
engine.Synchronize();
```

CPU workers obey the same rule: callbacks retain submitted owners through actual
execution; CUDA launchers hand off native allocation release with stream ordering.
A CPU lane is not a dedicated thread. Do not block a bounded worker pool waiting
for callbacks that need its occupied workers. Use stream waits for dependencies.
With zero workers, ready CPU work executes inline without changing the contract.

## Halo overlap and MPI

Owned and ghost Arrays have separate allocations. A core stencil may read only
owned values while a halo transfer writes ghost values. Classify core/ring cells
from actual adjacency during setup, and rebuild that classification after changes
to connectivity, partition or local numbering. A combined ArrayPair view does not
prove that a core stencil stays within owned rows.

The main thread can begin a transfer, submit core work, progress MPI, and then
submit ring work after the transfer event. Keep output separate from pending
communication writes. `BeginScheduled` permits later scheduled consumers but does
not finish an unfinished CPU producer. Complete that producer before beginning
MPI. For portable CUDA/MPI, progress the host MPI operation to completion before
asking a CUDA stream to wait on its portable event; an unfinished portable event
is rejected. CUDA-aware MPI is not assumed stream ordered or graph capturable.

Call route construction, payload setup, binding and `Begin` in matching collective
order on all participating ranks, including ranks with zero rows. Use MPI only on
threads permitted by the initialized MPI thread level. `MPIProgress` is explicitly
driven on its creating thread; drain it before `MPI_Finalize`. Do not wait on a
worker dependent on MPI completion while withholding the required host progress.
Use distinct matched tags for concurrently active bindings in a shared
`CommunicationContext`. Transient requests change setup cost, not these rules.

Contexts support `MPI_COMM_SELF` and reordered split communicators. When a
communicator does not carry `MPI_TAG_UB`, tag reservation uses the portable
range 0–32767; it does not assume that every communicator inherits world-model
environment attributes. This is the minimum range required by the
[MPI standard](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node245.htm).

`WaitDestinationReady()` makes the destination readable; the source can still be
in use until full completion. `Wait()` or successful `Test()` completes both.
Retain the operation or import it into a live `MPIProgress` owner. Check errors
explicitly, then coordinate recovery across ranks.

## Schemas, reductions and redistribution

An `ElementSchema` describes MPI fields with an extent equal to the POD element
stride. Its typemap must be valid and nonoverlapping within an element, with the
same signature and semantics on communicating ranks. Arithmetic and Python dtype
schemas carry checked type tags. Float32/64 and signed/unsigned 8/16/32/64-bit
integer schemas use the same names in C++ and Python, so their payloads and
reducers interoperate. Explicit untagged MPI schemas, raw Buffer binds and native
type-erased ArrayStorage imports remain caller promises: equal byte
size cannot establish a semantic type. Pointer/owning-object bytes are not
portable payloads. Import foreign storage once with a genuine shared owner; derive
aliases/ranges from that Allocation so they share one access controller.

Collision-free receives use copy. Collisions require an explicit reducer, using
receiver staging; unordered multiple writes are rejected. Custom reducers must
have the same named meaning and initialization policy across ranks, read only
transmitted fields, and obey the supplied execution context. Do not retain callback
pointers or call unmatched MPI collectives. Floating-point reductions need not be
bitwise invariant under repartitioning. `RowTransferPlan::Known` requires matching
send/receive row semantics: counts alone cannot detect a wrong permutation.

A route is immutable correspondence, a payload holds layouts, and a binding pins
specific allocations. Resetting an Array or replacing a mesh does not retarget
them. Retained views/aliases continue to refer to the old storage. After migration,
publish returned Arrays, rebuild layout-dependent payloads/bindings, and refresh
adjacency and core/ring classifications. Attach relevant mesh version stamps and
commit mutations; version checking detects stale declared dependencies but is not
a lock. Drain old operations before mutating their dependencies.

`DynamicRowTransfer` pins its source and privately builds a CSR destination;
construction and `Commit()` must follow matched collective order. It publishes
only after successful payload completion. Multiple fields are separate operations:
the application must commit and validate the complete field bundle before
publishing a new mesh epoch. There is no atomic multi-field transaction.

## Shape editing, Python and restart

Published row metadata is shared and immutable. `RowBuilder` permits independent
workers to edit distinct rows; callers join them before `Commit`. Resizing a row
invalidates its borrowed pointer. `ExclusiveRows` requires unique storage and no
active access; it blocks sharing until publication. Abandoning a destructive edit
invalidates the Array rather than rolling back its values. Do not publish a draft
or keep row pointers through resize/commit.

Python slices and element proxies retain ArrayStorage (values, metadata and access
controller), including through parent reset/destruction. They do not hold a
permanent lease. Each access acquires a lease; pending asynchronous writes must be
completed before synchronous indexing or NumPy snapshots. NumPy results and POD
class-value reads are copies. Mutate through a write lease, tracked slice assignment
or element proxy; changing a returned snapshot does not change the Array.

The Array I/O adapter uses the existing Serializer, stages CUDA snapshots on host,
and serializes logical rows without padding. Complete asynchronous producers first.
Parallel serializers require matching dataset/schema call order; per-rank
serializers are independent. Use stable explicit POD schema names and compatible
ABI bytes. Reopen the reader when selecting a different partition slice of a cached
dataset. Checkpoint the mesh IDs, state time level and all related fields together
at the application level; individual Array writes are not an atomic restart.

## What tests establish

Tests exercise immediate conflict errors, stale version/schema rejection,
cross-rank validation, failed-write invalidation, actual overlap, and independent
full-value oracles after joins. Missing-join tests deliberately finish a read
before opening a gated writer, demonstrating silent stale data **without** invoking
a data race. Bounds violations, invalid external pointers and genuinely racing
kernels remain caller errors; tests must not rely on undefined behavior to pass.

See [Array runtime](ArrayRuntime.md), [eager execution](EagerExecution.md),
[allocation lifetime](ArrayAllocationLifetime.md), [MPI transfers](ArrayTransientMPI.md)
and [row editing](ArrayRowEditing.md) for interfaces and ownership diagrams.
