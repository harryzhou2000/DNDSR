# Optional checked eager execution

`AccessTrackingContext` adds host submission checks to the existing eager engine.
It changes neither the kernel view nor the CPU/CUDA execution contract. Checking
does not insert a dependency, wait for numerical completion, repair a stencil,
or validate undeclared raw-pointer work. Unchecked eager mode remains available.

```cpp
namespace R = DNDS::Runtime;
auto checks = std::make_shared<R::AccessTrackingContext>();
R::EagerEngine engine(2, checks); // zero workers uses the same rules inline
auto compute = engine.CreateCPUStream(), comm = engine.CreateCPUStream();
R::SubmitArrays(engine, compute, producer, R::Write(u));
comm.WaitFor(compute.Record()); // required edge, supplied by the caller
R::SubmitArrays(engine, comm, consumer, R::Read(u), R::Write(result));
engine.ReleaseOn(std::move(u), comm);
engine.Synchronize();
```

Python accepts `Runtime.EagerEngine(host_threads=2,
tracking=Runtime.AccessTrackingContext())`. Retain/share that explicit context
across participating engines. Native CUDA streams, `SubmitArrays`, DOF engine
operators, `CopyAsync`, async allocation initialization, NCCL enqueue and ordered
release use the same bookkeeping. Numerical CPU/CUDA callbacks still receive
borrowed views and must use their supplied stream. Custom providers must declare
all accesses through `SubmitOperation` and retain/publish their actual completion;
plain `Submit` is a control/externally managed boundary, not pointer discovery.

## Order and lifetime

Each stream has a process-unique lane ID and a monotonically increasing position.
An event captures its immutable submitted prefix and transitive dependencies.
Waiting on an old event cannot authorize a later producer. A read must follow
the last pending writer; a write/release follows that writer and every pending
reader. Independent reads need no edge. Already observed actual completion is
also sufficient. Freeing on another stream is valid after the explicit fan-in.
Every potentially final alias release is checked conservatively; reference counts
cannot prove which release will be final. Rejection returns the moved handle.

Allocation identity covers Buffer subranges and all Array aliases. Whole-allocation
checking is deliberate: component selection is not a disjoint memory region.
Owned and ghost allocations remain distinct, enabling core/halo overlap. Stream
positions and access checks do not add row-mask checks inside a stencil loop.

Contexts share a process-wide allocation-use registry so two contexts cannot
independently grant conflicting access to the same Buffer. Stream frontiers remain
private; cross-context events explicitly convey order. The registry keeps weak
allocation identities, prunes completed reader records and expired allocations,
and never retains payload storage. Dropped stream handles retire issue metadata;
immutable event prefixes and native completion owners remain valid.

Acceptance is transactional. A lane's issue lock serializes its submission and
event export. The shared registry lock covers only conflict checking/publication,
never numerical execution, native waits or collective agreement. Rejection rolls
back the writer/readers and cannot leave a phantom access. Recursive submission
or event export from inside that lane's unaccepted callback fails explicitly.
Concurrent submissions on other lanes remain possible.

Immediate access uses the existing thread-safe leases: pending writers prevent
CPU/Python reads, and pending readers/writers prevent an immediate write. Mixed
checked/unchecked scheduled conflicts fail at reservation instead of silently
crossing policy boundaries. Finish the old conflicting uses before switching
policy. Read/read may coexist. This guard cannot intercept independently imported
foreign aliases or a caller using raw pointers outside the declared API.

`Array::MutationEpoch()` / `Buffer::MutationEpoch()` identify the latest issued
mutation intent. A write lease or scheduled write reservation advances this
epoch immediately, whereas `Version()` still advances at successful publication.
This lets a queued consumer capture input provenance after a producer is queued
without waiting for it. The caller must still supply the stream dependency.
An unchanged epoch proves neither readiness nor numerical success. Pair it with
storage identity and shape generation, and reject a failed allocation. Canceled
or rejected writes may conservatively advance the epoch; it never rolls back,
so a later unrelated write cannot masquerade as the canceled producer. Reads,
polling, synchronization and write completion do not change the epoch.
The epoch is process-local access bookkeeping, not a persistent entity ID or
restart datum. It is shared at the allocation/access-domain level, including
overlapping Buffer subranges; it intentionally does not infer disjoint rows.

## Portable MPI bridge

```cpp
// bridge contains only required producer/prior-consumer dependencies.
auto halo = progress.Import(binding.BeginChecked(bridge));
R::SubmitArrays(engine, core, coreKernel, R::Read(owned), R::Write(rhs));
ring.WaitFor(halo);
ring.WaitFor(core.Record());
R::SubmitArrays(engine, ring, ringKernel, R::Read(ghost), R::ReadWrite(rhs));
progress.Drain(); // MPI-owning thread, before waiting on dependent workers
engine.Synchronize();
```

`BeginChecked` is collective and uses a distinct lane for outstanding MPI work.
Later commands on `bridge` do not automatically follow MPI completion. It imports
the bridge's producer prefix. CPU producers are observed before MPI reads; Direct
and HostPacked CUDA MPI observe that native prefix before posting; DevicePacked
queues packing there and progresses sends/unpack on the host. Rank-local ordering
rejection is agreed before payload posting. A rejected rank leaves peer bindings
and access accounting usable for retry.

`MPIProgress::Import` preserves the checked prefix in its portable event. CPU
workers can enqueue a wait without entering MPI. CUDA still rejects waiting on an
unfinished portable event; the MPI-owning host must observe it before dependent
CUDA submission. The checked MPI footprint is conservatively retired at **full**
operation completion; `WaitDestinationReady` alone does not release the checker
footprint. Its immediate destination lease semantics remain unchanged. Fine-grained
checked receive/send retirement can be added independently later.

Graph capture, replay footprints and capture-time allocation manifests remain
planned work. Checking eager submission does not establish graph eligibility.

## MPI memory capability

CUDA awareness does not establish pool/VMM pointer support. The local OpenMPI
5.0.9 build has `OPAL_CUDA_VMM_SUPPORT=0`: its legacy pointer-context path receives
a null context for `cudaMallocAsync`, clears the active context and aborts during
derived-datatype packing. A direct async-pool trial exposed this; it is retained
as a failed control, not a passing transport test. No system MPI was modified.

Buffers retain a provider-reported `StreamOrderedMemory` property. Ordinary
preallocated device storage still uses Direct by default. Direct MPI rejects
pool/VMM endpoints collectively unless the caller sets the explicit verified
capability `TransferOptions::allowStreamOrderedDeviceMemory`. It never silently
packs. HostPacked/DevicePacked are explicit alternatives; NCCL owns its native
compatibility path. Imported foreign allocations and custom allocators must
report this memory property too. A true flag is a caller assertion about the
installed MPI, not an automatic probe or a portable guarantee.
