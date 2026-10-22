# Native Array runtime and gradual Geom migration

`DNDS::Runtime::Array<T, LayoutPolicy>` is the native interface. `DNDS::Array`
continues to provide its existing interface and deep-copy semantics; its host
allocation uses the common Buffer facility. Euler, CFV, ArrayPair and
ArrayTransformer can migrate incrementally. This runtime adds no PETSc, hypre,
LibTorch, Legion or mesh-library dependency.

## Responsibilities

The [caller contract](RuntimeCallerContract.md) specifies required stream joins,
raw-view lifetime, MPI participation, migration rebinding and checkpoint discipline.

| Component | Owns |
|---|---|
| Allocation | Shared native/foreign owner, device, identity, value generation and access controller |
| Buffer | A bounded byte range retaining its Allocation; overlapping ranges share exclusion |
| Allocator | Optional allocation/deallocation policy, including pools or pinned host storage |
| AllocationResource | Backend payload with explicit ordered-release capability and conservative RAII fallback |
| RowLayout | Immutable table, variable-length padded, or CSR row shape in element units |
| MetadataBuffer | Shared Buffer containing published host offsets, exposed only for reading |
| ArrayStorage | Shared record with a guarded publication boundary for values and immutable row metadata |
| Runtime::Array | Typed handle to ArrayStorage; optional fixed-width constraint |
| PythonArray | Runtime dtype/POD codec and the same ArrayStorage as C++ |
| ArrayAccess | Move-only read/write lease retaining both allocation and layout |
| ArrayView | Borrowed plain pointer/offsets for kernels; no tracking per element |
| RowBuilder / ExclusiveRows | Unpublished CPU row construction or unique shape edit; Commit publishes metadata |
| ArrayPair | Separate owned and ghost Arrays; optional combined read indexing |
| RowTransferPlan | Immutable distributed row relation, reversible and reusable across fields |
| ElementSchema | Retained MPI typemap, element extent and application type-signature identity |
| PayloadPlan | Row lengths, relative MPI datatypes, coalesced segments and contribution groups |
| CommunicationContext | Shared communicator and explicit matched-tag reservations |
| TransferBinding | Allocation identities, transport/reducer/workspace and transient or persistent request policy |
| TransferOperation | Active requests and leases; separate destination-ready and all-complete boundaries |
| DynamicRowTransfer | Pinned source and private CSR destination; collective publication after payload success |
| EagerEngine / StreamLauncher | Common ordered submission contract with inline/worker CPU and native CUDA providers |
| MPIProgress | Main-thread MPI retirement and explicit events observed by eager streams |
| ArrayIO | Schema-tagged row checkpoint adapter over the existing parallel serializer |
| VersionTracker | Selective mesh dependency generations and explicit mutation scopes |
| Geom::NativeRowTransfer / RegisterArray | Bridge from existing mesh permutation correspondence to native Arrays |

`Field` and `ExchangePlan` remain compatibility helpers for the initial prototype.
ExchangePlan now delegates to the same row-transfer implementation. New code
should retain route, payload and binding separately to reuse setup.

The detailed contracts and ownership graphs are in
[row editing](ArrayRowEditing.md), [eager execution](EagerExecution.md),
[allocation lifetime](ArrayAllocationLifetime.md), and
[transient MPI](ArrayTransientMPI.md). Geom migration remains optional; the
new runtime and its tests do not require replacing existing mesh fields.
The [2026-10-22 audit](../dev/array_runtime_audit_2026-10-22.md) summarizes
coverage, reviewed contracts and the retained-transfer-error correction.

## Element types and device dispatch

The [Euler workload and overlap audit](../dev/array_runtime_scenarios_2026-10-22.md)
maps production data patterns to CPU/MPI/CUDA tests, including detected type
mismatches and deliberately unchecked eager ordering mistakes.

The C++ interface is still templated: `Array<T, LayoutPolicy>` and its leased
`ArrayView<T>` preserve typed element access. For example:

```cpp
using namespace DNDS::Runtime;
struct CellInfo { int64_t globalId; int32_t flags; double volume; };
Array<double> state(RowLayout::Dense(100, 5));
Array<int64_t> ids(RowLayout::Dense(100, 1));
Array<CellInfo> metadata(RowLayout::Dense(100, 1));
```

Storage allocation, identity, access tracking and byte copying live in the
non-templated Buffer layer. Host/CUDA dispatch uses Device and ExecutionContext
at operation boundaries. Element loads through a kernel view are ordinary
typed pointer arithmetic; they do not branch on a runtime dtype or device.
Numerical kernels and reducer implementations may separately specialize on T
and backend, without templating the entire communication engine.

T must be trivially copyable and standard layout, with alignment at most 64
bytes. Allocation initializes bytes; it does not run per-element constructors
or destructors. Custom data structs are supported, while owning objects such
as std::string/std::vector are not. Pointer values also do not describe portable
MPI, checkpoint or device payloads. A custom MPI payload supplies ElementSchema
with a matching extent; a custom reduction supplies the appropriate callback.

Python uses a runtime dtype/POD codec over the exact native ArrayStorage record.
The public `Runtime/Array_bind.hpp` converter shares values, row metadata and
the controller with any bound `Array<T,L>`; no byte-Array intermediary or
width-specific Python class is needed. Numeric and structured POD dtypes are
supported. Bound POD classes can register a class-value codec once per type,
independent of Array widths. Object-reference NumPy dtypes remain rejected.
See [Python ownership and binding](ArrayPythonOwnership.md) for class/object
diagrams, the generic converter and tracked POD field editing.

`FixedRows<N>` adds a C++ width constraint; it does not require a new Python
Array class. `Dense(rows,width,stride)` is a uniform-width strided table, not
the legacy padded layout with variable row lengths and uniform capacity.
`Padded(lengths,capacity)` now provides that variable-size representation;
CSR remains available for packed variable rows. See [row construction and
exclusive edits](ArrayRowEditing.md) for Commit, retained snapshots and Python.
Existing legacy Array layouts remain available.

## Ownership, access and failure

Array copy/alias shares allocation and access state. `Clone()` makes independent
values; `To(device)` makes an explicit placement copy. `CopyFrom` returns an
operation for asynchronous copies. `Reset` replaces the handle's allocation and
layout; existing aliases, bindings and leases retain the previous ones.
`FromStorage` can attach a typed row layout to an owned Buffer, retaining its
controller; it does not wrap unmanaged pointers.

A read shares access with other reads. A write excludes both reads and writes
to the same allocation, including differently shaped aliases. A successful
write publishes one generation on completion. A failing write scope or
explicit `Fail()` marks the allocation unreadable; replace it with `Reset`
to recover. An unissued collective-transfer preflight is cancelled without
publishing a value change. There are no row-mask checks in production views.

```cpp
using namespace DNDS::Runtime;
Array<double, FixedRows<5>> u(RowLayout::Dense(nCells, 5));
{
    auto write = u.Write();
    auto view = write.View();             // unchecked kernel view
    for (size_t i = 0; i < view.Rows(); ++i)
        view(i, 0) = 1;
}
auto independent = u.Clone();
auto gpu = u.To(Device::CUDA(0));
auto context = ExecutionContext::CUDA(0, stream);
auto read = gpu.Read(context);
// launch<<<..., stream>>>(read.View());
read.Complete(Completion::Record(context));
read.Wait();
```

`At` checks host placement and bounds; `View<true>()` selects a separate
development checking policy. Raw views and `unsafe_address` are borrowed:
a caller must not retain/use them beyond the lease or after submission.
C++ cannot revoke a copied raw pointer.

CUDA leases retain storage until the recorded event completes. Without an
event, release fences that lease's stream. Streams are borrowed and must remain
alive. Allocations are initialized before publication. Host/CUDA copies are
explicit; cross-device ordinal copies are currently rejected. Python debugging
indexing is a deliberate convenience exception: it synchronizes selected
device values with host memory.

The device descriptor and execution context are distinct from Array shape,
routing and physics. New accelerators require implementing allocation,
completion/fence and copy dispatch in the backend layer and supplying their
kernels; no per-element virtual dispatch or mandatory host mirror is required.
Currently executable backends are Host and CUDA. Allocator policies must return
zero-initialized correctly placed storage and preserve ownership until the
last lease ends.

## Owned/ghost overlap

Owned and ghost remain separate allocations. A halo operation takes an owned
read lease and a ghost write lease. Core work receives only the owned view and
writes to an independent residual Array. `WaitDestinationReady()` drains
receives, combines contributions if needed and releases the ghost write lease.
Ring work can then read the ghost Array. `Wait()` also drains outgoing requests
and releases the owned send lease. An independent residual can retain one
write lease across core and ring work.

The application must classify core rows from its complete input stencil before
launch and invalidate that classification after connectivity/stencil changes.
A core view has no ghost pointer. This phase does not introduce a solver kernel
registry or claim automatic proof of arbitrary kernel addressing. Raw adjacency
mutation still requires explicit invalidation. General suballocation regional
leases are unnecessary for this owned/ghost pattern and remain deferred.

## MPI routing and payloads

All plan setup, binding and `Begin` calls are collective on the route
communicator and must be called in the same order. Structural orchestration is
serialized; this is not a multi-threaded distributed lock manager. By default
a binding has its own communicator and persistent requests. TransferOptions
can select transient requests and a shared context with an explicit matched
tag; see [the transfer policy guide](ArrayTransientMPI.md). Only one operation may be
active on a binding. Multiple bindings can share the same immutable route and
payload while retaining independent allocations and requests.

```cpp
RowTransferPlan route(mpi.comm, owned.Rows(), requests.size(), requests);
Array<double> ghost(route.DestinationLayout(owned.Layout()));
PayloadPlan payload(route, owned.Layout(), ghost.Layout(),
                    ElementSchema::Arithmetic<double>());
auto binding = payload.Bind(owned, ghost); // Direct is the default
auto transfer = binding.Begin();
// owned.Read().View() can feed a core kernel here
transfer.WaitDestinationReady();
// ghost.Read() is now permitted
transfer.Wait();
```

Each RowRequest names a source rank/row and a local destination row.
`Reverse()` transposes that relation. Ragged widths are exchanged at setup,
including empty rows and empty ranks. Copy destinations must be disjoint;
collisions are detected before starting requests. Shape replacement requires
a new payload/binding; an old binding intentionally continues to target its
retained allocations.

The supplied MPI datatype's **extent** is the element stride, and must match
the stored POD size. Its typemap must fit inside that extent. Schema identity
is an application assertion of matching type signatures across ranks, not
runtime reflection of arbitrary C++ structs. The caller supplies a valid
non-overlapping element typemap. Retained datatype ownership survives freeing
the caller's MPI handle.

Direct transport sends/receives derived datatypes against actual storage.
Adjacent segments merge; physically reordering sender rows outside MPI can
therefore produce a single segment. This is distinct from packaging.
`TransferTransport::HostPacked` explicitly uses MPI_Pack/Unpack and host
workspace, with host/device staging when necessary. It is a portable fallback,
not an automatic policy switch. Its host workspace is currently pageable;
pinned/device packing optimizations are not claimed. Peer packed messages are
limited by MPI's int packing API and rejected when too large.

Direct CUDA communication requires the MPI CUDA capability query to succeed.
Portable producer synchronization happens before MPI starts; receiver
completion happens before consuming kernels launch. This is not stream-native
MPI, GPUDirect performance evidence, or multi-node validation.

## Reductions

Default Copy rejects fan-in. Explicit reducers receive disjoint typed
contributions into operation workspace and combine at the receiver. Sum, min,
max and integer OR/XOR/AND have CPU and CUDA implementations. Custom reducers
can supply CPU and device callbacks; applications own their semantic identity.
Payload and binding own communication details; callbacks own no MPI requests.

`DestinationInit::Preserve` accumulates into existing values.
`Identity` initializes each touched destination once; unrequested rows remain
unchanged. Zero-width rows contribute no operations. Contributor order is
stable for a fixed route (source rank, then edge order). Floating min/max use
number-preferring fmin/fmax behavior. Integer sum wraps at its type width.
No partition-independent floating-point bitwise reproducibility is promised.

A failing custom reducer drains the requests, poisons partial output and
releases access. Explicit Wait reports the error. Destruction is a blocking
fallback, not an error-reporting API. MPI process/transport failure recovery
is outside this phase.

## Checkpoints

`WriteArray` / `ReadArray<T>` use the existing serializer's distributed CSR
path. Parallel HDF5 supports saved matching-rank partitions, EvenSplit at a
different rank count, and explicit contiguous global row slices. JSON is
per-rank. CUDA data is explicitly staged on host. Padding between rows is
omitted; reads reconstruct packed rows (or the requested fixed-width policy).

C++ and Python share format version 2, logical element byte size, schema
identity and byte-row encoding. Arithmetic schema names are stable
(`float64`, `int32`, etc.). Application PODs should provide an explicit stable
schema name; Python structured dtypes can use the same supplied schema.
The format requires compatible binary representation/endianness; it is not a
portable codec for arbitrary structs.

The existing shared-row-start serializer permits one resolved slice per
dataset in an open read session. Close/reopen the file to read a different
slice of that dataset. A matching-rank read rejects a different communicator
size; use EvenSplit or an explicit partition instead. Rank-local access
conflicts and schema disagreement are checked collectively before data I/O.
Ordinary MPI/HDF5 collective call discipline and allocation availability are
still required.

## Python debugging and production access

```python
import numpy as np
from DNDSR.DNDS import Runtime as R

a = R.Array(R.RowLayout.dense(4, 3), np.dtype("float64"))
a[:] = np.arange(12).reshape(4, 3)
column = a[:, 1]             # live controlled view of the same allocation
column[1:3] = [-5, -6]
assert a[1, 1] == -5
snapshot = column.numpy()   # independent NumPy copy

gpu = a.to(R.Device.cuda(0))
gpu[2, 1] = 17              # synchronized host-to-device element write
value = gpu[2, 1]           # synchronized device-to-host read
with gpu.write() as edit:   # batching control for debugging is also available
    edit[0, 0] = 3
```

Basic integer, negative, slice and ellipsis indexing is supported. Row/column
slices are live ArrayView objects, so `a[0][1] = value` updates the Array.
Views retain the original shared ArrayStorage after a parent Reset or
destruction, including logical metadata, CUDA CSR offsets and the controller.
Nested slices retain their own flat handle to that record. An idle view holds
no lease; each read/write acquires one. Fancy indexing, boolean masks and new
axes are omitted.
Ragged indexing supports a single row or a rectangle of equal-width rows.
These debugging views are not row-transfer payloads; SoA remains separate
Arrays sharing a route.

GPU indexing synchronizes and copies the selected elements; it is intended
for debugging, not performance kernels. Explicit live leases and transfer
operations still enforce conflicts, including on GPU. A NumPy snapshot cannot
mutate the Array. There is no untracked writable buffer export or implicit
DLPack bridge. Kernel integrations use a retained lease and explicit
`unsafe_address()`/completion with the usual borrowed-pointer responsibility.

MPIInfo, RowRequest, RowTransferPlan, ElementSchema (including mpi4py
`datatype.py2f()`), PayloadPlan, TransferBinding/Operation, reducers, device
copies and serializer operations are exposed. Python custom reducers receive
independent NumPy contribution/current rows and return the combined row.
ArrayPair preserves separate owned/ghost controllers.

## Geom and legacy compatibility

Geom continues to execute its existing local/distributed permutation machinery.
`NativeRowTransfer` extracts that exact correspondence for reuse.
`RegisterArray` migrates an owned dense/fixed/ragged native Array and publishes
the replacement only after its transfer finishes. Ghosts and connectivity
references are rebuilt/remapped by the appropriate Geom caller. Stable IDs and
histories are ordinary registered fields; opaque field values are not
automatically interpreted as indices.

Mesh mutation/version scopes remain at the guarded reorder boundaries.
Other legacy mesh mutations and raw legacy Array writes are not automatically
tracked. Preserve existing interfaces during migration and explicitly guard
new callers. In-place mesh mutation is allowed under its version scope;
field migration retains old allocations through the transition.

Validation commands and actual results belong in
[the phase report](../dev/array_geom_native_phase_2026-10-21.md).
