# Runtime allocation diagnostics

The Cauchy frontend enables tracking before mesh construction when
`output.diagnostics.allocations=true`. `allocationWarmupSteps` defaults to one
accepted step. Reports are `prefix-memory-R<rank>-S<step>.txt` and
`prefix-memory-R<rank>-final.txt`, emitted at host output boundaries without a
new collective. These are process-local totals, so colocated domains share the
same accounting. Injected meshes allocated before enabling tracking are not
retroactively discovered; enable tracking before constructing them.

Enable `Runtime::AllocationDiagnostics::Enable()` before solver setup. The
process-local registry observes subsequent Buffer allocations/imports; it does
not discover earlier allocations retrospectively. Disabling it clears records,
not storage. Neither enabling nor taking a snapshot calls MPI or synchronizes
CUDA. Allocation and final-release hooks are outside numerical element access.

```cpp
Runtime::AllocationDiagnostics::Enable();
Runtime::Array<real> state(Runtime::RowLayout::Dense(nOwned, 5));
state.SetName("D0/U/owned", DNDS_ALLOCATION_SITE());
// After warm-up and at a controller observation boundary:
Runtime::AllocationDiagnostics::Snapshot().Dump(std::cout);
```

`Array::Name` is an optional logical-storage label shared by aliases, including
Python aliases. Naming does not change values, shape or generations. Diagnostic
records keep all observed names for an allocation identity and count its
capacity once. `DNDS_ALLOCATION_SITE()` captures the calling function/file/line;
put it at the caller rather than in a default argument. Explicit `Array::To`
and `Clone` preserve a nonempty source label on the new allocation. Callers
can then assign a new role; recorded aliases still count the capacity once.
Other builders must name their newly allocated arrays explicitly.

For a provenance audit, set `DNDS_RUNTIME_ALLOCATION_STACKS=1` before enabling
tracking, or call `AllocationDiagnostics::CaptureStacks(true)`. Each subsequent
allocation records its creation stack, including unnamed allocations. Dumps
emit `allocation_stack identity=...` blocks and the process ID on Unix hosts.
Stack capture is off by default and incurs symbolization/setup overhead: do
not use a stack-enabled run as a performance measurement. It changes no kernel
arithmetic or allocation capacity. Creation paths identify where storage was
produced; they are not a complete enumeration of current sharing handles.
Use them together with the owning module's retained fields/read scopes to
attribute live allocations. Disabling tracking clears stacks and resets capture.

This registry does not call NVML, inspect other PIDs, or query whole-device
usage. Its records are created only by this process's Runtime Buffer lifecycle
hooks. A separate `nvidia-smi` reading can include other processes; a Runtime
report cannot, even when its arrays have no diagnostic name.

Records do not retain allocations. The activity callback captures only a weak
BufferState reference and reports current readers/writers and queued uses at
snapshot time. A submitted operation or a read scope can retain an allocation;
the diagnostics cannot. Snapshots are observations, not consistent distributed
checkpoints or access permissions. Labels and activity never authorize use.

Reported bytes are **declared addressable allocation capacities**, including
Array layout padding, not RSS or allocator pool reservation. CUDA alignment
over-allocation, provider-private AMG/NCCL storage, allocator caches and physical
page residency may add memory outside these capacities. An ordered release can
drop logical ownership before the device executes its queued free; this snapshot
does not instrument cudaFreeAsync completion or claim those bytes have already
returned to the pool. Pending read/write counts cover retained Runtime uses.
The solver-level warm-up report labels provider-private memory as unknown.
Array totals are never a substitute for independent device-memory measurements.

Python exposes `Runtime.enable_allocation_diagnostics()`,
`Runtime.allocation_snapshot()` and `array.name`. Snapshots expose allocation
identity, device, capacity, labels, source site and activity; they contain no
payload and can outlive the original Arrays safely.
