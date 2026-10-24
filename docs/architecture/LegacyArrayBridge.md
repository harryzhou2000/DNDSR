# Legacy Array host bridge

`DNDS/Runtime/LegacyArrayBridge.hpp` provides the narrow compatibility boundary
between `DNDS::Array` and `DNDS::Runtime::Array`. Existing Array construction,
copy, assignment, `CopyData` and raw row access keep their existing contracts.
Value copies remain deep; explicit sharing uses a named action.

```cpp
auto native = DNDS::Runtime::ShareLegacyHost(legacy);
{
    auto access = DNDS::Runtime::HostRowsWrite(native);
    auto rows = access.View();
    for (DNDS::index i = 0; i < rows.Size(); ++i)
        for (DNDS::rowsize j = 0; j < rows.RowSize(i); ++j)
            rows(i, j) *= 2;
}
{
    auto access = DNDS::Runtime::HostRowsRead(std::as_const(legacy));
    auto rows = access.NativeView(); // existing Runtime::ArrayView<const T>
    // Use rows only while access remains active.
}
```

`ShareLegacyHost` accepts a mutable source because the returned native handle
permits writes. `HostRowsRead` also accepts a const legacy source. Its scope
retains both immutable row metadata and a read lease; the corresponding write
scope uses the same allocation's write lease. Values edited through either side
are visible through the other side's next permitted access. Competing managed
writers/readers fail at acquisition through the existing Buffer controller.

`LegacyRowsView<T>` is a trivially copyable pointer/shape bundle with
`Size`, `RowSize`, `operator[]`, and `operator()`. Its query methods are
`DNDS_DEVICE_CALLABLE`; ownership, host placement checks and extent validation
remain outside loops. The host bridge does not acquire device storage. Device
callers can construct the same view bundle from a separately acquired native
device `ArrayView`, retaining its scope until actual device completion.

The bridge reuses the exact existing host Buffer, including its identity,
byte range and access controller. It never imports the raw address into another
controller. Static and runtime uniform tables retain their physical stride.
Padded rows retain capacity and copy/convert their `rowsize` lengths once into
immutable `size_t` metadata. Compressed CSR copies/converts validated `index`
offsets through row lengths into immutable `size_t` offsets. This metadata
conversion is a setup cost; value sharing does not copy values. Negative sizes,
decreasing offsets, invalid sentinels, overflowing extents, wrong metadata counts
and allocation/layout byte mismatches are rejected. Uncompressed CSR must be
explicitly published with `Compress` before flat sharing; a const read never
compresses it. PODs retain their actual C++ type and alignment. An optional
`ElementSchema` overload checks the MPI extent and registered semantic type;
untagged custom schema correctness remains its caller's responsibility.

Each alias publishes a layout snapshot. Legacy resize, assignment, swap,
decompression/recompression, clearing and destruction never retarget a retained
alias to a replacement allocation. A padded length edit leaves old aliases with
their old immutable lengths; acquire a new alias for the new layout. Native
`Reset` similarly replaces its handle and retains old scopes. Native
`ExclusiveRows` refuses in-place shape editing while the legacy allocation owner
or another alias survives. After detaching/releasing old owners, a new native
shape publication is independent of legacy state. A caller managing mesh epochs
must rebuild layout-dependent bindings after structural change; a retained
snapshot is not a live mirror of the source's current shape.

`getRowSizes()` still exposes legacy mutable metadata. A caller can therefore
publish a new padded row length over the same value allocation while an old
native alias retains its old length. The two aliases share values, including
padding; they do not share the meaning of which columns are logical entries.
That operation must occur at the setup phase barrier and invalidate the caller's
mesh/layout epoch. It does not grant permission to write the shared values while
an old reader runs. Legacy `Resize` instead publishes a fresh value allocation,
even when its dimensions happen to be unchanged. Do not infer shape identity
from Buffer identity alone: compare the publication/mesh epoch as well.

Empty uniform and padded arrays retain their existing tracked zero-byte
allocation. Default/empty CSR is initially an unpublished row-builder state;
explicit `Compress()` publishes its zero sentinel before sharing. Cleared or
moved-from vectors have no current allocation and reject sharing until resized
or assigned. A retained old alias remains usable independently of that state.

Separate legacy device mirrors are rejected on host alias acquisition until
`to_host` and `clear_device` have been called explicitly. Creating a non-Host
legacy mirror is rejected while native host aliases or access leases retain that
Buffer. The legacy Host backend aliases host memory and keeps the existing
contract. CSR decompression now takes a managed read lease before copying old
values, so it cannot race a managed native writer. There is no automatic
coherence between independent legacy and native CUDA allocations.

Legacy raw pointers, element writes, metadata escapes and device kernel views
remain phase-barrier responsibilities. Such writes bypass Buffer leases; do not
mix them with concurrent managed native work. Structural acquisition/rebinding
is setup work and must be externally serialized. The bridge does not turn every
legacy Array into a native storage wrapper, provide an owning reverse adapter,
or combine owned and ghost allocations. Pair owners adapt each side separately.

`test/cpp/DNDS/test_LegacyArrayBridge.cpp` exercises the supported layouts,
retained snapshots, controller conflicts, invalid publication, copy semantics
and registered POD extent checks.
