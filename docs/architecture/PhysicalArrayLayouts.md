# Physical Runtime Array layouts and borrowed access

`Runtime::Array<T>` owns a shared, type-erased `ArrayStorage`; it is not
instantiated for every width, tile size or physical format. The optional legacy
second alias argument does not constrain the allocation. A policy is selected
when acquiring a view, and validated before the hot loop.

```mermaid
flowchart TD
  A[Array T handles] --> S[shared ArrayStorage]
  P[PythonArray handle] --> S
  S --> B[Buffer range and access controller]
  B --> R[shared backend allocation owner]
  S --> M[shared immutable layout metadata]
  M --> Q[retained offsets and row sizes when needed]
  S --> L[access scope retains values and metadata]
  L --> V[borrowed ArrayView Policy]
  V --> W[borrowed RowSpan with component stride]
```

The access scope owns the lifetime. Copying a plain view or row does not extend
it. Device views contain only device-callable indexing and plain descriptors;
no shared ownership, mutex or allocation lookup is performed per element.

## Uniform physical formats

For logical row `i`, component `j`, width `C`, row count `N`:

| Format | Offset in elements | Physical extent | Component stride |
|---|---|---|---|
| `RowLayout::Dense(N,C,P)` | `i*P+j` | `N*P` | 1 |
| `RowLayout::ColMajor(N,C,options)` | `i+j*P` | `C*P` | P |
| `RowLayout::TiledColMajor(N,C,W)` | `(i/W)*C*W+(i%W)+j*W` | `ceil(N/W)*C*W` | W |

Column-major padding chooses `P=RoundUp(N+extraRows,rowPaddingMultiple)`;
zero rows normally allocate zero payload. `ColMajorWithPitch` imports an actual
pitch. Tiled tails are allocated but are not logical rows. Padding is never a
component or a DOF; loops traverse logical rows and row sizes.

Padded variable rows and CSR keep their existing formats. They are not ragged
SoA. `Stride()` is a physical format parameter (AoS row capacity, SoA plane
pitch or AoSoA tile size); generic code must not interpret it as an AoS row
stride. Use `ElementOffset` for storage adapters and `view(i,j)`/`view[i][j]`
for numerical element access.

```cpp
using namespace DNDS::Runtime;
Array<double> u(RowLayout::ColMajor(nCell, 5, {32, 1}));
auto write = u.Write();
auto rows = write.View<ColMajorTable>(); // placement/shape policy check once
for (DNDS::index iCell = 0; iCell < rows.RowCount(); ++iCell)
    rows[iCell][0] = 1.0;
```

`FixedWidthRows<ColMajorTable,5>` fixes logical width while retaining a dynamic
plane pitch. `TiledColMajorTable<32>` fixes the component stride. `TableRows`
retains a compile-time component stride of1; its existing `FixedRows<N>` and
static-capacity variants retain their AoS indexing. A projection that would
misrepresent physical placement fails before access. `UniformTableRows` accepts
any uniform physical format; `RowLayout` is the fully generic policy including
CSR and padded rows.

## Dense adapters and alignment

`DNDS/Algebra/EigenView.hpp` constructs borrowed column-major matrices from each
logical row. The adapter validates host placement and matrix capacity once and
passes the actual component stride to Eigen. This is not a payload copy or a
new numerical owner. Keep the originating lease while using a matrix map.
Raw row pointers are reserved for these centralized dense/interop/backend
adapters; ordinary numerical loops use stride-aware indexing.

An `AllocationOptions{alignmentBytes}` request controls the allocation base,
independently of table padding. A nonzero value must be a power of two. The
native CPU provider uses matching aligned allocation/deallocation; the native
CUDA provider retains the true allocation base when exposing an aligned
interior payload address. Imported pointers must already meet their advertised
type/alignment contract. Custom providers either honor the request or fail.
Row/column starts are aligned only when both the base and the physical pitch in
bytes meet the required alignment. Default Eigen maps are explicitly unaligned.

## Bounds and execution

Element assertions are a compile-time policy. `View<Policy,true>()` is checked;
`View<Policy,false>()` removes bounds checks. The configured default is controlled
by `DNDS_RUNTIME_ARRAY_BOUNDS_CHECKS`; structural/placement/lease validation is
never removed by that switch. Unchecked callers prove stencil bounds and stream
ordering. The view carries no cross-stream dependency ledger.
