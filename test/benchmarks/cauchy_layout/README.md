# Cauchy layout probe

Standalone FP64 simulated kernels; no production Array, CFV or solver changes.
The assessment plan/results live in the separate `workspace/dndsr-next-plan`
repository under `experiments/cauchy-layout`.

```sh
cmake -S test/benchmarks/cauchy_layout -B build-layout-probe \
  -DCAUCHY_LAYOUT_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-layout-probe -j4
build-layout-probe/cauchy_layout compute cpu 257 32 1 -1
build-layout-probe/cauchy_layout_cuda compute cuda 257 32 1 -1 1
mpirun -np 4 build-layout-probe/cauchy_layout comm cpu 1027 7 1 -1
```

Arguments: mode (`compute`, `comm`, `pipeline`), device (`cpu`, `cuda`), logical
rows, components, timed samples, layout, optional CUDA device ordinal, access
variant, and component-plane padding. Layouts
are 0=AoS, 1=SoA, 8/32=AoSoA tile width, 101=AoS flow5 + SoA extensions,
132=AoS flow5 + AoSoA32 extensions; -1 runs all. Compute uses one rank. CUDA
communication defaults to device ordinal equal to rank, so use one rank/device.
Do not run on busy GPUs. `run.py` rotates layout order across process-level trials
and writes commands, exit status, binary hash, raw CSV and compact manifests.

GPU compute access variants (the default0 preserves the original path):

| ID | Access implementation |
|---|---|
| 0 | Direct global access, one thread/cell |
| 1 | Cooperative shared-memory tiles; shared current/neighbor vectors for stencil |
| 2 | Double-buffered `cp.async` prefetch, four tiles/block |
| 3 | Same four-tile/double-buffer arrangement using synchronous loads |
| 8 / 32 | 8-/32-thread group per cell, component-parallel accesses and broadcasts |
| 64 / 128 / 256 | One block/cell with the specified thread count |
| 1256 | Block256 with an artificial shared reservation permitting only one resident block/SM |

For example, `... compute cuda 257 32 1 -1 1 2` checks every row of every layout
with async pipelining on GPU1. `run.py --variants 0,1,2,3,8,32` rotates layout/variant
pairs; variants require CUDA compute and an explicit `--device` ordinal. The new
`gpu_access.cuh` contains these benchmark-only implementations. CPU and MPI modes
retain variant0. `run.py --padding 0,1,8,17,64` holds logical N fixed while
varying only SoA component-plane pitch (including hybrid SoA extensions).
Other layouts run once per variant, with their ordinary tile padding. Report
the best repeated padded median and its extra bytes. A per-kernel best padding
is a tuning envelope; a shared field still needs one chosen physical pitch.
No general execution engine or production Array API is added.

Shared cell tiles use64 rows with odd component pitch. K2 uses component groups
and stages seven input vectors per row. Pipeline candidates retain two input
tiles plus one output tile; waits/barriers are device-side. All tail threads
participate in required warp collectives/block barriers before conditional stores.
Async input uses8-byte hardware copies; output still uses regular cooperative
shared-to-global stores. Resource descriptors on stderr report registers, local
memory, dynamic shared bytes, and the occupancy API block limit, not achieved
occupancy. The original mathematical body is unchanged for direct/shared cell
kernels; cooperative formulas are independently checked against that body.

K1 computes a gas/transport-like cell update including sqrt/power/exponential.
K2 gathers six irregular neighbors with matrix/limiter-like arithmetic. K3 applies
coupled flow triangular solves and flow/extension coupling. Inputs are immutable
and each timed iteration overwrites distinct output. These are access-pattern
probes, not numerical-method accuracy or solver-convergence tests. Matrix
coefficients are generated from a small fixed formula, not stored per edge.
Kernel dispatch is uniform across a launch; layout indexing is templated.

The layout oracle uses the same arithmetic with a pointer-free analytic input
generator and a contiguous small output. It validates indexing, not independently
the physics. It checks 257 distributed logical rows (all rows for N<=257).
N=257 tests every tile lane and the tail. CUDA warmup precedes timing. Device
events measure repeated launches; host wall time additionally includes submission
and completion overhead. CPU timing uses two applications/sample, CUDA twelve;
each row reports per-application time, not the repeated batch total.

Communication uses a periodic 3D Cartesian rank graph from MPI_Dims_create,
deduplicating self/duplicate neighbors. All peers send the same selected source
rows into distinct destination row ranges; this intentionally exercises repeated
source ownership without receive collisions. Source selection is contiguous or
deterministically scattered. Every received logical row/component is checked
exactly outside timing, using source rank and requested source-row identity.

Direct MPI typemaps are compiled in physical source order with corresponding
destination indices, coalescing adjacent sections. Packed transport explicitly
gathers a row-major packet then scatters after a contiguous MPI transfer. The
wire orders can differ while the logical result stays identical. GPU packing
uses device scratch and CUDA kernels; portable MPI has host-observed producer
and receiver boundaries. No host payload staging is added by this benchmark.
MPI may internally stage. Setup is excluded from steady transfer time and
reported separately. Requested sizes are 64KiB and 8MiB **per peer**, capped by
the available halo rows; use the actual reported byte count in comparisons.
Endpoints use the same layout within each trial. This is not a general
cross-layout wire-format adapter; a production payload plan must agree the
logical wire ordering explicitly when endpoint layouts differ.

Pipeline runs the same source, core and dependent arithmetic serially or with
core work after MPI posting. CPU polls requests between 4096-row tiles; GPU uses
nonblocking compute/packing streams and host MPI progress. This is not an eager
Runtime integration test. Dependent work starts after destination readiness.
Inputs use source-rank/row values; sampled output is checked analytically.

CSV fields (no header):

- Compute: kind, backend, layout, N, C, kernel, sample, host seconds,
  max scaled error, source+output bytes (including tile padding), device seconds,
  access variant, SoA component-plane padding in elements. Older follow-up
  files omit the last field, which means padding0.
- Comm/pipeline: kind, backend, layout, N, C, pattern/transport/schedule, sample,
  total seconds, pack seconds, exchange-window seconds, post-exchange seconds,
  bytes/peer, send datatype segments/peer, setup seconds, MPI ranks.

Distributed times are independent maximum-rank values; do not add their medians
as if the maximizing rank were the same. For pipeline, the exchange window can
include overlapped CPU core work and the post-exchange interval includes unpack
and remaining numeric work. Pure comm rows isolate transport. MPI uses transient
requests here; persistent/native Runtime wrapper costs are not measured. No
production transport/library default should be changed from a single probe.

## Species extension probe

`cauchy_species_cuda` adds two simulated FP64 kernels with16/32 species plus
five flow components: mixture reductions with temperature-dependent transport/
rates, and dense well-conditioned species matrix action. These are arithmetic
probes, without an external mechanism or chemical accuracy claim. The scalar
oracle shares formulas but uses analytic input rather than the measured layout.
N257 checks all logical rows; larger timings sample257 rows outside timing.

```sh
build-layout-probe/cauchy_species_cuda 257 16 1 1 2 1 1 17
python3 test/benchmarks/cauchy_layout/run_species.py \
  --exe build-layout-probe/cauchy_species_cuda --output data/species-probe \
  --species 16,32 --padding 0,1,8,17,64 --device 1
```

Arguments: logical rows, species count, layout (0/1/8/32), split (0/1), access
variant (0/1/2/8/32/128), timed samples, device ordinal, optional SoA plane padding.
Split mode uses genuinely separate AoS flow and selectable-layout species
allocations. Full mode uses one state allocation. Both update flow and species.
Variants are direct, shared tiles, double-buffered async tiles, group8, group32,
and block128. One-block/SM is not used for species. Timed batches use four
launches; allocation, initialization, validation and host copies are excluded.
Species CSV: kind, layout ID, split, N, species count, variant, kernel, sample,
device seconds, host seconds, max scaled error, combined input/output bytes,
SoA plane padding. Split allocation communication is not measured here.
The `cauchy_species` CPU target includes the same mathematical/layout source
with direct row loops and no CUDA/MPI dependency. Use `run_species.py --backend
cpu --variants 0 --device 0 --launcher '<pinned launcher>'`; its prefix is
`species-cpu`, device seconds are0 and host seconds time two applications/sample.
