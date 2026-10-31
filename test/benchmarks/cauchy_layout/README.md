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
rows, components, timed samples, layout, optional CUDA device ordinal. Layouts
are 0=AoS, 1=SoA, 8/32=AoSoA tile width, 101=AoS flow5 + SoA extensions,
132=AoS flow5 + AoSoA32 extensions; -1 runs all. Compute uses one rank. CUDA
communication defaults to device ordinal equal to rank, so use one rank/device.
Do not run on busy GPUs. `run.py` rotates layout order across process-level trials
and writes commands, exit status, binary hash, raw CSV and compact manifests.

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
  max scaled error, source+output bytes (including tile padding), device seconds.
- Comm/pipeline: kind, backend, layout, N, C, pattern/transport/schedule, sample,
  total seconds, pack seconds, exchange-window seconds, post-exchange seconds,
  bytes/peer, send datatype segments/peer, setup seconds, MPI ranks.

Distributed times are independent maximum-rank values; do not add their medians
as if the maximizing rank were the same. For pipeline, the exchange window can
include overlapped CPU core work and the post-exchange interval includes unpack
and remaining numeric work. Pure comm rows isolate transport. MPI uses transient
requests here; persistent/native Runtime wrapper costs are not measured. No
production transport/library default should be changed from a single probe.
