# E5 actual RHS and E6 nonlinear-update assessment

**Retain Direct transfers and the existing production executor.** Dedicated
MPI ownership is correct in this bounded SERIALIZED experiment, but its small
RHS benefit does not justify adding a production communication worker. Retain
assembled approximate Jacobians for the measured five-SGS-update workload.

BSSCA job `12443109`, exact source `9ccb3dbdbe80385dab870f12215b8f046f1d014b`,
completed `0:0` in **4m28s**. All **25 rows / 138 records** passed, with raw-log
hashes and records independently rechecked after fetching. Slurm selected
`amd_512`, node `o2702`: one exclusive node, 32 ranks, two physical cores/rank,
180G, OMP1/OPENBLAS1. Each case ran serially in this allocation. The requested
partitions included amd_256/amd_512/amd_1T/amd_2T; no node was pinned. The
independent 80-node job remained untouched. Login build used 16 workers and
completed without retry. GCC12.2/OpenMPI3.1.4/UCX, Release, assertions retained,
Runtime element bounds checks off, actual MPI_THREAD_SERIALIZED.

The [protocol](../rhs-stage-protocol.md) describes the public production
operators, benchmark-only adapters, ownership and validation. All work is
P3/intOrder5, five fixed Jacobi sweeps. Small flow5 has 32 cells/rank; both
large profiles have 5,000 cells/rank. Perfect-gas flow5 uses viscosity .001
and no limiter. The reacting profile uses actual mixture transport/chemistry,
separate flow5 and nine independent-species arrays, and WBAP for RHS timing.
The nonlinear profiles use no limiter. Both use Rusanov, polynomial viscous
gradients, explicit nonunit mixture scales, no PP and no face fallback. These
are generated frozen workloads, not physical time-integration experiments.

## E5 complete RHS

Median maximum-rank **milliseconds**, three uninstrumented trials, two complete
RHS calls per trial; warmup/reset/validation excluded:

| execution | transport | small flow5 | large flow5 | large reacting/WBAP |
| --- | --- | ---: | ---: | ---: |
| Original production/inline | Direct | 1.1125 | 104.285 | 1181.627 |
| Caller adapter | Direct | 1.1124 | 104.260 | 1169.837 |
| Compute worker | Direct | 1.1121 | 101.525 | 1160.462 |
| MPI worker | Direct | 1.0916 | 103.864 | 1167.855 |
| Caller adapter | HostPacked | 1.1229 | 106.584 | 1192.766 |
| Compute worker | HostPacked | 1.6033 | 103.846 | 1168.838 |
| MPI worker | HostPacked | 1.0947 | 103.093 | 1161.829 |

Against the matching HostPacked caller, compute/MPI workers improve large
flow5 by **2.57%/3.28%** and reacting RHS by **2.01%/2.59%**. Direct worker
gains are small: large flow5 improves about2.62%/0.38%, reacting about0.80%/0.17%
against the caller adapter. No general direct-transfer winner is established.
Small HostPacked compute-worker regresses substantially: 1.41–1.62ms versus
the caller's 1.117–1.125ms. Large Direct compute-worker flow5 also varies
101.33–107.46ms, overlapping caller104.02–104.29ms. Do not derive an automatic
dispatch threshold from three trials on one node.

The original control avoids public-provider substitution entirely. Its large
flow5 timing agrees closely with caller Direct; reacting original is about1%
slower than the adapter in this order. Adapter setup and separate communicators
therefore remain a limit on attributing every small difference to scheduling.
No production adoption or universal communication speedup is claimed.

All outputs compare elementwise **exactly** against the original Direct/inline
workspace after identical zero-start reconstruction work. Owned mean versions
remain unchanged. There are7 starts/RHS for flow5 and20 for reacting/WBAP,
counting both physical fields, with5/10 VR core calls respectively. Workspaces,
bindings, requests and worker lanes drain before MPI_Finalize. The MPI worker
owns request progression and hot MPI control RPCs; controller resource setup
and teardown happen without active payload requests. Every mode reserves the
same two physical cores; caller-only deliberately leaves one idle.

### Staging and scheduling attribution

Instrumented median per-RHS phase maxima in **milliseconds** for HostPacked;
phase maxima can come from different ranks and must not be summed:

| profile; placement | full-buffer copies | MPI Pack | post | progress | MPI Unpack | submit | host wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flow5; caller | 1.780 | .432 | .053 | 4.650 | .286 | 2.020 | 4.998 |
| flow5; compute worker | 1.969 | .402 | .095 | 9.356 | .178 | 2.301 | 9.554 |
| flow5; MPI worker | 1.629 | .419 | .115 | 10.053 | .171 | .119 | .003 |
| reacting; caller | 11.540 | 1.945 | .604 | 26.883 | .839 | 13.132 | 28.408 |
| reacting; compute worker | 10.110 | 1.686 | .201 | 84.037 | 1.151 | 11.562 | 85.384 |
| reacting; MPI worker | 10.620 | 1.921 | .744 | 60.579 | 1.326 | .105 | .012 |

Flow5 HostPacked copies14,528,800 logical bytes/RHS in21 CopyBytes calls versus
1,764,400 received payload bytes; reacting copies60,651,136 bytes in60 calls
versus7,365,568 received bytes. Direct has zero explicit CopyBytes staging or
MPI Pack/Unpack calls; MPI may internally pack derived datatypes. Original
control records zero adapter payload counters because it is not substituted;
that does not mean it sends no data.

MPI-worker submit time is lower, while its queue/progress work moves off the
controller. Progress/host-wait measurements include MPI scheduling and rank
imbalance; a larger progress clock under overlap is not an independent amount
of exposed communication to subtract from RHS time. The near-zero worker host
wait is observation after background completion, not zero communication cost.

Timed VR-core work intersects pending-transfer intervals in every adapter.
Caller intersections can span the full core call because no one polls until
the later wait; they do not prove network progress during computation. Worker
traces include producer/enqueue/post/receive/ghost-ready times, but host interval
intersection still is not a NIC utilization measurement. Only the completed
uninstrumented RHS times above establish net benefit. Timer-on/off records are
retained; the original reacting control differs about1% between their medians.

## E6 complete nonlinear update

Each call performs exactly **one capped correction, five SGS iterations and
two RHS evaluations**, mass1e6, scalar diagonal, no PP, fixed P3 reconstruction.
The manufactured source is immutable; perturbed mean states and coefficients
reset outside timing. Zero tolerances intentionally prohibit claiming convergence.
Median maximum-rank milliseconds:

| profile; storage | complete call | residual | Jacobian preparation | linear solve |
| --- | ---: | ---: | ---: | ---: |
| flow5; assembled | 256.078 | 206.837 | 22.001 | 22.625 |
| flow5; evaluated | 400.168 | 203.519 | 3.989 | 184.422 |
| reacting; assembled | 3803.223 | 2413.469 | 1049.123 | 156.013 |
| reacting; evaluated | 4409.138 | 2407.267 | 996.663 | 824.931 |

Assembled improves complete update cost by36.0% for flow5 and13.7% for reacting
in this workload. Evaluated preparation is cheaper but does not repay repeated
block evaluation in these five SGS iterations. Reacting source Jacobians retain
their reconstructed-quadrature producer; their cost is not an off-diagonal
storage optimization target. Profiles retain all eight existing disjoint phase
clocks; displayed maxima/medians are not additive.

All four assembled/evaluated complement solves have exactly identical complete
fields, maximum absolute difference0. Repeated reset solves match, input source
and reference states remain unchanged, and the energy-defect ratio decreases
to about4.62e-4 for flow5 and4.21e-4 for reacting (rank0 diagnostic). These are
bounded correction/oracle checks, not converged physical-stage acceptance.

## Disposition and reproducibility

This closes E5's requested representative composed-RHS/staging attribution and
E6's additional nonlinear-update timing with a retain recommendation. Existing
production executor, stable bindings, separate starts, Direct default and
assembled storage remain. Optional peer coalescing, early-ready-field grouping
delay, cross-node workers, general error recovery, production MPI execution
and fresh CUDA/capture qualification remain future work; none is inferred.

`summary.json` retains all trial ranges, exact source/cache/binary/mechanism
hashes and independent raw-record verification. `records.tar.gz` contains the
25-row manifest/logs, submitted script, exact command, flags/linkage, scheduler
records and compact local gates including the initial unsupported-default and
missing-interpreter failures. `reduce.py` independently rechecks raw hashes,
JSON identities and protocol assertions; run it from the measured source root.
`SHA256SUMS` covers these committed artifacts. The original build/configure
logs remain in the ignored local evidence tree and remote task directory;
their hashes are retained without committing verbose compiler diagnostics.
